#' Dense Cartesian point distances
#'
#' @param x,y Two-column numeric matrices or projected `sf`/`sfc` XY POINT inputs.
#' @param backend `"cpu"` (default), `"cuda"`, or `"metal"`. GPU backends
#'   must be explicitly selected and error when they are unavailable.
#' @param max_output_bytes Inclusive cap on the returned matrix payload, in bytes.
#' @param tile_bytes Inclusive cap on a GPU execution tile's two inputs and
#'   output, in bytes.
#' @return A numeric distance matrix for matrix inputs, or a `units` matrix for
#'   spatial inputs. The dimensions are `nrow(x)` by `nrow(y)`.
#' @export
sfgpu_distance <- function(x, y = x, backend = c("cpu", "cuda", "metal"),
                           max_output_bytes = 1024^3,
                           tile_bytes = 64 * 1024^2) {
  backend <- match.arg(backend)
  max_output_bytes <- .sfgpu_bytes(max_output_bytes, "max_output_bytes", 1)
  tile_bytes <- .sfgpu_bytes(tile_bytes, "tile_bytes", 40)
  inputs <- .sfgpu_prepare_inputs(x, y)
  x_mat <- inputs$x_mat
  y_mat <- inputs$y_mat

  nx <- nrow(x_mat)
  ny <- nrow(y_mat)
  # Both dimensions are R integers. The double product is exact for all
  # outputs admitted by the scalar byte cap (at most 2^53 bytes).
  pairs <- as.double(nx) * as.double(ny)
  if (pairs > floor(max_output_bytes / 8)) {
    stop("distance matrix payload exceeds max_output_bytes", call. = FALSE)
  }
  if (pairs > .Machine$integer.max * as.double(.Machine$integer.max) ||
      pairs > 2^52) {
    stop("distance matrix exceeds R's matrix length limit", call. = FALSE)
  }

  # This call allocates the result once, then fills it directly in column-major
  # order. GPU backends are not initialized until after the payload check above.
  out <- .Call(C_sfgpu_distance, x_mat, y_mat, backend, tile_bytes)
  if (inputs$spatial) {
    units::set_units(out, base::units(inputs$x_crs$ud_unit), mode = "standard")
  } else {
    out
  }
}

.sfgpu_prepare_inputs <- function(x, y) {
  x_spatial <- inherits(x, "sf") || inherits(x, "sfc")
  y_spatial <- inherits(y, "sf") || inherits(y, "sfc")
  if (x_spatial != y_spatial) {
    stop("x and y must both be matrices or both be sf/sfc spatial inputs", call. = FALSE)
  }
  if (!x_spatial) {
    return(list(x_mat = .sfgpu_matrix(x, "x"),
                y_mat = .sfgpu_matrix(y, "y"),
                spatial = FALSE, x_crs = NULL))
  }
  if (!requireNamespace("sf", quietly = TRUE) ||
      !requireNamespace("units", quietly = TRUE)) {
    stop("sf and units are required for spatial inputs", call. = FALSE)
  }
  x_geom <- if (inherits(x, "sf")) sf::st_geometry(x) else x
  y_geom <- if (inherits(y, "sf")) sf::st_geometry(y) else y
  x_crs <- sf::st_crs(x_geom)
  y_crs <- sf::st_crs(y_geom)
  if (is.na(x_crs) || is.na(y_crs)) {
    stop("spatial inputs require a known projected CRS", call. = FALSE)
  }
  if (!isTRUE(x_crs == y_crs)) {
    stop("spatial inputs must have the same CRS", call. = FALSE)
  }
  wkt <- x_crs$wkt
  projected <- startsWith(wkt, "PROJCRS[") || startsWith(wkt, "PROJCS[") ||
    (startsWith(wkt, "BOUNDCRS[") &&
     grepl("SOURCECRS\\[[[:space:]]*(PROJCRS|PROJCS)\\[", wkt))
  if (!isTRUE(projected) ||
      !identical(sf::st_is_longlat(x_geom), FALSE) ||
      !identical(sf::st_is_longlat(y_geom), FALSE) ||
      is.null(x_crs$ud_unit)) {
    stop("spatial inputs require a projected CRS with known distance units", call. = FALSE)
  }
  list(x_mat = .sfgpu_points(x_geom, "x"),
       y_mat = .sfgpu_points(y_geom, "y"),
       spatial = TRUE, x_crs = x_crs)
}

#' Query available sfgpu backends
#'
#' @return A named list with `cpu`, `cuda`, and `metal` entries. Each contains logical
#'   `compiled` and `available` scalars and character `device` and `reason`
#'   scalars. Inapplicable strings are `NA_character_`.
#' @export
sfgpu_backends <- function() {
  cuda <- .Call(C_sfgpu_cuda_info)
  metal <- .Call(C_sfgpu_metal_info)
  list(
    cpu = list(compiled = TRUE, available = TRUE,
               device = NA_character_, reason = NA_character_),
    cuda = list(compiled = isTRUE(cuda[[1L]]),
                available = isTRUE(cuda[[2L]]),
                device = cuda[[3L]], reason = cuda[[4L]]),
    metal = list(compiled = isTRUE(metal[[1L]]),
                 available = isTRUE(metal[[2L]]),
                 device = metal[[3L]], reason = metal[[4L]])
  )
}

.sfgpu_bytes <- function(value, name, minimum) {
  if (!is.numeric(value) || length(value) != 1L || is.na(value) ||
      !is.finite(value) || value != floor(value) || value < minimum ||
      value > 2^53) {
    stop(sprintf("%s must be one finite whole number of bytes between %s and 2^53",
                 name, minimum), call. = FALSE)
  }
  as.double(value)
}

.sfgpu_matrix <- function(value, name) {
  if (!is.matrix(value) || length(dim(value)) != 2L || ncol(value) != 2L ||
      !typeof(value) %in% c("double", "integer")) {
    stop(sprintf("%s must be a two-column numeric matrix", name), call. = FALSE)
  }
  if (anyNA(value) || any(!is.finite(value))) {
    stop(sprintf("%s coordinates must be finite and non-missing", name), call. = FALSE)
  }
  if (typeof(value) == "integer") storage.mode(value) <- "double"
  value
}

.sfgpu_points <- function(geom, name) {
  if (length(geom) == 0L) return(matrix(numeric(), ncol = 2L))
  types <- as.character(sf::st_geometry_type(geom))
  if (any(types != "POINT")) {
    stop(sprintf("%s must contain only POINT geometries", name), call. = FALSE)
  }
  if (any(sf::st_is_empty(geom))) {
    stop(sprintf("%s contains an empty POINT geometry", name), call. = FALSE)
  }
  dimensions <- vapply(geom, function(point) class(point)[[1L]], character(1L))
  if (any(dimensions != "XY")) {
    stop(sprintf("%s must contain only XY POINT geometries", name), call. = FALSE)
  }
  coords <- sf::st_coordinates(geom)
  .sfgpu_matrix(coords, name)
}
