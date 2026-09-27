#' Find points within a projected Euclidean radius
#'
#' @param x,y Two-column numeric matrices or projected `sf`/`sfc` XY POINT
#'   inputs. `y` defaults to `x`.
#' @param dist One finite, non-negative numeric distance. For matrices it is
#'   interpreted in the coordinate units. For spatial inputs it may be numeric
#'   in the CRS units, or a scalar `units` value convertible to those units.
#' @param backend `"cpu"` (default), `"cuda"`, or `"metal"`. GPU backends
#'   must be explicitly selected and error when unavailable.
#' @param max_output_bytes Inclusive cap on the output accounting: eight bytes
#'   per `x` list slot plus four bytes per returned `y` index. R object headers
#'   and temporary collection workspace are excluded. The default is \code{2^53} (unrestricted).
#' @param tile_bytes Inclusive GPU execution-tile budget, including inputs,
#'   output, and scratch. The default is 64 MiB; 49 bytes is the minimum for one
#'   candidate pair.
#' @return A list with one integer vector per row of `x`. Each vector contains
#'   the increasing, one-based indices of rows in `y` whose Euclidean distance
#'   is less than or equal to `dist`. Equal-coordinate duplicates are included.
#'   The list is named with the row names of `x`, when present.
#' @export
sfgpu_within_distance <- function(x, y = x, dist,
                                  backend = c("cpu", "cuda", "metal"),
                                  max_output_bytes = 2^53,
                                  tile_bytes = 64 * 1024^2) {
  backend <- match.arg(backend)
  max_output_bytes <- .sfgpu_bytes(max_output_bytes, "max_output_bytes", 1)
  tile_bytes <- .sfgpu_bytes(tile_bytes, "tile_bytes", 49)
  inputs <- .sfgpu_prepare_inputs(x, y)
  radius <- .sfgpu_radius(dist, inputs)
  if (backend == "cuda") cuda_ensure()
  .Call(C_sfgpu_within_distance, inputs$x_mat, inputs$y_mat, backend,
        radius, max_output_bytes, tile_bytes)
}

.sfgpu_radius <- function(dist, inputs) {
  if (inherits(dist, "units")) {
    if (!inputs$spatial) {
      stop("a units distance requires spatial inputs", call. = FALSE)
    }
    if (length(dist) != 1L || is.na(dist) || !is.finite(dist)) {
      stop("dist must be one finite non-negative distance", call. = FALSE)
    }
    dist <- tryCatch(
      units::set_units(dist, base::units(inputs$x_crs$ud_unit),
                       mode = "standard"),
      error = function(e) {
        stop("dist units must be convertible to the spatial CRS units", call. = FALSE)
      }
    )
    dist <- units::drop_units(dist)
  }
  if (!typeof(dist) %in% c("double", "integer") || length(dist) != 1L || is.na(dist) ||
      !is.finite(dist) || dist < 0) {
    stop("dist must be one finite non-negative numeric distance", call. = FALSE)
  }
  as.double(dist)
}
