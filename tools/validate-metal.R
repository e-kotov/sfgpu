# Independent numerical holdouts for a required Metal installation.
# CPU hypot is the binary64 subtraction reference; analytic grids and exact
# powers of two avoid relying only on a second implementation of the shader.
library(sfgpu)
info <- sfgpu_backends()$metal
stopifnot(isTRUE(info$compiled), isTRUE(info$available), !is.na(info$device))
cat("METAL_DEVICE=", info$device, "\n", sep = "")

stats <- function() {
  .Call(get("C_sfgpu_metal_stats", envir = asNamespace("sfgpu")))
}
checked <- 0L
gpu_pairs <- 0
cpu_pairs <- 0
check <- function(x, y, tile_bytes = 64 * 1024^2, require_gpu = FALSE) {
  reference <- sfgpu_distance(x, y, backend = "cpu")
  actual <- sfgpu_distance(x, y, backend = "metal", tile_bytes = tile_bytes)
  accounting <- stats()
  stopifnot(identical(dim(actual), dim(reference)),
            identical(is.infinite(actual), is.infinite(reference)),
            !anyNA(actual), all(actual >= 0))
  finite <- is.finite(reference)
  normal <- finite & reference >= .Machine$double.xmin
  zero <- finite & reference == 0
  # Relative checking here is deliberately tighter than the public absolute
  # floor for distances < 1; a float32 or double-single substitute must fail.
  if (!all(abs(actual[normal] / reference[normal] - 1) <=
           32 * .Machine$double.eps) || !all(actual[zero] == 0)) {
    stop("METAL_NUMERIC_MISMATCH")
  }
  tiny <- finite & !normal & !zero
  stopifnot(identical(as.numeric(actual[tiny]), as.numeric(reference[tiny])))
  stopifnot(accounting$gpu_pairs + accounting$cpu_pairs == length(actual))
  if (require_gpu) stopifnot(accounting$gpu_pairs >= 0.99 * length(actual))
  gpu_pairs <<- gpu_pairs + accounting$gpu_pairs
  cpu_pairs <<- cpu_pairs + accounting$cpu_pairs
  checked <<- checked + length(actual)
  invisible(actual)
}

set.seed(270926)
# Random, nonsymmetric projected coordinates plus centimetre-scale differences.
x <- cbind(5e6 + runif(37, -1000, 1000), 4e6 + runif(37, -1000, 1000))
y <- cbind(5e6 + runif(29, -1000, 1000), 4e6 + runif(29, -1000, 1000))
check(x, y, require_gpu = TRUE)
check(x, y, tile_bytes = 1024, require_gpu = TRUE)
check(y, x, tile_bytes = 88, require_gpu = TRUE)
check(rbind(c(1e7, 1e7)), rbind(c(1e7 + 0.01, 1e7 + 0.02)), require_gpu = TRUE)
# Force fine fractional roots, zero roots, sign changes and power-of-two scaling.
for (exponent in c(-1000, -700, -300, -40, 0, 40, 300, 700, 1000)) {
  scale <- 2^exponent
  a <- rbind(c(0, 0), c(3, 4), c(-2, 7), c(11, -5)) * scale
  b <- rbind(c(0, 0), c(1, 1), c(6, 8), c(-7, 3), c(11, -5)) * scale
  check(a, b, tile_bytes = 1024, require_gpu = abs(exponent) <= 800)
}
# General binary64 inputs within a narrow exponent band, across many scales.
for (exponent in seq(-900, 900, by = 100)) {
  a <- matrix(runif(26, 1, 2), ncol = 2) * 2^exponent
  b <- matrix(runif(22, 1, 2), ncol = 2) * 2^exponent
  check(a, b, require_gpu = abs(exponent) <= 800)
}
# Encoding rejection and boundary corrections must be correct and accounted.
smallest <- .Machine$double.xmin * .Machine$double.eps
boundary <- rbind(c(0, 0), c(smallest, 0), c(3 * smallest, 4 * smallest),
                  c(.Machine$double.xmin, 0), c(1e-200, -1e200),
                  c(1.3e308, 1.3e308), c(-1.3e308, 0))
check(boundary, boundary, tile_bytes = 1024)
cat("METAL_HOLDOUT_PASS comparisons=", checked, " gpu_pairs=", gpu_pairs,
    " cpu_pairs=", cpu_pairs, "\n", sep = "")
