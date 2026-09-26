#!/usr/bin/env Rscript
# Deterministic IEEE-754 patterns exercise the integer-key ordering independently
# of the index scheduler. The reference computes every distance on the CPU.
library(sfgpu)
stopifnot(isTRUE(sfgpu_backends()$metal$available))
set.seed(290926)
finite_patterns <- function(n) {
  bytes <- matrix(sample.int(256L, 8L * n, replace = TRUE) - 1L, 8L)
  exponent <- sample(c(0L, 1L, 1022L, 1023L, 2046L,
                       sample.int(2047L, n, replace = TRUE) - 1L),
                     n, replace = TRUE)
  bytes[7L, ] <- bytes[7L, ] %% 16L + 16L * (exponent %% 16L)
  bytes[8L, ] <- sample(c(0L, 128L), n, replace = TRUE) + exponent %/% 16L
  value <- readBin(as.raw(bytes), double(), n = n, size = 8L, endian = "little")
  stopifnot(all(is.finite(value)))
  value
}
subnormal <- .Machine$double.xmin * .Machine$double.eps
special <- rbind(c(-0.0, 0), c(0, -0.0), c(subnormal, -subnormal),
                 c(-subnormal, subnormal), c(.Machine$double.xmin, 0),
                 c(-.Machine$double.xmin, 0), c(.Machine$double.xmax, 0),
                 c(-.Machine$double.xmax, 0), c(-1, 0), c(2^-54, 0))
y <- rbind(matrix(finite_patterns(240L), ncol = 2L), special, special)
y <- y[sample.int(nrow(y)), , drop = FALSE]
x <- rbind(y[sample.int(nrow(y), 40L), ], special,
           matrix(finite_patterns(60L), ncol = 2L))
dense <- sfgpu_distance(x, y, backend = "cpu")
radii <- unique(c(0, subnormal, .Machine$double.xmin, 1,
                  median(dense[is.finite(dense)]), .Machine$double.xmax))
for (radius in radii) {
  expected <- lapply(seq_len(nrow(x)), function(i) which(dense[i, ] <= radius))
  actual <- sfgpu_within_distance(x, y, radius, backend = "metal")
  if (!identical(actual, expected)) stop("METAL_INDEX_KEY_MISMATCH")
  stats <- .Call(get("C_sfgpu_metal_radius_stats", asNamespace("sfgpu")))
  stopifnot(stats$indexed_waves > 0, stats$tiled_tiles == 0)
}
cat(sprintf("METAL_INDEX_KEYS_PASS comparisons=%d\n", length(dense) * length(radii)))
