#!/usr/bin/env Rscript
library(sfgpu)
backend <- Sys.getenv("SFGPU_RADIUS_BACKEND", "metal")
stopifnot(backend %in% c("cpu", "cuda", "metal"),
          isTRUE(sfgpu_backends()[[backend]]$available))

comparisons <- 0L
compare <- function(x, y, radii, tiles = c(49, 113, 4096, 64 * 1024^2)) {
  dense <- sfgpu_distance(x, y, backend = "cpu")
  for (radius in radii) {
    expected <- lapply(seq_len(nrow(x)), function(i) which(dense[i, ] <= radius))
    cpu <- sfgpu_within_distance(x, y, radius, backend = "cpu")
    stopifnot(identical(cpu, expected))
    for (tile in tiles) {
      cuda <- sfgpu_within_distance(x, y, radius, backend = backend,
                                    tile_bytes = tile)
      if (!identical(cuda, expected)) stop("RADIUS_MEMBERSHIP_MISMATCH")
      comparisons <<- comparisons + length(x[, 1]) * length(y[, 1])
    }
  }
}

# Exact represented distances, signed zero, duplicates and x/y tail tiles.
x <- rbind(c(0, 0), c(-0.0, 0), c(3, 4), c(5e6, 5e6),
           c(.Machine$double.xmax, 0), c(-.Machine$double.xmax, 0))
y <- rbind(c(0, -0.0), c(3, 4), c(5e6 + 0.01, 5e6 + 0.02),
           c(7, 0), c(.Machine$double.xmax, 0), c(1, 1), c(-5, -12))
compare(x, y, c(0, 1, 5, 13, .Machine$double.xmax))

# Subnormal and minimum normal coordinates exercise underflow and zero radius.
subnormal <- .Machine$double.xmin * .Machine$double.eps
tiny <- rbind(c(0, 0), c(subnormal, 0),
              c(.Machine$double.xmin, 0), c(-subnormal, subnormal))
compare(tiny, tiny[c(4, 1, 2, 3), ],
        c(0, subnormal, .Machine$double.xmin))

# Non-square random inputs across exponents; choose some radii from exact CPU
# pair distances to force inclusive boundary classification.
set.seed(260926)
for (scale in c(1e-200, 1e-9, 1, 1e9, 1e200)) {
  rx <- matrix(rnorm(34) * scale, ncol = 2)
  ry <- matrix(rnorm(38) * scale, ncol = 2)
  dense <- sfgpu_distance(rx, ry, backend = "cpu")
  compare(rx, ry, c(0, dense[1, 1], median(dense), max(dense)))
}

# The mask must perform material filtering while the CPU final predicate
# removes a point inside the axis box but outside the Euclidean circle.
sx <- matrix(c(0, 0), nrow = 1)
sy <- rbind(c(0.5, 0.5), c(0.9, 0.9), c(10, 10))
stopifnot(identical(sfgpu_within_distance(sx, sy, 1, backend = backend,
                                          tile_bytes = 49), list(1L)))
stats <- .Call(get("C_sfgpu_radius_stats", asNamespace("sfgpu")))
if (!identical(stats$total_pairs, 3) ||
    !identical(stats$candidate_pairs, 2) ||
    !identical(stats$accepted_pairs, 1)) stop("RADIUS_FILTER_MISMATCH")

# Inclusive payload accounting: one list slot and one integer index.
stopifnot(identical(sfgpu_within_distance(sx, sy, 1, backend = backend,
                                          max_output_bytes = 12), list(1L)))
over <- tryCatch(sfgpu_within_distance(sx, sy, 1, backend = backend,
                                        max_output_bytes = 11), error = identity)
stopifnot(inherits(over, "error"), grepl("max_output_bytes", conditionMessage(over)))

cat(sprintf("SFGPU_RADIUS_HOLDOUT=PASS comparisons=%d device=%s\n",
            comparisons, sfgpu_backends()[[backend]]$device))

# A rounded subtraction can include a point outside the exact-real interval.
# This case fails if the broadphase omits radius widening.
stopifnot(identical(sfgpu_within_distance(rbind(c(-1, 0)),
    rbind(c(2^-54, 0)), 1, backend = backend), list(1L)))
# Independent analytic lattice: integer squares are exact for these inputs.
a <- as.matrix(expand.grid(-4:4, -3:3))
b <- as.matrix(expand.grid(-5:5, -2:2))
for (radius in c(0, 1, 2, 5)) {
  expected <- lapply(seq_len(nrow(a)), function(i) {
    which((a[i, 1] - b[, 1])^2 + (a[i, 2] - b[, 2])^2 <= radius^2)
  })
  if (!identical(sfgpu_within_distance(a, b, radius, backend = backend), expected))
    stop("RADIUS_MEMBERSHIP_MISMATCH")
}
cat("RADIUS_ANALYTIC_BOUNDARY_PASS\n")
