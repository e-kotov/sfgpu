# CUDA or Metal indexed-radius holdouts. Keep expected results independent of
# the index scheduler; CUDA remains the default for the HPC gate.
library(sfgpu)
backend <- Sys.getenv("SFGPU_RADIUS_BACKEND", "cuda")
stopifnot(backend %in% c("cuda", "metal"),
          isTRUE(sfgpu_backends()[[backend]]$available))
stats_symbol <- switch(backend, cuda = "C_sfgpu_cuda_radius_stats",
                      metal = "C_sfgpu_metal_radius_stats")
path_stats <- function() .Call(get(stats_symbol, asNamespace("sfgpu")))
check <- function(x, y, radius, expected, tile_bytes = 64 * 1024^2,
                  max_output_bytes = 1024^3) {
  actual <- sfgpu_within_distance(x, y, radius, backend = backend,
                                  tile_bytes = tile_bytes,
                                  max_output_bytes = max_output_bytes)
  if (!identical(unname(actual), expected)) stop("RADIUS_INDEX_MISMATCH")
  invisible(actual)
}

# Original identifiers differ from sorted ranks. The distant-y point is inside
# the x interval: a kernel that bypasses its y filter must fail the counter check.
x <- rbind(c(0, 0), c(4, 0))
y <- rbind(c(4, 0), c(0, 10), c(-1, 0), c(0, 0), c(1, 0))
check(x, y, 1, list(3:5, 1L))
stopifnot(path_stats()$indexed_waves > 0, path_stats()$tiled_tiles == 0)
accounting <- .Call(get("C_sfgpu_radius_stats", asNamespace("sfgpu")))
if (accounting$candidate_pairs != 4) stop("RADIUS_INDEX_FILTER_MISMATCH")
check(x, y, 1, list(3:5, 1L), tile_bytes = 49)
stopifnot(path_stats()$indexed_waves == 0, path_stats()$tiled_tiles > 0)

# Signed zero, subnormals, the largest finite coordinates/radius, and a
# subtraction that rounds to the inclusive radius boundary.
check(rbind(c(0, 0), c(-0.0, 0)), rbind(c(-0.0, 0), c(0, -0.0)), 0,
      list(c(1L, 2L), c(1L, 2L)))
largest <- .Machine$double.xmax
extreme_y <- rbind(c(largest, 0), c(0, 0), c(-largest, 0))
check(matrix(c(largest, 0), nrow = 1L), extreme_y, largest, list(1:2))
check(matrix(c(-largest, 0), nrow = 1L), extreme_y, largest, list(2:3))
check(matrix(c(-1, 0), nrow = 1L), matrix(c(2^-54, 0), nrow = 1L), 1,
      list(1L))
subnormal <- .Machine$double.xmin * .Machine$double.eps
check(matrix(c(0, 0), nrow = 1L),
      rbind(c(0, 0), c(subnormal, 0), c(.Machine$double.xmin, 0)), 0,
      list(1L))
check(matrix(c(0, 0), nrow = 1L),
      rbind(c(0, 0), c(subnormal, 0), c(.Machine$double.xmin, 0)), subnormal,
      list(1:2))

# Analytic membership for a shuffled integer lattice, including duplicate x
# values and duplicate full coordinates. Integer squared lengths here are exact.
set.seed(280926)
a <- as.matrix(expand.grid(-8:8, -7:7))
b <- rbind(a, a[1:9, ])
b <- b[sample.int(nrow(b)), , drop = FALSE]
for (radius in c(0, 1, 2, 5)) {
  expected <- lapply(seq_len(nrow(a)), function(i) {
    which((a[i, 1] - b[, 1])^2 + (a[i, 2] - b[, 2])^2 <= radius^2)
  })
  check(a, b, radius, expected)
  check(a, b, radius, expected, tile_bytes = 8192)
}

# A single query has one more match than this backend's compaction wave.
# The index and bounded task output fit the default 64 MiB tile budget.
n <- if (backend == "metal") 65537L else 1048577L
one <- matrix(c(0, 0), nrow = 1L)
duplicates <- matrix(0, n, 2L)
wave_bytes <- 8 + 4 * n
check(one, duplicates, 0, list(seq_len(n)), max_output_bytes = wave_bytes)
stopifnot(path_stats()$indexed_waves >= 2, path_stats()$tiled_tiles == 0)
over <- tryCatch(sfgpu_within_distance(
  one, duplicates, 0, backend = backend, max_output_bytes = wave_bytes - 1
), error = identity)
stopifnot(inherits(over, "error"),
          grepl("max_output_bytes", conditionMessage(over)))
rm(duplicates)
gc()

# 100k unique coordinates, shuffled targets: exact expected identifiers and
# output accounting, without allocating a dense reference distance matrix.
n <- 100000L
p <- cbind(seq_len(n) * 16, (seq_len(n) %% 997L) * 16)
permutation <- sample.int(n)
q <- p[permutation, , drop = FALSE]
expected <- lapply(order(permutation), as.integer)
actual <- sfgpu_within_distance(p, q, 0, backend = backend,
                                max_output_bytes = 12 * n)
if (!identical(actual, expected)) stop("RADIUS_INDEX_MISMATCH")
stopifnot(path_stats()$indexed_waves > 0, path_stats()$tiled_tiles == 0)
err <- tryCatch(sfgpu_within_distance(p, q, 0, backend = backend,
                                      max_output_bytes = 12 * n - 1),
                error = identity)
stopifnot(inherits(err, "error"), grepl("max_output_bytes", conditionMessage(err)))
stopifnot(path_stats()$indexed_waves > 0, path_stats()$tiled_tiles == 0)
cat(sprintf("RADIUS_INDEX_HOLDOUT_PASS backend=%s device=%s\n", backend,
            sfgpu_backends()[[backend]]$device))
