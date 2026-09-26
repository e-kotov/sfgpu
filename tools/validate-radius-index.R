# CUDA index holdouts. Keep expected results independent of the index scheduler.
library(sfgpu)
stopifnot(isTRUE(sfgpu_backends()$cuda$available))
path_stats <- function() .Call(get("C_sfgpu_cuda_radius_stats", asNamespace("sfgpu")))
check <- function(x, y, radius, expected, tile_bytes = 64 * 1024^2) {
  actual <- sfgpu_within_distance(x, y, radius, backend = "cuda",
                                  tile_bytes = tile_bytes)
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

# A single query has more than a full compaction wave's worth of matches.
# The index is about 25 MiB; 64 MiB leaves space for the bounded task output.
n <- 1048577L
one <- matrix(c(0, 0), nrow = 1L)
duplicates <- matrix(0, n, 2L)
check(one, duplicates, 0, list(seq_len(n)))
stopifnot(path_stats()$indexed_waves >= 2, path_stats()$tiled_tiles == 0)
rm(duplicates)
gc()

# 100k unique coordinates, shuffled targets: exact expected identifiers and
# output accounting, without allocating a dense reference distance matrix.
n <- 100000L
p <- cbind(seq_len(n) * 16, (seq_len(n) %% 997L) * 16)
permutation <- sample.int(n)
q <- p[permutation, , drop = FALSE]
expected <- lapply(order(permutation), as.integer)
actual <- sfgpu_within_distance(p, q, 0, backend = "cuda",
                                max_output_bytes = 12 * n)
if (!identical(actual, expected)) stop("RADIUS_INDEX_MISMATCH")
stopifnot(path_stats()$indexed_waves > 0, path_stats()$tiled_tiles == 0)
err <- tryCatch(sfgpu_within_distance(p, q, 0, backend = "cuda",
                                      max_output_bytes = 12 * n - 1),
                error = identity)
stopifnot(inherits(err, "error"), grepl("max_output_bytes", conditionMessage(err)))
cat("RADIUS_INDEX_HOLDOUT_PASS\n")
