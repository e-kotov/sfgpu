test_that("indexed radius search restores original y order with duplicates", {
  x <- rbind(
    first = c(0, 0),
    same_x = c(0, 10),
    third = c(20, 0),
    duplicate_first = c(0, 0)
  )
  # Deliberately scrambled by x coordinate. Several x rows and y rows share
  # coordinates, and the two duplicate origins have distinct original indices.
  y <- rbind(
    near_first = c(0.5, 0),
    far = c(100, 0),
    origin_a = c(0, 0),
    third_exact = c(20, 0),
    second_exact = c(0, 10),
    first_boundary = c(-1, 0),
    second_boundary = c(0, 9),
    third_boundary = c(19, 0),
    bbox_corner_outside_circle = c(0.8, 0.8),
    origin_b = c(0, 0)
  )
  expected <- list(c(1L, 3L, 6L, 10L), c(5L, 7L), c(4L, 8L),
                   c(1L, 3L, 6L, 10L))
  names(expected) <- rownames(x)

  for (backend in contract_backends()) {
    for (tile in c(512, 4096)) {
      expect_identical(sfgpu_within_distance(x, y, dist = 1,
                                             backend = backend,
                                             tile_bytes = tile), expected)
    }
  }
})

test_that("indexed bounds retain cancellation and inclusive radius edges", {
  cancellation_x <- matrix(c(-1, 0), nrow = 1L)
  cancellation_y <- matrix(c(2^-54, 0), nrow = 1L)
  origin <- matrix(c(0, 0), nrow = 1L)
  # At one, the adjacent lower double is half an epsilon below; above is one
  # epsilon higher.
  edge_y <- rbind(c(1, 0), c(1 - .Machine$double.eps / 2, 0),
                  c(1 + .Machine$double.eps, 0))
  subnormal <- .Machine$double.xmin * .Machine$double.eps
  tiny_y <- rbind(c(0, 0), c(subnormal, 0),
                  c(.Machine$double.xmin, 0))

  for (backend in contract_backends()) {
    expect_identical(sfgpu_within_distance(cancellation_x, cancellation_y, 1,
                                           backend = backend), list(1L))
    expect_identical(sfgpu_within_distance(origin, edge_y, 1,
                                           backend = backend), list(1:2))
    expect_identical(sfgpu_within_distance(origin, tiny_y, 0,
                                           backend = backend), list(1L))
    expect_identical(sfgpu_within_distance(origin, tiny_y, subnormal,
                                           backend = backend), list(1:2))
  }
})

test_that("indexed bounds preserve signed zero and extreme coordinates", {
  signed_x <- rbind(c(0, 0), c(-0.0, 0))
  signed_y <- rbind(c(-0.0, 0), c(0, -0.0))
  largest <- .Machine$double.xmax
  positive_x <- matrix(c(largest, 0), nrow = 1L)
  negative_x <- matrix(c(-largest, 0), nrow = 1L)
  extreme_y <- rbind(c(largest, 0), c(0, 0), c(-largest, 0))

  for (backend in contract_backends()) {
    expect_identical(sfgpu_within_distance(signed_x, signed_y, 0,
                                           backend = backend),
                     list(1:2, 1:2))
    expect_identical(sfgpu_within_distance(positive_x, extreme_y, largest,
                                           backend = backend), list(1:2))
    expect_identical(sfgpu_within_distance(negative_x, extreme_y, largest,
                                           backend = backend), list(2:3))
  }
})

radius_index_gpus <- intersect(
  c("cuda", "metal"),
  names(sfgpu_backends())[vapply(sfgpu_backends(), function(info) {
    isTRUE(info$available)
  }, logical(1L))]
)

radius_index_path_stats <- function(backend) {
  symbol <- switch(backend,
                   cuda = "C_sfgpu_cuda_radius_stats",
                   metal = "C_sfgpu_metal_radius_stats")
  .Call(get(symbol, asNamespace("sfgpu")))
}

for (radius_index_backend in radius_index_gpus) {
  local({
    backend <- radius_index_backend

    test_that(sprintf("%s selects indexed or bounded tiled execution by budget",
                      backend), {
      x <- matrix(c(0, 0), nrow = 1L)
      y <- matrix(c(3, 4), nrow = 1L)

      expect_identical(sfgpu_within_distance(x, y, dist = 5,
                                             backend = backend), list(1L))
      stats <- radius_index_path_stats(backend)
      expect_gt(stats$indexed_waves, 0)
      expect_identical(stats$tiled_tiles, 0)

      expect_identical(sfgpu_within_distance(x, y, dist = 5,
                                             backend = backend,
                                             tile_bytes = 49), list(1L))
      stats <- radius_index_path_stats(backend)
      expect_gt(stats$tiled_tiles, 0)
      expect_identical(stats$indexed_waves, 0)
    })

    test_that(sprintf("%s indexed query batches restore shuffled original y ids",
                      backend), {
      n <- 4097L
      points <- cbind(seq_len(n), seq_len(n) %% 31L)
      permutation <- rev(seq_len(n))
      y <- points[permutation, , drop = FALSE]
      expected <- lapply(seq_len(n), function(i) {
        as.integer(match(i, permutation))
      })

      actual <- sfgpu_within_distance(points, y, dist = 0, backend = backend,
                                      max_output_bytes = 8 * n + 4 * n)
      expect_identical(actual, expected)
      stats <- radius_index_path_stats(backend)
      expect_gt(stats$indexed_waves, 0)
      expect_identical(stats$tiled_tiles, 0)
    })

    test_that(sprintf("%s indexed all-match output crosses a compaction wave",
                      backend), {
      # CUDA crosses its 1,048,576-pair wave with multiple x rows; Metal crosses
      # its 65,536-candidate command limit with one query.
      if (backend == "metal") {
        nx <- 1L
        ny <- 65537L
      } else {
        nx <- 1025L
        ny <- 1024L
      }
      x <- matrix(0, nrow = nx, ncol = 2L)
      y <- matrix(0, nrow = ny, ncol = 2L)
      output_bytes <- 8 * nx + 4 * nx * ny

      actual <- sfgpu_within_distance(x, y, dist = 0, backend = backend,
                                     max_output_bytes = output_bytes)
      expect_length(actual, nx)
      expect_true(all(vapply(actual, function(row) {
        length(row) == ny && identical(row, seq_len(ny))
      }, logical(1L))))

      stats <- .Call(get("C_sfgpu_radius_stats", asNamespace("sfgpu")))
      expect_identical(stats$total_pairs, as.double(nx) * ny)
      expect_identical(stats$candidate_pairs, as.double(nx) * ny)
      expect_identical(stats$accepted_pairs, as.double(nx) * ny)
      dispatch <- radius_index_path_stats(backend)
      expect_gte(dispatch$indexed_waves, 2)
      expect_identical(dispatch$tiled_tiles, 0)

      expect_error(sfgpu_within_distance(
        x, y, dist = 0, backend = backend,
        max_output_bytes = output_bytes - 1
      ), "max_output_bytes")
    })
  })
}
