test_that("backend preserves small offsets at large binary64 coordinates", {
  # The increments are exact at 2^52 (one-unit ULP), so the reference tests
  # subtraction from the original double inputs rather than decimal ideals.
  p <- c(2^52, -2^52)
  q <- c(p[1] + 1, p[2] + 2)
  x <- matrix(p, nrow = 1L)
  y <- matrix(q, nrow = 1L)
  expected <- sqrt((q[1] - p[1])^2 + (q[2] - p[2])^2)
  expect_identical(expected, sqrt(5))

  for (backend in contract_backends()) {
    actual <- sfgpu_distance(x, y, backend = backend)
    expect_true(is.finite(actual))
    expect_lte(abs(actual / expected - 1), 32 * .Machine$double.eps)
  }
})

test_that("backend retains distances across the binary64 exponent range", {
  # Keep every component and result representable, while spanning small normal
  # through near-maximum magnitudes. The CPU hypot backend supplies an
  # independent robust reference for each exact pair of input doubles.
  exponents <- c(-1020L, -900L, -600L, -300L, -1L, 0L, 1L,
                 300L, 600L, 900L, 1018L)
  scale <- 2^exponents
  origin <- matrix(c(0, 0), nrow = 1L)
  points <- cbind(0.6 * scale, 0.8 * scale)
  reference <- as.numeric(sfgpu_distance(origin, points, backend = "cpu"))
  expect_true(all(is.finite(reference) & reference > 0))

  for (backend in contract_backends()) {
    actual <- as.numeric(sfgpu_distance(origin, points, backend = backend))
    expect_true(all(is.finite(actual) & actual > 0))
    expect_true(all(abs(actual / reference - 1) <=
                      32 * .Machine$double.eps))
  }
})

test_that("backend matches robust hypot on subnormals and near-limit values", {
  least <- .Machine$double.xmin * .Machine$double.eps
  origin <- matrix(c(0, 0), nrow = 1L)
  tiny_points <- rbind(c(least, 0), c(least, least),
                       c(2 * least, 0), c(4 * least, 4 * least))
  tiny_reference <- as.numeric(sfgpu_distance(origin, tiny_points,
                                               backend = "cpu"))
  expect_true(all(tiny_reference > 0))

  largest <- .Machine$double.xmax
  large_points <- rbind(c(largest, 0), c(largest / 2, largest / 2),
                        c(largest, largest))
  large_reference <- as.numeric(sfgpu_distance(origin, large_points,
                                                backend = "cpu"))
  expect_true(all(is.finite(large_reference[1:2])))
  expect_true(is.infinite(large_reference[3]) && large_reference[3] > 0)

  for (backend in contract_backends()) {
    tiny_actual <- as.numeric(sfgpu_distance(origin, tiny_points,
                                              backend = backend))
    expect_true(all(tiny_actual > 0))
    expect_true(all(abs(tiny_actual / tiny_reference - 1) <=
                      32 * .Machine$double.eps))

    large_actual <- as.numeric(sfgpu_distance(origin, large_points,
                                               backend = backend))
    expect_true(all(is.finite(large_actual[1:2])))
    expect_true(all(abs(large_actual[1:2] / large_reference[1:2] - 1) <=
                      32 * .Machine$double.eps))
    expect_true(is.infinite(large_actual[3]) && large_actual[3] > 0)
  }
})

test_that("odd partial tiles preserve orientation and transposition", {
  x <- rbind(c(-3, 1), c(0, 0), c(7, -2), c(1, 5), c(1, 5))
  y <- rbind(c(2, 2), c(-4, 3), c(0, 9))
  reference <- sfgpu_distance(x, y, backend = "cpu")

  for (backend in contract_backends()) {
    # 88 bytes forces partial tiles with an odd asymmetric 5-by-3 result.
    actual <- sfgpu_distance(x, y, backend = backend, tile_bytes = 88)
    transposed <- sfgpu_distance(y, x, backend = backend, tile_bytes = 88)
    expect_identical(dim(actual), c(5L, 3L))
    expect_identical(dim(transposed), c(3L, 5L))
    expect_equal(actual, reference, tolerance = 32 * .Machine$double.eps)
    expect_equal(transposed, t(reference),
                 tolerance = 32 * .Machine$double.eps)
  }
})
