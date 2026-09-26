test_that("backend reporting distinguishes compiled and available", {
  info <- sfgpu_backends()
  expect_named(info, c("cpu", "cuda", "metal"))
  for (backend in info) {
    expect_type(backend$compiled, "logical")
    expect_length(backend$compiled, 1L)
    expect_type(backend$available, "logical")
    expect_length(backend$available, 1L)
    expect_type(backend$device, "character")
    expect_length(backend$device, 1L)
    expect_type(backend$reason, "character")
    expect_length(backend$reason, 1L)
  }
  expect_true(info$cpu$compiled)
  expect_true(info$cpu$available)

  for (backend in c("cuda", "metal")) {
    required <- identical(Sys.getenv(paste0("SFGPU_REQUIRE_", toupper(backend))),
                          "true")
    if (required) {
      expect_true(info[[backend]]$compiled, info[[backend]]$reason)
      expect_true(info[[backend]]$available, info[[backend]]$reason)
      expect_false(is.na(info[[backend]]$device))
    }
  }
})

test_that("analytic asymmetric inputs preserve matrix shape, order, and names", {
  x <- rbind(c(0, 0), c(3, 4), c(-2, 1))
  y <- rbind(c(0, 0), c(1, 0), c(0, 2), c(5, 4), c(-2, 1))
  rownames(x) <- c("origin", "345", "left")
  rownames(y) <- paste0("y", seq_len(nrow(y)))
  expected <- rbind(
    c(0, 1, 2, sqrt(41), sqrt(5)),
    c(5, 2 * sqrt(5), sqrt(13), 2, sqrt(34)),
    c(sqrt(5), sqrt(10), sqrt(5), sqrt(58), 0)
  )

  for (backend in contract_backends()) {
    actual <- sfgpu_distance(x, y, backend = backend)
    expect_type(actual, "double")
    expect_identical(dim(actual), c(3L, 5L))
    expect_identical(rownames(actual), rownames(x))
    expect_identical(colnames(actual), rownames(y))
    ordinary_close(actual, expected)

    transposed <- sfgpu_distance(y, x, backend = backend)
    expect_identical(dim(transposed), c(5L, 3L))
    ordinary_close(transposed, t(expected))

    ix <- c(3L, 1L)
    iy <- c(5L, 2L, 1L)
    reordered <- sfgpu_distance(x[ix, , drop = FALSE], y[iy, , drop = FALSE],
                                backend = backend)
    ordinary_close(reordered, expected[ix, iy, drop = FALSE])
  }
})

test_that("self distances and repeated points are exactly zero", {
  points <- rbind(c(0, 0), c(3, 4), c(0, 0), c(-2, 1))
  for (backend in contract_backends()) {
    actual <- sfgpu_distance(points, backend = backend)
    expect_identical(dim(actual), c(4L, 4L))
    expect_true(all(diag(actual) == 0))
    expect_identical(actual[1, 3], 0)
    expect_identical(actual[3, 1], 0)
    expect_equal(actual[1, ], actual[3, ])
    expect_equal(actual[, 1], actual[, 3])
  }
})

test_that("integer coordinate matrices are accepted as numeric input", {
  x <- matrix(as.integer(c(0, 0, 3, 4)), ncol = 2L, byrow = TRUE)
  y <- matrix(as.integer(c(0, 0, 5, 4)), ncol = 2L, byrow = TRUE)
  expect_type(sfgpu_distance(x, y), "double")
  ordinary_close(sfgpu_distance(x, y),
                 matrix(c(0, 5, sqrt(41), 2), nrow = 2L))
})

test_that("large offsets retain the distance between represented doubles", {
  p <- c(5e6, 5e6)
  q <- c(5e6 + 0.01, 5e6 + 0.02)
  dx <- q[1] - p[1]
  dy <- q[2] - p[2]
  reference <- sqrt(dx * dx + dy * dy)
  expect_gt(reference, 0)
  pair <- rbind(p, q)

  for (backend in contract_backends()) {
    actual <- sfgpu_distance(pair[1, , drop = FALSE],
                             pair[2, , drop = FALSE], backend = backend)
    ordinary_close(actual, matrix(reference, nrow = 1L))
  }
})

test_that("robust hypot handles large, tiny, and genuinely overflowing distances", {
  for (backend in contract_backends()) {
    large <- sfgpu_distance(rbind(c(0, 0)), rbind(c(3e200, 4e200)),
                            backend = backend)
    ordinary_close(large, matrix(5e200))

    tiny <- sfgpu_distance(rbind(c(0, 0)), rbind(c(3e-200, 4e-200)),
                           backend = backend)
    expect_true(is.finite(tiny))
    expect_gt(tiny, 0)
    expect_lte(abs(tiny / 5e-200 - 1), 32 * .Machine$double.eps)

    overflow <- sfgpu_distance(rbind(c(0, 0)), rbind(c(1.3e308, 1.3e308)),
                               backend = backend)
    expect_true(is.infinite(overflow))
    expect_gt(overflow, 0)
  }
})

test_that("empty matrix collections preserve every zero dimension", {
  empty <- matrix(numeric(), ncol = 2L)
  points <- rbind(c(0, 0), c(3, 4), c(-2, 1))
  for (backend in contract_backends()) {
    expect_identical(dim(sfgpu_distance(empty, points, backend = backend)),
                     c(0L, 3L))
    expect_identical(dim(sfgpu_distance(points, empty, backend = backend)),
                     c(3L, 0L))
    expect_identical(dim(sfgpu_distance(empty, empty, backend = backend)),
                     c(0L, 0L))
  }
})

test_that("result and tile byte limits are inclusive and checked", {
  a <- rbind(c(0, 0), c(1, 1))
  b <- rbind(c(0, 0), c(2, 2))
  for (backend in contract_backends()) {
    # A 2 by 2 double result occupies exactly 32 bytes.
    expect_identical(dim(sfgpu_distance(a, b, backend = backend,
                                        max_output_bytes = 32)), c(2L, 2L))
    expect_error(sfgpu_distance(a, b, backend = backend,
                                max_output_bytes = 31), "max_output_bytes")

    for (bad in list(NA_real_, Inf, 0, -1, 1.5, c(40, 80), "40")) {
      expect_error(sfgpu_distance(a, b, backend = backend,
                                  tile_bytes = bad), "tile_bytes")
    }
    for (bad in list(NA_real_, Inf, 0, -1, 1.5, c(32, 64), "32")) {
      expect_error(sfgpu_distance(a, b, backend = backend,
                                  max_output_bytes = bad), "max_output_bytes")
    }
    expect_error(sfgpu_distance(a, b, backend = backend, tile_bytes = 39),
                 "tile_bytes")
    expect_silent(sfgpu_distance(a[1, , drop = FALSE], b[1, , drop = FALSE],
                                 backend = backend))
  }
})

test_that("tiny and odd tile budgets preserve all edge tiles", {
  x <- rbind(c(0, 0), c(3, 4), c(-2, 1))
  y <- rbind(c(0, 0), c(1, 0), c(0, 2), c(5, 4), c(-2, 1))
  expected <- rbind(
    c(0, 1, 2, sqrt(41), sqrt(5)),
    c(5, 2 * sqrt(5), sqrt(13), 2, sqrt(34)),
    c(sqrt(5), sqrt(10), sqrt(5), sqrt(58), 0)
  )
  for (backend in contract_backends()) {
    for (budget in c(40, 88, 1024)) {
      actual <- sfgpu_distance(x, y, backend = backend, tile_bytes = budget)
      ordinary_close(actual, expected)
    }
  }
})

test_that("very skinny default tiles cover rows and columns beyond one CUDA grid", {
  x <- matrix(c(0, 0), ncol = 2L)
  y <- cbind(seq_len(1100000L), integer(1100000L))
  expected <- as.double(seq_len(1100000L))

  for (backend in contract_backends()) {
    across_columns <- sfgpu_distance(x, y, backend = backend)
    expect_identical(dim(across_columns), c(1L, 1100000L))
    expect_identical(as.numeric(across_columns), expected)

    across_rows <- sfgpu_distance(y, x, backend = backend)
    expect_identical(dim(across_rows), c(1100000L, 1L))
    expect_identical(as.numeric(across_rows), expected)
  }
})

test_that("unsupported matrices and backend names are rejected", {
  valid <- rbind(c(0, 0), c(1, 1))
  for (bad in list(matrix(1:6, ncol = 3), matrix(letters[1:4], ncol = 2),
                   c(1, 2), data.frame(x = 1:2, y = 1:2))) {
    expect_error(sfgpu_distance(bad), "two-column numeric matrix")
    expect_silent(sfgpu_distance(valid))
  }
  for (bad_value in c(NA_real_, NaN, Inf, -Inf)) {
    bad <- valid
    bad[1, 1] <- bad_value
    expect_error(sfgpu_distance(bad), "finite and non-missing")
  }
  expect_error(sfgpu_distance(valid, backend = "automatic"), "'arg'")
})

test_that("forced CUDA never falls back silently", {
  info <- sfgpu_backends()
  if (isTRUE(info$cuda$available)) {
    expect_equal(sfgpu_distance(rbind(c(0, 0)), rbind(c(3, 4)),
                                backend = "cuda"), matrix(5))
  } else {
    expect_error(sfgpu_distance(rbind(c(0, 0)), rbind(c(3, 4)),
                                backend = "cuda"), "CUDA")
  }
  expect_equal(sfgpu_distance(rbind(c(0, 0)), rbind(c(3, 4))), matrix(5))
})

test_that("forced Metal never falls back silently", {
  info <- sfgpu_backends()
  if (isTRUE(info$metal$available)) {
    expect_equal(sfgpu_distance(rbind(c(0, 0)), rbind(c(3, 4)),
                                backend = "metal"), matrix(5))
  } else {
    expect_error(sfgpu_distance(rbind(c(0, 0)), rbind(c(3, 4)),
                                backend = "metal"), "Metal")
  }
  expect_equal(sfgpu_distance(rbind(c(0, 0)), rbind(c(3, 4))), matrix(5))
})
