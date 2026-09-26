test_that("within-radius results are sorted y indices in x order", {
  x <- rbind(origin = c(0, 0), `345` = c(3, 4), left = c(-2, 1))
  y <- rbind(a = c(0, 0), b = c(1, 0), c = c(0, 2),
             d = c(5, 4), e = c(-2, 1))
  expected <- list(c(1L, 2L, 3L), 4L, 5L)
  names(expected) <- rownames(x)

  for (backend in contract_backends()) {
    actual <- sfgpu_within_distance(x, y, dist = 2, backend = backend)
    expect_type(actual, "list")
    expect_length(actual, nrow(x))
    expect_identical(actual, expected)
    expect_identical(sfgpu_within_distance(x, y, dist = 2,
                                           backend = backend), actual)
  }
})

test_that("zero radius includes coordinate duplicates and self matches", {
  x <- rbind(c(0, 0), c(3, 4), c(0, 0))
  y <- rbind(c(0, 0), c(1, 0), c(0, 0), c(3, 4))
  expected <- list(c(1L, 3L), 4L, c(1L, 3L))
  for (backend in contract_backends()) {
    expect_identical(sfgpu_within_distance(x, y, dist = 0, backend = backend),
                     expected)
    expect_identical(sfgpu_within_distance(x, dist = 0, backend = backend),
                     list(c(1L, 3L), 2L, c(1L, 3L)))
  }
})

test_that("empty collections retain one result slot per x row", {
  empty <- matrix(numeric(), ncol = 2L)
  x <- rbind(c(0, 0), c(3, 4))
  for (backend in contract_backends()) {
    expect_identical(sfgpu_within_distance(empty, x, 1, backend = backend),
                     list())
    expect_identical(sfgpu_within_distance(x, empty, 1, backend = backend),
                     list(integer(), integer()))
    expect_identical(sfgpu_within_distance(empty, empty, 1,
                                           backend = backend), list())
  }
})

test_that("output cap counts x slots and returned indices inclusively", {
  x <- matrix(c(0, 0), ncol = 2L)
  y <- rbind(c(0, 0), c(3, 4))
  for (backend in contract_backends()) {
    # One x slot plus two int indices: 8 + 2 * 4 bytes.
    expect_identical(sfgpu_within_distance(x, y, 5, backend = backend,
                                           max_output_bytes = 16),
                     list(1:2))
    expect_error(sfgpu_within_distance(x, y, 5, backend = backend,
                                       max_output_bytes = 15),
                 "max_output_bytes")
    # Even a zero-hit result accounts for the x list slots.
    expect_error(sfgpu_within_distance(x, matrix(numeric(), ncol = 2L), 5,
                                       backend = backend,
                                       max_output_bytes = 7),
                 "max_output_bytes")
    expect_identical(sfgpu_within_distance(x, matrix(numeric(), ncol = 2L), 5,
                                           backend = backend,
                                           max_output_bytes = 8),
                     list(integer()))
  }
})

test_that("radius, byte limits, and input forms are validated", {
  x <- rbind(c(0, 0), c(1, 1))
  for (backend in contract_backends()) {
    for (bad in list(NA_real_, NaN, Inf, -Inf, -1, c(1, 2), "1")) {
      expect_error(sfgpu_within_distance(x, dist = bad, backend = backend),
                   "dist")
    }
    for (bad in list(NA_real_, Inf, 0, -1, 1.5, c(49, 80), "49")) {
      expect_error(sfgpu_within_distance(x, dist = 1, backend = backend,
                                         tile_bytes = bad), "tile_bytes")
    }
    expect_error(sfgpu_within_distance(x, dist = 1, backend = backend,
                                       tile_bytes = 48), "tile_bytes")
    expect_error(sfgpu_within_distance(x, dist = 1, backend = backend,
                                       max_output_bytes = -1),
                 "max_output_bytes")
    expect_error(sfgpu_within_distance(x, dist = 1, backend = backend,
                                       max_output_bytes = 16,
                                       tile_bytes = 49), "max_output_bytes")
    if (requireNamespace("units", quietly = TRUE)) {
      expect_error(sfgpu_within_distance(x, dist = units::set_units(1, m),
                                         backend = backend), "units distance")
    }
    expect_error(sfgpu_within_distance(x, matrix(1:6, ncol = 3), 1,
                                       backend = backend), "two-column numeric")
  }
})

test_that("projected sf points accept and convert units radii", {
  skip_if_not_installed("sf")
  skip_if_not_installed("units")
  x <- sf::st_as_sf(
    data.frame(easting = c(500000, 500003), northing = c(5700000, 5700004)),
    coords = c("easting", "northing"), crs = 32632
  )
  y <- sf::st_as_sf(
    data.frame(easting = c(500000, 500005), northing = c(5700000, 5700004)),
    coords = c("easting", "northing"), crs = 32632
  )
  expected <- list(1L, c(1L, 2L))
  for (backend in contract_backends()) {
    numeric_radius <- sfgpu_within_distance(x, y, dist = 5, backend = backend)
    converted_radius <- sfgpu_within_distance(
      x, y, dist = units::set_units(500, cm), backend = backend
    )
    expect_identical(numeric_radius, expected)
    expect_identical(converted_radius, expected)
    expect_identical(converted_radius, numeric_radius)
  }
  expect_error(sfgpu_within_distance(x, y, dist = units::set_units(1, s)),
               "convertible")
  expect_error(sfgpu_within_distance(x, y, dist = units::set_units(NA_real_, m)),
               "finite")
})

test_that("forced radius GPU requests never fall back silently", {
  x <- matrix(c(0, 0), ncol = 2L)
  for (backend in c("cuda", "metal")) {
    info <- sfgpu_backends()[[backend]]
    if (isTRUE(info$available)) {
      expect_identical(sfgpu_within_distance(x, dist = 1, backend = backend),
                       list(1L))
    } else {
      expect_error(sfgpu_within_distance(x, dist = 1, backend = backend),
                   backend, ignore.case = TRUE)
    }
    expect_identical(sfgpu_within_distance(x, dist = 1), list(1L))
  }
})
