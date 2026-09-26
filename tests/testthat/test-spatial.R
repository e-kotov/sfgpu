test_that("projected XY POINT inputs preserve CRS units", {
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

  for (backend in contract_backends()) {
    actual <- sfgpu_distance(x, y, backend = backend)
    reference <- sf::st_distance(x, y)
    expect_true(inherits(actual, "units"))
    expect_identical(dim(actual), c(2L, 2L))
    expect_identical(attr(actual, "units"), attr(reference, "units"))
    ordinary_close(units::drop_units(actual), units::drop_units(reference))

    # sf and sfc are both accepted spatial input forms.
    mixed_wrappers <- sfgpu_distance(sf::st_geometry(x), y, backend = backend)
    ordinary_close(units::drop_units(mixed_wrappers),
                   units::drop_units(reference))
  }
})

test_that("projected non-metre coordinates retain their CRS distance unit", {
  skip_if_not_installed("sf")
  skip_if_not_installed("units")

  x <- sf::st_sfc(sf::st_point(c(0, 0)), crs = 2263)
  y <- sf::st_sfc(sf::st_point(c(3, 4)), crs = 2263)
  reference <- sf::st_distance(x, y)
  for (backend in contract_backends()) {
    actual <- sfgpu_distance(x, y, backend = backend)
    expect_identical(attr(actual, "units"), attr(reference, "units"))
    ordinary_close(units::drop_units(actual), units::drop_units(reference))
  }
})

test_that("empty projected spatial collections preserve zero dimensions", {
  skip_if_not_installed("sf")
  empty <- sf::st_sfc(crs = 32632)
  points <- sf::st_sfc(sf::st_point(c(0, 0)), sf::st_point(c(3, 4)),
                       crs = 32632)
  for (backend in contract_backends()) {
    expect_identical(dim(sfgpu_distance(empty, points, backend = backend)),
                     c(0L, 2L))
    expect_identical(dim(sfgpu_distance(points, empty, backend = backend)),
                     c(2L, 0L))
    expect_identical(dim(sfgpu_distance(empty, empty, backend = backend)),
                     c(0L, 0L))
  }
})

test_that("unsupported spatial inputs and CRS states are rejected", {
  skip_if_not_installed("sf")
  points <- sf::st_sfc(sf::st_point(c(0, 0)), sf::st_point(c(1, 1)),
                       crs = 32632)
  good <- sf::st_as_sf(points)
  geographic <- sf::st_sfc(sf::st_point(c(0, 0)), crs = 4326)
  unknown_crs <- sf::st_sfc(sf::st_point(c(0, 0)))
  other_projected <- sf::st_sfc(sf::st_point(c(0, 0)), crs = 32631)
  geocentric <- sf::st_sfc(sf::st_point(c(0, 0)), crs = 4978)
  xyz <- sf::st_sfc(sf::st_point(c(0, 0, 1)), crs = 32632)
  xym <- sf::st_sfc(sf::st_point(c(0, 0, 1), dim = "XYM"), crs = 32632)
  line <- sf::st_sfc(sf::st_linestring(rbind(c(0, 0), c(1, 1))), crs = 32632)
  polygon <- sf::st_sfc(sf::st_polygon(list(rbind(
    c(0, 0), c(1, 0), c(1, 1), c(0, 0)
  ))), crs = 32632)
  empty_point <- sf::st_sfc(sf::st_point(), crs = 32632)
  matrix_input <- matrix(c(0, 0), ncol = 2L)

  crs_rejected <- list(
    list(value = geographic, pattern = "projected CRS"),
    list(value = unknown_crs, pattern = "known projected CRS"),
    list(value = geocentric, pattern = "projected CRS")
  )
  for (fixture in crs_rejected) {
    # Matching each bad CRS to itself reaches the CRS predicate under test;
    # pairing it with `good` would fail earlier as a CRS mismatch.
    expect_error(sfgpu_distance(fixture$value, fixture$value), fixture$pattern)
  }

  geometry_rejected <- list(
    list(value = xyz, pattern = "XY POINT"),
    list(value = xym, pattern = "XY POINT"),
    list(value = line, pattern = "only POINT"),
    list(value = polygon, pattern = "only POINT"),
    list(value = empty_point, pattern = "empty POINT")
  )

  for (fixture in geometry_rejected) {
    expect_error(sfgpu_distance(fixture$value, good), fixture$pattern)
    expect_error(sfgpu_distance(good, fixture$value), fixture$pattern)
  }
  expect_error(sfgpu_distance(good, other_projected), "same CRS")
  expect_error(sfgpu_distance(other_projected, good), "same CRS")
  expect_error(sfgpu_distance(matrix_input, good), "both be matrices")
  expect_error(sfgpu_distance(good, matrix_input), "both be matrices")
})

test_that("valid projected spatial calls work after rejected calls", {
  skip_if_not_installed("sf")
  good <- sf::st_sfc(sf::st_point(c(0, 0)), sf::st_point(c(3, 4)),
                     crs = 32632)
  geographic <- sf::st_sfc(sf::st_point(c(0, 0)), crs = 4326)
  expect_error(sfgpu_distance(geographic, geographic), "projected CRS")
  expect_silent(sfgpu_distance(good, good))
})
