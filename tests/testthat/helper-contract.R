ordinary_close <- function(actual, reference) {
  testthat::expect_true(all(is.finite(actual)))
  testthat::expect_true(all(
    abs(actual - reference) <=
      32 * .Machine$double.eps * pmax(1, abs(reference))
  ))
}

contract_backends <- function() {
  info <- sfgpu_backends()
  if (identical(Sys.getenv("SFGPU_REQUIRE_CUDA"), "true")) {
    testthat::expect_true(info$cuda$compiled, info$cuda$reason)
    testthat::expect_true(info$cuda$available, info$cuda$reason)
    testthat::expect_false(is.na(info$cuda$device))
    return(c("cpu", "cuda"))
  }
  c("cpu", if (isTRUE(info$cuda$available)) "cuda")
}
