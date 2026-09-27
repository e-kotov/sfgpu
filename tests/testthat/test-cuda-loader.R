test_that("sfgpu_install_cuda rejects non-Windows calls", {
  if (.Platform$OS.type != "windows") {
    expect_error(sfgpu_install_cuda(), "sfgpu_install_cuda\\(\\) is only needed on Windows")
    expect_error(install_cuda(), "sfgpu_install_cuda\\(\\) is only needed on Windows")
  }
})

test_that("cuda_manifest returns expected fields", {
  m <- sfgpu:::cuda_manifest()
  expect_true(is.list(m))
  expect_identical(m$abi, "1")
  expect_true(nzchar(m$url))
  expect_true(nzchar(m$sha256))
  expect_true(nzchar(m$cuda))
})

test_that("cuda_dll_path respects SFGPU_CUDA_DLL", {
  old <- Sys.getenv("SFGPU_CUDA_DLL", unset = NA)
  on.exit({
    if (is.na(old)) Sys.unsetenv("SFGPU_CUDA_DLL") else Sys.setenv(SFGPU_CUDA_DLL = old)
  }, add = TRUE)
  Sys.setenv(SFGPU_CUDA_DLL = "C:/test/custom_sfgpu_cuda.dll")
  expect_identical(sfgpu:::cuda_dll_path(), "C:/test/custom_sfgpu_cuda.dll")
})

test_that(".sha256_file calculates consistent hash", {
  tf <- tempfile()
  on.exit(unlink(tf), add = TRUE)
  writeBin(charToRaw("sfgpu-test-vector"), tf)
  # echo -n "sfgpu-test-vector" | shasum -a 256
  expected_hash <- "2cb0ea3cf00bef022dd0c81a0c7c954194bc82572c14dbd5a9703031474cab58"
  expect_identical(sfgpu:::.sha256_file(tf), expected_hash)
})

test_that("C_sfgpu_cuda_loaded reports FALSE when uninitialized", {
  loaded <- .Call(sfgpu:::C_sfgpu_cuda_loaded)
  expect_false(isTRUE(loaded))
})
