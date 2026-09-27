#' Install and enable CUDA companion DLL on Windows
#'
#' On Windows, standard R package builds (such as those from R-universe) are
#' compiled with MinGW GCC without NVIDIA CUDA support, because CUDA requires MSVC.
#' This function downloads the matching pre-compiled CUDA companion DLL from GitHub
#' Releases into the user data directory, verifies its SHA-256 checksum, and loads it.
#'
#' On Linux, CUDA should be built from source using \code{R CMD INSTALL --configure-args=--enable-cuda .}.
#' On macOS, Apple Metal acceleration is compiled automatically on Apple Silicon.
#'
#' @param force Logical; if \code{TRUE}, re-download and overwrite any existing DLL.
#'   If a CUDA DLL is already loaded in the current R session, R must be restarted first.
#' @param quiet Logical; if \code{TRUE}, suppress download progress messages.
#' @return Invisibly returns the destination file path of the installed DLL.
#' @export
sfgpu_install_cuda <- function(force = FALSE, quiet = FALSE) {
  if (.Platform$OS.type != "windows" && !nzchar(Sys.getenv("SFGPU_ALLOW_INSTALL_CUDA_TEST"))) {
    stop("sfgpu_install_cuda() is only needed on Windows; on Linux reinstall with --enable-cuda, and on macOS Metal is built-in.", call. = FALSE)
  }

  m <- cuda_manifest()
  dest <- cuda_dll_path()

  if (file.exists(dest) && !force) {
    file_hash <- .sha256_file(dest)
    if (identical(file_hash, m$sha256) || startsWith(m$sha256, "00000000")) {
      cuda_ensure()
      message("CUDA companion DLL is already installed at: ", dest)
      return(invisible(dest))
    }
  }

  if (isTRUE(.Call(C_sfgpu_cuda_loaded))) {
    stop("A CUDA DLL is already loaded in this R session; restart R and rerun sfgpu_install_cuda().", call. = FALSE)
  }

  tmp <- tempfile(fileext = ".dll")
  on.exit(unlink(tmp), add = TRUE)

  utils::download.file(m$url, tmp, mode = "wb", quiet = quiet)

  download_hash <- .sha256_file(tmp)
  if (!startsWith(m$sha256, "00000000") && !identical(download_hash, m$sha256)) {
    stop(sprintf("Checksum mismatch for %s\nExpected: %s\nActual:   %s\nRefusing to install.",
                 m$url, m$sha256, download_hash), call. = FALSE)
  }

  dir.create(dirname(dest), recursive = TRUE, showWarnings = FALSE)
  success <- file.copy(tmp, dest, overwrite = TRUE)
  if (!success) {
    stop("Failed to copy downloaded DLL to: ", dest, call. = FALSE)
  }

  cuda_ensure()
  message("Successfully installed sfgpu CUDA companion DLL to: ", dest)
  print(sfgpu_backends())
  invisible(dest)
}

#' @rdname sfgpu_install_cuda
#' @export
install_cuda <- sfgpu_install_cuda

cuda_manifest <- function() {
  manifest_path <- system.file("cuda-manifest.dcf", package = "sfgpu")
  if (!nzchar(manifest_path) || !file.exists(manifest_path)) {
    manifest_path <- file.path("inst", "cuda-manifest.dcf")
  }
  if (!file.exists(manifest_path)) {
    stop("cuda-manifest.dcf could not be found", call. = FALSE)
  }
  dcf <- read.dcf(manifest_path)
  as.list(as.data.frame(dcf, stringsAsFactors = FALSE))
}

cuda_dll_path <- function() {
  env <- Sys.getenv("SFGPU_CUDA_DLL")
  if (nzchar(env)) return(env)
  abi <- tryCatch(cuda_manifest()$abi, error = function(e) "1")
  file.path(tools::R_user_dir("sfgpu", "data"), "cuda",
            paste0("abi-", abi), "sfgpu_cuda.dll")
}

cuda_ensure <- function() {
  if (.Platform$OS.type != "windows" && !nzchar(Sys.getenv("SFGPU_CUDA_DLL"))) {
    return(invisible(TRUE))
  }
  if (isTRUE(.Call(C_sfgpu_cuda_loaded))) {
    return(invisible(TRUE))
  }
  path <- cuda_dll_path()
  if (file.exists(path)) {
    .Call(C_sfgpu_cuda_load, enc2utf8(normalizePath(path, mustWork = TRUE)))
  }
  invisible(TRUE)
}

.sha256_file <- function(file) {
  if (exists("sha256sum", asNamespace("tools"))) {
    unname(tools::sha256sum(file))
  } else if (requireNamespace("openssl", quietly = TRUE)) {
    as.character(openssl::sha256(file(file, "rb")))
  } else if (requireNamespace("digest", quietly = TRUE)) {
    digest::digest(file, algo = "sha256", file = TRUE)
  } else {
    stop("Neither tools::sha256sum(), 'openssl', nor 'digest' package is available for SHA-256 verification.", call. = FALSE)
  }
}
