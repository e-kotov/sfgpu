\name{sfgpu_install_cuda}
\alias{sfgpu_install_cuda}
\alias{install_cuda}
\title{Install and Enable CUDA Companion DLL on Windows}
\usage{
sfgpu_install_cuda(force = FALSE, quiet = FALSE)

install_cuda(force = FALSE, quiet = FALSE)
}
\arguments{
\item{force}{Logical; if \code{TRUE}, re-download and overwrite any existing DLL.
If a CUDA DLL is already loaded in the current R session, R must be restarted first.}

\item{quiet}{Logical; if \code{TRUE}, suppress download progress messages.}
}
\value{
Invisibly returns the destination file path of the installed DLL.
}
\description{
On Windows, standard binary builds of \code{sfgpu} (such as those from R-universe)
are built without NVIDIA CUDA support because standard Rtools uses MinGW GCC,
whereas CUDA requires MSVC. \code{sfgpu_install_cuda()} downloads the pre-compiled
companion DLL from GitHub Releases into the user data directory, verifies its
SHA-256 checksum against the package manifest, and dynamically loads it.

On Linux, CUDA acceleration is enabled at install time via \code{R CMD INSTALL --configure-args=--enable-cuda .}.
On macOS, Apple Metal acceleration is compiled automatically on Apple Silicon.
}
\seealso{\code{sfgpu_backends}}
