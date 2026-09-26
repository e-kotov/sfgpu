\name{sfgpu_distance}
\alias{sfgpu_distance}
\title{Compute a Dense Matrix of XY Point Distances}
\usage{
sfgpu_distance(
  x,
  y = x,
  backend = c("cpu", "cuda", "metal"),
  max_output_bytes = 1024^3,
  tile_bytes = 64 * 1024^2
)
}
\arguments{
  \item{x}{A numeric two-column matrix of Cartesian coordinates, or an
  \code{sf}/\code{sfc} collection of nonempty XY \code{POINT} geometries in a
  known projected CRS.}
  \item{y}{A two-column numeric matrix or spatial input matching the input
  category of \code{x}. Spatial inputs may be \code{sf} or \code{sfc} and
  must have the same known projected CRS. Defaults to \code{x}.}
  \item{backend}{Backend to use: \code{"cpu"}, \code{"cuda"}, or
  \code{"metal"}. The default selects CPU. An explicit GPU request errors if
  the package was built without that backend or no device is available; it
  never falls back to CPU.}
  \item{max_output_bytes}{Inclusive cap on the dense double result payload in
  bytes. The default is 1 GiB. This does not cap total process memory.}
  \item{tile_bytes}{Inclusive byte budget for both coordinate input tiles and
  the output tile. The default is 64 MiB; at least 40 bytes are needed for one
  pair.}
}
\value{
A numeric matrix with \code{nrow(x)} rows and \code{nrow(y)} columns. Input
matrix dimnames are propagated. For spatial inputs, the result is a distance
matrix carrying units appropriate to the CRS. If \code{y} is omitted, all
pairwise distances within \code{x} are returned.
}
\description{
Computes Cartesian Euclidean distances between two coordinate collections.
Matrix coordinates use common caller-selected units and carry no CRS
interpretation. Spatial inputs must both be spatial objects (either \code{sf}
or \code{sfc}), with equal known projected CRSs and only nonempty XY point geometries. The
function does not transform coordinates or drop extra dimensions. Empty
collections are valid. Geographic or unknown CRSs, mismatched CRSs, mixed
input kinds, missing or non-finite coordinates, extra dimensions, unsupported
geometry types, and empty individual points are errors.

Results are binary64 doubles. CPU and CUDA use double-precision hypot-style
arithmetic. Metal uses exact scaled-integer encoding and wide integer distance
arithmetic. Pairs that cannot be encoded exactly, approach a rounding boundary,
or have a result exponent outside [-900, 900] are corrected using CPU hypot.
The corrected fraction depends on the input and tile size and can reach 100\%.
A missing Metal device or command failure still errors. The
\code{max_output_bytes} limit applies to result payload only; tiling bounds
execution buffers but does not make the returned matrix out-of-core.
}
\details{
The default installation is CPU-only. CUDA and Metal are optional build
features. Use \code{sfgpu_backends()} to inspect whether each backend is
compiled and available at runtime. A GPU-enabled build may still have no usable
device.
}
\examples{
x <- rbind(c(0, 0), c(3, 4))
y <- rbind(c(0, 0), c(5, 4))
sfgpu_distance(x, y)
sfgpu_distance(x)
\donttest{
if (sfgpu_backends()$cuda$available) {
  sfgpu_distance(x, y, backend = "cuda")
}
}
}
\seealso{\code{sf::st_distance}, \code{sfgpu_backends}}
