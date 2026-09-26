\name{sfgpu_within_distance}
\alias{sfgpu_within_distance}
\title{Find Points Within a Projected Euclidean Radius}
\usage{
sfgpu_within_distance(
  x,
  y = x,
  dist,
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
  \item{dist}{One finite, non-negative numeric radius. Numeric values use
  coordinate units for matrices and CRS units for spatial inputs. Spatial
  inputs also accept one \code{units} value convertible to the CRS units.}
  \item{backend}{Backend to use: \code{"cpu"}, \code{"cuda"}, or
  \code{"metal"}. The default selects CPU. An explicit GPU request errors if
  that backend was not compiled or no device is available; it never falls back
  to CPU.}
  \item{max_output_bytes}{Inclusive cap on the output accounting: eight bytes
  per \code{x} list slot plus four bytes per returned index. R object headers
  and temporary collection workspace are excluded. The default is 1 GiB.}
  \item{tile_bytes}{Inclusive budget for the GPU execution tile, including
  inputs, output, and scratch. The default is 64 MiB; at least 49 bytes are
  needed for one candidate pair.}
}
\value{
A list with one integer vector per row of \code{x}. Each vector contains the
increasing, one-based indices of rows in \code{y} whose Euclidean distance is
less than or equal to \code{dist}. Equal-coordinate duplicates are included.
The list is named with the row names of \code{x}, when present.
}
\description{
Finds all pairs of points at Euclidean distance less than or equal to the
specified radius. Matrix inputs use common caller-selected coordinate units.
Spatial inputs must both be \code{sf} or \code{sfc}, have the same known
projected CRS, and contain only nonempty XY points. The function does not
transform coordinates. Empty collections are valid. Geographic or unknown
CRSs, mismatched CRSs, mixed input kinds, missing or non-finite coordinates,
extra dimensions, unsupported geometry types, and empty individual points are
errors. The function returns sparse indices and does not allocate the full
pairwise distance matrix. An output cap failure errors without returning a
partial result.
}
\details{
The default installation is CPU-only. CUDA and Metal are optional build
features. Use \code{sfgpu_backends()} to inspect whether each backend is
compiled and available at runtime. Distances are inclusive at the radius
boundary. A zero radius includes all coordinate duplicates, including the
point at the same row position when \code{y} is omitted.

The output byte accounting includes eight bytes per list slot and four bytes
per returned integer index, but excludes R object headers and temporary
collection workspace. It is not a cap on total process memory. For spatial
inputs, a numeric radius is interpreted in CRS units; a scalar \code{units}
radius is converted to CRS units. For matrix inputs, a \code{units} radius is
invalid.

Membership is decided on the represented binary64 coordinates by the inclusive
CPU double-precision predicate \code{hypot(dx, dy) <= dist}. CUDA and Metal
generate candidates using conservative rectangular coordinate bounds, then the
host applies that final predicate before returning indices. The CPU backend
sorts and indexes the \code{y} points by x coordinate to prune candidates
before applying the same predicate, with work proportional to the index build
and candidates examined. CUDA and Metal candidate scans can still
examine \code{nrow(x) * nrow(y)} pairs, though they do not create the dense
distance matrix. No speed advantage is guaranteed.
}
\examples{
x <- rbind(c(0, 0), c(3, 4))
y <- rbind(c(0, 0), c(5, 4))
sfgpu_within_distance(x, y, dist = 2)
sfgpu_within_distance(x, dist = 0)
\donttest{
if (sfgpu_backends()$cuda$available) {
  sfgpu_within_distance(x, y, dist = 5, backend = "cuda")
}
}
}
\seealso{\code{sfgpu_distance}, \code{sf::st_is_within_distance},
\code{sfgpu_backends}}
