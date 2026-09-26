# sfgpu

> [!WARNING]
> **Experimental development package.** This version supports Cartesian XY
> point distances and within-radius matches on CPU and, when enabled, NVIDIA CUDA and
> Apple Metal GPUs.
> The API and implementation may change. It is not a CRAN release and has no
> performance guarantee.

`sfgpu` is an opt-in companion to [`sf`](https://r-spatial.github.io/sf/).
It computes dense distance matrices and sparse within-radius matches between
two point collections. CPU is the default backend; CUDA and Metal are selected
explicitly. The package does not modify `sf`.

All development happens on the `dev` branch and on feature branches. The
`main` branch remains README-only until the package is stable enough to try.

## Install

The default build is CPU-only and does not require a GPU toolchain:

```sh
R CMD INSTALL .
```

CUDA support is opt-in. Build with a working NVIDIA CUDA compiler and toolkit:

```sh
R CMD INSTALL --configure-args=--enable-cuda .
```

Apple Metal support is opt-in and requires Apple's Metal SDK and compiler:

```sh
R CMD INSTALL --configure-args=--enable-metal .
```

The configure script accepts `NVCC`, `CUDA_HOME`, and `CUDA_ARCH_FLAGS` to
select the compiler, toolkit location, and architecture flags when needed.
Those flags must be supported by the installed `nvcc`. A GPU-enabled build
does not guarantee that a usable device is available at runtime. Inspect build
and runtime state with `sfgpu_backends()`; a forced GPU call errors if the
backend was not compiled or no device is available. Device failures are errors;
the Metal numerical correction path described below is separate.
CPU remains the default backend in every build.

## Use

For numeric matrices, each input must have exactly two numeric columns. The
coordinates are Cartesian and must already use common caller-selected units;
the function does not infer CRS or units for matrices.

```r
library(sfgpu)

x <- rbind(c(0, 0), c(3, 4))
y <- rbind(c(0, 0), c(5, 4))
sfgpu_distance(x, y)                         # CPU by default
sfgpu_distance(x, y, backend = "cuda")      # explicitly require CUDA
sfgpu_distance(x, y, backend = "metal")     # explicitly require Metal
sfgpu_within_distance(x, y, dist = 3)        # sparse radius matches
sfgpu_backends()                             # compiled and available backends
```

`sfgpu_distance()` returns a numeric matrix with `nrow(x)` rows and `nrow(y)`
columns; for spatial inputs, the matrix carries the CRS distance units. Matrix
dimnames are propagated. With `y` omitted, it returns all pairwise distances
within `x`:

```r
sfgpu_distance(x)
```

Spatial inputs to either operation must both be spatial (`sf` or `sfc`), have
the same known projected CRS, and contain nonempty XY POINT geometries. Neither
operation transforms coordinates or discards Z/M dimensions. Geographic CRS,
unknown or mismatched CRS, mixed matrix/spatial arguments, missing or non-finite
coordinates, and unsupported or empty geometries are errors. Empty collections
are supported. `sfgpu_distance()` returns CRS distance units; the radius
function returns indices, with a numeric radius interpreted in CRS units.

```r
points <- sf::st_as_sf(
  data.frame(x = c(0, 3), y = c(0, 4)),
  coords = c("x", "y"), crs = 32632
)
sfgpu_distance(points)                      # units follow projected CRS
sfgpu_within_distance(points, dist = units::set_units(5, m))
```

## Memory limits

For `sfgpu_distance()`, `max_output_bytes` inclusively limits the dense result
matrix payload. Its default is 1 GiB. It does not cap total R process memory.
For example, a 100,000 by 100,000 double matrix needs 80 GB for the result
alone. Requests over the limit are rejected before result allocation or GPU
startup.

For `sfgpu_distance()`, `tile_bytes` is an inclusive execution budget for the
two coordinate input tiles plus the distance output tile. Its default is 64
MiB; 40 bytes is the minimum budget for one pair. Smaller budgets create
smaller tiles but do not make the returned matrix out-of-core.

`sfgpu_within_distance()` returns one sorted integer vector per row of `x`,
containing the one-based indices in `y` at Euclidean distance less than or equal
to `dist`. A zero radius includes all coordinate duplicates, including a point's
own index when `y` is omitted. Numeric radii use coordinate units for matrices
and CRS units for spatial inputs; spatial inputs also accept a scalar `units`
radius convertible to the CRS units. The result remains sparse and the function
does not allocate a full pairwise distance matrix.

For radius matches, `max_output_bytes` defaults to 1 GiB and inclusively counts
eight bytes per `x` list slot plus four bytes per returned index; R object
headers and temporary collection workspace are excluded, so this is an output
accounting limit rather than a cap on total process memory. The function errors
if the cap would be exceeded, before returning a partial result. The radius
operation's `tile_bytes` budget includes its candidate inputs, output, and
scratch; 49 bytes is the minimum for one pair.

For CUDA and Metal radius calls, when the sorted `y` index and bounded
query/task/output workspace fit `tile_bytes`, sfgpu sorts the `y` coordinates
on the host, retains original row indices, transfers the sorted index to the
device, binary-searches x-coordinate ranges for each query, and compacts the
remaining rectangular candidates. Metal stores ordered binary64 coordinate
keys in its index. The host applies the same final inclusive Euclidean
predicate for both backends. If the full index workspace does not fit, CUDA and
Metal use their bounded tiled GPU scans instead; a 49-byte tile selects that
path. CUDA compacts at most 1,048,576 candidates per wave, while Metal compacts
at most 65,536 per command to bound command duration. The CPU backend uses its
own sorted-y index and remains the default. This dispatch does not promise a
speed advantage for either GPU; results depend on the data, output density,
device, and tile budget.

```r
sfgpu_distance(x, y, max_output_bytes = 1024, tile_bytes = 40)
```

For `sfgpu_distance()`, the result is a binary64 matrix. CPU and CUDA use
double-precision, robust hypot-style arithmetic. Metal uses exact
scaled-integer coordinates, wide integer squared distances and an integer
square root; it does not use native double arithmetic, which Metal lacks.
Pairs that cannot be encoded exactly, lie near a rounding boundary, or fall
outside the GPU result exponent range [-900, 900] are recomputed using CPU
double-precision hypot. The fraction corrected depends on the coordinates and
tile size and can reach 100%. `tools/validate-metal.R` reports accepted GPU and
corrected CPU pair counts. Metal commands are capped at 65,536 pairs to bound
command duration; no physical-Mac speedup has been established.

For `sfgpu_within_distance()`, every backend makes the final membership test
with CPU double-precision `hypot(dx, dy) <= dist` on the represented binary64
coordinates. CUDA and Metal indexed paths binary-search x-coordinate ranges on
device-side sorted-y indexes, filter by the y-coordinate bound, and compact
candidates before the host applies the final hypot test. Metal represents its
sorted coordinates as ordered binary64 keys. If the index workspace does not
fit the tile budget, either GPU uses a bounded full rectangle scan that can
inspect O(nrow(x) * nrow(y)) pairs. The CPU backend sorts and indexes the `y`
points by x coordinate to prune candidates before applying the same predicate.
None of these paths carries a speed guarantee or automatic backend choice.

Comparisons use the represented binary64 coordinates; precision already lost
when coordinates were created cannot be recovered. For dense distances,
ordinary finite results are checked against an absolute bound of
`32 * .Machine$double.eps * pmax(1, abs(reference))`; very small nonzero values
need relative checks.

## Scope

This version does not implement geographic/geodesic distances, other geometry
types, out-of-core results, or automatic backend selection. Radius matches are
sparse, while `sfgpu_distance()` returns a dense matrix. CUDA, Metal, and CPU-only
portability are separate capabilities. The hosted macOS
Metal CI runner validates Metal compilation and correctness on its virtualized
Apple device; it does not establish performance on a physical Mac. Performance
has not been established by this README.

## Licence

MIT. See the full text in `LICENSE.md`.
