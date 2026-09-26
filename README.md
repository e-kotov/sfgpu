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

The result has `nrow(x)` rows and `nrow(y)` columns. Matrix dimnames are
propagated. With `y` omitted, the function returns all pairwise distances
within `x`:

```r
sfgpu_distance(x)
```

Spatial inputs must both be spatial (`sf` or `sfc`), have the same known
projected CRS, and contain nonempty XY POINT geometries. The result carries
the distance units for that CRS. The function does not transform coordinates
or discard Z/M dimensions. Geographic CRS, unknown or mismatched CRS, mixed
matrix/spatial arguments, missing or non-finite coordinates, and unsupported
or empty geometries are errors. Empty collections are supported.

```r
points <- sf::st_as_sf(
  data.frame(x = c(0, 3), y = c(0, 4)),
  coords = c("x", "y"), crs = 32632
)
sfgpu_distance(points)                      # units follow projected CRS
sfgpu_within_distance(points, dist = units::set_units(5, m))
```

## Memory limits

`max_output_bytes` limits the dense result matrix payload and is inclusive.
Its default is 1 GiB. It does not cap total R process memory. For example, a
100,000 by 100,000 double matrix needs 80 GB for the result alone. Requests
over the limit are rejected before result allocation or GPU startup.

`tile_bytes` is an inclusive execution budget for the two coordinate input
tiles plus the distance output tile. Its default is 64 MiB; 40 bytes is the
minimum budget for one pair. Smaller budgets create smaller tiles but do not
make the returned matrix out-of-core.

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

```r
sfgpu_distance(x, y, max_output_bytes = 1024, tile_bytes = 40)
```

Results are binary64 doubles. CPU and CUDA use double-precision, robust
hypot-style arithmetic. Metal uses exact scaled-integer coordinates, wide
integer squared distances and an integer square root; it does not use native
double arithmetic, which Metal lacks. Pairs that cannot be encoded exactly,
lie near a rounding boundary, or fall outside the GPU result exponent range
[-900, 900] are recomputed using CPU double-precision hypot. The fraction
corrected depends on the coordinates and tile size and can reach 100%.
`tools/validate-metal.R` reports accepted GPU and corrected CPU pair counts.
Metal commands are capped at 65,536 pairs to bound command duration; no
physical-Mac speedup has been established.

Comparisons
are defined for the represented binary64 coordinates; precision already lost
when coordinates were created cannot be recovered. Ordinary finite results
are checked against an absolute bound of
`32 * .Machine$double.eps * pmax(1, abs(reference))`; very small nonzero values
need relative checks.

## Scope

This version does not implement geographic/geodesic distances, other geometry
types, sparse or out-of-core results, or automatic backend selection. CUDA,
Metal, and CPU-only portability are separate capabilities. The hosted macOS
Metal CI runner validates Metal compilation and correctness on its virtualized
Apple device; it does not establish performance on a physical Mac. Performance
has not been established by this README.

## Licence

MIT. See the full text in `LICENSE.md`.
