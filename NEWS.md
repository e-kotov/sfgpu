# sfgpu 0.1.0

- Add `sfgpu_within_distance()` for bounded-memory, sparse point matches within
  an inclusive Euclidean radius on CPU and available CUDA or Metal backends.
- Accept coordinate-unit numeric radii for matrices and spatial inputs, plus
  convertible `units` radii for projected spatial inputs.
- Use a sorted-y index, device binary search, and candidate compaction for CUDA
  radius calls when the index workspace fits `tile_bytes`; CUDA otherwise keeps
  the bounded tiled scan, and Metal continues to use that scan. This adds no
  performance guarantee.
