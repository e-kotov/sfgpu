# sfgpu 0.1.0

- Add `sfgpu_within_distance()` for bounded-memory, sparse point matches within
  an inclusive Euclidean radius on CPU and available CUDA or Metal backends.
- Accept coordinate-unit numeric radii for matrices and spatial inputs, plus
  convertible `units` radii for projected spatial inputs.
- Use sorted-y indexes, device binary search, and candidate compaction for CUDA
  and Metal radius calls when the index workspace fits `tile_bytes`; either GPU
  otherwise uses its bounded tiled scan. Metal compares ordered binary64 keys
  and caps each compaction command at 65,536 candidates. This adds no performance
  guarantee.
