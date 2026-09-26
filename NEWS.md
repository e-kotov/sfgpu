# sfgpu 0.1.0

- Add `sfgpu_within_distance()` for bounded-memory, sparse point matches within
  an inclusive Euclidean radius on CPU and available CUDA or Metal backends.
- Accept coordinate-unit numeric radii for matrices and spatial inputs, plus
  convertible `units` radii for projected spatial inputs.
