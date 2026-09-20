# sfgpu

> [!WARNING]
> **Highly experimental.** This package is at a very early stage. Its API, its
> internals, and even its name will keep changing and breaking without notice
> until further notice. Do not depend on it.

`sfgpu` is a planned companion package to [`sf`](https://r-spatial.github.io/sf/)
(in the same spirit as [`lwgeom`](https://r-spatial.github.io/lwgeom/)) that lets
selected spatial operations run on a GPU. `sf` itself stays unchanged; `sfgpu` is
opt-in.

## Status

Nothing is usable yet. The `main` branch intentionally contains no code. All
development happens on the `dev` branch and on feature branches, and code will
reach `main` only once something is stable enough to try.

## Licence

Not decided yet. The licence will be chosen together with the GPU backend design,
because the choice of backend determines which licences are possible.
