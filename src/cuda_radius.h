#ifndef SFGPU_CUDA_RADIUS_H
#define SFGPU_CUDA_RADIUS_H

#include "radius.h"

struct SfgpuCudaRadiusStats {
  double indexed_waves = 0;
  double tiled_tiles = 0;
};

SfgpuCudaRadiusStats sfgpu_cuda_radius_stats();

void sfgpu_cuda_radius(const double* x, std::size_t nx,
                       const double* y, std::size_t ny, double radius,
                       std::size_t tile_bytes, SfgpuRadiusEmit emit,
                       void* state, bool (*interrupt)());

#endif
