#ifndef SFGPU_CUDA_RADIUS_H
#define SFGPU_CUDA_RADIUS_H

#include "radius.h"

void sfgpu_cuda_radius(const double* x, std::size_t nx,
                       const double* y, std::size_t ny, double radius,
                       std::size_t tile_bytes, SfgpuRadiusEmit emit,
                       void* state, bool (*interrupt)());

#endif
