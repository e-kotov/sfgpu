#ifndef SFGPU_CUDA_DISTANCE_H
#define SFGPU_CUDA_DISTANCE_H

#include <cstddef>
#include <string>

bool sfgpu_cuda_info(std::string& device, std::string& reason);
void sfgpu_cuda_distance(const double* x, std::size_t nx,
                         const double* y, std::size_t ny, double* out,
                         std::size_t tile_bytes, bool (*interrupt)());

#endif
