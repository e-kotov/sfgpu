#ifndef SFGPU_METAL_DISTANCE_H
#define SFGPU_METAL_DISTANCE_H

#include <cstddef>
#include <cstdint>
#include <string>

struct SfgpuMetalStats {
  double gpu_pairs = 0;
  double cpu_pairs = 0;
  double tiles = 0;
};

bool sfgpu_metal_info(std::string& device, std::string& reason);
void sfgpu_metal_distance(const double* x, std::size_t nx,
                          const double* y, std::size_t ny, double* out,
                          std::size_t tile_bytes, bool (*interrupt)());
SfgpuMetalStats sfgpu_metal_stats();

#endif
