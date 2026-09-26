#include "cuda_radius.h"
#include "cuda_distance.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void cuda_check(cudaError_t status, const char* action) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA ") + action + ": " +
                             cudaGetErrorString(status));
  }
}

template <typename T> struct DeviceBuffer {
  T* data = nullptr;
  explicit DeviceBuffer(std::size_t count) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)),
               "radius allocation failed");
  }
  ~DeviceBuffer() { if (data) cudaFree(data); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

__global__ void candidate_kernel(const SfgpuRadiusBounds* x,
                                 const double* y,
                                 unsigned char* candidate,
                                 std::size_t nx, std::size_t ny,
                                 std::size_t y_stride) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t j = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (i < nx && j < ny) {
    const SfgpuRadiusBounds b = x[i];
    const double yx = y[j];
    const double yy = y[y_stride + j];
    candidate[j * nx + i] =
        static_cast<unsigned char>(yx >= b.xmin && yx <= b.xmax &&
                                   yy >= b.ymin && yy <= b.ymax);
  }
}

std::size_t tile_bytes_for(std::size_t nx, std::size_t ny) {
  return 32 * nx + 16 * ny + nx * ny;
}

}  // namespace

void sfgpu_cuda_radius(const double* x, std::size_t nx,
                       const double* y, std::size_t ny, double radius,
                       std::size_t tile_bytes, SfgpuRadiusEmit emit,
                       void* state, bool (*interrupt)()) {
  std::string device;
  std::string reason;
  if (!sfgpu_cuda_info(device, reason)) {
    throw std::runtime_error("CUDA backend unavailable: " + reason);
  }
  if (!nx || !ny) return;

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "radius memory query failed");
  const std::size_t budget = std::min(tile_bytes, free_bytes / 2);
  if (budget < 49) {
    throw std::runtime_error("CUDA memory cannot fit the minimum 49-byte radius tile");
  }

  // The pair mask is one byte per pair; x bounds and y coordinates are 32
  // and 16 bytes per point. Bound the grid as well as the allocation.
  constexpr std::size_t pair_cap = 1048576;
  std::size_t tx = std::min<std::size_t>(nx, 256);
  std::size_t ty = std::min<std::size_t>(ny, pair_cap / tx);
  while (tile_bytes_for(tx, ty) > budget) {
    if (tx == 1 && ty == 1) {
      throw std::runtime_error("CUDA radius tile budget cannot fit one point pair");
    }
    if (tx >= ty && tx > 1) tx = (tx + 1) / 2;
    else ty = (ty + 1) / 2;
  }

  std::vector<SfgpuRadiusBounds> hx(tx);
  std::vector<double> hy(2 * ty);
  std::vector<unsigned char> hout(tx * ty);
  DeviceBuffer<SfgpuRadiusBounds> dx(tx);
  DeviceBuffer<double> dy(2 * ty);
  DeviceBuffer<unsigned char> dout(tx * ty);
  const dim3 block(16, 16);
  for (std::size_t j0 = 0; j0 < ny; j0 += ty) {
    const std::size_t jy = std::min(ty, ny - j0);
    for (std::size_t j = 0; j < jy; ++j) {
      hy[j] = y[j0 + j];
      hy[ty + j] = y[ny + j0 + j];
    }
    cuda_check(cudaMemcpy(dy.data, hy.data(), 2 * ty * sizeof(double),
                          cudaMemcpyHostToDevice), "copying radius y tile failed");
    for (std::size_t i0 = 0; i0 < nx; i0 += tx) {
      if (!interrupt()) throw std::runtime_error("radius computation interrupted");
      const std::size_t ix = std::min(tx, nx - i0);
      for (std::size_t i = 0; i < ix; ++i) {
        hx[i] = sfgpu_radius_bounds(x[i0 + i], x[nx + i0 + i], radius);
      }
      cuda_check(cudaMemcpy(dx.data, hx.data(), tx * sizeof(SfgpuRadiusBounds),
                            cudaMemcpyHostToDevice), "copying radius x tile failed");
      const dim3 grid(static_cast<unsigned>((ix + block.x - 1) / block.x),
                      static_cast<unsigned>((jy + block.y - 1) / block.y));
      candidate_kernel<<<grid, block>>>(dx.data, dy.data, dout.data, ix, jy, ty);
      cuda_check(cudaGetLastError(), "radius kernel launch failed");
      cuda_check(cudaMemcpy(hout.data(), dout.data, ix * jy,
                            cudaMemcpyDeviceToHost), "copying radius mask failed");
      for (std::size_t j = 0; j < jy; ++j) {
        for (std::size_t i = 0; i < ix; ++i) {
          if (hout[j * ix + i]) emit(i0 + i, j0 + j, state);
        }
      }
    }
  }
}
