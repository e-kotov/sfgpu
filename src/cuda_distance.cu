#include "cuda_distance.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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

struct DeviceBuffer {
  double* data = nullptr;
  explicit DeviceBuffer(std::size_t count) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(double)),
               "allocation failed");
  }
  ~DeviceBuffer() { if (data) cudaFree(data); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

__global__ void distance_kernel(const double* x, const double* y, double* out,
                                std::size_t x_stride, std::size_t y_stride,
                                std::size_t nx, std::size_t ny) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t j = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (i < nx && j < ny) {
    const double dx = x[i] - y[j];
    const double dy = x[x_stride + i] - y[y_stride + j];
    out[j * x_stride + i] = hypot(dx, dy);
  }
}

}  // namespace

bool sfgpu_cuda_info(std::string& device, std::string& reason) {
  device.clear();
  reason.clear();
  int count = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) {
    reason = std::string("CUDA device query failed: ") + cudaGetErrorString(status);
    return false;
  }
  if (count < 1) {
    reason = "no CUDA device is available";
    return false;
  }
  int device_index = 0;
  status = cudaGetDevice(&device_index);
  if (status != cudaSuccess) {
    reason = std::string("CUDA current-device query failed: ") + cudaGetErrorString(status);
    return false;
  }
  cudaDeviceProp prop{};
  status = cudaGetDeviceProperties(&prop, device_index);
  if (status != cudaSuccess) {
    reason = std::string("CUDA device properties failed: ") + cudaGetErrorString(status);
    return false;
  }
  if (prop.major < 7) {
    reason = "CUDA device is below the package's sm_70 minimum";
    return false;
  }
  device = prop.name;
  return true;
}

void sfgpu_cuda_distance(const double* x, std::size_t nx,
                         const double* y, std::size_t ny, double* out,
                         std::size_t tile_bytes, bool (*interrupt)()) {
  std::string device;
  std::string reason;
  if (!sfgpu_cuda_info(device, reason)) {
    throw std::runtime_error("CUDA backend unavailable: " + reason);
  }
  if (nx == 0 || ny == 0) return;

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "memory query failed");
  // Reserve at least half the reported free memory for the driver and other
  // processes; the requested tile_bytes remains an upper bound.
  const std::size_t budget = std::min(tile_bytes, free_bytes / 2);
  if (budget < 40) {
    throw std::runtime_error("CUDA memory cannot fit the minimum 40-byte tile");
  }

  std::size_t tx = std::min(nx, std::max<std::size_t>(1,
      static_cast<std::size_t>(std::sqrt(static_cast<double>(budget) / 8.0))));
  std::size_t ty = 0;
  while (tx >= 1) {
    if (budget >= 16 * tx) ty = std::min(ny, (budget - 16 * tx) / (8 * tx + 16));
    if (ty >= 1) break;
    tx /= 2;
  }
  if (tx == 0 || ty == 0) {
    throw std::runtime_error("CUDA tile budget cannot fit one point pair");
  }
  // CUDA grid.y is limited to 65535 blocks on supported devices.
  ty = std::min<std::size_t>(ty, 65535U * 16U);

  // Host staging and device allocations are each bounded by tile_bytes:
  // 16*tx + 16*ty + 8*tx*ty bytes. They are reused across all tiles.
  std::vector<double> hx(2 * tx);
  std::vector<double> hy(2 * ty);
  std::vector<double> hout(tx * ty);
  DeviceBuffer dx(2 * tx);
  DeviceBuffer dy(2 * ty);
  DeviceBuffer dout(tx * ty);
  const dim3 block(16, 16);
  for (std::size_t j0 = 0; j0 < ny; j0 += ty) {
    const std::size_t jy = std::min(ty, ny - j0);
    for (std::size_t j = 0; j < jy; ++j) {
      hy[j] = y[j0 + j];
      hy[ty + j] = y[ny + j0 + j];
    }
    cuda_check(cudaMemcpy(dy.data, hy.data(), 2 * ty * sizeof(double),
                          cudaMemcpyHostToDevice), "copying y tile failed");
    for (std::size_t i0 = 0; i0 < nx; i0 += tx) {
      const std::size_t ix = std::min(tx, nx - i0);
      for (std::size_t i = 0; i < ix; ++i) {
        hx[i] = x[i0 + i];
        hx[tx + i] = x[nx + i0 + i];
      }
      cuda_check(cudaMemcpy(dx.data, hx.data(), 2 * tx * sizeof(double),
                            cudaMemcpyHostToDevice), "copying x tile failed");
      const dim3 grid(static_cast<unsigned>((ix + block.x - 1) / block.x),
                      static_cast<unsigned>((jy + block.y - 1) / block.y));
      distance_kernel<<<grid, block>>>(dx.data, dy.data, dout.data,
                                       tx, ty, ix, jy);
      cuda_check(cudaGetLastError(), "kernel launch failed");
      cuda_check(cudaMemcpy(hout.data(), dout.data, tx * ty * sizeof(double),
                            cudaMemcpyDeviceToHost), "copying output tile failed");
      for (std::size_t j = 0; j < jy; ++j) {
        for (std::size_t i = 0; i < ix; ++i) {
          out[(j0 + j) * nx + i0 + i] = hout[j * tx + i];
        }
      }
      if (!interrupt()) throw std::runtime_error("distance computation interrupted");
    }
  }
}
