#include "cuda_radius.h"
#include "cuda_distance.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

SfgpuCudaRadiusStats last_stats;

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

// A full sorted-y index is used only when it and the complete bounded query
// workspace fit the caller's tile budget. All indices fit uint32 because R
// matrix row counts are signed 32-bit integers.
struct IndexedY {
  double x, y;
  std::uint32_t original_j, padding;
};
struct Range { std::uint32_t first, last; };
struct Task { std::uint32_t local_i, first, count; };
struct Pair { std::uint32_t local_i, original_j; };
static_assert(sizeof(IndexedY) == 24, "CUDA radius y-index layout changed");
static_assert(sizeof(SfgpuRadiusBounds) == 32, "CUDA radius bounds layout changed");
static_assert(sizeof(Range) == 8, "CUDA radius range layout changed");
static_assert(sizeof(Task) == 12, "CUDA radius task layout changed");
static_assert(sizeof(Pair) == 8, "CUDA radius pair layout changed");

__global__ void indexed_ranges(const SfgpuRadiusBounds* bounds,
                               const IndexedY* y, Range* ranges,
                               std::uint32_t nx, std::uint32_t ny) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nx) return;
  const SfgpuRadiusBounds b = bounds[i];
  std::uint32_t low = 0, high = ny;
  while (low < high) {
    const std::uint32_t mid = low + (high - low) / 2;
    if (y[mid].x < b.xmin) low = mid + 1;
    else high = mid;
  }
  const std::uint32_t first = low;
  high = ny;
  while (low < high) {
    const std::uint32_t mid = low + (high - low) / 2;
    if (y[mid].x <= b.xmax) low = mid + 1;
    else high = mid;
  }
  ranges[i] = {first, low};
}

// Each task covers at most 256 candidates from one x-range. Each active
// thread may append at most one pair; total task lengths never exceed capacity.
__global__ void indexed_filter(const SfgpuRadiusBounds* bounds,
                               const IndexedY* y, const Task* tasks,
                               Pair* output, std::uint32_t* count,
                               std::uint32_t capacity) {
  const Task task = tasks[blockIdx.x];
  const std::uint32_t k = threadIdx.x;
  if (k >= task.count) return;
  const IndexedY point = y[task.first + k];
  const SfgpuRadiusBounds b = bounds[task.local_i];
  if (point.y >= b.ymin && point.y <= b.ymax) {
    const std::uint32_t slot = atomicAdd(count, 1U);
    if (slot < capacity) output[slot] = {task.local_i, point.original_j};
  }
}

// The original GPU path remains available when the full index does not fit
// tile_bytes, including the 49-byte minimum. It never falls back to CPU.
__global__ void tiled_candidates(const SfgpuRadiusBounds* x,
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

std::size_t tiled_bytes_for(std::size_t nx, std::size_t ny) {
  return 32 * nx + 16 * ny + nx * ny;
}

void tiled_radius(const double* x, std::size_t nx,
                  const double* y, std::size_t ny, double radius,
                  std::size_t budget, SfgpuRadiusEmit emit,
                  void* state, bool (*interrupt)()) {
  constexpr std::size_t pair_cap = 1048576;
  std::size_t tx = std::min<std::size_t>(nx, 256);
  std::size_t ty = std::min<std::size_t>(ny, pair_cap / tx);
  while (tiled_bytes_for(tx, ty) > budget) {
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
      tiled_candidates<<<grid, block>>>(dx.data, dy.data, dout.data, ix, jy, ty);
      cuda_check(cudaGetLastError(), "radius kernel launch failed");
      cuda_check(cudaMemcpy(hout.data(), dout.data, ix * jy,
                            cudaMemcpyDeviceToHost), "copying radius mask failed");
      ++last_stats.tiled_tiles;
      for (std::size_t j = 0; j < jy; ++j) {
        for (std::size_t i = 0; i < ix; ++i) {
          if (hout[j * ix + i]) emit(i0 + i, j0 + j, state);
        }
      }
    }
  }
}

struct IndexPlan {
  std::size_t queries = 0;
  std::size_t tasks = 0;
  std::size_t descriptors = 0;
};

IndexPlan index_plan(std::size_t nx, std::size_t ny, std::size_t budget) {
  if (ny > budget / sizeof(IndexedY)) return {};
  const std::size_t remaining = budget - sizeof(IndexedY) * ny;
  for (std::size_t tx = std::min<std::size_t>(nx, 4096); tx >= 1;
       tx = tx == 1 ? 0 : (tx + 1) / 2) {
    // Avoid an 8 MiB output allocation for tiny calls. The division tests
    // the product before multiplying, even on narrow size_t platforms.
    std::size_t maximum_capacity = 256;
    if (ny > 1048576 / tx) {
      maximum_capacity = 1048576;
    } else {
      const std::size_t possible_pairs = tx * ny;
      while (maximum_capacity < possible_pairs) maximum_capacity *= 2;
    }
    for (std::size_t capacity = maximum_capacity; capacity >= 256;
         capacity /= 2) {
      // Per wave, each query can contribute one partial descriptor; all full
      // descriptors account for 256 tasks. One extra covers the wave edge.
      const std::size_t max_descriptors = tx + (capacity + 255) / 256 + 1;
      const std::size_t fixed = sizeof(SfgpuRadiusBounds) * tx +
          sizeof(Range) * tx + sizeof(Task) * max_descriptors +
          sizeof(std::uint32_t);
      if (fixed <= remaining &&
          capacity <= (remaining - fixed) / sizeof(Pair)) {
        return {tx, capacity, max_descriptors};
      }
    }
  }
  return {};
}

void indexed_radius(const double* x, std::size_t nx,
                    const double* y, std::size_t ny, double radius,
                    const IndexPlan& plan, SfgpuRadiusEmit emit,
                    void* state, bool (*interrupt)()) {
  std::vector<IndexedY> sorted_y(ny);
  for (std::size_t j = 0; j < ny; ++j) {
    sorted_y[j] = {y[j], y[ny + j], static_cast<std::uint32_t>(j), 0};
  }
  std::sort(sorted_y.begin(), sorted_y.end(),
            [](const IndexedY& a, const IndexedY& b) {
    return a.x < b.x || (a.x == b.x && a.original_j < b.original_j);
  });

  std::vector<SfgpuRadiusBounds> bounds(plan.queries);
  std::vector<Range> ranges(plan.queries);
  std::vector<Task> tasks;
  tasks.reserve(plan.descriptors);
  std::vector<Pair> output(plan.tasks);
  DeviceBuffer<IndexedY> dy(ny);
  DeviceBuffer<SfgpuRadiusBounds> dx(plan.queries);
  DeviceBuffer<Range> dranges(plan.queries);
  DeviceBuffer<Task> dtasks(plan.descriptors);
  DeviceBuffer<Pair> doutput(plan.tasks);
  DeviceBuffer<std::uint32_t> dcount(1);
  cuda_check(cudaMemcpy(dy.data, sorted_y.data(), ny * sizeof(IndexedY),
                        cudaMemcpyHostToDevice), "copying sorted radius y index failed");
  for (std::size_t i0 = 0; i0 < nx; i0 += plan.queries) {
    if (!interrupt()) throw std::runtime_error("radius computation interrupted");
    const std::size_t ix = std::min(plan.queries, nx - i0);
    for (std::size_t i = 0; i < ix; ++i) {
      bounds[i] = sfgpu_radius_bounds(x[i0 + i], x[nx + i0 + i], radius);
    }
    cuda_check(cudaMemcpy(dx.data, bounds.data(), ix * sizeof(SfgpuRadiusBounds),
                          cudaMemcpyHostToDevice), "copying radius query bounds failed");
    indexed_ranges<<<static_cast<unsigned>((ix + 255) / 256), 256>>>(
        dx.data, dy.data, dranges.data, static_cast<std::uint32_t>(ix),
        static_cast<std::uint32_t>(ny));
    cuda_check(cudaGetLastError(), "radius index search launch failed");
    cuda_check(cudaMemcpy(ranges.data(), dranges.data, ix * sizeof(Range),
                          cudaMemcpyDeviceToHost), "copying radius index ranges failed");

    std::size_t query = 0;
    std::uint32_t cursor = ranges[0].first;
    while (query < ix) {
      tasks.clear();
      std::size_t task_count = 0;
      while (query < ix && task_count < plan.tasks) {
        const Range range = ranges[query];
        if (range.first > range.last || range.last > ny || cursor < range.first ||
            cursor > range.last) {
          throw std::runtime_error("CUDA radius index returned an invalid range");
        }
        if (cursor == range.last) {
          ++query;
          if (query < ix) cursor = ranges[query].first;
          continue;
        }
        const std::size_t count = std::min<std::size_t>(
            {256, static_cast<std::size_t>(range.last - cursor),
             plan.tasks - task_count});
        if (tasks.size() >= plan.descriptors || count == 0) {
          throw std::runtime_error("CUDA radius task planner exceeded its budget");
        }
        tasks.push_back({static_cast<std::uint32_t>(query), cursor,
                         static_cast<std::uint32_t>(count)});
        cursor += static_cast<std::uint32_t>(count);
        task_count += count;
      }
      if (tasks.empty()) continue;
      cuda_check(cudaMemcpy(dtasks.data, tasks.data(), tasks.size() * sizeof(Task),
                            cudaMemcpyHostToDevice), "copying radius tasks failed");
      cuda_check(cudaMemset(dcount.data, 0, sizeof(std::uint32_t)),
                 "resetting radius candidate counter failed");
      indexed_filter<<<static_cast<unsigned>(tasks.size()), 256>>>(
          dx.data, dy.data, dtasks.data, doutput.data, dcount.data,
          static_cast<std::uint32_t>(plan.tasks));
      cuda_check(cudaGetLastError(), "radius indexed filter launch failed");
      std::uint32_t emitted = 0;
      cuda_check(cudaMemcpy(&emitted, dcount.data, sizeof(emitted),
                            cudaMemcpyDeviceToHost), "copying radius candidate count failed");
      ++last_stats.indexed_waves;
      if (emitted > task_count || emitted > plan.tasks) {
        throw std::runtime_error("CUDA radius candidate count exceeds task capacity");
      }
      if (emitted) {
        cuda_check(cudaMemcpy(output.data(), doutput.data,
                              emitted * sizeof(Pair), cudaMemcpyDeviceToHost),
                   "copying compact radius candidates failed");
      }
      for (std::size_t k = 0; k < emitted; ++k) {
        const Pair pair = output[k];
        if (pair.local_i >= ix || pair.original_j >= ny) {
          throw std::runtime_error("CUDA radius emitted an invalid index");
        }
        emit(i0 + pair.local_i, pair.original_j, state);
      }
      if (!interrupt()) throw std::runtime_error("radius computation interrupted");
    }
  }
}

}  // namespace

SfgpuCudaRadiusStats sfgpu_cuda_radius_stats() { return last_stats; }

void sfgpu_cuda_radius(const double* x, std::size_t nx,
                       const double* y, std::size_t ny, double radius,
                       std::size_t tile_bytes, SfgpuRadiusEmit emit,
                       void* state, bool (*interrupt)()) {
  last_stats = SfgpuCudaRadiusStats{};
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
  const IndexPlan plan = index_plan(nx, ny, budget);
  if (plan.queries) {
    indexed_radius(x, nx, y, ny, radius, plan, emit, state, interrupt);
  } else {
    tiled_radius(x, nx, y, ny, radius, budget, emit, state, interrupt);
  }
}
