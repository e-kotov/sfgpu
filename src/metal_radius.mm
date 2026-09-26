#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_radius.h"
#include "metal_radius_kernel.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct KeyBounds { std::uint64_t xmin, xmax, ymin, ymax; };
struct KeyCoordinates { std::uint64_t x, y; };
struct IndexedY {
  std::uint64_t x, y;
  std::uint32_t original_j, padding;
};
struct Range { std::uint32_t first, last; };
struct Task { std::uint32_t local_i, first, count; };
struct Pair { std::uint32_t local_i, original_j; };
static_assert(sizeof(KeyBounds) == 32 && sizeof(KeyCoordinates) == 16 &&
              sizeof(IndexedY) == 24 && sizeof(Range) == 8 &&
              sizeof(Task) == 12 && sizeof(Pair) == 8,
              "Metal radius host/shader buffer layout mismatch");

SfgpuMetalRadiusStats last_stats;

std::runtime_error failure(const char* operation, NSError* error = nil) {
  const char* detail = error ? [[error localizedDescription] UTF8String] : nullptr;
  return std::runtime_error(std::string("Metal radius ") + operation +
                            (detail ? std::string(": ") + detail : " failed"));
}

struct RadiusContext {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> tiled_pipeline;
  id<MTLComputePipelineState> ranges_pipeline;
  id<MTLComputePipelineState> filter_pipeline;

  RadiusContext() {
    device = MTLCreateSystemDefaultDevice();
    if (!device) throw failure("device discovery");
    queue = [device newCommandQueue];
    if (!queue) throw failure("command queue creation");
    NSError* error = nil;
    MTLCompileOptions* options = [MTLCompileOptions new];
    id<MTLLibrary> library = [device newLibraryWithSource:
        [NSString stringWithUTF8String:sfgpu_metal_radius_source]
        options:options error:&error];
    if (!library) throw failure("shader compilation", error);
    tiled_pipeline = pipeline(library, @"radius_kernel");
    ranges_pipeline = pipeline(library, @"radius_index_ranges");
    filter_pipeline = pipeline(library, @"radius_index_filter");
  }

  id<MTLComputePipelineState> pipeline(id<MTLLibrary> library, NSString* name) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) throw failure("kernel lookup");
    NSError* error = nil;
    id<MTLComputePipelineState> result =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!result) throw failure("pipeline creation", error);
    return result;
  }
};

RadiusContext& context() {
  static RadiusContext state;
  return state;
}

std::size_t max_buffer(RadiusContext& ctx) {
  return static_cast<std::size_t>(ctx.device.maxBufferLength);
}

void completed(id<MTLCommandBuffer> command) {
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted) {
    throw failure("command execution", command.error);
  }
}

id<MTLBuffer> buffer(RadiusContext& ctx, std::size_t bytes) {
  id<MTLBuffer> result = [ctx.device newBufferWithLength:bytes
                         options:MTLResourceStorageModeShared];
  if (!result) throw failure("buffer allocation");
  if (![result contents]) throw failure("buffer mapping");
  return result;
}

KeyBounds key_bounds(double x, double y, double radius) {
  const auto b = sfgpu_radius_bounds(x, y, radius);
  return {sfgpu_radius_ordered_key(b.xmin),
          sfgpu_radius_ordered_key(b.xmax),
          sfgpu_radius_ordered_key(b.ymin),
          sfgpu_radius_ordered_key(b.ymax)};
}

void tiled_radius(RadiusContext& ctx, const double* x, std::size_t nx,
                  const double* y, std::size_t ny, double radius,
                  std::size_t tile_bytes, SfgpuRadiusEmit emit,
                  void* state, bool (*interrupt)()) {
  // Preserve the old bounded GPU path for tiny budgets or large y indices.
  constexpr std::size_t pair_cap = 65536;
  std::size_t tx = std::min<std::size_t>(nx, 256);
  std::size_t ty = std::min(ny, pair_cap / tx);
  tx = std::min(nx, pair_cap / ty);
  const std::size_t limit = max_buffer(ctx);
  while (32 * tx + 16 * ty + tx * ty > tile_bytes ||
         32 * tx > limit || 16 * ty > limit || tx * ty > limit) {
    if (tx == 1 && ty == 1) throw failure("buffer limit");
    if (tx >= ty && tx > 1) tx = (tx + 1) / 2;
    else ty = (ty + 1) / 2;
  }
  id<MTLBuffer> bx = buffer(ctx, 32 * tx);
  id<MTLBuffer> by = buffer(ctx, 16 * ty);
  id<MTLBuffer> bo = buffer(ctx, tx * ty);
  auto* px = static_cast<KeyBounds*>([bx contents]);
  auto* py = static_cast<KeyCoordinates*>([by contents]);
  auto* mask = static_cast<unsigned char*>([bo contents]);
  const NSUInteger max_threads = ctx.tiled_pipeline.maxTotalThreadsPerThreadgroup;
  if (!max_threads) throw failure("threadgroup limit");
  const NSUInteger width = std::min<NSUInteger>(16, max_threads);
  const NSUInteger height = std::min<NSUInteger>(16, max_threads / width);
  const MTLSize group = MTLSizeMake(width, height, 1);
  for (std::size_t j = 0; j < ny; j += ty) {
    const auto cols = static_cast<std::uint32_t>(std::min(ty, ny - j));
    for (std::size_t c = 0; c < cols; ++c) {
      py[c] = {sfgpu_radius_ordered_key(y[j + c]),
               sfgpu_radius_ordered_key(y[ny + j + c])};
    }
    for (std::size_t i = 0; i < nx; i += tx) {
      @autoreleasepool {
        if (!interrupt()) throw std::runtime_error("Metal radius computation interrupted");
        const auto rows = static_cast<std::uint32_t>(std::min(tx, nx - i));
        for (std::size_t r = 0; r < rows; ++r) {
          px[r] = key_bounds(x[i + r], x[nx + i + r], radius);
        }
        const std::size_t pairs = static_cast<std::size_t>(rows) * cols;
        std::fill(mask, mask + pairs, static_cast<unsigned char>(2));
        id<MTLCommandBuffer> command = [ctx.queue commandBuffer];
        if (!command) throw failure("command creation");
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) throw failure("encoder creation");
        [encoder setComputePipelineState:ctx.tiled_pipeline];
        [encoder setBuffer:bx offset:0 atIndex:0];
        [encoder setBuffer:by offset:0 atIndex:1];
        [encoder setBuffer:bo offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake((rows + width - 1) / width,
                                                 (cols + height - 1) / height, 1)
                threadsPerThreadgroup:group];
        [encoder endEncoding];
        completed(command);
        ++last_stats.tiled_tiles;
        for (std::size_t c = 0; c < cols; ++c) {
          for (std::size_t r = 0; r < rows; ++r) {
            const unsigned char candidate = mask[c * rows + r];
            if (candidate > 1) throw failure("invalid shader output");
            if (candidate) emit(i + r, j + c, state);
          }
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

IndexPlan index_plan(std::size_t nx, std::size_t ny, std::size_t budget,
                     std::size_t limit) {
  if (ny > budget / sizeof(IndexedY) || ny > limit / sizeof(IndexedY)) return {};
  const std::size_t remaining = budget - sizeof(IndexedY) * ny;
  for (std::size_t tx = std::min<std::size_t>(nx, 4096); tx >= 1;
       tx = tx == 1 ? 0 : (tx + 1) / 2) {
    std::size_t maximum_capacity = 256;
    if (ny > 65536 / tx) maximum_capacity = 65536;
    else {
      const std::size_t possible_pairs = tx * ny;
      while (maximum_capacity < possible_pairs) maximum_capacity *= 2;
    }
    for (std::size_t capacity = maximum_capacity; capacity >= 256;
         capacity /= 2) {
      // At most one partial descriptor per query plus one wave-edge fragment.
      const std::size_t descriptors = tx + (capacity + 255) / 256 + 1;
      const std::size_t bx = sizeof(KeyBounds) * tx;
      const std::size_t br = sizeof(Range) * tx;
      const std::size_t bt = sizeof(Task) * descriptors;
      const std::size_t bo = sizeof(Pair) * capacity;
      if (bx > limit || br > limit || bt > limit || bo > limit ||
          sizeof(std::uint32_t) > limit) continue;
      const std::size_t fixed = bx + br + bt + sizeof(std::uint32_t);
      if (fixed <= remaining && bo <= remaining - fixed) {
        return {tx, capacity, descriptors};
      }
    }
  }
  return {};
}

void indexed_radius(RadiusContext& ctx, const double* x, std::size_t nx,
                    const double* y, std::size_t ny, double radius,
                    const IndexPlan& plan, SfgpuRadiusEmit emit,
                    void* state, bool (*interrupt)()) {
  std::vector<IndexedY> sorted_y(ny);
  for (std::size_t j = 0; j < ny; ++j) {
    sorted_y[j] = {sfgpu_radius_ordered_key(y[j]),
                   sfgpu_radius_ordered_key(y[ny + j]),
                   static_cast<std::uint32_t>(j), 0};
  }
  std::sort(sorted_y.begin(), sorted_y.end(),
            [](const IndexedY& a, const IndexedY& b) {
    return a.x < b.x || (a.x == b.x && a.original_j < b.original_j);
  });

  id<MTLBuffer> by = buffer(ctx, sizeof(IndexedY) * ny);
  id<MTLBuffer> bx = buffer(ctx, sizeof(KeyBounds) * plan.queries);
  id<MTLBuffer> br = buffer(ctx, sizeof(Range) * plan.queries);
  id<MTLBuffer> bt = buffer(ctx, sizeof(Task) * plan.descriptors);
  id<MTLBuffer> bo = buffer(ctx, sizeof(Pair) * plan.tasks);
  id<MTLBuffer> bc = buffer(ctx, sizeof(std::uint32_t));
  auto* py = static_cast<IndexedY*>([by contents]);
  auto* px = static_cast<KeyBounds*>([bx contents]);
  auto* ranges = static_cast<Range*>([br contents]);
  auto* tasks_out = static_cast<Task*>([bt contents]);
  auto* output = static_cast<Pair*>([bo contents]);
  auto* counter = static_cast<std::uint32_t*>([bc contents]);
  std::copy(sorted_y.begin(), sorted_y.end(), py);
  sorted_y.clear();
  sorted_y.shrink_to_fit();

  const NSUInteger range_width = std::min<NSUInteger>(
      256, ctx.ranges_pipeline.maxTotalThreadsPerThreadgroup);
  const NSUInteger filter_width = std::min<NSUInteger>(
      256, ctx.filter_pipeline.maxTotalThreadsPerThreadgroup);
  if (!range_width || !filter_width) throw failure("indexed threadgroup limit");
  std::vector<Task> tasks;
  tasks.reserve(plan.descriptors);
  for (std::size_t i0 = 0; i0 < nx; i0 += plan.queries) {
    @autoreleasepool {
      if (!interrupt()) throw std::runtime_error("Metal radius computation interrupted");
      const auto ix = static_cast<std::uint32_t>(std::min(plan.queries, nx - i0));
      for (std::size_t i = 0; i < ix; ++i) {
        px[i] = key_bounds(x[i0 + i], x[nx + i0 + i], radius);
      }
      const auto y_count = static_cast<std::uint32_t>(ny);
      id<MTLCommandBuffer> command = [ctx.queue commandBuffer];
      if (!command) throw failure("index command creation");
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      if (!encoder) throw failure("index encoder creation");
      [encoder setComputePipelineState:ctx.ranges_pipeline];
      [encoder setBuffer:bx offset:0 atIndex:0];
      [encoder setBuffer:by offset:0 atIndex:1];
      [encoder setBuffer:br offset:0 atIndex:2];
      [encoder setBytes:&ix length:sizeof(ix) atIndex:3];
      [encoder setBytes:&y_count length:sizeof(y_count) atIndex:4];
      [encoder dispatchThreadgroups:MTLSizeMake((ix + range_width - 1) / range_width, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(range_width, 1, 1)];
      [encoder endEncoding];
      completed(command);

      std::size_t query = 0;
      std::uint32_t cursor = ranges[0].first;
      while (query < ix) {
        @autoreleasepool {
          tasks.clear();
          std::size_t task_count = 0;
          while (query < ix && task_count < plan.tasks) {
            const Range range = ranges[query];
            if (range.first > range.last || range.last > ny ||
                cursor < range.first || cursor > range.last) {
              throw failure("invalid index range");
            }
            if (cursor == range.last) {
              ++query;
              if (query < ix) cursor = ranges[query].first;
              continue;
            }
            const std::size_t count = std::min<std::size_t>(
                {256, static_cast<std::size_t>(range.last - cursor),
                 plan.tasks - task_count});
            if (!count || tasks.size() >= plan.descriptors) {
              throw failure("task planner exceeded budget");
            }
            tasks.push_back({static_cast<std::uint32_t>(query), cursor,
                             static_cast<std::uint32_t>(count)});
            cursor += static_cast<std::uint32_t>(count);
            task_count += count;
          }
          if (tasks.empty()) continue;
          std::copy(tasks.begin(), tasks.end(), tasks_out);
          *counter = 0;
          std::fill(output, output + task_count,
                    Pair{UINT32_MAX, UINT32_MAX});
          const auto capacity = static_cast<std::uint32_t>(plan.tasks);
          id<MTLCommandBuffer> filter = [ctx.queue commandBuffer];
          if (!filter) throw failure("filter command creation");
          id<MTLComputeCommandEncoder> filter_encoder = [filter computeCommandEncoder];
          if (!filter_encoder) throw failure("filter encoder creation");
          [filter_encoder setComputePipelineState:ctx.filter_pipeline];
          [filter_encoder setBuffer:bx offset:0 atIndex:0];
          [filter_encoder setBuffer:by offset:0 atIndex:1];
          [filter_encoder setBuffer:bt offset:0 atIndex:2];
          [filter_encoder setBuffer:bo offset:0 atIndex:3];
          [filter_encoder setBuffer:bc offset:0 atIndex:4];
          [filter_encoder setBytes:&capacity length:sizeof(capacity) atIndex:5];
          [filter_encoder dispatchThreadgroups:MTLSizeMake(tasks.size(), 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(filter_width, 1, 1)];
          [filter_encoder endEncoding];
          completed(filter);
          const std::uint32_t emitted = *counter;
          if (emitted > task_count || emitted > plan.tasks) {
            throw failure("candidate count exceeds task capacity");
          }
          for (std::size_t k = 0; k < emitted; ++k) {
            const Pair pair = output[k];
            if (pair.local_i >= ix || pair.original_j >= ny) {
              throw failure("invalid compact candidate");
            }
            emit(i0 + pair.local_i, pair.original_j, state);
          }
          ++last_stats.indexed_waves;
          if (!interrupt()) throw std::runtime_error("Metal radius computation interrupted");
        }
      }
    }
  }
}

}  // namespace

SfgpuMetalRadiusStats sfgpu_metal_radius_stats() { return last_stats; }

void sfgpu_metal_radius(const double* x, std::size_t nx,
                        const double* y, std::size_t ny, double radius,
                        std::size_t tile_bytes, SfgpuRadiusEmit emit,
                        void* state, bool (*interrupt)()) {
  last_stats = SfgpuMetalRadiusStats{};
  @autoreleasepool {
    RadiusContext& ctx = context();
    if (!nx || !ny) return;
    if (tile_bytes < 49) throw std::runtime_error("Metal radius tile_bytes must be at least 49");
    const IndexPlan plan = index_plan(nx, ny, tile_bytes, max_buffer(ctx));
    if (plan.queries) indexed_radius(ctx, x, nx, y, ny, radius, plan,
                                    emit, state, interrupt);
    else tiled_radius(ctx, x, nx, y, ny, radius, tile_bytes,
                      emit, state, interrupt);
  }
}
