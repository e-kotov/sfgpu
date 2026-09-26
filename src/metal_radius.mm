#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_radius.h"
#include "metal_radius_kernel.h"
#include <algorithm>
#include <stdexcept>
#include <string>

namespace {
struct KeyBounds { std::uint64_t xmin, xmax, ymin, ymax; };
struct KeyCoordinates { std::uint64_t x, y; };
static_assert(sizeof(KeyBounds) == 32 && sizeof(KeyCoordinates) == 16,
              "Metal radius buffer layout mismatch");

std::runtime_error failure(const char* operation, NSError* error = nil) {
  const char* detail = error ? [[error localizedDescription] UTF8String] : nullptr;
  return std::runtime_error(std::string("Metal radius ") + operation +
                            (detail ? std::string(": ") + detail : " failed"));
}

struct RadiusContext {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> pipeline;
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
    id<MTLFunction> function = [library newFunctionWithName:@"radius_kernel"];
    if (!function) throw failure("kernel lookup");
    pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) throw failure("pipeline creation", error);
  }
};

RadiusContext& context() {
  static RadiusContext state;
  return state;
}
}  // namespace

void sfgpu_metal_radius(const double* x, std::size_t nx,
                        const double* y, std::size_t ny, double radius,
                        std::size_t tile_bytes, SfgpuRadiusEmit emit,
                        void* state, bool (*interrupt)()) {
  @autoreleasepool {
    RadiusContext& ctx = context();
    if (!nx || !ny) return;
    if (tile_bytes < 49) throw std::runtime_error("Metal radius tile_bytes must be at least 49");
    // Bound each command's pair count as well as allocation size.
    constexpr std::size_t pair_cap = 65536;
    std::size_t tx = std::min<std::size_t>(nx, 256);
    std::size_t ty = std::min(ny, pair_cap / tx);
    tx = std::min(nx, pair_cap / ty);
    const auto max_buffer = static_cast<std::size_t>(ctx.device.maxBufferLength);
    while (32 * tx + 16 * ty + tx * ty > tile_bytes ||
           32 * tx > max_buffer || 16 * ty > max_buffer || tx * ty > max_buffer) {
      if (tx == 1 && ty == 1) throw failure("buffer limit");
      if (tx >= ty && tx > 1) tx = (tx + 1) / 2;
      else ty = (ty + 1) / 2;
    }
    id<MTLBuffer> bx = [ctx.device newBufferWithLength:32 * tx options:MTLResourceStorageModeShared];
    id<MTLBuffer> by = [ctx.device newBufferWithLength:16 * ty options:MTLResourceStorageModeShared];
    id<MTLBuffer> bo = [ctx.device newBufferWithLength:tx * ty options:MTLResourceStorageModeShared];
    if (!bx || !by || !bo) throw failure("tile allocation");
    auto* px = static_cast<KeyBounds*>([bx contents]);
    auto* py = static_cast<KeyCoordinates*>([by contents]);
    auto* mask = static_cast<unsigned char*>([bo contents]);
    if (!px || !py || !mask) throw failure("buffer mapping");
    const NSUInteger max_threads = ctx.pipeline.maxTotalThreadsPerThreadgroup;
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
            const auto b = sfgpu_radius_bounds(x[i + r], x[nx + i + r], radius);
            px[r] = {sfgpu_radius_ordered_key(b.xmin), sfgpu_radius_ordered_key(b.xmax),
                     sfgpu_radius_ordered_key(b.ymin), sfgpu_radius_ordered_key(b.ymax)};
          }
          const std::size_t pairs = static_cast<std::size_t>(rows) * cols;
          std::fill(mask, mask + pairs, static_cast<unsigned char>(2));
          id<MTLCommandBuffer> command = [ctx.queue commandBuffer];
          if (!command) throw failure("command creation");
          id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
          if (!encoder) throw failure("encoder creation");
          [encoder setComputePipelineState:ctx.pipeline];
          [encoder setBuffer:bx offset:0 atIndex:0];
          [encoder setBuffer:by offset:0 atIndex:1];
          [encoder setBuffer:bo offset:0 atIndex:2];
          [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
          [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
          [encoder dispatchThreadgroups:MTLSizeMake((rows + width - 1) / width,
                                                   (cols + height - 1) / height, 1)
                  threadsPerThreadgroup:group];
          [encoder endEncoding];
          [command commit];
          [command waitUntilCompleted];
          if (command.status != MTLCommandBufferStatusCompleted) {
            throw failure("command execution", command.error);
          }
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
}
