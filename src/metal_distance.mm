#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_distance.h"
#include "metal_kernel.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {
struct Point { std::int64_t x, y; };
static_assert(sizeof(Point) == 16, "Metal point layout must be two int64 values");
constexpr std::uint64_t correction = UINT64_MAX;
constexpr std::uint64_t unwritten = UINT64_C(0x7ff8000000000000);
SfgpuMetalStats last_stats;

std::runtime_error failure(const char* operation, NSError* error = nil) {
  const char* detail = error ? [[error localizedDescription] UTF8String] : nullptr;
  return std::runtime_error(std::string("Metal ") + operation +
                            (detail ? std::string(": ") + detail : " failed"));
}

struct Context {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> pipeline;
  Context() {
    device = MTLCreateSystemDefaultDevice();
    if (!device) throw std::runtime_error("Metal has no available device");
    queue = [device newCommandQueue];
    if (!queue) throw failure("command queue creation");
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.fastMathEnabled = NO;
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:
        [NSString stringWithUTF8String:sfgpu_metal_source]
        options:options error:&error];
    if (!library) throw failure("shader compilation", error);
    id<MTLFunction> function = [library newFunctionWithName:@"distance_kernel"];
    if (!function) throw failure("distance kernel lookup");
    pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) throw failure("pipeline creation", error);
  }
};

Context& context() {
  // R calls the backend on its main thread. ARC owns the cached objects and
  // releases them when this shared library is unloaded.
  static Context state;
  return state;
}

bool encode(double value, int quantum, std::int64_t& result) {
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const unsigned biased = static_cast<unsigned>((bits >> 52) & 0x7ffU);
  std::uint64_t significand = bits & UINT64_C(0x000fffffffffffff);
  if (biased == 0x7ffU) return false;
  int exponent = -1074;
  if (biased) {
    significand |= UINT64_C(1) << 52;
    exponent = static_cast<int>(biased) - 1023 - 52;
  }
  if (!significand) { result = 0; return true; }
  const int shift = exponent - quantum;
  constexpr std::uint64_t bound = UINT64_C(1) << 61;
  std::uint64_t magnitude;
  if (shift >= 0) {
    if (shift > 61 || significand > (bound >> shift)) return false;
    magnitude = significand << shift;
  } else {
    const int drop = -shift;
    if (drop >= 64 || (significand & ((UINT64_C(1) << drop) - 1))) return false;
    magnitude = significand >> drop;
  }
  if (magnitude > bound) return false;
  result = static_cast<std::int64_t>(magnitude);
  if (bits >> 63) result = -result;
  // Independent round-trip check also protects future changes to the encoder.
  return std::ldexp(static_cast<double>(result), quantum) == value;
}

void pack(const double* input, std::size_t stride, std::size_t offset,
          std::size_t count, int quantum, Point* output) {
  for (std::size_t i = 0; i < count; ++i) {
    if (!encode(input[offset + i], quantum, output[i].x) ||
        !encode(input[stride + offset + i], quantum, output[i].y)) {
      output[i].x = std::numeric_limits<std::int64_t>::min();
      output[i].y = std::numeric_limits<std::int64_t>::min();
    }
  }
}

std::size_t bytes(std::size_t nx, std::size_t ny) {
  // Callers cap each dimension and the pair count at 65536 before using this.
  return 16 * nx + 16 * ny + 8 * nx * ny;
}
}  // namespace

bool sfgpu_metal_info(std::string& device, std::string& reason) {
  @autoreleasepool {
    try {
      Context& ctx = context();
      device = [[ctx.device name] UTF8String];
      reason.clear();
      return true;
    } catch (const std::exception& e) {
      device.clear();
      reason = e.what();
      return false;
    }
  }
}

SfgpuMetalStats sfgpu_metal_stats() { return last_stats; }

void sfgpu_metal_distance(const double* x, std::size_t nx,
                          const double* y, std::size_t ny, double* out,
                          std::size_t tile_bytes, bool (*interrupt)()) {
  last_stats = SfgpuMetalStats{};
  @autoreleasepool {
    Context& ctx = context(); // Explicit backend requires a device even if empty.
    if (!nx || !ny) return;
    if (tile_bytes < 40) throw std::runtime_error("Metal tile_bytes must be at least 40");
    const std::size_t pair_cap = 65536; // Bound command duration; not a tuned crossover.
    std::size_t tx = std::min<std::size_t>(nx, 256);
    std::size_t ty = std::min(ny, pair_cap / tx);
    tx = std::min(nx, pair_cap / ty);
    const std::size_t max_buffer = static_cast<std::size_t>(ctx.device.maxBufferLength);
    while (bytes(tx, ty) > tile_bytes || 16 * tx > max_buffer ||
           16 * ty > max_buffer || 8 * tx * ty > max_buffer) {
      if (tx == 1 && ty == 1) throw std::runtime_error("Metal device buffer limit is too small");
      if (tx >= ty && tx > 1) tx = (tx + 1) / 2;
      else ty = (ty + 1) / 2;
    }
    id<MTLBuffer> bx = [ctx.device newBufferWithLength:16 * tx options:MTLResourceStorageModeShared];
    id<MTLBuffer> by = [ctx.device newBufferWithLength:16 * ty options:MTLResourceStorageModeShared];
    id<MTLBuffer> bo = [ctx.device newBufferWithLength:8 * tx * ty options:MTLResourceStorageModeShared];
    if (!bx || !by || !bo) throw failure("tile buffer allocation");
    auto* px = static_cast<Point*>([bx contents]);
    auto* py = static_cast<Point*>([by contents]);
    auto* po = static_cast<std::uint64_t*>([bo contents]);
    if (!px || !py || !po) throw failure("shared buffer mapping");
    const NSUInteger max_threads = ctx.pipeline.maxTotalThreadsPerThreadgroup;
    if (!max_threads) throw failure("threadgroup limit");
    const NSUInteger width = std::min<NSUInteger>(16, max_threads);
    const NSUInteger height = std::min<NSUInteger>(16, max_threads / width);
    const MTLSize group = MTLSizeMake(width, height, 1);
    for (std::size_t j = 0; j < ny; j += ty) {
      const std::uint32_t cols = static_cast<std::uint32_t>(std::min(ty, ny - j));
      for (std::size_t i = 0; i < nx; i += tx) {
        @autoreleasepool {
          if (!interrupt()) throw std::runtime_error("Metal distance computation interrupted");
          const std::uint32_t rows = static_cast<std::uint32_t>(std::min(tx, nx - i));
          double max_abs = 0;
          for (std::size_t k = 0; k < rows; ++k) {
            max_abs = std::max(max_abs, std::abs(x[i + k]));
            max_abs = std::max(max_abs, std::abs(x[nx + i + k]));
          }
          for (std::size_t k = 0; k < cols; ++k) {
            max_abs = std::max(max_abs, std::abs(y[j + k]));
            max_abs = std::max(max_abs, std::abs(y[ny + j + k]));
          }
          int exponent = 0;
          if (max_abs) std::frexp(max_abs, &exponent);
          const std::int32_t quantum = max_abs ? exponent - 61 : 0;
          pack(x, nx, i, rows, quantum, px);
          pack(y, ny, j, cols, quantum, py);
          const std::size_t pairs = static_cast<std::size_t>(rows) * cols;
          std::fill(po, po + pairs, unwritten);
          id<MTLCommandBuffer> command = [ctx.queue commandBuffer];
          if (!command) throw failure("command buffer creation");
          id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
          if (!encoder) throw failure("compute encoder creation");
          [encoder setComputePipelineState:ctx.pipeline];
          [encoder setBuffer:bx offset:0 atIndex:0];
          [encoder setBuffer:by offset:0 atIndex:1];
          [encoder setBuffer:bo offset:0 atIndex:2];
          [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
          [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
          [encoder setBytes:&quantum length:sizeof(quantum) atIndex:5];
          [encoder dispatchThreadgroups:MTLSizeMake((rows + width - 1) / width,
                                                   (cols + height - 1) / height, 1)
                  threadsPerThreadgroup:group];
          [encoder endEncoding];
          [command commit];
          [command waitUntilCompleted];
          if (command.status != MTLCommandBufferStatusCompleted) {
            throw failure("command execution", command.error);
          }
          ++last_stats.tiles;
          for (std::size_t c = 0; c < cols; ++c) {
            for (std::size_t r = 0; r < rows; ++r) {
              const std::uint64_t bits = po[c * rows + r];
              double value;
              if (bits == correction) {
                value = std::hypot(x[i + r] - y[j + c], x[nx + i + r] - y[ny + j + c]);
                ++last_stats.cpu_pairs;
              } else {
                std::memcpy(&value, &bits, sizeof(value));
                if (!std::isfinite(value) || value < 0) throw failure("invalid shader output");
                ++last_stats.gpu_pairs;
              }
              out[(j + c) * nx + i + r] = value;
            }
          }
        }
      }
    }
  }
}
