#ifndef SFGPU_METAL_RADIUS_KERNEL_H
#define SFGPU_METAL_RADIUS_KERNEL_H

static const char* sfgpu_metal_radius_source = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct Bounds { ulong xmin, xmax, ymin, ymax; };
struct Coordinates { ulong x, y; };
kernel void radius_kernel(
    device const Bounds* bounds [[buffer(0)]],
    device const Coordinates* points [[buffer(1)]],
    device uchar* mask [[buffer(2)]],
    constant uint& nx [[buffer(3)]],
    constant uint& ny [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= nx || gid.y >= ny) return;
  Bounds b = bounds[gid.x];
  Coordinates p = points[gid.y];
  mask[ulong(gid.y) * ulong(nx) + ulong(gid.x)] =
      (p.x >= b.xmin && p.x <= b.xmax && p.y >= b.ymin && p.y <= b.ymax);
}

struct IndexedY { ulong x, y; uint original_j, padding; };
struct Range { uint first, last; };
struct Task { uint local_i, first, count; };
struct Pair { uint local_i, original_j; };

kernel void radius_index_ranges(
    device const Bounds* bounds [[buffer(0)]],
    device const IndexedY* points [[buffer(1)]],
    device Range* ranges [[buffer(2)]],
    constant uint& nx [[buffer(3)]],
    constant uint& ny [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= nx) return;
  Bounds b = bounds[gid];
  uint low = 0U, high = ny;
  while (low < high) {
    uint mid = low + (high - low) / 2U;
    if (points[mid].x < b.xmin) low = mid + 1U;
    else high = mid;
  }
  uint first = low;
  high = ny;
  while (low < high) {
    uint mid = low + (high - low) / 2U;
    if (points[mid].x <= b.xmax) low = mid + 1U;
    else high = mid;
  }
  ranges[gid] = {first, low};
}

kernel void radius_index_filter(
    device const Bounds* bounds [[buffer(0)]],
    device const IndexedY* points [[buffer(1)]],
    device const Task* tasks [[buffer(2)]],
    device Pair* output [[buffer(3)]],
    device atomic_uint* count [[buffer(4)]],
    constant uint& capacity [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group_size [[threads_per_threadgroup]]) {
  Task task = tasks[group.x];
  Bounds b = bounds[task.local_i];
  for (uint k = lane; k < task.count; k += group_size.x) {
    IndexedY point = points[task.first + k];
    if (point.y >= b.ymin && point.y <= b.ymax) {
      uint slot = atomic_fetch_add_explicit(count, 1U, memory_order_relaxed);
      if (slot < capacity) output[slot] = {task.local_i, point.original_j};
    }
  }
}
)MSL";
#endif
