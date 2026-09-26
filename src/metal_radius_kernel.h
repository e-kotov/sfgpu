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
)MSL";
#endif
