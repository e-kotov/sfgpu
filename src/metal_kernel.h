#ifndef SFGPU_METAL_KERNEL_H
#define SFGPU_METAL_KERNEL_H

// Independently written fixed-point Euclidean distance kernel. Each accepted
// input coordinate is an exact signed integer multiple of 2^quantum. The host
// rejects coordinates that cannot be represented within the 61-bit bound.
// This source is compiled at runtime, so source installs need no xcrun metal.
static const char* sfgpu_metal_source = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Point { long x; long y; };
struct U128 { ulong lo; ulong hi; };

constant ulong correction = 0xffffffffffffffffUL;
constant long invalid_point = (-9223372036854775807L - 1L);

U128 square64(ulong a) {
  // Four 32 x 32 products, with explicit carries; no truncated 64-bit square.
  ulong a0 = a & 0xffffffffUL;
  ulong a1 = a >> 32;
  ulong p00 = a0 * a0;
  ulong p01 = a0 * a1;
  ulong p11 = a1 * a1;
  ulong lo = p00;
  ulong add = p01 << 32;
  lo += add;
  ulong carry = ulong(lo < add);
  ulong previous = lo;
  lo += add;
  carry += ulong(lo < previous);
  U128 value = {lo, p11 + 2UL * (p01 >> 32) + carry};
  return value;
}

U128 add128(U128 a, U128 b) {
  U128 result;
  result.lo = a.lo + b.lo;
  result.hi = a.hi + b.hi + ulong(result.lo < a.lo);
  return result;
}

bool le128(U128 a, U128 b) {
  return a.hi < b.hi || (a.hi == b.hi && a.lo <= b.lo);
}

int bit_length64(ulong x) {
  int bits = 0;
  while (x != 0UL) { ++bits; x >>= 1; }
  return bits;
}

int bit_length128(U128 x) {
  return x.hi == 0UL ? bit_length64(x.lo) : 64 + bit_length64(x.hi);
}

U128 left128(U128 x, int shift) {
  if (shift == 0) return x;
  if (shift < 64) {
    U128 result = {x.lo << shift, (x.hi << shift) | (x.lo >> (64 - shift))};
    return result;
  }
  U128 result = {0UL, x.lo << (shift - 64)};
  return result;
}

ulong isqrt128(U128 number) {
  // Binary search of the floor square root. Normalization below proves the
  // answer fits in 63 bits, so trial*trial is represented by U128.
  ulong root = 0UL;
  for (int bit = 62; bit >= 0; --bit) {
    ulong trial = root | (1UL << bit);
    if (le128(square64(trial), number)) root = trial;
  }
  return root;
}

ulong exact_axis_bits(ulong magnitude, int quantum) {
  if (magnitude == 0UL) return 0UL;
  int bits = bit_length64(magnitude);
  int exponent = quantum + bits - 1;
  if (exponent < -900 || exponent > 900) return correction;
  ulong significand;
  if (bits <= 53) {
    significand = magnitude << (53 - bits);
  } else {
    int drop_bits = bits - 53;
    ulong mask = (1UL << drop_bits) - 1UL;
    ulong dropped = magnitude & mask;
    ulong halfway = 1UL << (drop_bits - 1);
    significand = magnitude >> drop_bits;
    if (dropped > halfway ||
        (dropped == halfway && (significand & 1UL) != 0UL)) ++significand;
    if (significand == (1UL << 53)) {
      significand >>= 1;
      ++exponent;
    }
  }
  if (exponent > 900) return correction;
  return (ulong(exponent + 1023) << 52) | (significand & ((1UL << 52) - 1UL));
}

ulong distance_bits(Point a, Point b, int quantum) {
  if (a.x == invalid_point || a.y == invalid_point ||
      b.x == invalid_point || b.y == invalid_point) return correction;
  long signed_dx = a.x - b.x;
  long signed_dy = a.y - b.y;
  ulong dx = signed_dx < 0L ? ulong(-signed_dx) : ulong(signed_dx);
  ulong dy = signed_dy < 0L ? ulong(-signed_dy) : ulong(signed_dy);
  if (dx == 0UL) return exact_axis_bits(dy, quantum);
  if (dy == 0UL) return exact_axis_bits(dx, quantum);
  U128 sum = add128(square64(dx), square64(dy));
  int bits = bit_length128(sum);
  if (bits == 0) return 0UL;
  // An even shift preserves sqrt(sum) * 2^quantum while yielding a 63-bit
  // integer root. The floor root is <1 unit below the true scaled root.
  int shift = ((126 - bits) / 2) * 2;
  U128 scaled = left128(sum, shift);
  ulong root = isqrt128(scaled);
  int exponent = quantum - shift / 2 + 62;
  if (exponent < -900 || exponent > 900) return correction;
  // root has exactly 63 significant bits. Its unknown fractional remainder is
  // <1; correct on the host if it could straddle a binary64 midpoint.
  ulong dropped = root & 1023UL;
  if (dropped == 511UL || dropped == 512UL) return correction;
  ulong significand = root >> 10;
  if (dropped > 512UL) ++significand;
  if (significand == (1UL << 53)) {
    significand >>= 1;
    ++exponent;
  }
  if (exponent > 900) return correction;
  return (ulong(exponent + 1023) << 52) | (significand & ((1UL << 52) - 1UL));
}

kernel void distance_kernel(device const Point* x [[buffer(0)]],
                            device const Point* y [[buffer(1)]],
                            device ulong* out [[buffer(2)]],
                            constant uint& nx [[buffer(3)]],
                            constant uint& ny [[buffer(4)]],
                            constant int& quantum [[buffer(5)]],
                            uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= nx || gid.y >= ny) return;
  ulong result = distance_bits(x[gid.x], y[gid.y], quantum);
  out[ulong(gid.y) * ulong(nx) + ulong(gid.x)] = result;
}
)METAL";

#endif
