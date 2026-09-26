#ifndef SFGPU_RADIUS_H
#define SFGPU_RADIUS_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

struct SfgpuRadiusBounds {
  double xmin, xmax, ymin, ymax;
};

// A reported CPU hypot can be one rounding step below the mathematical
// length. Widen the radius, then round each coordinate endpoint outwards twice.
// The second endpoint step covers rounding of the coordinate subtraction.
inline SfgpuRadiusBounds sfgpu_radius_bounds(double x, double y, double radius) {
  const double negative = -std::numeric_limits<double>::infinity();
  const double positive = std::numeric_limits<double>::infinity();
  const double expanded = std::nextafter(radius, positive);
  auto lower = [negative, expanded](double value) {
    double bound = value - expanded;
    if (bound != negative) {
      bound = std::nextafter(std::nextafter(bound, negative), negative);
    }
    return bound;
  };
  auto upper = [positive, expanded](double value) {
    double bound = value + expanded;
    if (bound != positive) {
      bound = std::nextafter(std::nextafter(bound, positive), positive);
    }
    return bound;
  };
  return {lower(x), upper(x), lower(y), upper(y)};
}

// IEEE binary64 bit order mapped to monotonically increasing unsigned keys.
// All supplied coordinates are finite. The bound endpoints may be infinite.
inline std::uint64_t sfgpu_radius_ordered_key(double value) {
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits >> 63) ? ~bits : (bits ^ UINT64_C(0x8000000000000000));
}

using SfgpuRadiusEmit = void (*)(std::size_t i, std::size_t j, void* state);

#endif
