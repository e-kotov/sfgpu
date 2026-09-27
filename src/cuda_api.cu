#include "sfgpu_cuda_api.h"
#include "cuda_distance.h"
#include "cuda_radius.h"

#include <cstdio>
#include <exception>
#include <string>

#ifndef SFGPU_BUILD_ID
#define SFGPU_BUILD_ID "cuda12-abi1"
#endif

#ifdef _WIN32
#define SFGPU_EXPORT __declspec(dllexport)
#else
#define SFGPU_EXPORT __attribute__((visibility("default")))
#endif

namespace {

struct CallbackFailed {};

sfgpu_interrupt_fn g_interrupt = nullptr;
bool interrupt_tramp() {
  if (!g_interrupt) return true;
  return g_interrupt() != 0;
}

struct EmitCtx {
  sfgpu_emit_fn fn;
  void* state;
};

void emit_tramp(std::size_t i, std::size_t j, void* p) {
  auto* c = static_cast<EmitCtx*>(p);
  if (c->fn(static_cast<uint64_t>(i), static_cast<uint64_t>(j), c->state) != 0) {
    throw CallbackFailed{};
  }
}

void copy_str(char* dst, size_t n, const std::string& s) {
  if (dst && n > 0) {
    std::snprintf(dst, n, "%s", s.c_str());
  }
}

template <class F>
int guarded(char* err, size_t n, F&& f) {
  try {
    f();
    return SFGPU_CUDA_OK;
  } catch (const CallbackFailed&) {
    return SFGPU_CUDA_CALLBACK_FAILED;
  } catch (const std::exception& e) {
    copy_str(err, n, e.what());
    if (std::string(e.what()).find("interrupted") != std::string::npos) {
      return SFGPU_CUDA_INTERRUPTED;
    }
    return SFGPU_CUDA_ERROR;
  } catch (...) {
    copy_str(err, n, "unknown CUDA error");
    return SFGPU_CUDA_ERROR;
  }
}

int api_info(char* dev, size_t dev_len, char* reason, size_t reason_len) {
  std::string device;
  std::string why;
  int ok = 0;
  int rc = guarded(reason, reason_len, [&] {
    ok = sfgpu_cuda_info(device, why) ? 1 : 0;
  });
  copy_str(dev, dev_len, device);
  if (rc == SFGPU_CUDA_OK) {
    copy_str(reason, reason_len, why);
    return ok;
  }
  return 0;
}

int api_distance(const double* x, uint64_t nx, const double* y, uint64_t ny,
                 double* out, uint64_t tile_bytes, sfgpu_interrupt_fn intr,
                 char* err, size_t err_len) {
  g_interrupt = intr;
  return guarded(err, err_len, [&] {
    sfgpu_cuda_distance(x, static_cast<std::size_t>(nx),
                        y, static_cast<std::size_t>(ny),
                        out, static_cast<std::size_t>(tile_bytes),
                        interrupt_tramp);
  });
}

int api_radius(const double* x, uint64_t nx, const double* y, uint64_t ny,
               double radius, uint64_t tile_bytes, sfgpu_emit_fn emit, void* state,
               sfgpu_interrupt_fn intr, char* err, size_t err_len) {
  g_interrupt = intr;
  EmitCtx ctx{emit, state};
  return guarded(err, err_len, [&] {
    sfgpu_cuda_radius(x, static_cast<std::size_t>(nx),
                      y, static_cast<std::size_t>(ny),
                      radius, static_cast<std::size_t>(tile_bytes),
                      emit_tramp, &ctx, interrupt_tramp);
  });
}

void api_radius_stats(double* indexed_waves, double* tiled_tiles) {
  auto stats = sfgpu_cuda_radius_stats();
  if (indexed_waves) *indexed_waves = stats.indexed_waves;
  if (tiled_tiles) *tiled_tiles = stats.tiled_tiles;
}

const sfgpu_cuda_api kApi = {
  SFGPU_CUDA_ABI_VERSION,
  sizeof(sfgpu_cuda_api),
  SFGPU_BUILD_ID,
  api_info,
  api_distance,
  api_radius,
  api_radius_stats
};

}  // namespace

extern "C" SFGPU_EXPORT const sfgpu_cuda_api* sfgpu_cuda_get_api(uint32_t abi) {
  return (abi == SFGPU_CUDA_ABI_VERSION) ? &kApi : nullptr;
}
