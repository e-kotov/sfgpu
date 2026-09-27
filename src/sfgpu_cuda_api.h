#ifndef SFGPU_CUDA_API_H
#define SFGPU_CUDA_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SFGPU_CUDA_ABI_VERSION 1u

enum {
  SFGPU_CUDA_OK = 0,
  SFGPU_CUDA_ERROR = 1,
  SFGPU_CUDA_INTERRUPTED = 2,
  SFGPU_CUDA_CALLBACK_FAILED = 3
};

/* Both callbacks are invoked only on the calling (R main) thread and must not throw. */
typedef int (*sfgpu_interrupt_fn)(void);                          /* 1 = continue, 0 = interrupt */
typedef int (*sfgpu_emit_fn)(uint64_t i, uint64_t j, void* state); /* 0 = ok, non-zero = fail */

typedef struct sfgpu_cuda_api {
  uint32_t abi_version;
  uint32_t struct_size;
  const char* build_id; /* static string owned by the DLL: source SHA, CUDA version, archs */

  /* Returns 1 if device available, 0 if unavailable or error. Fills device and reason buffers. */
  int (*info)(char* device, size_t device_len, char* reason, size_t reason_len);

  /* Distance computation: returns SFGPU_CUDA_* status code. */
  int (*distance)(const double* x, uint64_t nx, const double* y, uint64_t ny,
                  double* out, uint64_t tile_bytes, sfgpu_interrupt_fn interrupt,
                  char* err, size_t err_len);

  /* Radius computation: returns SFGPU_CUDA_* status code. */
  int (*radius)(const double* x, uint64_t nx, const double* y, uint64_t ny,
                double radius, uint64_t tile_bytes, sfgpu_emit_fn emit, void* state,
                sfgpu_interrupt_fn interrupt, char* err, size_t err_len);

  /* Radius statistics */
  void (*radius_stats)(double* indexed_waves, double* tiled_tiles);
} sfgpu_cuda_api;

typedef const sfgpu_cuda_api* (*sfgpu_cuda_get_api_fn)(uint32_t requested_abi);

#ifdef __cplusplus
}
#endif

#endif /* SFGPU_CUDA_API_H */
