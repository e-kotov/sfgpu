#include <R.h>
#include <Rinternals.h>
#include <R_ext/Utils.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstddef>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef SFGPU_WITH_CUDA
#include "cuda_distance.h"
#endif
#include "sfgpu_cuda_api.h"
#include "metal_distance.h"

extern const sfgpu_cuda_api* g_cuda;
extern std::string g_cuda_load_error;

namespace {

void check_interrupt(void*) { R_CheckUserInterrupt(); }

bool interrupt_ok() {
  // R_ToplevelExec catches R's non-local jump and returns control to C++.
  // The caller then throws a C++ exception so CUDA RAII destructors run.
  return R_ToplevelExec(check_interrupt, nullptr) == TRUE;
}

SEXP row_names(SEXP matrix) {
  SEXP names = Rf_getAttrib(matrix, R_DimNamesSymbol);
  return names == R_NilValue ? R_NilValue : VECTOR_ELT(names, 0);
}

void cpu_distance(const double* x, std::size_t nx,
                  const double* y, std::size_t ny, double* out) {
  for (std::size_t j = 0; j < ny; ++j) {
    const double y0 = y[j];
    const double y1 = y[ny + j];
    for (std::size_t i = 0; i < nx; ++i) {
      out[j * nx + i] = std::hypot(x[i] - y0, x[nx + i] - y1);
    }
    if ((j & 255U) == 0U && !interrupt_ok()) {
      throw std::runtime_error("distance computation interrupted");
    }
  }
}

}  // namespace

extern "C" SEXP C_sfgpu_distance(SEXP x, SEXP y, SEXP backend, SEXP tile) {
  if (TYPEOF(x) != REALSXP || TYPEOF(y) != REALSXP ||
      TYPEOF(backend) != STRSXP || XLENGTH(backend) != 1 ||
      TYPEOF(tile) != REALSXP || XLENGTH(tile) != 1) {
    Rf_error("invalid native distance arguments");
  }
  SEXP xd = Rf_getAttrib(x, R_DimSymbol);
  SEXP yd = Rf_getAttrib(y, R_DimSymbol);
  if (xd == R_NilValue || yd == R_NilValue || LENGTH(xd) != 2 ||
      LENGTH(yd) != 2 || INTEGER(xd)[1] != 2 || INTEGER(yd)[1] != 2) {
    Rf_error("native distance inputs must be two-column matrices");
  }
  const int nx = INTEGER(xd)[0];
  const int ny = INTEGER(yd)[0];
  if (nx < 0 || ny < 0 ||
      static_cast<double>(nx) * static_cast<double>(ny) >
          static_cast<double>(R_XLEN_T_MAX)) {
    Rf_error("distance matrix exceeds R's matrix length limit");
  }
  const char* backend_name = CHAR(STRING_ELT(backend, 0));
  const bool use_cuda = std::strcmp(backend_name, "cuda") == 0;
  const bool use_metal = std::strcmp(backend_name, "metal") == 0;
  if (!use_cuda && !use_metal && std::strcmp(backend_name, "cpu") != 0) {
    Rf_error("unknown distance backend");
  }
  const double tile_value = REAL(tile)[0];
  if (!std::isfinite(tile_value) || tile_value < 40 ||
      tile_value > 9007199254740992.0 || std::floor(tile_value) != tile_value ||
      tile_value > static_cast<double>(std::numeric_limits<std::size_t>::max())) {
    Rf_error("native tile_bytes must be a whole number from 40 to 2^53");
  }
#if !defined(SFGPU_WITH_CUDA) && !defined(SFGPU_CUDA_DYNAMIC)
  if (use_cuda) {
    Rf_error("CUDA support is not compiled; reinstall with --enable-cuda");
  }
#elif defined(SFGPU_CUDA_DYNAMIC)
  if (use_cuda && !g_cuda) {
    Rf_error("%s", g_cuda_load_error.c_str());
  }
#endif
#ifndef SFGPU_WITH_METAL
  if (use_metal) {
    Rf_error("Metal support is not compiled; reinstall with --enable-metal");
  }
#endif
  SEXP out = PROTECT(Rf_allocMatrix(REALSXP, nx, ny));
  SEXP names = PROTECT(Rf_allocVector(VECSXP, 2));
  SET_VECTOR_ELT(names, 0, row_names(x));
  SET_VECTOR_ELT(names, 1, row_names(y));
  if (VECTOR_ELT(names, 0) != R_NilValue ||
      VECTOR_ELT(names, 1) != R_NilValue) {
    Rf_setAttrib(out, R_DimNamesSymbol, names);
  }

  char error_text[1024] = {0};
  {
    try {
      if (use_cuda) {
#ifdef SFGPU_WITH_CUDA
        sfgpu_cuda_distance(REAL(x), static_cast<std::size_t>(nx),
                            REAL(y), static_cast<std::size_t>(ny), REAL(out),
                            static_cast<std::size_t>(tile_value),
                            interrupt_ok);
#elif defined(SFGPU_CUDA_DYNAMIC)
        if (!g_cuda) {
          throw std::runtime_error(g_cuda_load_error);
        }
        char cuda_err[1024] = {0};
        auto intr_fn = []() -> int { return interrupt_ok() ? 1 : 0; };
        int rc = g_cuda->distance(REAL(x), static_cast<uint64_t>(nx),
                                  REAL(y), static_cast<uint64_t>(ny), REAL(out),
                                  static_cast<uint64_t>(tile_value),
                                  intr_fn, cuda_err, sizeof(cuda_err));
        if (rc == SFGPU_CUDA_INTERRUPTED) {
          throw std::runtime_error("distance computation interrupted");
        } else if (rc != SFGPU_CUDA_OK) {
          throw std::runtime_error(cuda_err[0] ? cuda_err : "CUDA distance computation failed");
        }
#endif
      } else if (use_metal) {
#ifdef SFGPU_WITH_METAL
        sfgpu_metal_distance(REAL(x), static_cast<std::size_t>(nx),
                             REAL(y), static_cast<std::size_t>(ny), REAL(out),
                             static_cast<std::size_t>(tile_value), interrupt_ok);
#endif
      } else {
        cpu_distance(REAL(x), static_cast<std::size_t>(nx), REAL(y),
                     static_cast<std::size_t>(ny), REAL(out));
      }
    } catch (const std::exception& e) {
      std::snprintf(error_text, sizeof(error_text), "%s", e.what());
    } catch (...) {
      std::snprintf(error_text, sizeof(error_text), "%s", "unknown native distance error");
    }
  }
  if (error_text[0] != '\0') {
    UNPROTECT(2);
    Rf_error("%s", error_text);
  }
  UNPROTECT(2);
  return out;
}

extern "C" SEXP C_sfgpu_metal_info() {
  bool compiled = false;
  bool available = false;
  char device_text[256] = {0};
  char reason_text[1024] = {0};
#ifdef SFGPU_WITH_METAL
  compiled = true;
  try {
    std::string device;
    std::string reason;
    available = sfgpu_metal_info(device, reason);
    std::snprintf(device_text, sizeof(device_text), "%s", device.c_str());
    std::snprintf(reason_text, sizeof(reason_text), "%s", reason.c_str());
  } catch (const std::exception& e) {
    std::snprintf(reason_text, sizeof(reason_text), "%s", e.what());
  }
#else
  std::snprintf(reason_text, sizeof(reason_text), "%s",
                "Metal support is not compiled; reinstall with --enable-metal");
#endif
  SEXP ans = PROTECT(Rf_allocVector(VECSXP, 4));
  SET_VECTOR_ELT(ans, 0, Rf_ScalarLogical(compiled));
  SET_VECTOR_ELT(ans, 1, Rf_ScalarLogical(available));
  SET_VECTOR_ELT(ans, 2, device_text[0] == '\0' ? Rf_ScalarString(NA_STRING) : Rf_mkString(device_text));
  SET_VECTOR_ELT(ans, 3, reason_text[0] == '\0' ? Rf_ScalarString(NA_STRING) : Rf_mkString(reason_text));
  UNPROTECT(1);
  return ans;
}

extern "C" SEXP C_sfgpu_metal_stats() {
  SfgpuMetalStats stats;
#ifdef SFGPU_WITH_METAL
  stats = sfgpu_metal_stats();
#endif
  SEXP ans = PROTECT(Rf_allocVector(VECSXP, 3));
  SEXP names = PROTECT(Rf_allocVector(STRSXP, 3));
  SET_VECTOR_ELT(ans, 0, Rf_ScalarReal(stats.gpu_pairs));
  SET_VECTOR_ELT(ans, 1, Rf_ScalarReal(stats.cpu_pairs));
  SET_VECTOR_ELT(ans, 2, Rf_ScalarReal(stats.tiles));
  SET_STRING_ELT(names, 0, Rf_mkChar("gpu_pairs"));
  SET_STRING_ELT(names, 1, Rf_mkChar("cpu_pairs"));
  SET_STRING_ELT(names, 2, Rf_mkChar("tiles"));
  Rf_setAttrib(ans, R_NamesSymbol, names);
  UNPROTECT(2);
  return ans;
}

extern "C" SEXP C_sfgpu_cuda_info() {
  bool compiled = false;
  bool available = false;
  char device_text[256] = {0};
  char reason_text[1024] = {0};
#if defined(SFGPU_WITH_CUDA)
  compiled = true;
  {
    try {
      std::string device;
      std::string reason;
      available = sfgpu_cuda_info(device, reason);
      std::snprintf(device_text, sizeof(device_text), "%s", device.c_str());
      std::snprintf(reason_text, sizeof(reason_text), "%s", reason.c_str());
    } catch (const std::exception& e) {
      std::snprintf(reason_text, sizeof(reason_text), "%s", e.what());
    }
  }
#elif defined(SFGPU_CUDA_DYNAMIC)
  compiled = true;
  if (!g_cuda) {
    available = false;
    std::snprintf(reason_text, sizeof(reason_text), "%s", g_cuda_load_error.c_str());
  } else {
    try {
      available = (g_cuda->info(device_text, sizeof(device_text),
                                reason_text, sizeof(reason_text)) != 0);
    } catch (...) {
      available = false;
      std::snprintf(reason_text, sizeof(reason_text), "%s", "unknown error querying CUDA device");
    }
  }
#else
  std::snprintf(reason_text, sizeof(reason_text), "%s",
                "CUDA support is not compiled; reinstall with --enable-cuda");
#endif
  SEXP ans = PROTECT(Rf_allocVector(VECSXP, 4));
  SET_VECTOR_ELT(ans, 0, Rf_ScalarLogical(compiled));
  SET_VECTOR_ELT(ans, 1, Rf_ScalarLogical(available));
  SET_VECTOR_ELT(ans, 2, device_text[0] == '\0' ? Rf_ScalarString(NA_STRING) : Rf_mkString(device_text));
  SET_VECTOR_ELT(ans, 3, reason_text[0] == '\0' ? Rf_ScalarString(NA_STRING) : Rf_mkString(reason_text));
  UNPROTECT(1);
  return ans;
}
