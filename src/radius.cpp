#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Utils.h>

#include "radius.h"
#include "cuda_radius.h"
#ifdef SFGPU_WITH_METAL
#include "metal_radius.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

struct RadiusStats {
  double total_pairs = 0;
  double candidate_pairs = 0;
  double accepted_pairs = 0;
};

RadiusStats last_stats;

void check_interrupt(void*) { R_CheckUserInterrupt(); }

bool interrupt_ok() {
  return R_ToplevelExec(check_interrupt, nullptr) == TRUE;
}

struct Collector {
  const double* x;
  const double* y;
  std::size_t nx;
  std::size_t ny;
  double radius;
  std::size_t max_hits;
  std::size_t hits = 0;
  std::size_t candidates = 0;
  std::vector<std::vector<int>> rows;

  Collector(const double* x_, const double* y_, std::size_t nx_,
            std::size_t ny_, double radius_, std::size_t max_hits_)
      : x(x_), y(y_), nx(nx_), ny(ny_), radius(radius_),
        max_hits(max_hits_), rows(nx_) {}

  void accept(std::size_t i, std::size_t j) {
    if (i >= nx || j >= ny) {
      throw std::runtime_error("radius backend emitted an out-of-range index");
    }
    ++candidates;
    if (std::hypot(x[i] - y[j], x[nx + i] - y[ny + j]) > radius) return;
    if (hits >= max_hits) {
      throw std::runtime_error("radius neighbour payload exceeds max_output_bytes");
    }
    rows[i].push_back(static_cast<int>(j + 1));
    ++hits;
  }

  static void emit(std::size_t i, std::size_t j, void* state) {
    static_cast<Collector*>(state)->accept(i, j);
  }
};

void cpu_radius(const double* x, std::size_t nx,
                const double* y, std::size_t ny, double radius,
                Collector& collector) {
  std::vector<std::size_t> order(ny);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [y](std::size_t a, std::size_t b) {
    return y[a] < y[b] || (y[a] == y[b] && a < b);
  });
  for (std::size_t i = 0; i < nx; ++i) {
    const SfgpuRadiusBounds bounds =
        sfgpu_radius_bounds(x[i], x[nx + i], radius);
    auto first = std::lower_bound(order.begin(), order.end(), bounds.xmin,
        [y](std::size_t j, double bound) { return y[j] < bound; });
    auto last = std::upper_bound(first, order.end(), bounds.xmax,
        [y](double bound, std::size_t j) { return bound < y[j]; });
    for (auto it = first; it != last; ++it) {
      const std::size_t j = *it;
      if (y[ny + j] >= bounds.ymin && y[ny + j] <= bounds.ymax) {
        collector.accept(i, j);
      }
    }
    if ((i & 255U) == 0U && !interrupt_ok()) {
      throw std::runtime_error("radius computation interrupted");
    }
  }
}

SEXP row_names(SEXP matrix) {
  SEXP names = Rf_getAttrib(matrix, R_DimNamesSymbol);
  return names == R_NilValue ? R_NilValue : VECTOR_ELT(names, 0);
}

struct OutputState {
  const std::vector<std::vector<int>>* rows;
  SEXP out;
};

// R allocation failures must return through C++ before native rows are destroyed.
// The outer list is protected by the caller before any native resources exist.
void fill_output(void* data) {
  auto* state = static_cast<OutputState*>(data);
  for (std::size_t i = 0; i < state->rows->size(); ++i) {
    const auto& row = (*state->rows)[i];
    SEXP neighbours = PROTECT(Rf_allocVector(INTSXP, row.size()));
    std::copy(row.begin(), row.end(), INTEGER(neighbours));
    SET_VECTOR_ELT(state->out, i, neighbours);
    UNPROTECT(1);
  }
}

}  // namespace

extern "C" SEXP C_sfgpu_within_distance(SEXP x, SEXP y, SEXP backend,
                                          SEXP radius, SEXP cap, SEXP tile) {
  if (TYPEOF(x) != REALSXP || TYPEOF(y) != REALSXP ||
      TYPEOF(backend) != STRSXP || XLENGTH(backend) != 1 ||
      TYPEOF(radius) != REALSXP || XLENGTH(radius) != 1 ||
      TYPEOF(cap) != REALSXP || XLENGTH(cap) != 1 ||
      TYPEOF(tile) != REALSXP || XLENGTH(tile) != 1) {
    Rf_error("invalid native radius arguments");
  }
  SEXP xd = Rf_getAttrib(x, R_DimSymbol);
  SEXP yd = Rf_getAttrib(y, R_DimSymbol);
  if (xd == R_NilValue || yd == R_NilValue || LENGTH(xd) != 2 ||
      LENGTH(yd) != 2 || INTEGER(xd)[1] != 2 || INTEGER(yd)[1] != 2) {
    Rf_error("native radius inputs must be two-column matrices");
  }
  const int nx_int = INTEGER(xd)[0];
  const int ny_int = INTEGER(yd)[0];
  if (nx_int < 0 || ny_int < 0) Rf_error("invalid native matrix dimensions");
  const std::size_t nx = static_cast<std::size_t>(nx_int);
  const std::size_t ny = static_cast<std::size_t>(ny_int);
  const double radius_value = REAL(radius)[0];
  if (!std::isfinite(radius_value) || radius_value < 0) {
    Rf_error("native radius must be finite and nonnegative");
  }
  const double cap_value = REAL(cap)[0];
  if (!std::isfinite(cap_value) || cap_value < 0 ||
      cap_value > 9007199254740992.0 || std::floor(cap_value) != cap_value) {
    Rf_error("native max_output_bytes must be a whole number from 0 to 2^53");
  }
  const double tile_value = REAL(tile)[0];
  if (!std::isfinite(tile_value) || tile_value < 49 ||
      tile_value > 9007199254740992.0 || std::floor(tile_value) != tile_value ||
      tile_value > static_cast<double>(std::numeric_limits<std::size_t>::max())) {
    Rf_error("native tile_bytes must be a whole number from 49 to 2^53");
  }
  const char* backend_name = CHAR(STRING_ELT(backend, 0));
  const bool use_cuda = std::strcmp(backend_name, "cuda") == 0;
  const bool use_metal = std::strcmp(backend_name, "metal") == 0;
  if (!use_cuda && !use_metal && std::strcmp(backend_name, "cpu") != 0) {
    Rf_error("unknown radius backend");
  }
#ifndef SFGPU_WITH_CUDA
  if (use_cuda) Rf_error("CUDA support is not compiled; reinstall with --enable-cuda");
#endif
#ifndef SFGPU_WITH_METAL
  if (use_metal) Rf_error("Metal support is not compiled; reinstall with --enable-metal");
#endif
  const double base_bytes = 8.0 * static_cast<double>(nx);
  if (base_bytes > cap_value) {
    Rf_error("radius neighbour payload exceeds max_output_bytes");
  }
  const std::size_t max_hits =
      static_cast<std::size_t>(std::floor((cap_value - base_bytes) / 4.0));

  SEXP out = PROTECT(Rf_allocVector(VECSXP, nx_int));
  SEXP names = row_names(x);
  if (names != R_NilValue) Rf_setAttrib(out, R_NamesSymbol, names);
  char error_text[1024] = {0};
  {
    std::vector<std::vector<int>> rows;
    try {
      Collector collector(REAL(x), REAL(y), nx, ny, radius_value, max_hits);
      if (use_cuda) {
#ifdef SFGPU_WITH_CUDA
        sfgpu_cuda_radius(REAL(x), nx, REAL(y), ny, radius_value,
                          static_cast<std::size_t>(tile_value),
                          Collector::emit, &collector, interrupt_ok);
#endif
      } else if (use_metal) {
#ifdef SFGPU_WITH_METAL
        sfgpu_metal_radius(REAL(x), nx, REAL(y), ny, radius_value,
                           static_cast<std::size_t>(tile_value),
                           Collector::emit, &collector, interrupt_ok);
#endif
      } else {
        cpu_radius(REAL(x), nx, REAL(y), ny, radius_value, collector);
      }
      for (auto& row : collector.rows) std::sort(row.begin(), row.end());
      rows = std::move(collector.rows);
      OutputState output{&rows, out};
      if (!R_ToplevelExec(fill_output, &output)) {
        throw std::runtime_error("radius result allocation failed");
      }
      last_stats = {static_cast<double>(nx) * static_cast<double>(ny),
                    static_cast<double>(collector.candidates),
                    static_cast<double>(collector.hits)};
    } catch (const std::exception& e) {
      std::snprintf(error_text, sizeof(error_text), "%s", e.what());
    } catch (...) {
      std::snprintf(error_text, sizeof(error_text), "%s", "unknown native radius error");
    }
  } // Native row storage is released before any R non-local jump.
  if (error_text[0]) {
    UNPROTECT(1);
    Rf_error("%s", error_text);
  }

  UNPROTECT(1);
  return out;
}

extern "C" SEXP C_sfgpu_radius_stats() {
  SEXP out = PROTECT(Rf_allocVector(VECSXP, 3));
  SEXP names = PROTECT(Rf_allocVector(STRSXP, 3));
  SET_VECTOR_ELT(out, 0, Rf_ScalarReal(last_stats.total_pairs));
  SET_VECTOR_ELT(out, 1, Rf_ScalarReal(last_stats.candidate_pairs));
  SET_VECTOR_ELT(out, 2, Rf_ScalarReal(last_stats.accepted_pairs));
  SET_STRING_ELT(names, 0, Rf_mkChar("total_pairs"));
  SET_STRING_ELT(names, 1, Rf_mkChar("candidate_pairs"));
  SET_STRING_ELT(names, 2, Rf_mkChar("accepted_pairs"));
  Rf_setAttrib(out, R_NamesSymbol, names);
  UNPROTECT(2);
  return out;
}

extern "C" SEXP C_sfgpu_cuda_radius_stats() {
  SEXP out = PROTECT(Rf_allocVector(VECSXP, 2));
  SEXP names = PROTECT(Rf_allocVector(STRSXP, 2));
  SfgpuCudaRadiusStats stats;
#ifdef SFGPU_WITH_CUDA
  stats = sfgpu_cuda_radius_stats();
#endif
  SET_VECTOR_ELT(out, 0, Rf_ScalarReal(stats.indexed_waves));
  SET_VECTOR_ELT(out, 1, Rf_ScalarReal(stats.tiled_tiles));
  SET_STRING_ELT(names, 0, Rf_mkChar("indexed_waves"));
  SET_STRING_ELT(names, 1, Rf_mkChar("tiled_tiles"));
  Rf_setAttrib(out, R_NamesSymbol, names);
  UNPROTECT(2);
  return out;
}
