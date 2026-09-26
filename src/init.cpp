#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

extern "C" SEXP C_sfgpu_distance(SEXP, SEXP, SEXP, SEXP);
extern "C" SEXP C_sfgpu_cuda_info();

static const R_CallMethodDef call_methods[] = {
    {"C_sfgpu_distance", reinterpret_cast<DL_FUNC>(&C_sfgpu_distance), 4},
    {"C_sfgpu_cuda_info", reinterpret_cast<DL_FUNC>(&C_sfgpu_cuda_info), 0},
    {nullptr, nullptr, 0}};

extern "C" void R_init_sfgpu(DllInfo* dll) {
  R_registerRoutines(dll, nullptr, call_methods, nullptr, nullptr);
  R_useDynamicSymbols(dll, FALSE);
  R_forceSymbols(dll, TRUE);
}
