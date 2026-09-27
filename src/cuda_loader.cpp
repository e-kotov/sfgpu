#include <R.h>
#include <Rinternals.h>
#include <string>

#include "sfgpu_cuda_api.h"

const sfgpu_cuda_api* g_cuda = nullptr;
std::string g_cuda_load_error = "CUDA companion DLL not loaded; run sfgpu::sfgpu_install_cuda()";

#if defined(_WIN32) || defined(SFGPU_CUDA_DYNAMIC)

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

extern "C" SEXP C_sfgpu_cuda_loaded() {
  return Rf_ScalarLogical(g_cuda != nullptr);
}

extern "C" SEXP C_sfgpu_cuda_load(SEXP path) {
  if (g_cuda != nullptr) {
    return Rf_ScalarLogical(TRUE);
  }
  if (TYPEOF(path) != STRSXP || XLENGTH(path) != 1) {
    Rf_error("path must be a single string");
  }

  const char* p = Rf_translateCharUTF8(STRING_ELT(path, 0));
  int wide_len = MultiByteToWideChar(CP_UTF8, 0, p, -1, nullptr, 0);
  if (wide_len <= 0) {
    g_cuda_load_error = "failed to convert DLL path to UTF-16";
    return Rf_ScalarLogical(FALSE);
  }

  std::wstring wide_path(static_cast<std::size_t>(wide_len), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, p, -1, &wide_path[0], wide_len);

  // Strip trailing null if present
  if (!wide_path.empty() && wide_path.back() == L'\0') {
    wide_path.pop_back();
  }

  HMODULE h = LoadLibraryExW(
      wide_path.c_str(), nullptr,
      LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);

  if (!h) {
    DWORD err = GetLastError();
    g_cuda_load_error = "LoadLibraryExW failed for '" + std::string(p) +
                        "' (Win32 error " + std::to_string(err) + ")";
    return Rf_ScalarLogical(FALSE);
  }

  // Intermediate cast avoids -Wcast-function-type under strict GCC compiler flags
  auto get_api = reinterpret_cast<sfgpu_cuda_get_api_fn>(
      reinterpret_cast<void (*)(void)>(GetProcAddress(h, "sfgpu_cuda_get_api")));

  if (!get_api) {
    g_cuda_load_error = "entrypoint 'sfgpu_cuda_get_api' not found in '" +
                        std::string(p) + "'";
    return Rf_ScalarLogical(FALSE);
  }

  const sfgpu_cuda_api* api = get_api(SFGPU_CUDA_ABI_VERSION);
  if (!api || api->struct_size < sizeof(sfgpu_cuda_api)) {
    g_cuda_load_error = "CUDA companion DLL ABI mismatch; run sfgpu::sfgpu_install_cuda(force = TRUE)";
    return Rf_ScalarLogical(FALSE);
  }

  g_cuda = api;
  return Rf_ScalarLogical(TRUE);
}

#else
// Non-Windows dynamic stub (e.g. for development/mocking)
extern "C" SEXP C_sfgpu_cuda_loaded() {
  return Rf_ScalarLogical(g_cuda != nullptr);
}

extern "C" SEXP C_sfgpu_cuda_load(SEXP path) {
  g_cuda_load_error = "dynamic CUDA loading is only supported on Windows";
  return Rf_ScalarLogical(FALSE);
}
#endif

#else

// Stubs when SFGPU_CUDA_DYNAMIC is not active (e.g. standard Linux / macOS builds)
extern "C" SEXP C_sfgpu_cuda_loaded() {
  return Rf_ScalarLogical(FALSE);
}

extern "C" SEXP C_sfgpu_cuda_load(SEXP path) {
  return Rf_ScalarLogical(FALSE);
}

#endif
