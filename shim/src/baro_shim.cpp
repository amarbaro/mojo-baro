#include "baro_shim.h"

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <string>

namespace {
thread_local std::string g_last_error;

void set_error(const char *what) { g_last_error = what; }
} // namespace

struct baro_ctx {
  hipStream_t stream;
  hipblasLtHandle_t lt;
  int device_id;
};

extern "C" baro_ctx *baro_init(int device_id) {
  if (hipSetDevice(device_id) != hipSuccess) {
    set_error("hipSetDevice failed");
    return nullptr;
  }
  auto *ctx = new baro_ctx{};
  ctx->device_id = device_id;
  if (hipStreamCreate(&ctx->stream) != hipSuccess) {
    set_error("hipStreamCreate failed");
    delete ctx;
    return nullptr;
  }
  if (hipblasLtCreate(&ctx->lt) != HIPBLAS_STATUS_SUCCESS) {
    set_error("hipblasLtCreate failed");
    (void)hipStreamDestroy(ctx->stream);
    delete ctx;
    return nullptr;
  }
  return ctx;
}

extern "C" void baro_destroy(baro_ctx *ctx) {
  if (!ctx)
    return;
  hipblasLtDestroy(ctx->lt);
  (void)hipStreamDestroy(ctx->stream);
  delete ctx;
}

extern "C" int baro_gemm_f16(baro_ctx *ctx, int m, int n, int k,
                             const void *a, const void *b, void *c,
                             float alpha, float beta) {
  if (!ctx || !a || !b || !c) {
    set_error("null argument");
    return -1;
  }

  // Row-major A(m,k) * B(k,n) is computed as the column-major product
  // B'(n,k) * A'(k,m), which hipBLASLt sees as an ordinary NN GEMM.
  hipblasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  hipblasLtMatmulDesc_t desc = nullptr;
  int rc = -1;

  auto cleanup = [&] {
    if (desc) hipblasLtMatmulDescDestroy(desc);
    if (lc) hipblasLtMatrixLayoutDestroy(lc);
    if (lb) hipblasLtMatrixLayoutDestroy(lb);
    if (la) hipblasLtMatrixLayoutDestroy(la);
  };

  if (hipblasLtMatrixLayoutCreate(&lb, HIP_R_16F, n, k, n) !=
          HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatrixLayoutCreate(&la, HIP_R_16F, k, m, k) !=
          HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatrixLayoutCreate(&lc, HIP_R_16F, n, m, n) !=
          HIPBLAS_STATUS_SUCCESS) {
    set_error("hipblasLtMatrixLayoutCreate failed");
    cleanup();
    return rc;
  }

  if (hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) !=
      HIPBLAS_STATUS_SUCCESS) {
    set_error("hipblasLtMatmulDescCreate failed");
    cleanup();
    return rc;
  }

  hipblasOperation_t op = HIPBLAS_OP_N;
  hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op,
                                  sizeof(op));
  hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op,
                                  sizeof(op));

  hipblasLtMatmulPreference_t pref = nullptr;
  if (hipblasLtMatmulPreferenceCreate(&pref) != HIPBLAS_STATUS_SUCCESS) {
    set_error("hipblasLtMatmulPreferenceCreate failed");
    cleanup();
    return rc;
  }
  const uint64_t workspace_size = 32 * 1024 * 1024;
  hipblasLtMatmulPreferenceSetAttribute(
      pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size,
      sizeof(workspace_size));

  hipblasLtMatmulHeuristicResult_t heuristic{};
  int returned = 0;
  auto hs = hipblasLtMatmulAlgoGetHeuristic(ctx->lt, desc, lb, la, lc, lc, pref,
                                            1, &heuristic, &returned);
  hipblasLtMatmulPreferenceDestroy(pref);
  if (hs != HIPBLAS_STATUS_SUCCESS || returned == 0) {
    set_error("no hipBLASLt algorithm for this problem shape");
    cleanup();
    return rc;
  }

  void *workspace = nullptr;
  if (heuristic.workspaceSize &&
      hipMalloc(&workspace, heuristic.workspaceSize) != hipSuccess) {
    set_error("workspace hipMalloc failed");
    cleanup();
    return rc;
  }

  if (hipblasLtMatmul(ctx->lt, desc, &alpha, b, lb, a, la, &beta, c, lc, c, lc,
                      &heuristic.algo, workspace, heuristic.workspaceSize,
                      ctx->stream) != HIPBLAS_STATUS_SUCCESS) {
    set_error("hipblasLtMatmul failed");
  } else {
    rc = 0;
  }

  if (workspace)
    (void)hipFree(workspace);
  cleanup();
  return rc;
}

extern "C" int baro_sync(baro_ctx *ctx) {
  if (!ctx) {
    set_error("null context");
    return -1;
  }
  if (hipStreamSynchronize(ctx->stream) != hipSuccess) {
    set_error("hipStreamSynchronize failed");
    return -1;
  }
  return 0;
}

extern "C" const char *baro_last_error(void) { return g_last_error.c_str(); }
