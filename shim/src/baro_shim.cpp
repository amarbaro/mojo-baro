#include "baro_shim.h"

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <string>

namespace {
thread_local std::string g_last_error;

void set_error(const char *what) { g_last_error = what; }
} // namespace

extern "C" int baro_sync(baro_ctx *ctx);

namespace {
} // namespace

struct baro_ctx {
  hipStream_t stream;
  hipblasLtHandle_t lt;
  int device_id;
  /* Workspace and algorithm selection are cached across calls. Doing the
     heuristic lookup and a hipMalloc per call costs more than the GEMM at
     these sizes and made hipBLASLt benchmark below a naive kernel. */
  void *workspace;
  size_t workspace_size;
  bool have_algo;
  hipblasLtMatmulAlgo_t algo;
  size_t algo_ws;
  int cached_m, cached_n, cached_k;
  hipDataType cached_dt;
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
  ctx->workspace_size = 32 * 1024 * 1024;
  if (hipMalloc(&ctx->workspace, ctx->workspace_size) != hipSuccess) {
    set_error("workspace hipMalloc failed");
    (void)hipStreamDestroy(ctx->stream);
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
  if (ctx->workspace)
    (void)hipFree(ctx->workspace);
  hipblasLtDestroy(ctx->lt);
  (void)hipStreamDestroy(ctx->stream);
  delete ctx;
}

namespace {
int gemm_impl(baro_ctx *ctx, int m, int n, int k, const void *a, const void *b,
              void *c, float alpha, float beta, hipDataType dt) {
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

  if (hipblasLtMatrixLayoutCreate(&lb, dt, n, k, n) !=
          HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatrixLayoutCreate(&la, dt, k, m, k) !=
          HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatrixLayoutCreate(&lc, dt, n, m, n) !=
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

  const bool shape_changed = !ctx->have_algo || ctx->cached_m != m ||
                             ctx->cached_n != n || ctx->cached_k != k ||
                             ctx->cached_dt != dt;
  if (shape_changed) {
    hipblasLtMatmulPreference_t pref = nullptr;
    if (hipblasLtMatmulPreferenceCreate(&pref) != HIPBLAS_STATUS_SUCCESS) {
      set_error("hipblasLtMatmulPreferenceCreate failed");
      cleanup();
      return rc;
    }
    hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ctx->workspace_size,
        sizeof(ctx->workspace_size));

    hipblasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    auto hs = hipblasLtMatmulAlgoGetHeuristic(ctx->lt, desc, lb, la, lc, lc,
                                              pref, 1, &heuristic, &returned);
    hipblasLtMatmulPreferenceDestroy(pref);
    if (hs != HIPBLAS_STATUS_SUCCESS || returned == 0) {
      set_error("no hipBLASLt algorithm for this problem shape");
      cleanup();
      return rc;
    }
    ctx->algo = heuristic.algo;
    ctx->algo_ws = heuristic.workspaceSize;
    ctx->cached_m = m;
    ctx->cached_n = n;
    ctx->cached_k = k;
    ctx->cached_dt = dt;
    ctx->have_algo = true;
  }

  if (hipblasLtMatmul(ctx->lt, desc, &alpha, b, lb, a, la, &beta, c, lc, c, lc,
                      &ctx->algo, ctx->workspace, ctx->algo_ws,
                      ctx->stream) != HIPBLAS_STATUS_SUCCESS) {
    set_error("hipblasLtMatmul failed");
  } else {
    rc = 0;
  }

  cleanup();
  return rc;
}
} // namespace

extern "C" int baro_gemm_f16(baro_ctx *ctx, int m, int n, int k, const void *a,
                             const void *b, void *c, float alpha, float beta) {
  return gemm_impl(ctx, m, n, k, a, b, c, alpha, beta, HIP_R_16F);
}

extern "C" int baro_gemm_f32(baro_ctx *ctx, int m, int n, int k, const void *a,
                             const void *b, void *c, float alpha, float beta) {
  return gemm_impl(ctx, m, n, k, a, b, c, alpha, beta, HIP_R_32F);
}

extern "C" void *baro_device_alloc(size_t bytes) {
  void *ptr = nullptr;
  if (hipMalloc(&ptr, bytes) != hipSuccess) {
    set_error("hipMalloc failed");
    return nullptr;
  }
  return ptr;
}

extern "C" void baro_device_free(void *ptr) {
  if (ptr)
    (void)hipFree(ptr);
}

extern "C" int baro_upload(baro_ctx *ctx, void *dst, const void *src,
                           size_t bytes) {
  if (!ctx) {
    set_error("null context");
    return -1;
  }
  if (hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst),
                         const_cast<void *>(src), bytes,
                         ctx->stream) != hipSuccess) {
    set_error("hipMemcpyHtoDAsync failed");
    return -1;
  }
  return baro_sync(ctx);
}

extern "C" int baro_download(baro_ctx *ctx, void *dst, const void *src,
                             size_t bytes) {
  if (!ctx) {
    set_error("null context");
    return -1;
  }
  if (hipMemcpyDtoHAsync(dst,
                         reinterpret_cast<hipDeviceptr_t>(const_cast<void *>(src)),
                         bytes, ctx->stream) != hipSuccess) {
    set_error("hipMemcpyDtoHAsync failed");
    return -1;
  }
  return baro_sync(ctx);
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
