#include "baro_shim.h"

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#define BARO_MAX_ALGOS 32
#define BARO_TUNE_REPS 5

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
  int n_algos, chosen_algo;
  /* Best (algorithm, splitK, workgroup-mapping) found for the cached shape.
     splitK and wgm are only reachable through the ext API and are worth far
     more than algorithm choice alone on gfx1100. */
  int best_splitk, best_wgm;
  bool have_tuned;
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

/* Row-major A(m,k)*B(k,n) is issued as the column-major product B'(n,k)*A'(k,m),
   so hipBLASLt sees a plain NN GEMM with the operands swapped. */
int gemm_impl(baro_ctx *ctx, int m, int n, int k, const void *a, const void *b,
              void *c, float alpha, float beta, hipDataType dt) {
  if (!ctx || !a || !b || !c) {
    set_error("null argument");
    return -1;
  }

  hipblaslt_ext::Gemm gemm(ctx->lt, HIPBLAS_OP_N, HIPBLAS_OP_N, dt, dt, dt, dt,
                           HIPBLAS_COMPUTE_32F);
  hipblaslt_ext::GemmEpilogue epilogue;
  hipblaslt_ext::GemmInputs inputs;
  inputs.setA(const_cast<void *>(b));
  inputs.setB(const_cast<void *>(a));
  inputs.setC(c);
  inputs.setD(c);
  inputs.setAlpha(&alpha);
  inputs.setBeta(&beta);

  if (gemm.setProblem(n, m, k, 1, epilogue, inputs) !=
      HIPBLAS_STATUS_SUCCESS) {
    set_error("ext setProblem failed");
    return -1;
  }

  const bool shape_changed = !ctx->have_tuned || ctx->cached_m != m ||
                             ctx->cached_n != n || ctx->cached_k != k ||
                             ctx->cached_dt != dt;

  if (shape_changed) {
    hipblaslt_ext::GemmPreference pref;
    pref.setMaxWorkspaceBytes(ctx->workspace_size);
    std::vector<hipblasLtMatmulHeuristicResult_t> cand;
    if (gemm.algoGetHeuristic(BARO_MAX_ALGOS, pref, cand) !=
            HIPBLAS_STATUS_SUCCESS ||
        cand.empty()) {
      set_error("no hipBLASLt algorithm for this problem shape");
      return -1;
    }

    /* Search (algorithm, splitK, wgm). splitK matters most at small sizes,
       where a single-split GEMM cannot fill the GPU with workgroups. Paid
       once per shape. */
    static const int kSplitK[] = {0, 1, 2, 4, 8, 16, 32};
    static const int kWgm[] = {0, 1, 2, 4, 8, 16};

    hipEvent_t ev0, ev1;
    if (hipEventCreate(&ev0) != hipSuccess ||
        hipEventCreate(&ev1) != hipSuccess) {
      set_error("hipEventCreate failed");
      return -1;
    }

    float best_ms = -1.0f;
    int best_algo = -1, best_sk = 0, best_wgm = 0;

    for (size_t i = 0; i < cand.size(); ++i) {
      for (int sk : kSplitK) {
        for (int wg : kWgm) {
          hipblaslt_ext::GemmTuning tuning;
          tuning.setSplitK(static_cast<uint16_t>(sk));
          tuning.setWgm(static_cast<int16_t>(wg));
          size_t need = 0;
          if (gemm.isAlgoSupported(cand[i].algo, tuning, need) !=
                  HIPBLAS_STATUS_SUCCESS ||
              need > ctx->workspace_size)
            continue;
          if (gemm.initialize(cand[i].algo, tuning, ctx->workspace, true,
                              ctx->stream) != HIPBLAS_STATUS_SUCCESS)
            continue;
          if (gemm.run(ctx->stream) != HIPBLAS_STATUS_SUCCESS)
            continue;
          (void)hipStreamSynchronize(ctx->stream);

          (void)hipEventRecord(ev0, ctx->stream);
          for (int rep = 0; rep < BARO_TUNE_REPS; ++rep)
            (void)gemm.run(ctx->stream);
          (void)hipEventRecord(ev1, ctx->stream);
          (void)hipEventSynchronize(ev1);

          float ms = 0.0f;
          if (hipEventElapsedTime(&ms, ev0, ev1) == hipSuccess &&
              (best_ms < 0.0f || ms < best_ms)) {
            best_ms = ms;
            best_algo = static_cast<int>(i);
            best_sk = sk;
            best_wgm = wg;
          }
        }
      }
    }
    (void)hipEventDestroy(ev0);
    (void)hipEventDestroy(ev1);

    if (best_algo < 0) {
      set_error("no supported (algo, splitK, wgm) combination");
      return -1;
    }

    ctx->algo = cand[best_algo].algo;
    ctx->best_splitk = best_sk;
    ctx->best_wgm = best_wgm;
    ctx->n_algos = static_cast<int>(cand.size());
    ctx->chosen_algo = best_algo;
    ctx->cached_m = m;
    ctx->cached_n = n;
    ctx->cached_k = k;
    ctx->cached_dt = dt;
    ctx->have_tuned = true;
    ctx->have_algo = true;
  }

  hipblaslt_ext::GemmTuning tuning;
  tuning.setSplitK(static_cast<uint16_t>(ctx->best_splitk));
  tuning.setWgm(static_cast<int16_t>(ctx->best_wgm));
  if (gemm.initialize(ctx->algo, tuning, ctx->workspace, true, ctx->stream) !=
      HIPBLAS_STATUS_SUCCESS) {
    set_error("ext initialize failed");
    return -1;
  }
  if (gemm.run(ctx->stream) != HIPBLAS_STATUS_SUCCESS) {
    set_error("ext run failed");
    return -1;
  }
  return 0;
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

extern "C" int baro_algo_count(baro_ctx *ctx) {
  return ctx ? ctx->n_algos : 0;
}

extern "C" int baro_splitk(baro_ctx *ctx) { return ctx ? ctx->best_splitk : -1; }

extern "C" int baro_wgm(baro_ctx *ctx) { return ctx ? ctx->best_wgm : -1; }

extern "C" int baro_algo_chosen(baro_ctx *ctx) {
  return ctx ? ctx->chosen_algo : -1;
}

extern "C" const char *baro_last_error(void) { return g_last_error.c_str(); }
