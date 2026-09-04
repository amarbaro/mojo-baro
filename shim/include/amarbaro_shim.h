#ifndef BARO_SHIM_H
#define BARO_SHIM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct amarbaro_ctx amarbaro_ctx;

/* Lifecycle: one context owns a hipBLASLt handle and a HIP stream. */
amarbaro_ctx *amarbaro_init(int device_id);
void amarbaro_destroy(amarbaro_ctx *ctx);

/* C = alpha * A(m,k) * B(k,n) + beta * C, row-major, fp16 in / fp32 accum. */
int amarbaro_gemm_f16(amarbaro_ctx *ctx, int m, int n, int k, const void *a,
                      const void *b, void *c, float alpha, float beta);

/* Same, fp32 in / fp32 accum. Used as the vendor baseline in benchmarks. */
int amarbaro_gemm_f32(amarbaro_ctx *ctx, int m, int n, int k, const void *a,
                      const void *b, void *c, float alpha, float beta);

/* Same as amarbaro_gemm_f16, but b is [n][k] row-major (ldb = k) instead of
   [k][n]; opB = HIPBLAS_OP_T. Its own cached algo/splitK/wgm search, keyed
   separately from the NN entry. */
int amarbaro_gemm_f16_nt(amarbaro_ctx *ctx, int m, int n, int k, const void *a,
                         const void *b, void *c, float alpha, float beta);

/* Device memory. amarbaro_upload/amarbaro_download are synchronous on ctx's stream. */
void *amarbaro_device_alloc(size_t bytes);
void amarbaro_device_free(void *ptr);
int amarbaro_upload(amarbaro_ctx *ctx, void *dst, const void *src, size_t bytes);
int amarbaro_download(amarbaro_ctx *ctx, void *dst, const void *src, size_t bytes);

/* Blocks until all work on the context stream has retired. */
int amarbaro_sync(amarbaro_ctx *ctx);

const char *amarbaro_last_error(void);

/* Diagnostics for the last tuned shape: how many algorithms the heuristic
   offered, and which one measured fastest. */
int amarbaro_algo_count(amarbaro_ctx *ctx);
int amarbaro_algo_chosen(amarbaro_ctx *ctx);
int amarbaro_splitk(amarbaro_ctx *ctx);
int amarbaro_wgm(amarbaro_ctx *ctx);
/* VRAM currently held for GEMM workspace, after right-sizing. */
size_t amarbaro_workspace_bytes(amarbaro_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif /* BARO_SHIM_H */
