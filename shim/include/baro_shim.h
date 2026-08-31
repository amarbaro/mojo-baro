#ifndef BARO_SHIM_H
#define BARO_SHIM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct baro_ctx baro_ctx;

/* Lifecycle: one context owns a hipBLASLt handle and a HIP stream. */
baro_ctx *baro_init(int device_id);
void baro_destroy(baro_ctx *ctx);

/* C = alpha * A(m,k) * B(k,n) + beta * C, row-major, fp16 in / fp32 accum. */
int baro_gemm_f16(baro_ctx *ctx, int m, int n, int k, const void *a,
                  const void *b, void *c, float alpha, float beta);

/* Same, fp32 in / fp32 accum. Used as the vendor baseline in benchmarks. */
int baro_gemm_f32(baro_ctx *ctx, int m, int n, int k, const void *a,
                  const void *b, void *c, float alpha, float beta);

/* Device memory. baro_upload/baro_download are synchronous on ctx's stream. */
void *baro_device_alloc(size_t bytes);
void baro_device_free(void *ptr);
int baro_upload(baro_ctx *ctx, void *dst, const void *src, size_t bytes);
int baro_download(baro_ctx *ctx, void *dst, const void *src, size_t bytes);

/* Blocks until all work on the context stream has retired. */
int baro_sync(baro_ctx *ctx);

const char *baro_last_error(void);

/* Diagnostics for the last tuned shape: how many algorithms the heuristic
   offered, and which one measured fastest. */
int baro_algo_count(baro_ctx *ctx);
int baro_algo_chosen(baro_ctx *ctx);
int baro_splitk(baro_ctx *ctx);
int baro_wgm(baro_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif /* BARO_SHIM_H */
