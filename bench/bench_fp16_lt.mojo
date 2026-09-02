"""hipBLASLt fp16 GEMM through the shim: the vendor bar for kernels/matmul_wmma_lds.mojo.

C is fp16 (the shim passes one dtype for A/B/C/D), so the gate is err <= 1.0:
with |sum| <= 4096 the fp16 ulp is 2. Prints the algo/splitK/wgm receipt (P1).
Build with the shim link flags from run-tests.sh; patch M/N/K like bench_fp16.
"""

from std.ffi import external_call
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, HostBuffer
from amarbaro import AmarBaro

comptime M = 512
comptime N = 512
comptime K = 512
comptime ITERS = 200
comptime WARMUP_SECONDS = 10.0


def check(a: HostBuffer[DType.float16], b: HostBuffer[DType.float16], c: HostBuffer[DType.float16]) -> Float64:
    comptime RSTEP = 37 if M <= 2048 else (M // 32)
    comptime CSTEP = 41 if N <= 2048 else (N // 32)
    var worst = Float64(0)
    for r in range(0, M, RSTEP):
        for col in range(0, N, CSTEP):
            var want = Float64(0)
            for p in range(K):
                want += Float64(a[r * K + p]) * Float64(b[p * N + col])
            var err = abs(Float64(c[r * N + col]) - want)
            if err > worst:
                worst = err
    return worst


def main() raises:
    var ctx = DeviceContext()
    var ah = ctx.enqueue_create_host_buffer[DType.float16](M * K)
    var bh = ctx.enqueue_create_host_buffer[DType.float16](K * N)
    var ch = ctx.enqueue_create_host_buffer[DType.float16](M * N)
    ctx.synchronize()
    # values in {-1,0,1} so K=4096 sums stay exact in fp16 output range
    for i in range(M * K):
        ah[i] = Scalar[DType.float16]((i % 3) - 1)
    for i in range(K * N):
        bh[i] = Scalar[DType.float16]((i % 3) - 1)
    var ad = ctx.enqueue_create_buffer[DType.float16](M * K)
    var bd = ctx.enqueue_create_buffer[DType.float16](K * N)
    var cd = ctx.enqueue_create_buffer[DType.float16](M * N)
    ctx.enqueue_copy(dst_buf=ad, src_buf=ah)
    ctx.enqueue_copy(dst_buf=bd, src_buf=bh)
    ctx.synchronize()

    var baro = AmarBaro()
    var pa = Int(ad.unsafe_ptr())
    var pb = Int(bd.unsafe_ptr())
    var pc = Int(cd.unsafe_ptr())
    var w0 = perf_counter_ns()
    while Float64(perf_counter_ns() - w0) / 1.0e9 < WARMUP_SECONDS:
        for _ in range(20):
            baro.gemm_f16(M, N, K, pa, pb, pc)
        baro.sync()
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        baro.gemm_f16(M, N, K, pa, pb, pc)
    baro.sync()
    var ms = Float64(perf_counter_ns() - t0) / 1.0e6 / Float64(ITERS)
    ctx.enqueue_copy(dst_buf=ch, src_buf=cd)
    ctx.synchronize()
    var err = check(ah, bh, ch)
    comptime FLOPS = 2.0 * Float64(M) * Float64(N) * Float64(K)
    var out = String("{")
    out += '"variant": "hipblaslt_fp16", '
    out += '"m": ' + String(M) + ', "n": ' + String(N) + ', "k": ' + String(K) + ", "
    out += '"ms": ' + String(ms) + ', "gflops": ' + String(FLOPS / (ms * 1.0e6)) + ", "
    out += '"correct": ' + ("true" if err <= 1.0 else "false") + ', "max_err": ' + String(err) + ", "
    out += '"algo_count": ' + String(external_call["amarbaro_algo_count", Int32](baro._ctx)) + ", "
    out += '"algo_chosen": ' + String(external_call["amarbaro_algo_chosen", Int32](baro._ctx)) + ", "
    out += '"splitk": ' + String(external_call["amarbaro_splitk", Int32](baro._ctx)) + ", "
    out += '"wgm": ' + String(external_call["amarbaro_wgm", Int32](baro._ctx)) + ", "
    out += '"iters": ' + String(ITERS) + ', "warmup_s": ' + String(WARMUP_SECONDS) + ', '
    out += '"dtype": "float16", "c_dtype": "float16", "gate": "err<=1.0: fp16 C, |sum|<=4096 has ulp 2"}'
    print(out)
