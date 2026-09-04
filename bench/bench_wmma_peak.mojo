"""WMMA-only roofline microbenchmark (kernels/wmma_peak.mojo). No global memory
traffic in the timed loop: measures matrix-core issue rate alone, not a GEMM.
"""

from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

from wmma_peak import wmma_peak_kernel, NACC, ITERS, WAVES_PER_BLOCK, BLOCKS_PER_CU

comptime NUM_CU = 96
comptime NUM_BLOCKS = NUM_CU * BLOCKS_PER_CU
comptime NTHREADS = WAVES_PER_BLOCK * 32
comptime NOUT = NUM_BLOCKS * NTHREADS
comptime c_layout = row_major[NOUT, 1]()
comptime WARMUP_SECONDS = 10.0


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()

    var ch = ctx.enqueue_create_host_buffer[DType.float32](NOUT)
    ctx.synchronize()

    var cd = ctx.enqueue_create_buffer[DType.float32](NOUT)
    ctx.synchronize()

    var C = TileTensor(cd, c_layout)

    comptime kernel = wmma_peak_kernel[type_of(c_layout)]

    var w0 = perf_counter_ns()
    while Float64(perf_counter_ns() - w0) / 1.0e9 < WARMUP_SECONDS:
        ctx.enqueue_function[kernel](C, grid_dim=NUM_BLOCKS, block_dim=NTHREADS)
        ctx.synchronize()

    var t0 = perf_counter_ns()
    ctx.enqueue_function[kernel](C, grid_dim=NUM_BLOCKS, block_dim=NTHREADS)
    ctx.synchronize()
    var ms = Float64(perf_counter_ns() - t0) / 1.0e6

    ctx.enqueue_copy(dst_buf=ch, src_buf=cd)
    ctx.synchronize()

    var ok = True
    for i in range(NOUT):
        var v = Float64(ch[i])
        if v != v or v == 0.0:
            ok = False

    comptime FLOP_PER_MMA = 8192.0
    comptime TOTAL_MMA = Float64(NUM_BLOCKS) * Float64(WAVES_PER_BLOCK) * Float64(ITERS) * Float64(NACC)
    var gflops = (TOTAL_MMA * FLOP_PER_MMA) / (ms * 1.0e6)

    var out = String("{")
    out += '"variant": "wmma_peak", '
    out += '"nacc": ' + String(NACC) + ', "iters": ' + String(ITERS) + ", "
    out += '"waves_per_block": ' + String(WAVES_PER_BLOCK) + ', "blocks_per_cu": ' + String(BLOCKS_PER_CU) + ", "
    out += '"grid": ' + String(NUM_BLOCKS) + ', "block": ' + String(NTHREADS) + ", "
    out += '"ms": ' + String(ms) + ", "
    out += '"gflops": ' + String(gflops) + ", "
    out += '"correct": ' + ("true" if ok else "false") + "}"
    print(out)
