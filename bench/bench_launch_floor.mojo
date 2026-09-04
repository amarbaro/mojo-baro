"""Stage 0 of bench/launch-fusion-protocol.md: per-launch floor on this box.

(a) empty kernel, grid 1 block 64
(b) empty kernel, grid 96 block 256
(c) the five small ssm kernels back-to-back at engine shapes (zero data)
(d) one launch, grid NH_V block SSTATE, five barrier-separated phases touching
    the same buffers (fused-shape feasibility, not the bit-exact kernel)

No synchronize between launches; ITERS launches per arm, wall time / ITERS.
"""

from std.gpu import block_idx, thread_idx
from std.math import ceildiv
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major

from matmul_skinny import SPLITK, SM
from ssm import (
    amar_ssm_reduce_gates, amar_ssm_conv, amar_ssm_qk_l2norm,
    amar_ssm_delta_step, amar_ssm_gated_out_bf16, CONV, NH_V, SSTATE,
)

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime H = 4096
comptime ITERS = 2000
comptime WARM = 200

comptime p_32 = row_major[SPLITK, SM, NH_V]()
comptime g32_layout = row_major[NH_V]()
comptime conv_layout = row_major[CONV]()
comptime cs_layout = row_major[3, CONV]()
comptime cw_layout = row_major[CONV, 4]()
comptime s_layout = row_major[NH_V, SSTATE, SSTATE]()
comptime o_layout = row_major[NH_V, SSTATE]()
comptime n128_layout = row_major[SSTATE]()
comptime h_layout = row_major[H]()


def empty_k():
    pass


def fused_shape_k[
    SLayout: TensorLayout, CLayout: TensorLayout, OLayout: TensorLayout
](
    S0: TileTensor[f32, SLayout, MutAnyOrigin],
    S1: TileTensor[f32, SLayout, MutAnyOrigin],
    Conv: TileTensor[f32, CLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
):
    comptime assert S0.flat_rank == 3 and Conv.flat_rank == 1 and O.flat_rank == 2
    var h = block_idx.x
    var j = thread_idx.x
    var c = rebind[Scalar[f32]](Conv[h * SSTATE + j])
    barrier()
    c = c * 0.5 + 1.0
    Conv[h * SSTATE + j] = rebind[Conv.ElementType](c)
    barrier()
    var acc: Scalar[f32] = 0
    for i in range(SSTATE):
        var s = rebind[Scalar[f32]](S0[h, i, j])
        s = s * 0.99 + c
        S1[h, i, j] = rebind[S1.ElementType](s)
        acc += s
    barrier()
    O[h, j] = rebind[O.ElementType](acc)
    barrier()
    O[h, j] = rebind[O.ElementType](acc * c)


def time_us(ctx: DeviceContext, n: Int, t0: Int) raises -> Float64:
    ctx.synchronize()
    return Float64(perf_counter_ns() - t0) / 1e3 / Float64(n)


def main() raises:
    comptime assert has_accelerator()
    var ctx = DeviceContext()

    var pab = ctx.enqueue_create_buffer[f32](SPLITK * SM * NH_V)
    var pab2 = ctx.enqueue_create_buffer[f32](SPLITK * SM * NH_V)
    var eg = ctx.enqueue_create_buffer[f32](NH_V)
    var beta = ctx.enqueue_create_buffer[f32](NH_V)
    var ssma = ctx.enqueue_create_buffer[f32](NH_V)
    var dtb = ctx.enqueue_create_buffer[f32](NH_V)
    var qkv = ctx.enqueue_create_buffer[f32](CONV)
    var cs = ctx.enqueue_create_buffer[f32](3 * CONV)
    var cs_w = ctx.enqueue_create_buffer[f32](3 * CONV)
    var cw = ctx.enqueue_create_buffer[f32](CONV * 4)
    var conv = ctx.enqueue_create_buffer[f32](CONV)
    var s0 = ctx.enqueue_create_buffer[f32](NH_V * SSTATE * SSTATE)
    var s1 = ctx.enqueue_create_buffer[f32](NH_V * SSTATE * SSTATE)
    var so = ctx.enqueue_create_buffer[f32](NH_V * SSTATE)
    var z = ctx.enqueue_create_buffer[f32](H)
    var nw = ctx.enqueue_create_buffer[f32](SSTATE)
    var res = ctx.enqueue_create_buffer[bf16](H)
    for b in [pab, pab2, eg, beta, ssma, dtb, qkv, cs, cs_w, cw, conv, s0, s1, so, z, nw]:
        ctx.enqueue_memset(b, 0)
    ctx.enqueue_memset(res, 0)
    ctx.synchronize()

    var Pab = TileTensor(pab, p_32)
    var Pab2 = TileTensor(pab2, p_32)
    var Eg = TileTensor(eg, g32_layout)
    var Beta = TileTensor(beta, g32_layout)
    var SsmA = TileTensor(ssma, g32_layout)
    var DtB = TileTensor(dtb, g32_layout)
    var Qkv = TileTensor(qkv, conv_layout)
    var Cs = TileTensor(cs, cs_layout)
    var Cs_w = TileTensor(cs_w, cs_layout)
    var Cw = TileTensor(cw, cw_layout)
    var Conv = TileTensor(conv, conv_layout)
    var S0 = TileTensor(s0, s_layout)
    var S1 = TileTensor(s1, s_layout)
    var So = TileTensor(so, o_layout)
    var Z = TileTensor(z, h_layout)
    var Nw = TileTensor(nw, n128_layout)
    var Res = TileTensor(res, h_layout)

    comptime rgates_k = amar_ssm_reduce_gates[type_of(p_32), type_of(g32_layout), type_of(g32_layout)]
    comptime conv_k = amar_ssm_conv[type_of(conv_layout), type_of(cs_layout), type_of(cw_layout), type_of(conv_layout)]
    comptime l2_k = amar_ssm_qk_l2norm[type_of(conv_layout)]
    comptime delta_k = amar_ssm_delta_step[type_of(s_layout), type_of(conv_layout), type_of(g32_layout), type_of(o_layout)]
    comptime gated_k = amar_ssm_gated_out_bf16[type_of(o_layout), type_of(h_layout), type_of(n128_layout), type_of(h_layout)]
    comptime fused_k = fused_shape_k[type_of(s_layout), type_of(conv_layout), type_of(o_layout)]

    # (a)
    for _ in range(WARM):
        ctx.enqueue_function[empty_k](grid_dim=1, block_dim=64)
    ctx.synchronize()
    var t = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[empty_k](grid_dim=1, block_dim=64)
    var a = time_us(ctx, ITERS, t)

    # (b)
    for _ in range(WARM):
        ctx.enqueue_function[empty_k](grid_dim=96, block_dim=256)
    ctx.synchronize()
    t = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[empty_k](grid_dim=96, block_dim=256)
    var b = time_us(ctx, ITERS, t)

    # (c)
    for _ in range(WARM):
        ctx.enqueue_function[rgates_k](Pab, Pab2, Eg, Beta, SsmA, DtB, Int32(0), grid_dim=1, block_dim=NH_V)
        ctx.enqueue_function[conv_k](Qkv, Cs, Cw, Conv, Cs_w, grid_dim=ceildiv(CONV, 256), block_dim=256)
        ctx.enqueue_function[l2_k](Conv, grid_dim=NH_V, block_dim=SSTATE)
        ctx.enqueue_function[delta_k](S0, S1, Conv, Eg, Beta, So, grid_dim=NH_V, block_dim=SSTATE)
        ctx.enqueue_function[gated_k](So, Z, Nw, Res, grid_dim=NH_V, block_dim=SSTATE)
    ctx.synchronize()
    t = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[rgates_k](Pab, Pab2, Eg, Beta, SsmA, DtB, Int32(0), grid_dim=1, block_dim=NH_V)
        ctx.enqueue_function[conv_k](Qkv, Cs, Cw, Conv, Cs_w, grid_dim=ceildiv(CONV, 256), block_dim=256)
        ctx.enqueue_function[l2_k](Conv, grid_dim=NH_V, block_dim=SSTATE)
        ctx.enqueue_function[delta_k](S0, S1, Conv, Eg, Beta, So, grid_dim=NH_V, block_dim=SSTATE)
        ctx.enqueue_function[gated_k](So, Z, Nw, Res, grid_dim=NH_V, block_dim=SSTATE)
    var c = time_us(ctx, ITERS, t)

    # (d)
    for _ in range(WARM):
        ctx.enqueue_function[fused_k](S0, S1, Conv, So, grid_dim=NH_V, block_dim=SSTATE)
    ctx.synchronize()
    t = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[fused_k](S0, S1, Conv, So, grid_dim=NH_V, block_dim=SSTATE)
    var d = time_us(ctx, ITERS, t)

    print("iters:", ITERS, "warm:", WARM)
    print("a empty g1 b64 us/launch:", a)
    print("b empty g96 b256 us/launch:", b)
    print("c ssm5 chain us/chain:", c, "us/launch:", c / 5)
    print("d fused-shape us/launch:", d)
    print("ceiling_pct 646 launches x floor(b) / 24 ms:", 646.0 * b / 24000.0 * 100.0)
