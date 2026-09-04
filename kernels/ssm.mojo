from std.gpu import block_idx, global_idx, lane_id, thread_idx, WARP_SIZE
from std.gpu.primitives import warp
from std.math import exp, log1p, rsqrt, sqrt
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation

from matmul_skinny import SPLITK

comptime f32 = DType.float32
comptime CONV = 8192
comptime KDIM = 2048
comptime NH_K = 16
comptime NH_V = 32
comptime SSTATE = 128
comptime SSM_EPS = Float32(1e-6)


def amar_cast_bf16[
    XLayout: TensorLayout, OLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    O: TileTensor[DType.bfloat16, OLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert X.flat_rank == 1 and O.flat_rank == 1
    var i = global_idx.x
    if i < Int(n):
        O[i] = rebind[O.ElementType](
            rebind[Scalar[f32]](X[i]).cast[DType.bfloat16]()
        )


def amar_residual_add[
    XLayout: TensorLayout, YLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    Y: TileTensor[f32, YLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert X.flat_rank == 1 and Y.flat_rank == 1
    var i = global_idx.x
    if i < Int(n):
        Y[i] = rebind[Y.ElementType](
            rebind[Scalar[f32]](Y[i]) + rebind[Scalar[f32]](X[i])
        )


def amar_ssm_gates[
    ALayout: TensorLayout, BLayout: TensorLayout, GLayout: TensorLayout,
    WLayout: TensorLayout, DLayout: TensorLayout
](
    AlphaRaw: TileTensor[f32, ALayout, MutAnyOrigin],
    BetaRaw: TileTensor[f32, BLayout, MutAnyOrigin],
    EgOut: TileTensor[f32, GLayout, MutAnyOrigin],
    BetaOut: TileTensor[f32, WLayout, MutAnyOrigin],
    SsmA: TileTensor[f32, DLayout, MutAnyOrigin],
    DtBias: TileTensor[f32, DLayout, MutAnyOrigin],
):
    comptime assert AlphaRaw.flat_rank == 1 and BetaRaw.flat_rank == 1
    comptime assert EgOut.flat_rank == 1 and BetaOut.flat_rank == 1
    comptime assert SsmA.flat_rank == 1 and DtBias.flat_rank == 1
    var h = global_idx.x
    if h >= NH_V:
        return
    var braw = rebind[Scalar[f32]](BetaRaw[h])
    BetaOut[h] = rebind[BetaOut.ElementType](1 / (1 + exp(-braw)))
    var araw = rebind[Scalar[f32]](AlphaRaw[h]) + rebind[Scalar[f32]](DtBias[h])
    var sp = log1p(exp(araw))
    EgOut[h] = rebind[EgOut.ElementType](exp(sp * rebind[Scalar[f32]](SsmA[h])))


def amar_ssm_reduce_gates[
    PLayout: TensorLayout, GLayout: TensorLayout, DLayout: TensorLayout
](
    Ap: TileTensor[f32, PLayout, MutAnyOrigin],
    Bp: TileTensor[f32, PLayout, MutAnyOrigin],
    EgOut: TileTensor[f32, GLayout, MutAnyOrigin],
    BetaOut: TileTensor[f32, GLayout, MutAnyOrigin],
    SsmA: TileTensor[f32, DLayout, MutAnyOrigin],
    DtBias: TileTensor[f32, DLayout, MutAnyOrigin],
    m: Int32,
):
    comptime assert Ap.flat_rank == 3 and Bp.flat_rank == 3
    comptime assert EgOut.flat_rank == 2 and BetaOut.flat_rank == 2
    comptime assert SsmA.flat_rank == 1 and DtBias.flat_rank == 1
    var h = global_idx.x
    if h >= NH_V:
        return
    var sa = rebind[Scalar[f32]](SsmA[h])
    var db = rebind[Scalar[f32]](DtBias[h])
    for r in range(Int(m)):
        var araw: Scalar[f32] = 0
        var braw: Scalar[f32] = 0
        comptime for s in range(SPLITK):
            araw += rebind[Scalar[f32]](Ap[s, r, h])
            braw += rebind[Scalar[f32]](Bp[s, r, h])
        BetaOut[r, h] = rebind[BetaOut.ElementType](1 / (1 + exp(-braw)))
        var asum = araw + db
        var sp = log1p(exp(asum))
        EgOut[r, h] = rebind[EgOut.ElementType](exp(sp * sa))


def amar_ssm_conv[
    QLayout: TensorLayout, SLayout: TensorLayout, WLayout: TensorLayout,
    OLayout: TensorLayout
](
    Qkv: TileTensor[f32, QLayout, MutAnyOrigin],
    ConvState: TileTensor[f32, SLayout, MutAnyOrigin],
    ConvW: TileTensor[f32, WLayout, MutAnyOrigin],
    Out: TileTensor[f32, OLayout, MutAnyOrigin],
    ring: Int32, ssm_i: Int32, slots: Int32, m: Int32,
):
    comptime assert Qkv.flat_rank == 2 and ConvState.flat_rank == 4
    comptime assert ConvW.flat_rank == 2 and Out.flat_rank == 2
    var c = global_idx.x
    if c >= CONV:
        return
    var si = Int(ssm_i)
    var sl = Int(slots)
    var rg = Int(ring)
    var cw0 = rebind[Scalar[f32]](ConvW[c, 0])
    var cw1 = rebind[Scalar[f32]](ConvW[c, 1])
    var cw2 = rebind[Scalar[f32]](ConvW[c, 2])
    var cw3 = rebind[Scalar[f32]](ConvW[c, 3])
    for r in range(Int(m)):
        var rs = (rg + r) % sl
        var ws = (rg + r + 1) % sl
        var w0 = rebind[Scalar[f32]](ConvState[rs, si, 0, c])
        var w1 = rebind[Scalar[f32]](ConvState[rs, si, 1, c])
        var w2 = rebind[Scalar[f32]](ConvState[rs, si, 2, c])
        var w3 = rebind[Scalar[f32]](Qkv[r, c])
        var acc = w0 * cw0 + w1 * cw1 + w2 * cw2 + w3 * cw3
        Out[r, c] = rebind[Out.ElementType](acc / (1 + exp(-acc)))
        ConvState[ws, si, 0, c] = rebind[ConvState.ElementType](w1)
        ConvState[ws, si, 1, c] = rebind[ConvState.ElementType](w2)
        ConvState[ws, si, 2, c] = rebind[ConvState.ElementType](w3)


def amar_ssm_qk_l2norm[
    XLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    m: Int32,
):
    comptime assert X.flat_rank == 2
    var head = block_idx.x
    var tid = thread_idx.x
    var base = head * SSTATE
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[SSTATE // WARP_SIZE]()
    )
    for r in range(Int(m)):
        var v = rebind[Scalar[f32]](X[r, base + tid])
        var ssq = warp.sum(v * v)
        if lane_id() == 0:
            sums[tid // WARP_SIZE] = rebind[sums.ElementType](ssq)
        barrier()
        var total: Float32 = 0
        comptime for w in range(SSTATE // WARP_SIZE):
            total += rebind[Scalar[f32]](sums[w])
        var inv = rsqrt(total + SSM_EPS)
        if head < NH_K:
            inv = inv / sqrt(Float32(SSTATE))
        X[r, base + tid] = rebind[X.ElementType](v * inv)
        barrier()


def amar_ssm_delta_step[
    MR: Int,
    S0Layout: TensorLayout, CLayout: TensorLayout, GLayout: TensorLayout,
    OLayout: TensorLayout
](
    SAll: TileTensor[f32, S0Layout, MutAnyOrigin],
    ConvOut: TileTensor[f32, CLayout, MutAnyOrigin],
    Eg: TileTensor[f32, GLayout, MutAnyOrigin],
    Beta: TileTensor[f32, GLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    ring: Int32, ssm_i: Int32, slots: Int32,
):
    comptime assert SAll.flat_rank == 5 and ConvOut.flat_rank == 2
    comptime assert Eg.flat_rank == 2 and O.flat_rank == 3
    var h = block_idx.x
    var j = thread_idx.x
    var kh = h % NH_K
    var si = Int(ssm_i)
    var sl = Int(slots)
    var rg = Int(ring)

    var kq = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[2, SSTATE]()
    )
    comptime for r in range(MR):
        var rs = (rg + r) % sl
        var ws = (rg + r + 1) % sl
        kq[0, j] = rebind[kq.ElementType](ConvOut[r, kh * SSTATE + j])
        kq[1, j] = rebind[kq.ElementType](ConvOut[r, KDIM + kh * SSTATE + j])
        barrier()

        var eg = rebind[Scalar[f32]](Eg[r, h])
        var beta = rebind[Scalar[f32]](Beta[r, h])
        var vj = rebind[Scalar[f32]](ConvOut[r, 2 * KDIM + h * SSTATE + j])

        var col = SIMD[f32, SSTATE]()
        comptime for i in range(SSTATE):
            col[i] = rebind[Scalar[f32]](SAll[rs, si, h, i, j])

        var sk: Float32 = 0
        comptime for i in range(SSTATE):
            sk += col[i] * eg * rebind[Scalar[f32]](kq[1, i])
        var d = (vj - sk) * beta

        var o: Float32 = 0
        comptime for i in range(SSTATE):
            var s = col[i] * eg + rebind[Scalar[f32]](kq[1, i]) * d
            SAll[ws, si, h, i, j] = rebind[SAll.ElementType](s)
            o += s * rebind[Scalar[f32]](kq[0, i])
        O[r, h, j] = rebind[O.ElementType](o)
        barrier()


def amar_ssm_gated_out[
    OLayout: TensorLayout, ZLayout: TensorLayout, NLayout: TensorLayout,
    RLayout: TensorLayout
](
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    Z: TileTensor[f32, ZLayout, MutAnyOrigin],
    NormW: TileTensor[f32, NLayout, MutAnyOrigin],
    Res: TileTensor[f32, RLayout, MutAnyOrigin],
):
    comptime assert O.flat_rank == 2 and Z.flat_rank == 1
    comptime assert NormW.flat_rank == 1 and Res.flat_rank == 1
    var h = block_idx.x
    var j = thread_idx.x
    var v = rebind[Scalar[f32]](O[h, j])
    var ssq = warp.sum(v * v)
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[SSTATE // WARP_SIZE]()
    )
    if lane_id() == 0:
        sums[j // WARP_SIZE] = rebind[sums.ElementType](ssq)
    barrier()
    var total: Float32 = 0
    comptime for w in range(SSTATE // WARP_SIZE):
        total += rebind[Scalar[f32]](sums[w])
    var scale = rsqrt(total / Float32(SSTATE) + SSM_EPS)
    var z = rebind[Scalar[f32]](Z[h * SSTATE + j])
    Res[h * SSTATE + j] = rebind[Res.ElementType](
        v * scale * rebind[Scalar[f32]](NormW[j]) * (z / (1 + exp(-z)))
    )


def amar_ssm_gated_out_bf16[
    OLayout: TensorLayout, ZLayout: TensorLayout, NLayout: TensorLayout,
    RLayout: TensorLayout
](
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    Z: TileTensor[f32, ZLayout, MutAnyOrigin],
    NormW: TileTensor[f32, NLayout, MutAnyOrigin],
    Res: TileTensor[DType.bfloat16, RLayout, MutAnyOrigin],
    m: Int32,
):
    comptime assert O.flat_rank == 3 and Z.flat_rank == 2
    comptime assert NormW.flat_rank == 1 and Res.flat_rank == 2
    var h = block_idx.x
    var j = thread_idx.x
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[SSTATE // WARP_SIZE]()
    )
    var nwj = rebind[Scalar[f32]](NormW[j])
    for r in range(Int(m)):
        var v = rebind[Scalar[f32]](O[r, h, j])
        var ssq = warp.sum(v * v)
        if lane_id() == 0:
            sums[j // WARP_SIZE] = rebind[sums.ElementType](ssq)
        barrier()
        var total: Float32 = 0
        comptime for w in range(SSTATE // WARP_SIZE):
            total += rebind[Scalar[f32]](sums[w])
        var scale = rsqrt(total / Float32(SSTATE) + SSM_EPS)
        var z = rebind[Scalar[f32]](Z[r, h * SSTATE + j])
        Res[r, h * SSTATE + j] = rebind[Res.ElementType](
            (
                v * scale * nwj * (z / (1 + exp(-z)))
            ).cast[DType.bfloat16]()
        )
        barrier()
