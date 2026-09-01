from std.gpu import block_idx, global_idx, lane_id, thread_idx, WARP_SIZE
from std.gpu.primitives import warp
from std.math import cos, exp, log, rsqrt, sin
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation

comptime f32 = DType.float32
comptime HD = 256
comptime NQH = 16
comptime NKVH = 4
comptime MAX_T = 1024


def head_rmsnorm[
    XLayout: TensorLayout, GLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    G: TileTensor[f32, GLayout, MutAnyOrigin],
    eps: Float32,
):
    comptime assert X.flat_rank == 2 and G.flat_rank == 1
    var h = block_idx.x
    var d = thread_idx.x
    var v = rebind[Scalar[f32]](X[h, d])
    var ssq = warp.sum(v * v)
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[HD // WARP_SIZE]()
    )
    if lane_id() == 0:
        sums[d // WARP_SIZE] = rebind[sums.ElementType](ssq)
    barrier()
    var total: Float32 = 0
    comptime for w in range(HD // WARP_SIZE):
        total += rebind[Scalar[f32]](sums[w])
    X[h, d] = rebind[X.ElementType](
        v * rsqrt(total / Float32(HD) + eps) * rebind[Scalar[f32]](G[d])
    )


def attn_decode[
    QLayout: TensorLayout, KLayout: TensorLayout, OLayout: TensorLayout
](
    Q: TileTensor[f32, QLayout, MutAnyOrigin],
    Kc: TileTensor[f32, KLayout, MutAnyOrigin],
    Vc: TileTensor[f32, KLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    t_len: Int32,
    scale: Float32,
):
    comptime assert Q.flat_rank == 2 and Kc.flat_rank == 3 and O.flat_rank == 2
    var h = block_idx.x
    var tid = thread_idx.x
    var kvh = h // (NQH // NKVH)
    var T = Int(t_len)

    var qs = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[HD]()
    )
    var scores = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[MAX_T]()
    )
    qs[tid] = rebind[qs.ElementType](Q[h, tid])
    barrier()

    var t = tid
    while t < T:
        var acc: Float32 = 0
        for d in range(HD):
            acc += rebind[Scalar[f32]](qs[d]) * rebind[Scalar[f32]](
                Kc[kvh, t, d]
            )
        scores[t] = rebind[scores.ElementType](acc * scale)
        t += HD
    barrier()

    var local_max = Float32(-3.4e38)
    t = tid
    while t < T:
        var s = rebind[Scalar[f32]](scores[t])
        if s > local_max:
            local_max = s
        t += HD
    var wmax = warp.max(local_max)
    var red = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[HD // WARP_SIZE]()
    )
    if lane_id() == 0:
        red[tid // WARP_SIZE] = rebind[red.ElementType](wmax)
    barrier()
    var row_max = Float32(-3.4e38)
    comptime for w in range(HD // WARP_SIZE):
        var s = rebind[Scalar[f32]](red[w])
        if s > row_max:
            row_max = s
    barrier()

    var partial: Float32 = 0
    t = tid
    while t < T:
        var e = exp(rebind[Scalar[f32]](scores[t]) - row_max)
        scores[t] = rebind[scores.ElementType](e)
        partial += e
        t += HD
    var wsum = warp.sum(partial)
    if lane_id() == 0:
        red[tid // WARP_SIZE] = rebind[red.ElementType](wsum)
    barrier()
    var total: Float32 = 0
    comptime for w in range(HD // WARP_SIZE):
        total += rebind[Scalar[f32]](red[w])
    var inv = 1 / total
    barrier()

    var o: Float32 = 0
    for tt in range(T):
        o += rebind[Scalar[f32]](scores[tt]) * rebind[Scalar[f32]](
            Vc[kvh, tt, tid]
        )
    O[h, tid] = rebind[O.ElementType](o * inv)


def gate_mul[
    XLayout: TensorLayout, GLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    Gate: TileTensor[f32, GLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert X.flat_rank == 1 and Gate.flat_rank == 1
    var i = global_idx.x
    if i >= Int(n):
        return
    var g = rebind[Scalar[f32]](Gate[i])
    X[i] = rebind[X.ElementType](
        rebind[Scalar[f32]](X[i]) * (1 / (1 + exp(-g)))
    )


comptime NROT = 64
comptime YARN_LOW = Float32(14.0)
comptime YARN_HIGH = Float32(22.0)
comptime FREQ_BASE = Float32(1e7)
comptime FREQ_SCALE = Float32(0.25)
comptime MSCALE = Float32(1.1386294361119891)


def qgate_split[
    FLayout: TensorLayout, QLayout: TensorLayout, GLayout: TensorLayout
](
    Qfull: TileTensor[f32, FLayout, MutAnyOrigin],
    Q: TileTensor[f32, QLayout, MutAnyOrigin],
    Gate: TileTensor[f32, GLayout, MutAnyOrigin],
):
    comptime assert Qfull.flat_rank == 1 and Q.flat_rank == 2 and Gate.flat_rank == 1
    var h = block_idx.x
    var d = thread_idx.x
    Q[h, d] = rebind[Q.ElementType](Qfull[h * 2 * HD + d])
    Gate[h * HD + d] = rebind[Gate.ElementType](Qfull[h * 2 * HD + HD + d])


def rope_yarn[
    XLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    pos: Int32,
):
    comptime assert X.flat_rank == 2
    var h = block_idx.x
    var j = thread_idx.x
    var theta_ex = Float32(Int(pos)) * exp(
        Float32(-2 * j) / Float32(NROT) * log(FREQ_BASE)
    )
    var theta_in = FREQ_SCALE * theta_ex
    var ramp = (Float32(j) - YARN_LOW) / max(YARN_HIGH - YARN_LOW, 0.001)
    ramp = min(max(ramp, 0), 1)
    var theta = theta_in * (1 - ramp) + theta_ex * ramp
    var c = cos(theta) * MSCALE
    var s = sin(theta) * MSCALE
    var x0 = rebind[Scalar[f32]](X[h, j])
    var x1 = rebind[Scalar[f32]](X[h, j + NROT // 2])
    X[h, j] = rebind[X.ElementType](x0 * c - x1 * s)
    X[h, j + NROT // 2] = rebind[X.ElementType](x0 * s + x1 * c)


def kv_append[
    CLayout: TensorLayout, NLayout: TensorLayout
](
    Cache: TileTensor[f32, CLayout, MutAnyOrigin],
    New: TileTensor[f32, NLayout, MutAnyOrigin],
    t_idx: Int32,
):
    comptime assert Cache.flat_rank == 3 and New.flat_rank == 2
    var h = block_idx.x
    var d = thread_idx.x
    Cache[h, Int(t_idx), d] = rebind[Cache.ElementType](New[h, d])
