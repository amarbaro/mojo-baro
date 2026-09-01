
from std.gpu import block_dim, block_idx, global_idx, lane_id, thread_idx, WARP_SIZE
from std.gpu.primitives import warp
from std.math import cos, exp, log, rsqrt, sin
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation

comptime EW_THREADS = 256
comptime f32 = DType.float32


def rmsnorm[
    XLayout: TensorLayout, GLayout: TensorLayout, OLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    G: TileTensor[f32, GLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    n: Int32,
    eps: Float32,
):
    comptime assert X.flat_rank == 2 and G.flat_rank == 1 and O.flat_rank == 2

    var N = Int(n)
    var row = block_idx.x
    var tid = thread_idx.x

    var partial: Float32 = 0
    var i = tid
    while i < N:
        var v = rebind[Scalar[f32]](X[row, i])
        partial += v * v
        i += EW_THREADS

    var sums = stack_allocation[
        f32, address_space = AddressSpace.SHARED
    ](row_major[EW_THREADS // WARP_SIZE]())
    var wsum = warp.sum(partial)
    if lane_id() == 0:
        sums[tid // WARP_SIZE] = rebind[sums.ElementType](wsum)
    barrier()
    var total: Float32 = 0
    comptime for w in range(EW_THREADS // WARP_SIZE):
        total += rebind[Scalar[f32]](sums[w])

    var scale = rsqrt(total / Float32(N) + eps)
    i = tid
    while i < N:
        O[row, i] = rebind[O.ElementType](
            rebind[Scalar[f32]](X[row, i])
            * scale
            * rebind[Scalar[f32]](G[i])
        )
        i += EW_THREADS


def rmsnorm_cast[
    XLayout: TensorLayout, GLayout: TensorLayout, OLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    G: TileTensor[f32, GLayout, MutAnyOrigin],
    O: TileTensor[DType.bfloat16, OLayout, MutAnyOrigin],
    n: Int32,
    eps: Float32,
):
    comptime assert X.flat_rank == 2 and G.flat_rank == 1 and O.flat_rank == 2

    var N = Int(n)
    var row = block_idx.x
    var tid = thread_idx.x

    var partial: Float32 = 0
    var i = tid
    while i < N:
        var v = rebind[Scalar[f32]](X[row, i])
        partial += v * v
        i += EW_THREADS

    var sums = stack_allocation[
        f32, address_space = AddressSpace.SHARED
    ](row_major[EW_THREADS // WARP_SIZE]())
    var wsum = warp.sum(partial)
    if lane_id() == 0:
        sums[tid // WARP_SIZE] = rebind[sums.ElementType](wsum)
    barrier()
    var total: Float32 = 0
    comptime for w in range(EW_THREADS // WARP_SIZE):
        total += rebind[Scalar[f32]](sums[w])

    var scale = rsqrt(total / Float32(N) + eps)
    i = tid
    while i < N:
        O[row, i] = rebind[O.ElementType](
            (
                rebind[Scalar[f32]](X[row, i])
                * scale
                * rebind[Scalar[f32]](G[i])
            ).cast[DType.bfloat16]()
        )
        i += EW_THREADS


def swiglu[
    GLayout: TensorLayout, ULayout: TensorLayout, OLayout: TensorLayout
](
    Gate: TileTensor[f32, GLayout, MutAnyOrigin],
    Up: TileTensor[f32, ULayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    count: Int32,
):
    comptime assert Gate.flat_rank == 1 and Up.flat_rank == 1 and O.flat_rank == 1

    var idx = global_idx.x
    if idx >= Int(count):
        return
    var g = rebind[Scalar[f32]](Gate[idx])
    var u = rebind[Scalar[f32]](Up[idx])
    var silu = g / (1 + exp(-g))
    O[idx] = rebind[O.ElementType](silu * u)


def rope_rows[
    XLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    n_heads: Int32,
    head_dim: Int32,
    pos: Int32,
    theta_base: Float32,
):
    comptime assert X.flat_rank == 2

    var row = block_idx.x
    var half = Int(head_dim) // 2
    var pairs = Int(n_heads) * half
    var idx = thread_idx.x
    while idx < pairs:
        var h = idx // half
        var i = idx % half
        var base = h * Int(head_dim)
        var freq = exp(
            Float32(-2 * i) / Float32(head_dim) * log(theta_base)
        )
        var angle = Float32(Int(pos) + row) * freq
        var c = cos(angle)
        var s = sin(angle)
        var x0 = rebind[Scalar[f32]](X[row, base + i])
        var x1 = rebind[Scalar[f32]](X[row, base + half + i])
        X[row, base + i] = rebind[X.ElementType](x0 * c - x1 * s)
        X[row, base + half + i] = rebind[X.ElementType](x0 * s + x1 * c)
        idx += block_dim.x


def softmax_rows[
    XLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert X.flat_rank == 2

    var N = Int(n)
    var row = block_idx.x
    var tid = thread_idx.x

    var scratch = stack_allocation[
        f32, address_space = AddressSpace.SHARED
    ](row_major[EW_THREADS // WARP_SIZE]())

    var local_max = Float32(-3.4e38)
    var i = tid
    while i < N:
        var v = rebind[Scalar[f32]](X[row, i])
        if v > local_max:
            local_max = v
        i += EW_THREADS
    var wmax = warp.max(local_max)
    if lane_id() == 0:
        scratch[tid // WARP_SIZE] = rebind[scratch.ElementType](wmax)
    barrier()
    var row_max = Float32(-3.4e38)
    comptime for w in range(EW_THREADS // WARP_SIZE):
        var v = rebind[Scalar[f32]](scratch[w])
        if v > row_max:
            row_max = v
    barrier()

    var partial: Float32 = 0
    i = tid
    while i < N:
        var e = exp(rebind[Scalar[f32]](X[row, i]) - row_max)
        X[row, i] = rebind[X.ElementType](e)
        partial += e
        i += EW_THREADS
    var wsum = warp.sum(partial)
    if lane_id() == 0:
        scratch[tid // WARP_SIZE] = rebind[scratch.ElementType](wsum)
    barrier()
    var total: Float32 = 0
    comptime for w in range(EW_THREADS // WARP_SIZE):
        total += rebind[Scalar[f32]](scratch[w])

    var inv = 1 / total
    i = tid
    while i < N:
        X[row, i] = rebind[X.ElementType](rebind[Scalar[f32]](X[row, i]) * inv)
        i += EW_THREADS


def embed_lookup[
    TLayout: TensorLayout, OLayout: TensorLayout
](
    Table: TileTensor[DType.bfloat16, TLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    token: Int32,
    n: Int32,
):
    comptime assert Table.flat_rank == 2 and O.flat_rank == 2

    var idx = global_idx.x
    if idx >= Int(n):
        return
    O[block_idx.y, idx] = rebind[O.ElementType](
        rebind[Scalar[DType.bfloat16]](Table[Int(token), idx]).cast[f32]()
    )


def embed_lookup_pos[
    TLayout: TensorLayout, OLayout: TensorLayout, KLayout: TensorLayout
](
    Table: TileTensor[DType.bfloat16, TLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    Toks: TileTensor[DType.int32, KLayout, MutAnyOrigin],
    pos: Int32,
    n: Int32,
):
    comptime assert Table.flat_rank == 2 and O.flat_rank == 2 and Toks.flat_rank == 1

    var idx = global_idx.x
    if idx >= Int(n):
        return
    var token = Int(rebind[Scalar[DType.int32]](Toks[Int(pos)]))
    O[block_idx.y, idx] = rebind[O.ElementType](
        rebind[Scalar[DType.bfloat16]](Table[token, idx]).cast[f32]()
    )


def argmax_pos[
    XLayout: TensorLayout, OLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    Out: TileTensor[DType.int32, OLayout, MutAnyOrigin],
    n: Int32,
    wpos: Int32,
):
    comptime assert X.flat_rank == 2 and Out.flat_rank == 1

    var N = Int(n)
    var row = block_idx.x
    var tid = thread_idx.x

    var best_v = Float32(-3.4e38)
    var best_i: Int32 = 0
    var i = tid
    while i < N:
        var v = rebind[Scalar[f32]](X[row, i])
        if v > best_v:
            best_v = v
            best_i = Int32(i)
        i += EW_THREADS

    var vals = stack_allocation[
        f32, address_space = AddressSpace.SHARED
    ](row_major[EW_THREADS]())
    var idxs = stack_allocation[
        DType.int32, address_space = AddressSpace.SHARED
    ](row_major[EW_THREADS]())
    vals[tid] = rebind[vals.ElementType](best_v)
    idxs[tid] = rebind[idxs.ElementType](best_i)
    barrier()

    if tid == 0:
        var bv = Float32(-3.4e38)
        var bi: Int32 = 0
        comptime for t in range(EW_THREADS):
            var v = rebind[Scalar[f32]](vals[t])
            var ix = rebind[Scalar[DType.int32]](idxs[t])
            if v > bv or (v == bv and ix < bi):
                bv = v
                bi = ix
        Out[Int(wpos)] = rebind[Out.ElementType](bi)


def argmax_row[
    XLayout: TensorLayout, OLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    Out: TileTensor[DType.int32, OLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert X.flat_rank == 2 and Out.flat_rank == 1

    var N = Int(n)
    var row = block_idx.x
    var tid = thread_idx.x

    var best_v = Float32(-3.4e38)
    var best_i: Int32 = 0
    var i = tid
    while i < N:
        var v = rebind[Scalar[f32]](X[row, i])
        if v > best_v:
            best_v = v
            best_i = Int32(i)
        i += EW_THREADS

    var vals = stack_allocation[
        f32, address_space = AddressSpace.SHARED
    ](row_major[EW_THREADS]())
    var idxs = stack_allocation[
        DType.int32, address_space = AddressSpace.SHARED
    ](row_major[EW_THREADS]())
    vals[tid] = rebind[vals.ElementType](best_v)
    idxs[tid] = rebind[idxs.ElementType](best_i)
    barrier()

    if tid == 0:
        var bv = Float32(-3.4e38)
        var bi: Int32 = 0
        comptime for t in range(EW_THREADS):
            var v = rebind[Scalar[f32]](vals[t])
            var ix = rebind[Scalar[DType.int32]](idxs[t])
            if v > bv or (v == bv and ix < bi):
                bv = v
                bi = ix
        Out[row] = rebind[Out.ElementType](bi)
