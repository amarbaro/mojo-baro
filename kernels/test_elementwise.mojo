"""Host-reference checks for the elementwise/decode kernel pack.

Every kernel runs on real decode shapes (rows=8, hidden=4096 or vocab-sized)
against a straightforward fp64 host implementation. Gate per kernel:
max rel err < 1e-4 (fp32 vs fp64 reference), exact for argmax.
"""
from std.math import ceildiv, cos, exp, log, sin, sqrt
from std.sys import has_accelerator

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

from elementwise import (
    amar_rmsnorm, amar_swiglu, amar_rope_rows, amar_softmax_rows, amar_embed_lookup, amar_argmax_row,
    EW_THREADS,
)

comptime R = 8
comptime H = 4096
comptime V = 1024      # vocab stand-in for argmax/softmax width
comptime HEADS = 8
comptime HDIM = 128

comptime f32 = DType.float32


def fail(name: String, detail: Float64) raises:
    print("FAIL:", name, detail)
    raise Error("elementwise parity failure: " + name)


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()

    comptime x_layout = row_major[R, H]()
    comptime g_layout = row_major[H]()
    comptime flat_layout = row_major[R * H]()
    comptime v_layout = row_major[R, V]()
    comptime tok_layout = row_major[R]()
    comptime table_layout = row_major[V, H]()

    var x_host = ctx.enqueue_create_host_buffer[f32](R * H)
    var g_host = ctx.enqueue_create_host_buffer[f32](H)
    var o_host = ctx.enqueue_create_host_buffer[f32](R * H)
    ctx.synchronize()
    for i in range(R * H):
        x_host[i] = Scalar[f32]((i % 37) - 18) * 0.11
    for i in range(H):
        g_host[i] = Scalar[f32]((i % 11) - 5) * 0.3

    var x_dev = ctx.enqueue_create_buffer[f32](R * H)
    var g_dev = ctx.enqueue_create_buffer[f32](H)
    var o_dev = ctx.enqueue_create_buffer[f32](R * H)
    ctx.enqueue_copy(dst_buf=x_dev, src_buf=x_host)
    ctx.enqueue_copy(dst_buf=g_dev, src_buf=g_host)
    ctx.synchronize()

    var X = TileTensor(x_dev, x_layout)
    var G = TileTensor(g_dev, g_layout)
    var O = TileTensor(o_dev, x_layout)

    # --- amar_rmsnorm ---
    comptime rms_k = amar_rmsnorm[type_of(x_layout), type_of(g_layout), type_of(x_layout)]
    ctx.enqueue_function[rms_k](
        X, G, O, Int32(H), Float32(1e-6), grid_dim=R, block_dim=EW_THREADS
    )
    ctx.enqueue_copy(dst_buf=o_host, src_buf=o_dev)
    ctx.synchronize()
    var worst = Float64(0)
    for r in range(R):
        var ss = Float64(0)
        for i in range(H):
            ss += Float64(x_host[r * H + i]) * Float64(x_host[r * H + i])
        var scale = 1.0 / sqrt(ss / Float64(H) + 1e-6)
        for i in range(0, H, 7):
            var want = Float64(x_host[r * H + i]) * scale * Float64(g_host[i])
            var err = abs(Float64(o_host[r * H + i]) - want) / (abs(want) + 1e-6)
            if err > worst:
                worst = err
    if worst > 1e-4:
        fail("amar_rmsnorm", worst)
    print("amar_rmsnorm ok", worst)

    # --- amar_swiglu (reuse x as gate, g broadcast? use two flat tensors) ---
    var u_host = ctx.enqueue_create_host_buffer[f32](R * H)
    ctx.synchronize()
    for i in range(R * H):
        u_host[i] = Scalar[f32]((i % 23) - 11) * 0.2
    var u_dev = ctx.enqueue_create_buffer[f32](R * H)
    ctx.enqueue_copy(dst_buf=u_dev, src_buf=u_host)
    ctx.synchronize()
    var Xf = TileTensor(x_dev, flat_layout)
    var Uf = TileTensor(u_dev, flat_layout)
    var Of = TileTensor(o_dev, flat_layout)
    comptime swiglu_k = amar_swiglu[
        type_of(flat_layout), type_of(flat_layout), type_of(flat_layout)
    ]
    ctx.enqueue_function[swiglu_k](
        Xf, Uf, Of, Int32(R * H),
        grid_dim=ceildiv(R * H, EW_THREADS), block_dim=EW_THREADS,
    )
    ctx.enqueue_copy(dst_buf=o_host, src_buf=o_dev)
    ctx.synchronize()
    worst = 0
    for i in range(0, R * H, 13):
        var gv = Float64(x_host[i])
        var want = gv / (1.0 + exp(-gv)) * Float64(u_host[i])
        var err = abs(Float64(o_host[i]) - want) / (abs(want) + 1e-6)
        if err > worst:
            worst = err
    if worst > 1e-4:
        fail("amar_swiglu", worst)
    print("amar_swiglu ok", worst)

    # --- rope (in place on a copy of x) ---
    ctx.enqueue_copy(dst_buf=o_dev, src_buf=x_host)
    ctx.synchronize()
    comptime rope_k = amar_rope_rows[type_of(x_layout)]
    comptime pos = 5
    comptime theta = 10000.0
    ctx.enqueue_function[rope_k](
        O, Int32(HEADS), Int32(HDIM), Int32(pos), Float32(theta),
        grid_dim=R, block_dim=EW_THREADS,
    )
    ctx.enqueue_copy(dst_buf=o_host, src_buf=o_dev)
    ctx.synchronize()
    worst = 0
    for r in range(R):
        for h in range(HEADS):
            for i in range(0, HDIM // 2, 5):
                var base = h * HDIM
                var freq = exp(Float64(-2 * i) / Float64(HDIM) * log(Float64(theta)))
                var ang = Float64(pos + r) * freq
                var x0 = Float64(x_host[r * H + base + i])
                var x1 = Float64(x_host[r * H + base + HDIM // 2 + i])
                var w0 = x0 * cos(ang) - x1 * sin(ang)
                var w1 = x0 * sin(ang) + x1 * cos(ang)
                var e0 = abs(Float64(o_host[r * H + base + i]) - w0)
                var e1 = abs(Float64(o_host[r * H + base + HDIM // 2 + i]) - w1)
                var err = max(e0, e1) / (max(abs(w0), abs(w1)) + 1e-6)
                if err > worst:
                    worst = err
    if worst > 1e-4:
        fail("rope", worst)
    print("rope ok", worst)

    # NOTE: RoPE beyond head 0..HEADS covers only HEADS*HDIM = 1024 of H;
    # remaining columns are untouched by the kernel and unchecked by design.

    # --- softmax + argmax on [R, V] ---
    var s_host = ctx.enqueue_create_host_buffer[f32](R * V)
    var t_host = ctx.enqueue_create_host_buffer[DType.int32](R)
    ctx.synchronize()
    for i in range(R * V):
        s_host[i] = Scalar[f32](((i * 131) % 997) - 498) * 0.01
    var s_dev = ctx.enqueue_create_buffer[f32](R * V)
    var t_dev = ctx.enqueue_create_buffer[DType.int32](R)
    ctx.enqueue_copy(dst_buf=s_dev, src_buf=s_host)
    ctx.synchronize()
    var S = TileTensor(s_dev, v_layout)
    var T = TileTensor(t_dev, tok_layout)

    comptime argmax_k = amar_argmax_row[type_of(v_layout), type_of(tok_layout)]
    ctx.enqueue_function[argmax_k](
        S, T, Int32(V), grid_dim=R, block_dim=EW_THREADS
    )
    comptime softmax_k = amar_softmax_rows[type_of(v_layout)]
    ctx.enqueue_function[softmax_k](S, Int32(V), grid_dim=R, block_dim=EW_THREADS)
    ctx.enqueue_copy(dst_buf=s_host, src_buf=s_dev)
    ctx.enqueue_copy(dst_buf=t_host, src_buf=t_dev)
    ctx.synchronize()
    worst = 0
    for r in range(R):
        var mx = Float64(-1e30)
        var best = 0
        for i in range(V):
            var raw = Float64(((r * V + i) * 131 % 997) - 498) * 0.01
            if raw > mx:
                mx = raw
                best = i
        if Int(t_host[r]) != best:
            fail("argmax", Float64(Int(t_host[r])))
        var tot = Float64(0)
        for i in range(V):
            var raw = Float64(((r * V + i) * 131 % 997) - 498) * 0.01
            tot += exp(raw - mx)
        for i in range(0, V, 17):
            var raw = Float64(((r * V + i) * 131 % 997) - 498) * 0.01
            var want = exp(raw - mx) / tot
            var err = abs(Float64(s_host[r * V + i]) - want) / (want + 1e-9)
            if err > worst:
                worst = err
    if worst > 1e-4:
        fail("softmax", worst)
    print("softmax+argmax ok", worst)

    # --- embed lookup from a bf16 table ---
    var tab_host = ctx.enqueue_create_host_buffer[DType.bfloat16](V * H)
    ctx.synchronize()
    for i in range(V * H):
        tab_host[i] = (Scalar[f32]((i % 19) - 9) * 0.125).cast[DType.bfloat16]()
    var tab_dev = ctx.enqueue_create_buffer[DType.bfloat16](V * H)
    ctx.enqueue_copy(dst_buf=tab_dev, src_buf=tab_host)
    ctx.synchronize()
    var Tab = TileTensor(tab_dev, table_layout)
    comptime embed_k = amar_embed_lookup[type_of(table_layout), type_of(x_layout)]
    comptime token = 123
    ctx.enqueue_function[embed_k](
        Tab, O, Int32(token), Int32(H),
        grid_dim=(ceildiv(H, EW_THREADS), 1), block_dim=EW_THREADS,
    )
    ctx.enqueue_copy(dst_buf=o_host, src_buf=o_dev)
    ctx.synchronize()
    worst = 0
    for i in range(0, H, 3):
        var want = Float64(tab_host[token * H + i].cast[f32]())
        var err = abs(Float64(o_host[i]) - want)
        if err > worst:
            worst = err
    if worst > 0:
        fail("embed", worst)
    print("embed ok", worst)

    print("PASS: all elementwise kernels match host references")
