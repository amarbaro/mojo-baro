"""Numeric check: shim GEMM on gfx1100 vs a host reference."""
from std.memory import alloc

from baro import Baro


def main() raises:
    comptime M = 4
    comptime K = 3
    comptime N = 2

    var a_host = alloc[Float16](M * K)
    var b_host = alloc[Float16](K * N)
    var c_host = alloc[Float16](M * N)

    for i in range(M):
        for j in range(K):
            a_host[unsafe_offset=i * K + j] = Float16(i + j)
    for i in range(K):
        for j in range(N):
            b_host[unsafe_offset=i * N + j] = Float16(i - j)

    var baro = Baro()
    var a_dev = baro.device_alloc(M * K * 2)
    var b_dev = baro.device_alloc(K * N * 2)
    var c_dev = baro.device_alloc(M * N * 2)

    baro.upload(a_dev, a_host.unsafe_bitcast[UInt8](), M * K * 2)
    baro.upload(b_dev, b_host.unsafe_bitcast[UInt8](), K * N * 2)
    baro.gemm_f16(M, N, K, a_dev, b_dev, c_dev)
    baro.download(c_host.unsafe_bitcast[UInt8](), c_dev, M * N * 2)

    var failures = 0
    for i in range(M):
        for j in range(N):
            var want = Float32(0)
            for p in range(K):
                want += Float32(i + p) * Float32(p - j)
            var got = Float32(c_host[unsafe_offset=i * N + j])
            if abs(got - want) > 0.01:
                print("MISMATCH at", i, j, "got", got, "want", want)
                failures += 1

    baro.device_free(a_dev)
    baro.device_free(b_dev)
    baro.device_free(c_dev)
    a_host.unsafe_free()
    b_host.unsafe_free()
    c_host.unsafe_free()

    if failures:
        raise Error("GEMM mismatch: ", failures, " element(s)")
    print("GEMM OK —", M, "x", K, "@", K, "x", N, "matches host reference")
