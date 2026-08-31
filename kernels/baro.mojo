"""Mojo binding over libbaro_shim.so — the C-ABI edge to hipBLASLt.

The shim context is carried as an integer address. Mojo 1.0's `Pointer` is
non-null by design, and an opaque C handle we never dereference gains nothing
from pointer typing, so the address is the honest representation.
"""

from std.ffi import external_call


struct Baro(Movable):
    var _ctx: Int

    def __init__(out self, device_id: Int = 0) raises:
        self._ctx = external_call["baro_init", Int](Int32(device_id))
        if self._ctx == 0:
            raise Error("baro_init failed: ", Self.last_error())

    def __deinit__(deinit self):
        external_call["baro_destroy", NoneType](self._ctx)

    @staticmethod
    def last_error() -> String:
        var p = external_call["baro_last_error", UnsafePointer[UInt8, MutUntrackedOrigin]]()
        return String(unsafe_from_utf8_ptr=p)

    def device_alloc(self, bytes: Int) raises -> Int:
        var p = external_call["baro_device_alloc", Int](bytes)
        if p == 0:
            raise Error("device_alloc failed: ", Self.last_error())
        return p

    def device_free(self, ptr: Int):
        external_call["baro_device_free", NoneType](ptr)

    def upload(self, dst: Int, src: UnsafePointer[UInt8, MutUntrackedOrigin], bytes: Int) raises:
        if external_call["baro_upload", Int32](self._ctx, dst, src, bytes) != 0:
            raise Error("upload failed: ", Self.last_error())

    def download(self, dst: UnsafePointer[UInt8, MutUntrackedOrigin], src: Int, bytes: Int) raises:
        if external_call["baro_download", Int32](self._ctx, dst, src, bytes) != 0:
            raise Error("download failed: ", Self.last_error())

    def sync(self) raises:
        if external_call["baro_sync", Int32](self._ctx) != 0:
            raise Error("sync failed: ", Self.last_error())

    def gemm_f32(
        self, m: Int, n: Int, k: Int, a: Int, b: Int, c: Int,
        alpha: Float32 = 1.0, beta: Float32 = 0.0,
    ) raises:
        """Vendor fp32 GEMM. Pointers are raw device addresses; they may come
        from this shim or from a MAX DeviceBuffer via unsafe_ptr()."""
        var rc = external_call["baro_gemm_f32", Int32](
            self._ctx, Int32(m), Int32(n), Int32(k), a, b, c, alpha, beta
        )
        if rc != 0:
            raise Error("gemm_f32 failed: ", Self.last_error())

    def gemm_f16(
        self, m: Int, n: Int, k: Int, a: Int, b: Int, c: Int,
        alpha: Float32 = 1.0, beta: Float32 = 0.0,
    ) raises:
        var rc = external_call["baro_gemm_f16", Int32](
            self._ctx, Int32(m), Int32(n), Int32(k), a, b, c, alpha, beta
        )
        if rc != 0:
            raise Error("gemm_f16 failed: ", Self.last_error())


def main() raises:
    var baro = Baro()
    print("baro context up on device 0")
