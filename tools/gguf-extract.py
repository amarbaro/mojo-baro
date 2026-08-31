#!/usr/bin/env python3
"""Extract tensors from a GGUF file for kernel parity tests.

Parses header/metadata/tensor-info per docs/gguf-format-notes.md, then for
each requested tensor writes its raw bytes to .work/gguf/<safe-name>.bin.
With --ref it also writes a random bf16 activation matrix A (M x K) and the
fp32 reference product C = A @ W^T (W stored [out, in] row-major, so the
GEMM output is [M, out]) for the Mojo test to compare against.

Usage: tools/gguf-extract.py MODEL.gguf blk.0.ffn_up.weight --ref 8
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

ALIGN_DEFAULT = 32
GGML_BYTES = {0: ("f32", 4), 1: ("f16", 2), 30: ("bf16", 2)}


def read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8")


SCALAR_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
              6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}


def read_value(f, vtype, want=True):
    """Read (or skip, want=False) one metadata value."""
    if vtype in SCALAR_FMT:
        fmt = SCALAR_FMT[vtype]
        raw = f.read(struct.calcsize(fmt))
        return struct.unpack(fmt, raw)[0] if want else None
    if vtype == 8:
        s = read_str(f)
        return s if want else None
    if vtype == 9:
        (etype,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        # Huge token arrays are only materialized when asked for.
        keep = want and n <= 4096
        vals = [read_value(f, etype, keep) for _ in range(n)]
        return vals if keep else f"<array len={n}>"
    raise ValueError(f"unknown kv type {vtype}")


def parse(path):
    f = open(path, "rb")
    magic, version = struct.unpack("<4sI", f.read(8))
    assert magic == b"GGUF" and version == 3, (magic, version)
    n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
    align = ALIGN_DEFAULT
    kv = {}
    for _ in range(n_kv):
        key = read_str(f)
        (vtype,) = struct.unpack("<I", f.read(4))
        kv[key] = read_value(f, vtype)
    align = kv.get("general.alignment", ALIGN_DEFAULT)
    infos = {}
    for _ in range(n_tensors):
        name = read_str(f)
        (nd,) = struct.unpack("<I", f.read(4))
        dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
        ttype, offset = struct.unpack("<IQ", f.read(4 + 8))
        infos[name] = (dims, ttype, offset)
    data_start = (f.tell() + align - 1) // align * align
    return f, infos, data_start, kv


def main():
    argv = sys.argv[1:]
    ref_m = 0
    if "--ref" in argv:
        i = argv.index("--ref")
        ref_m = int(argv[i + 1])
        del argv[i : i + 2]
    dump_meta = "--meta" in argv
    if dump_meta:
        argv.remove("--meta")
    model, names = argv[0], argv[1:]
    out_dir = Path(__file__).resolve().parent.parent / ".work/gguf"
    out_dir.mkdir(parents=True, exist_ok=True)
    f, infos, data_start, kv = parse(model)
    if dump_meta:
        print(json.dumps(kv, indent=1, default=str))
        if not names:
            return
    meta = {}
    for name in names:
        dims, ttype, offset = infos[name]
        tname, esize = GGML_BYTES[ttype]
        n_elem = int(np.prod(dims))
        f.seek(data_start + offset)
        raw = f.read(n_elem * esize)
        safe = name.replace(".", "_").replace("/", "_")
        (out_dir / f"{safe}.bin").write_bytes(raw)
        # dims are innermost-first; logical (torch) shape is reversed
        shape = list(reversed(dims))
        meta[name] = {"file": f"{safe}.bin", "shape": shape, "type": tname}
        if ref_m and tname == "bf16" and len(shape) == 2:
            out_f, in_f = shape  # W is [out, in] row-major
            w16 = np.frombuffer(raw, dtype=np.uint16).reshape(out_f, in_f)
            w32 = (w16.astype(np.uint32) << 16).view(np.float32)
            rng = np.random.default_rng(7)
            a32 = rng.standard_normal((ref_m, in_f)).astype(np.float32)
            a16 = (a32.view(np.uint32) >> 16).astype(np.uint16)  # truncate to bf16
            a32t = (a16.astype(np.uint32) << 16).view(np.float32)
            c32 = a32t @ w32.T  # [M, out]
            (out_dir / f"{safe}.a.bin").write_bytes(a16.tobytes())
            (out_dir / f"{safe}.c.bin").write_bytes(c32.astype(np.float32).tobytes())
            meta[name]["ref_m"] = ref_m
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
