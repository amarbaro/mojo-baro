#!/usr/bin/env python3
"""Embed baro kernel sources into a GGUF's metadata (self-describing model).

Writes a NEW file — never touches the source GGUF. Adds string KV pairs:
  baro.kernel.arch, baro.kernel.commit, baro.kernel.files (comma list),
  baro.kernel.src.<basename> = full source text
GGUF readers ignore unknown keys; tensor offsets are relative to the data
region so only the header padding changes.

Usage: tools/gguf-embed.py SRC.gguf DST.gguf kernels/*.mojo
"""
import os
import struct
import subprocess
import sys
from pathlib import Path

ALIGN = 32


def w_str(out, s):
    b = s.encode("utf-8")
    out.write(struct.pack("<Q", len(b)))
    out.write(b)


def src_key(k):
    """Kernel/engine sources are keyed by basename (engine imports them flat);
    anything else keeps its repo-relative path so the closure can rebuild it."""
    rel = k.resolve().relative_to(Path(__file__).resolve().parent.parent).as_posix()
    return k.name if rel.startswith(("kernels/", "serve/")) else rel


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    kfiles = [Path(p) for p in sys.argv[3:]]
    assert src.exists() and not dst.exists(), "dst must not exist"

    f = open(src, "rb")
    magic, version = struct.unpack("<4sI", f.read(8))
    assert magic == b"GGUF" and version == 3
    n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

    commit = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parent.parent),
         "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    new_kv = [("baro.kernel.arch", "gfx1100"),
              ("baro.kernel.commit", commit),
              ("baro.kernel.files", ",".join(src_key(k) for k in kfiles))]
    if os.environ.get("BARO_KERNEL_PARENT"):
        new_kv.append(("baro.kernel.parent", os.environ["BARO_KERNEL_PARENT"]))
    for k in kfiles:
        new_kv.append((f"baro.kernel.src.{src_key(k)}", k.read_text()))

    out = open(dst, "wb")
    out.write(struct.pack("<4sI", b"GGUF", 3))
    out.write(struct.pack("<QQ", n_tensors, n_kv + len(new_kv)))

    # copy existing KV region + tensor infos verbatim by re-parsing bounds
    kv_start = f.tell()
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("ge", Path(__file__).parent / "gguf-extract.py")
    ge = module_from_spec(spec)
    spec.loader.exec_module(ge)
    _, infos, data_start, kv = ge.parse(src)
    f.seek(kv_start)
    for _ in range(n_kv):
        ge.read_str(f)
        (vtype,) = struct.unpack("<I", f.read(4))
        ge.read_value(f, vtype, want=False)
    kv_end = f.tell()
    for _ in range(n_tensors):
        ge.read_str(f)
        (nd,) = struct.unpack("<I", f.read(4))
        f.seek(8 * nd + 4 + 8, 1)
    info_end = f.tell()

    f.seek(kv_start)
    out.write(f.read(kv_end - kv_start))
    for key, val in new_kv:
        w_str(out, key)
        out.write(struct.pack("<I", 8))
        w_str(out, val)
    f.seek(kv_end)
    out.write(f.read(info_end - kv_end))

    pad = (ALIGN - out.tell() % ALIGN) % ALIGN
    out.write(b"\x00" * pad)
    f.seek(data_start)
    while True:
        chunk = f.read(1 << 24)
        if not chunk:
            break
        out.write(chunk)
    out.close()
    print(f"wrote {dst} (+{len(new_kv)} kv)")


if __name__ == "__main__":
    main()
