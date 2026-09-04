#!/usr/bin/env python3
"""Q0 gate for bench/draft-q4-protocol.md: the q4-draft engine pack entry
must be byte-equal to llama.cpp's Q4_0 quantization of the same tensor.

Usage: tools/q4-check.py PACKDIR Q4_0.gguf name [name ...]
For each pack name: int4 nibbles and fp16 scales compared element-wise
against the q4_0 blocks (18 bytes: fp16 d + 16 packed-nibble bytes) of the
llama-quantize file. A pack name ending in ".q4draft" is looked up in the
GGUF under its name with that suffix stripped (the pack stores the draft
head's q4 copy under a distinct index name so it doesn't collide with the
trunk's own "output.weight" entry).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("ge", Path(__file__).parent / "gguf-extract.py")
ge = module_from_spec(spec)
spec.loader.exec_module(ge)

Q4_0_TYPE = 2


def main():
    packdir, gguf, names = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    idx = {}
    for line in (packdir / "index.txt").read_text().splitlines():
        n, dt, off, ne = line.split()
        idx[n] = (dt, int(off), int(ne))
    pack = np.memmap(packdir / "pack.bin", dtype=np.uint8, mode="r")
    f, infos, data_start, _ = ge.parse(gguf)
    ok = True
    for name in names:
        dt, off, ne = idx[name]
        assert dt == "q4", (name, dt)
        nqs = ne // 2
        pq = np.frombuffer(pack[off:off + nqs], dtype=np.uint8)
        pd = np.frombuffer(pack[off + nqs:off + nqs + (ne // 32) * 2], dtype=np.float16)
        gguf_name = name[: -len(".q4draft")] if name.endswith(".q4draft") else name
        dims, ttype, toff = infos[gguf_name]
        assert ttype == Q4_0_TYPE, (gguf_name, ttype)
        nb = ne // 32
        f.seek(data_start + toff)
        blk = np.frombuffer(f.read(nb * 18), dtype=np.uint8).reshape(nb, 18)
        gd = blk[:, :2].copy().view(np.float16).reshape(-1)
        gq = blk[:, 2:].copy().view(np.uint8).reshape(-1)
        dq = int((pq != gq).sum())
        dd = int((pd.view(np.uint16) != gd.view(np.uint16)).sum())
        print(f"{name}: nibbles mismatch {dq}/{nqs}  scales mismatch {dd}/{nb}")
        ok = ok and dq == 0 and dd == 0
    print("Q0 gate:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
