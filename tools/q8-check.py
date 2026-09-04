#!/usr/bin/env python3
"""Q0 gate for bench/q8-protocol.md: the q8 engine pack must be bit-equal to
llama.cpp's Q8_0 quantization of the same model.

Usage: tools/q8-check.py PACKDIR Q8_0.gguf name [name ...]
For each tensor: int8 quants and fp16 scales compared element-wise against the
q8_0 blocks (34 bytes: fp16 d + 32 int8) of the llama-quantize file.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("ge", Path(__file__).parent / "gguf-extract.py")
ge = module_from_spec(spec)
spec.loader.exec_module(ge)

Q8_0_TYPE = 8


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
        assert dt == "q8", (name, dt)
        pq = np.frombuffer(pack[off:off + ne], dtype=np.int8)
        pd = np.frombuffer(pack[off + ne:off + ne + (ne // 32) * 2], dtype=np.float16)
        dims, ttype, toff = infos[name]
        assert ttype == Q8_0_TYPE, (name, ttype)
        nb = ne // 32
        f.seek(data_start + toff)
        blk = np.frombuffer(f.read(nb * 34), dtype=np.uint8).reshape(nb, 34)
        gd = blk[:, :2].copy().view(np.float16).reshape(-1)
        gq = blk[:, 2:].copy().view(np.int8).reshape(-1)
        dq = int((pq != gq).sum())
        dd = int((pd.view(np.uint16) != gd.view(np.uint16)).sum())
        print(f"{name}: quants mismatch {dq}/{ne}  scales mismatch {dd}/{nb}")
        ok = ok and dq == 0 and dd == 0
    print("Q0 gate:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
