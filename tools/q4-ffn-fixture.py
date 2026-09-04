#!/usr/bin/env python3
"""Bench fixture for bench_coldcache_mrow.mojo's q4row arm: a standalone
ggml-exact Q4_0 quantization of blk.0.ffn_gate.weight (the ffn shape used by
every other cold-cache arm in that bench), written next to the tensor's
existing bf16 extraction. Not part of the engine pack -- this shape isn't
the draft head, it's just the fixed shape the M0/M4 cold-cache receipts
compare against.

Usage: tools/q4-ffn-fixture.py MODEL.gguf
Writes .work/gguf/blk_0_ffn_gate_weight.q4.bin (nibbles, [N, K/2]) and
.q4scales.bin (fp16, [N, K/32]).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("ge", Path(__file__).parent / "gguf-extract.py")
ge = module_from_spec(spec)
spec.loader.exec_module(ge)
spec2 = spec_from_file_location("epack", Path(__file__).parent / "engine-pack.py")
epack = module_from_spec(spec2)
spec2.loader.exec_module(epack)

NAME = "blk.0.ffn_gate.weight"


def main():
    model = sys.argv[1]
    out_dir = Path(__file__).resolve().parent.parent / ".work/gguf"
    out_dir.mkdir(parents=True, exist_ok=True)
    f, infos, data_start, _ = ge.parse(model)
    dims, ttype, toff = infos[NAME]
    tname, esize = ge.GGML_BYTES[ttype]
    assert tname == "bf16", tname
    shape = list(reversed(dims))
    n_elem = int(np.prod(dims))
    f.seek(data_start + toff)
    w = np.frombuffer(f.read(n_elem * esize), dtype=np.uint16).reshape(shape)
    q, d = epack.quantize_q4_0(w)
    safe = NAME.replace(".", "_")
    (out_dir / f"{safe}.q4.bin").write_bytes(q.tobytes())
    (out_dir / f"{safe}.q4scales.bin").write_bytes(d.tobytes())
    print(f"{safe}: nibbles {q.nbytes} bytes, scales {d.nbytes} bytes")


if __name__ == "__main__":
    main()
