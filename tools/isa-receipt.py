#!/usr/bin/env python3
"""ISA receipt for a Mojo-built binary: per gfx code object, kernel resource
usage (VGPR/SGPR/LDS/scratch/spills) and an instruction histogram.

usage: tools/isa-receipt.py <binary> [--hist N] [--dump DIR]
Extracts every embedded e_machine=224 (AMDGPU) ELF, reads its metadata notes
with llvm-readelf and disassembles with llvm-objdump --mcpu=gfx1100.
"""
import argparse, collections, os, re, struct, subprocess, sys, tempfile

LLVM = "/opt/rocm/llvm/bin"
KEYS = (".vgpr_count", ".sgpr_count", ".group_segment_fixed_size",
        ".private_segment_fixed_size", ".vgpr_spill_count", ".sgpr_spill_count",
        ".max_flat_workgroup_size")


def code_objects(path):
    d = open(path, "rb").read()
    i = 0
    while True:
        i = d.find(b"\x7fELF", i)
        if i < 0:
            return
        if struct.unpack_from("<H", d, i + 18)[0] == 224:
            shoff, = struct.unpack_from("<Q", d, i + 40)
            se, sn = struct.unpack_from("<HH", d, i + 58)
            yield d[i:i + shoff + se * sn]
        i += 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--hist", type=int, default=25)
    ap.add_argument("--dump", help="keep .co files and .s here")
    a = ap.parse_args()
    out = a.dump or tempfile.mkdtemp(prefix="isa-")
    os.makedirs(out, exist_ok=True)
    n = 0
    for blob in code_objects(a.binary):
        p = os.path.join(out, f"co{n}.co")
        open(p, "wb").write(blob)
        notes = subprocess.run([f"{LLVM}/llvm-readelf", "--notes", p],
                               capture_output=True, text=True).stdout
        name = None
        for line in notes.splitlines():
            s = line.strip()
            if s.startswith(".name:"):
                name = s.split(":", 1)[1].strip()
            for k in KEYS:
                if s.startswith(k + ":"):
                    print(f"co{n} {name} {k[1:]}={s.split(':',1)[1].strip()}")
        asm = subprocess.run([f"{LLVM}/llvm-objdump", "-d", "--mcpu=gfx1100", p],
                             capture_output=True, text=True).stdout
        if a.dump:
            open(os.path.join(out, f"co{n}.s"), "w").write(asm)
        hist = collections.Counter()
        for line in asm.splitlines():
            m = re.match(r"\s*([a-z_0-9]+)\s", line)
            if m and ("_" in m.group(1)):
                hist[m.group(1)] += 1
        total = sum(hist.values())
        print(f"co{n} instructions={total}")
        for op, c in hist.most_common(a.hist):
            print(f"  {c:6d}  {op}")
        n += 1
    if n == 0:
        print("no AMDGPU code objects found", file=sys.stderr)
        sys.exit(1)
    if a.dump:
        print(f"dumped to {out}")


if __name__ == "__main__":
    main()
