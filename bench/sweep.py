#!/usr/bin/env python3
"""Sweep the regtile kernel's compile-time tuning constants.

Rewrites the comptime constants in kernels/matmul.mojo, rebuilds, benchmarks,
and restores the original file. Numeric search belongs in a loop, not in an
agent: each point is exact, costs no tokens, and takes a few seconds.
"""
import itertools
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "kernels" / "matmul.mojo"
OUT = ROOT / "bench" / "sweep.jsonl"

# (BM, BN, BK, TM, TN) -- threads = (BM/TM)*(BN/TN), must be a sane block size.
GRID = dict(BM=[32, 64, 128], BN=[32, 64, 128], BK=[8, 16, 32], TM=[2, 4, 8], TN=[2, 4, 8])


def patch(**vals):
    text = SRC.read_text()
    for k, v in vals.items():
        text = re.sub(rf"^comptime {k} = \d+$", f"comptime {k} = {v}", text, flags=re.M)
    SRC.write_text(text)


def measure():
    b = subprocess.run(
        [str(ROOT / ".venv/bin/mojo"), "build", "bench/bench.mojo",
         "-o", str(ROOT / ".work/sweep_bin"), "-I", "kernels"],
        cwd=ROOT, text=True, capture_output=True)
    if b.returncode:
        return None, "build failed"
    r = subprocess.run([str(ROOT / ".work/sweep_bin")], cwd=ROOT, text=True,
                       capture_output=True, timeout=300)
    if r.returncode:
        return None, "run failed"
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and '"regtile"' in line:
            rec = json.loads(line)
            if not rec["correct"]:
                return None, f"INCORRECT (err {rec['max_err']:.3g})"
            return rec["gflops"], "ok"
    return None, "no regtile record"


def main():
    original = SRC.read_text()
    combos = []
    for bm, bn, bk, tm, tn in itertools.product(*GRID.values()):
        if bm % tm or bn % tn:
            continue
        threads = (bm // tm) * (bn // tn)
        if not (64 <= threads <= 1024) or threads % 64:
            continue
        # Shared memory: (BM*BK + BK*BN) floats, keep under 64 KB LDS.
        if (bm * bk + bk * bn) * 4 > 64 * 1024:
            continue
        # Staging loops assume the tiles divide evenly across the block.
        if (bm * bk) % threads or (bk * bn) % threads:
            continue
        combos.append((bm, bn, bk, tm, tn))

    print(f"{len(combos)} valid configurations")
    results = []
    try:
        with OUT.open("a") as fh:
            for i, (bm, bn, bk, tm, tn) in enumerate(combos, 1):
                patch(BM=bm, BN=bn, BK=bk, TM=tm, TN=tn)
                gflops, status = measure()
                rec = dict(BM=bm, BN=bn, BK=bk, TM=tm, TN=tn,
                           threads=(bm // tm) * (bn // tn),
                           gflops=gflops, status=status)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                if gflops:
                    results.append(rec)
                print(f"[{i}/{len(combos)}] BM{bm} BN{bn} BK{bk} TM{tm} TN{tn} "
                      f"-> {gflops:.0f} GFLOP/s" if gflops else
                      f"[{i}/{len(combos)}] BM{bm} BN{bn} BK{bk} TM{tm} TN{tn} -> {status}")
    finally:
        SRC.write_text(original)

    results.sort(key=lambda r: -r["gflops"])
    print("\ntop 10:")
    for r in results[:10]:
        print(f"  {r['gflops']:8.0f}  BM{r['BM']} BN{r['BN']} BK{r['BK']} "
              f"TM{r['TM']} TN{r['TN']}  ({r['threads']} thr)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
