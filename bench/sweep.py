#!/usr/bin/env python3
"""Sweep a kernel's compile-time tuning constants.

Rewrites the comptime constants in the kernel source, rebuilds, benchmarks, and
restores the original file. Numeric search belongs in a loop, not in an agent:
each point is exact, costs no tokens, and takes a few seconds.

Two targets:
  regtile  fp32 register-tiled kernel   (BM/BN/BK/TM/TN)
  wmma     fp16 matrix-core kernel      (WARPS_M/N, WTILE_M/N)
"""
import argparse
import itertools
import json
import pathlib
import re
import os
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENV = {**os.environ}
ENV.setdefault("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT", "10")

TARGETS = {
    "regtile": dict(
        src=ROOT / "kernels" / "matmul.mojo",
        bench="bench/bench.mojo",
        variant="regtile",
        out=ROOT / "bench" / "sweep.jsonl",
    ),
    "wmma": dict(
        src=ROOT / "kernels" / "matmul_wmma_lds.mojo",
        bench="bench/bench_fp16.mojo",
        variant="wmma_lds_fp16",
        out=ROOT / "bench" / "sweep-wmma.jsonl",
    ),
}

# (BM, BN, BK, TM, TN) -- threads = (BM/TM)*(BN/TN), must be a sane block size.
GRID = dict(BM=[32, 64, 128], BN=[32, 64, 128], BK=[8, 16, 32], TM=[2, 4, 8], TN=[2, 4, 8])

# WMMA: block is WARPS_M*WARPS_N waves; each wave owns WTILE_M x WTILE_N
# 16x16 output tiles. Block tile is WARPS_M*WTILE_M*16 by WARPS_N*WTILE_N*16.
WMMA_GRID = dict(WARPS_M=[1, 2, 4], WARPS_N=[1, 2, 4],
                 WTILE_M=[1, 2, 4], WTILE_N=[1, 2, 4])


def wmma_combos():
    for wm, wn, tm, tn in itertools.product(*WMMA_GRID.values()):
        threads = wm * wn * 32
        if not (64 <= threads <= 512):
            continue
        blk_m, blk_n, blk_k = wm * tm * 16, wn * tn * 16, 16
        # Staging loops assume the tiles divide evenly across the block.
        if (blk_m * blk_k) % threads or (blk_k * blk_n) % threads:
            continue
        # fp16 LDS footprint; keep well under 64 KB so occupancy stays high.
        if (blk_m * blk_k + blk_k * blk_n) * 2 > 32 * 1024:
            continue
        # Accumulator registers per lane.
        if tm * tn * 8 > 128:
            continue
        yield dict(WARPS_M=wm, WARPS_N=wn, WTILE_M=tm, WTILE_N=tn)


def patch(SRC, **vals):
    text = SRC.read_text()
    for k, v in vals.items():
        text = re.sub(rf"^comptime {k} = \d+$", f"comptime {k} = {v}", text, flags=re.M)
    SRC.write_text(text)


def measure(T):
    b = subprocess.run(
        [str(ROOT / ".venv/bin/mojo"), "build",
         T["bench"], "-o", str(ROOT / ".work/sweep_bin"), "-I", "kernels",
         "-Xlinker", f"-L{ROOT / '.work/shim-build'}", "-Xlinker", "-lamarbaro_shim",
         "-Xlinker", "-rpath", "-Xlinker", str(ROOT / ".work/shim-build")],
        cwd=ROOT, text=True, capture_output=True)
    if b.returncode:
        return None, "build failed"
    r = subprocess.run([str(ROOT / ".work/sweep_bin")], cwd=ROOT, text=True,
                       capture_output=True, timeout=300, env=ENV)
    if r.returncode:
        return None, "run failed"
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and f'"{T["variant"]}"' in line:
            rec = json.loads(line)
            if not rec["correct"]:
                return None, f"INCORRECT (err {rec['max_err']:.3g})"
            return rec["gflops"], "ok"
    return None, "no regtile record"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default="regtile", choices=sorted(TARGETS))
    args = ap.parse_args()
    T = TARGETS[args.target]
    SRC, OUT = T["src"], T["out"]

    original = SRC.read_text()
    if args.target == "wmma":
        combos = list(wmma_combos())
        label = lambda c: (f"WARPS {c['WARPS_M']}x{c['WARPS_N']} "
                           f"WTILE {c['WTILE_M']}x{c['WTILE_N']} "
                           f"({c['WARPS_M']*c['WTILE_M']*16}x"
                           f"{c['WARPS_N']*c['WTILE_N']*16} tile, "
                           f"{c['WARPS_M']*c['WARPS_N']*32} thr)")
    else:
        combos = []
        for bm, bn, bk, tm, tn in itertools.product(*GRID.values()):
            if bm % tm or bn % tn:
                continue
            threads = (bm // tm) * (bn // tn)
            if not (64 <= threads <= 1024) or threads % 64:
                continue
            if (bm * bk + bk * bn) * 4 > 64 * 1024:
                continue
            if (bm * bk) % threads or (bk * bn) % threads:
                continue
            combos.append(dict(BM=bm, BN=bn, BK=bk, TM=tm, TN=tn))
        label = lambda c: (f"BM{c['BM']} BN{c['BN']} BK{c['BK']} "
                           f"TM{c['TM']} TN{c['TN']}")

    print(f"{len(combos)} valid configurations for {args.target}", flush=True)
    results = []
    try:
        with OUT.open("a") as fh:
            for i, cfg in enumerate(combos, 1):
                patch(SRC, **cfg)
                gflops, status = measure(T)
                rec = dict(cfg)
                rec.update(gflops=gflops, status=status, target=args.target)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                if gflops:
                    results.append(rec)
                shown = f"{gflops:.0f} GFLOP/s" if gflops else status
                print(f"[{i}/{len(combos)}] {label(cfg)} -> {shown}", flush=True)
    finally:
        SRC.write_text(original)

    results.sort(key=lambda r: -r["gflops"])
    print("\ntop 10:")
    for r in results[:10]:
        print(f"  {r['gflops']:8.0f}  {label(r)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
