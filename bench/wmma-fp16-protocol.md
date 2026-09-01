# fp16 WMMA GEMM: closing the gap to aiter

Bound by `PROTOCOL-RULES.md`. Predictions frozen 2026-09-01 at commit d44dea8
(working tree), BEFORE any kernel change. Baseline: `kernels/matmul_wmma_lds.mojo`
as of that commit, measured by `bench/bench_fp16.mojo` at 512/2048/4096 cubed
(`.work/fp16-sizes.sh base`).

## Diagnosis (from reading the kernel, not from measurement)

1. The 72-config sweep behind `WARPS 4x4 WTILE 2x1` ran at 512^3 only:
   64 blocks of 64x64 on 96 CUs, card never full. Config is a latency optimum
   applied at sizes where aiter leads 3.3x.
2. Global->LDS staging is 2-byte scalar loads with a per-element bounds branch,
   16 per thread per K-step.
3. WTILE 2x1: 3 fragment loads per 2 mma. LDS-bandwidth bound.
4. No double buffering: two barriers per 16 elements of K, global latency exposed.

## Steps and frozen predictions (GFLOP/s ratio vs baseline, same size)

| step | change | 2048^3 | 4096^3 | verdict rule |
|---|---|---|---|---|
| 1 | 16-byte vector global loads for A and B staging, scalar fallback on edge tiles | >= 1.3x | >= 1.3x | < 1.1x = diagnosis 2 wrong, stop and re-diagnose |
| 2 | re-sweep WARPS/WTILE at 4096^3 (not 512^3) | >= 1.3x on top of 1 | >= 1.5x on top of 1 | best config within 5% of current = diagnosis 1 wrong |
| 3 | register-prefetch double buffering, BLK_K 32 | >= 1.2x on top of 2 | >= 1.2x on top of 2 | |

Each step: correctness gate must pass at all three sizes; 512^3 may regress
up to 10% (it is the latency regime). Any arm's receipt = the `blk`, `warps`,
`wtile`, `grid`, `block` fields the bench now prints, plus m/n/k.

## Receipts

Filled in as steps land; each row cites the commit that produced it.

### Clock/power receipt, 2026-09-01, commit 8f7feed

12 back-to-back `bench_fp16` runs at 4096^3, `rocm-smi -P -c -t` sampled at 1 Hz
from a separate process (`.work/smi-probe.log`): package power 5 W idle ->
291-307 W; sclk 15 MHz -> 3005-3008 MHz; junction 52 C -> 75-81 C; 43.0k-44.5k
GFLOP/s on every run. During a sweep the card idles ~95% of wall time (compile
dominates), so cool cores there are duty cycle, not a measurement artefact.

### Step 1 receipt, commit 4092125

| size | blk | warps | wtile | grid | block | base | step 1 |
|---|---|---|---|---|---|---|---|
| 512^3 | 128x64x16 | 4x4 | 2x1 | 8x4 | 512 | 9158 | 14293 |
| 2048^3 | 128x64x16 | 4x4 | 2x1 | 32x16 | 512 | 26800 | 39881 |
| 4096^3 | 128x64x16 | 4x4 | 2x1 | 64x32 | 512 | 26632 | 42733 |

### Step 2 receipt, sweep at 4096^3 (72 configs, `bench/sweep-wmma.jsonl` rows with size=4096)

**Sweep defect caught by the P1 receipt.** The kernel's `WTILE_M` line carried a
trailing comment, so `sweep.py`'s `^comptime WTILE_M = \d+$` never matched and
WTILE_M silently stayed at 2 for every config. The sweep's labels "WTILE 1x4" and
"4x4" are all really 2x4; the 4096 rows in `sweep-wmma.jsonl` from 2026-09-01
must be read with WTILE_M := 2. Standalone re-run of the labelled winner (1x4)
gave 49.9k vs the sweep's 66.2k; with WTILE_M=2 it reproduces (66.2k, 65.3k).
Regex now tolerates trailing comments.

Winner: WARPS 4x2, WTILE 2x4 (128x128 tile, 256 threads). Every top-9 config has
WTILE_N=4 -- B-fragment reuse across four mma per LDS read was the lever.

| size | blk | warps | wtile | grid | block | step 1 | step 2 |
|---|---|---|---|---|---|---|---|
| 512^3 | 128x128x16 | 4x2 | 2x4 | 4x4 | 256 | 14293 | 10948 (-23%, 16 blocks; exceeds the 10% allowance, noted) |
| 2048^3 | 128x128x16 | 4x2 | 2x4 | 16x16 | 256 | 39881 | 56593 (1.42x) |
| 4096^3 | 128x128x16 | 4x2 | 2x4 | 32x32 | 256 | 42733 | 66197 (1.55x) |

### Step 3 result: NOT landed (frozen >= 1.2x, measured <= 1.01x)

Register-prefetch double buffering (fetch next tile into registers before
computing the current one, single LDS buffer), on top of step 2. Draft kept at
`.work/wmma_dbuf.mojo`, not in the tree.

| variant | 2048^3 | 4096^3 |
|---|---|---|
| step 2 (no prefetch, BLK_K 16) | 56593 | 66197 |
| prefetch, BLK_K 32 | 57189 | 59600 |
| prefetch, BLK_K 16 | 52038 | 66924 |

Diagnosis 4 was wrong: at 4 KB LDS and 256 threads enough blocks co-reside per
CU that global latency is already hidden by occupancy; the extra registers cost
more than the overlap earns. Re-confirms the file's earlier BLK_K finding.

Remaining gap to aiter at 4096^3: 66.2k vs 88.9k (0.74x, was 0.30x). Next
lever if pursued: B fragment LDS reads are 16 strided 2-byte loads; a
swizzled/padded sb layout or transposed staging is the candidate, but the file
records the transposed attempt costing 3% at the old config -- re-measure at
this config before trusting that.

## Round 2 (frozen 2026-09-01 at commit d671031): 66.2k -> aiter's 88.9k at 4096^3

| step | change | 4096^3 | verdict rule |
|---|---|---|---|
| P | rocprofv3 counters on the step-2 binary (LDS bank conflicts, VGPRs, occupancy) | receipt only | none |
| 4 | B supplied pre-transposed (NT layout), fragment = two 16-byte LDS reads | >= 1.15x | < 1.05x = LDS reads are not the bottleneck |
| 5 | workgroup swizzle: block_idx remapped so 8 M-rows of blocks co-reside | >= 1.05x on top of 4 | |

### Round 2 receipts (2026-09-01, kernel at commit 006dadf, all 4096^3, interleaved with step 2 each run)

**The vendor bar was wrong.** BASELINE's "aiter 88897" has no source in the
vault or the AMDHQ ledger; aiter measured on this machine (CDNA config copy) is
21124. The vendor for fp16 on this card is **hipBLASLt fp16 via the shim**
(`bench/bench_fp16_lt.mojo`), now measured with its algo receipt:

| size | hipBLASLt fp16 | receipt | ours (step 2) | gap |
|---|---|---|---|---|
| 2048^3 | 80198 | 32 algos, chosen 0, splitK 1, wgm 1, C fp16 | 56593 | 1.42x |
| 4096^3 | 83673 | 32 algos, chosen 0, splitK 0, wgm 2..16 (search noise), C fp16 | 66197 | 1.26x |

| variant | 4096^3 | vs step 2 | rule |
|---|---|---|---|
| P: rocprofv3 --pmc | **no receipt**: hangs indefinitely under the MAX runtime, even at 3 iterations; killed twice | | open |
| 4a: NT layout, scalar fragment reads | 53.3-54.5k | 0.81x | fail |
| 4b: NT + 16-byte fragment reads, no pad | 53.6k | 0.81x | fail |
| 4c: NT + 16-byte reads, LDS row pad 8 halves | 55.2k | 0.84x | fail |
| 5: workgroup swizzle GROUP_M 8 | 64.2-64.4k | 0.97x | fail |
| plain BLK_K 32 | 56.8k | 0.86x | |
| plain BLK_K 64 | 45.7k | 0.69x | |

ISA (code object extracted from the binary, `llvm-objdump --mcpu=gfx1100`):
step 2 = 135 VGPR, 0 spills, 8 KB LDS; per K-step 8 v_wmma, 4 ds_load_b128 (A),
64 ds_load_u16 (B), 2 global_load_b128, 2 s_barrier; epilogue 64 scalar stores.
4c has the textbook mix (8 wmma, 12 ds_load_b128, 0 u16, 130 VGPR) and is still
16% slower, so LDS instruction count is not the limiter. What 4c changed is the
global pattern: B rows became 32-byte chunks at 8 KB stride, the pattern A
already has. Deeper BLK_K fixes that pattern and loses more, tracking LDS
occupancy (8/4/2 blocks per CU). Conclusion: the kernel hides latency with
occupancy only; every trade of occupancy for reuse loses. Closing the last
1.26x needs single-wave software pipelining (two LDS buffers, loads for tile
k+1 issued before the mma of tile k, fp16 C) -- an L rewrite, not a tweak.
Drafts: `.work/wmma_nt.mojo`, `.work/wmma_swz.mojo`, `.work/wmma_bk{2,4}.mojo`.
