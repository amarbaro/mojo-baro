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

## Round 3 (2026-09-02): pipelined rewrite, `kernels/matmul_wmma_pipe.mojo`

Vendor receipt first (`TENSILE_DB=0x8000 .work/bench_lt16_4096`): hipBLASLt runs
`MT96x96x32 WG32_4_1 MIWT3_3 PGR2 PLR1 LDSB0 TLDS1 LPA16` at 4096^3, i.e. 4 waves,
48x48 wave tiles, DepthU 32, two LDS buffers, global prefetch two deep, transposed
operand in LDS. Ours was 8 waves, K16, 2 barriers per K-step, no pipelining.
Hardware receipts: `sharedMemPerMultiprocessor` = 65536 (64 KB LDS per WGP);
the compiler caps VGPRs at 192 because `max_flat_workgroup_size` defaults to 1024.

Exploration (each row one build+run, 4096^3, ISA receipt via `tools/isa-receipt.py`,
driver `bench/pipe-sweep.sh`; `.work/pipe-sweep.jsonl` has every row):

| step | config | GFLOP/s | receipt | what it taught |
|---|---|---|---|---|
| 0 | old `wmma_lds` 4x2/2x4 K16 | 69-72k | 135 VGPR, 8 KB LDS | same-morning baseline |
| 1 | 2x2 warps 4x4 K32 2-stage | 14.2k | 192 VGPR, 356 spills | 128 acc + staging spills at the 192 cap |
| 2 | 2x2 4x2 K32 2-stage, B row-major | 73.9k | 146 VGPR, 28 KB | pipelining alone +5% |
| 2b | same, B transposed + pad 8 | 62.2k | 155 VGPR | transposed store 16-way bank conflicts |
| 2c | same, PAD_A 0 (no swizzle) | 57.1k | | A fragment reads 4-way conflicted: -23% |
| 3 | 2x2 4x2 K32, B transposed, XOR swizzle | 72.2k | 165 VGPR | conflict-free transposed B = parity, not a win |
| 4 | 4x2 2x4 K32 2-stage (128x128, 8 waves) | 76.8k | 184 VGPR, 36 KB | 36 KB = ONE block per 64 KB WGP |
| 5 | + A XOR swizzle, PAD_A 0 (32 KB) | 81.8k | 167 VGPR | two blocks per WGP |
| 6 | + swizzled transposed B | 84.5k | 183 VGPR | now B transposed pays (+3%) |
| 7 | + ALIGNED (no edge bounds branches) | 89.7k | 165 VGPR | branchy loads cost 6% |
| 8 | + fp16 C | 89.6k | | C dtype is noise at 4096 |
| 9 | + `rocdl.flat_work_group_size` metadata, 2x2 warps 4x4 K32 | **91.3k** | 252 VGPR, 0 spills | cap lifted to 256; 4 waves, 64x64 per wave |

Swizzles (row stride 32 halves = 16 dwords, chunk = 8 halves):
A: chunk' = chunk ^ ((row >> 1) & 3). B stored [n][k] with loader lanes along k,
chunk' = chunk ^ (((n >> 1) ^ (n >> 3)) & 3). Both conflict-free for the b128
fragment reads and for the stores (b128 for A, b16 for B).

Launch bounds: `@__llvm_metadata(`rocdl.flat_work_group_size`=StaticTuple[Int32, 1](NTHREADS))`
(receipt `max_flat_workgroup_size=256` in the code-object notes). Trials of
`MAX_THREADS_PER_BLOCK_METADATA`, `rocdl.max_flat_work_group_size` and integer
values all fail to compile in Mojo 1.0; this form works.

### Confirmation race, `bench/race-fp16.sh 5 ...` (interleaved, medians of 5, min-max)

| size | vendor hipBLASLt fp16 | ours fp16 C | ours fp32 C | verdict |
|---|---|---|---|---|
| 4096^3 | 90121 (89150-91389) | **90954** (90675-91261) | 90098 (89858-90501) | +0.9%, ranges overlap: tie |
| 2048^3 | 81995 (81823-82066) | **84607** (84562-84683) | 84285 (84240-84367) | +3.2%, disjoint: win |
| 512^3, 128x128 tile | 27203 (27126-27332) | 17172 | 17323 | 16 blocks on 96 CUs: loss |
| 512^3, 32x64 tile (2x2 warps, 1x2) | 27203 | **31841** (31762-31966) | | +17%, disjoint: win |

Correctness gate (exact small-integer products) passes on every row. 512 needs a
size-dispatched config; the kernel default is the 128x128 4x4 config.

### Round 3b (2026-09-02): two-deep global prefetch (vendor PGR2)

Frozen before building: 4x4 champion at 252 VGPR will spill with a second
staging set (fail); 8-wave 2x4 config lands only if >= +2%.

| arm | 4096^3 | VGPR / spills | note |
|---|---|---|---|
| 2x2 4x4 PGR1 (champion) | 91.5k | 252 / 0 | |
| 2x2 4x4 PGR2 | 45.8k | 256 / 76 | spills, as predicted |
| 4x2 2x4 PGR1, cap lifted | 83.4k | 208 / 0 | the metadata *hurts* this config |
| 4x2 2x4 PGR2, cap lifted | 87.1k | 244 / 0 | |
| 4x2 2x4 PGR1, default cap | 88.8k | 165 / 0 | |
| **4x2 2x4 PGR2, default cap** | **98.9k** | 188 / 0 | fits under 192 by 4 registers |

The launch-bounds metadata is therefore per-config: it enables 4x4 wave tiles
but makes the 8-wave kernel slower (the allocator spends registers it does not
need). `LB` comptime knob; new default is `LB=0, PGR=2, WARPS 4x2, WTILE 2x4`.

Confirmation race (`bench/race-fp16.sh 5`, interleaved, median of 5, min-max):

| size | vendor | PGR2 8-wave | 4x4 PGR1 | verdict |
|---|---|---|---|---|
| 4096^3 | 89716 (88537-91306) | **97957** (97841-98466) | 91059 | +9.2%, disjoint |
| 2048^3 | 81698 (80596-81769) | **93841** (93728-94051) | 84823 | +14.9%, disjoint |
| 512^3 | 27203 | 21619 (128x128) | | still needs the 32x64 size dispatch |
