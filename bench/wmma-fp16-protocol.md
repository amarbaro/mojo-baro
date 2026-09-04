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

### Round 4 (2026-09-02): 64x64 tile for small grids -- frozen BEFORE building

Diagnosis: at 1024^3 the 128x128 tile gives 64 blocks for 96 CUs (one partial
wave of work, tail-bound); at 512^3 only 16 blocks. `bench/fp16-templates.sh`
(10 s warm-up, commit `b5680b7`): 1024^3 0.98x, 512^3 0.86x, 256^3 0.78x vs
hipBLASLt, ahead everywhere >= 1536^3.

Change: second instantiation of `amar_matmul_wmma_pipe` with WARPS 2x2, WTILE
2x2 (64x64 block, 4 waves, same PGR2/K32 pipeline), selected by a size
dispatch (small tile when the 128x128 grid would be < 96 blocks).

| size | prediction | fail |
|---|---|---|
| 1024^3 | >= 1.10x hipBLASLt | < 1.00x = diagnosis wrong |
| 512^3 | >= 1.00x | |
| 256^3 | still behind (launch floor dominates below ~50 us) | |
| >= 1536^3 | unchanged (dispatch must not touch these) | any regression > 2% |

Second step only if 64x64 alone does not clear 512^3: split-K.

**Round 4 result** (`19a7b19`, then the three-tier dispatch commit;
`.work/fp16-templates-round4.log`, `-round4b.log`):

64x64 alone: 256^3 1.10x, 512^3 1.21x, 768^3 1.17x, **1024^3 1.01x -- missed
the >= 1.10x bar.** 1024^3 wants 8 waves back: six configs at 1024^3 (server
up, relative only): 64x128 (2x4 warps, 2x2 wtile, 256 threads) 74.7k, 128x64
70.3k, 64x64 65.5k, 128x64 4-wave 60.6k, 64x64 4x2/1x2 58.7k, 32x32 28.5k.
64x128 loses to 64x64 at 256/512/768 (fewer blocks). Dispatch on B128 =
blocks the 128x128 tile would launch: >= 96 -> 128x128 8-wave; >= 64 ->
64x128 8-wave; else 64x64 4-wave.

| size | ours | hipBLASLt | ratio | prediction |
|---|---|---|---|---|
| 256^3 | 6372 | 6288 | 1.01 | "still behind" -- wrong, ahead (barely) |
| 512^3 | 30642 | 26324 | 1.16 | >= 1.00 met |
| 768^3 | 64204 | 54924 | 1.17 | -- |
| 1024^3 | 74824 | 63623 | 1.18 | >= 1.10 met after the 64x128 tier |
| 1536..4096^3 | unchanged within noise | | 1.02-1.35 | met |

Split-K not needed; not built.

### Round 5 (2026-09-02): clock under load -- where the remaining headroom is

Trigger: an external review of `kernels/matmul_wmma.mojo` (the retired direct
kernel) read as if it were the shipped one, claiming large headroom from LDS
staging, XOR swizzle, fp16 register packing and a vectorized epilogue. Against
the pipe kernel: staging, swizzle and edge-branch removal are rounds 3-3b;
packing is forced by the receipts above (252 VGPR / 0 spills for 4x4 is
impossible unpacked: 128 acc + 128 bfr + 64 staging before addressing, and
WMMA operands are packed-half VGPR tuples by ISA definition); the epilogue is
64 scalar stores per wave against ~8000 main-loop instructions at 4096^3
(<= 1%), and the D layout (lane owns rows 2i+half of one column) needs a
permute or an LDS-staged transpose before any store can widen. The direct
kernel's A loads were already vectorized by the compiler (`ea4eff2`); its
scalar path was B, and the 16-byte staging fix was step 1 of round 1 (1.5-1.6x).

Not on record until now: the clock the card holds under WMMA load.
`bench/clock-probe.sh <cmd>` samples rocm-smi around any command and prints the
busy-window receipt. Two runs of the 4096^3 128x128 8-wave binary (llama-server
resident but idle; diagnostic, not a number of record; arm read back from the
JSON: pgr 2, lb 0, blk 128x128x32, warps 4x2, wtile 2x4, grid 32x32, 256
threads, iters 200, warm 10 s, correct, max_err 2.2e-4):

| run | gflops | sclk min/med/max, busy | power | junction max |
|---|---|---|---|---|
| 1 | 91754 | 2608/2615/2667 MHz | 281-302 W | 77 C |
| 2 | 89459 | 2559/2606/2709 MHz | 278-320 W | 77 C |

The 2026-09-01 44k kernel held 3005-3008 MHz at 291-307 W (line 38). The
matrix path is power-bound: peak at 2.6 GHz = 96 x 512 x 2.6e9 = 128 TFLOP/s
(512 FLOP/clk/CU = 122.8 TFLOPS spec / 2.5 GHz / 96 CU). Utilisation: 70-72%
in these probes, 76% at the 97957 race median, 82% at 3584^3 (105786). The
remaining kernel-side headroom is that 15-20%, and being power-bound it is
reached by fewer instructions per FLOP (which lifts the clock), not by more
overlap. Open: the 4096^3 number of record spans 90.7k-98.0k across runs
(templates table vs race median); the clock spread seen here is the likely
cause and has not been separated from thermal order.

### Round 6 (2026-09-04): frozen BEFORE running -- spread receipt and WMMA-only roofline

Predictions frozen against plan `luminous-growing-kitten` (2026-09-02) and
commit 9ac5f3a/73adaba (tooling only, kernel untouched). Power cap read back:
`power1_cap` 290 W (LACT). Fail rules stated per row.

| step | arm(s) | prediction | fail rule |
|---|---|---|---|
| 0.2 spread | `PROBE=1 race-fp16.sh 5 vendor=.work/fp16_lt_4096 ours=.work/fp16_clockprobe_4096` | ours median 92-98k GFLOP/s; sample gflops tracks sclk_med (r > 0.8); vendor-first order moves ours <= 2% | r < 0.5 -> clock is not the spread's cause, look at thermal order instead |
| 0.3 roofline R | `.work/wmma_peak_n8` / `_n2` under probe, 3 samples each | n8: 480-512 FLOP/clk/CU, sclk 2.4-2.7 GHz, R ~ 120-130 TFLOP/s; n2 <= 60% of n8 | n8 >= 900 FLOP/clk/CU -> peak is 1024/clk/CU not 512: STOP, re-plan (stop rule a) |

Every fp16 number from here is quoted as a fraction of R with sclk_med and
FLOP/clk/CU beside it; target for the levers of Phase 1-3 is >= 0.90 R at
2048^3/3072^3/4096^3.

**Round 6 result, step 0.2** (2026-09-04, llama-server resident and idle,
`.work/r6/race-0.2-4096.log`, all arms `correct: true`):

| arm | median GFLOP/s (min-max, 5) | sclk_med | FLOP/clk/CU | power |
|---|---|---|---|---|
| hipBLASLt NN | 82375 (81580-82842) | 2870-2913 MHz | 293-299 | 273-330 W |
| ours 128x128 8-wave | 89500 (89228-90354) | 2603-2619 MHz | 355-362 | 277-317 W |

Prediction 92-98k missed low (89.5k) with a 1.3% spread; the templates-table
record (90.7-98.0k) was taken with llama-server killed, so this is the
resident-idle number and the two are not the same arm. The sclk-vs-gflops
correlation test is moot at this spread (sclk 2603-2619, 0.6%). What the
receipt does establish: at the same power cap the vendor kernel holds
2.9 GHz and ours 2.6 GHz, i.e. ours draws more energy per clock and is
throttled 10% deeper; ours does 356 FLOP/clk/CU = 70% of the 512 peak, the
vendor 296 = 58%. The clock deficit is the round-5 diagnosis confirmed under
protocol: the lever is energy per FLOP, not overlap.

**Round 6 result, step 0.3 -- roofline R** (2026-09-04, `.work/r6/roofline-0.3.log`,
`kernels/wmma_peak.mojo` commit ee6ab9e, 16 waves/CU, ITERS 4096, correct: true):

| arm | median GFLOP/s (3) | sclk_med | FLOP/clk/CU | power |
|---|---|---|---|---|
| n8 (8 independent acc) | 125375 (124856-125530) | 2988-3002 MHz | 434-437 | 265-317 W |
| n2 (2 acc, dependent chain) | 126021 (125737-126865) | 2981-2989 MHz | 438-443 | 238-293 W |

**R = 125.4 TFLOP/s.** Both predictions missed: FLOP/clk/CU is 436, not
480-512 (the WMMA pipe issues at ~85% of the 512 nominal even with nothing
else in flight), and the 2-accumulator chain is NOT slower (prediction
<= 60%): back-to-back dependent `v_wmma` costs nothing extra on gfx1100, so
accumulator count is not a lever. Stop rule (a) not hit (436 < 900).
The WMMA-only kernel holds 3.0 GHz at the same 290 W cap where the GEMM
holds 2.6 GHz: the matrix units alone do not pull the card to the cap, the
GEMM's LDS/VALU/address traffic does. Champion at 4096^3 = 89.5k =
**0.71 R** (0.82 R per clock). Target 0.90 R = 113k GFLOP/s.

### Round 6 step 0.4 -- ablation ladder, frozen BEFORE running

Arms `.work/fp16_abl{0,1,2,3}_4096` (commit eaf05f0), 3 probed samples each,
interleaved. With R = 125.4 TFLOP/s, t_R = 1.096 ms. Predictions (ms, from
the plan, re-based on measured R): abl3 1.15-1.25; abl2 1.35-1.45;
abl0 1.40-1.50 (= 89-98k); abl1 0.7-0.9. Reading: t3 - t_R = loop/barrier
overhead; t2 - t3 = LDS read + address cost; t0 - t2 = exposed global
latency; t1 = memory skeleton. Receipt caveat recorded before running: abl2
spills 7 VGPR and abl3 spills 69 (abl0/abl1 spill 0), so t3 is an upper
bound on the register-resident cost, not a clean number. Fail rule: if
t0 - t2 > 0.15 ms the plan's ordering (diet before prefetch) is wrong and
Phase 3 moves ahead of Phase 1.

### Round 7 -- Phase 1 levers, frozen BEFORE building (2026-09-04)

Baseline arm for every race below: `.work/fp16_abl0_4096` (= champion,
89.5k resident-idle, 0.71 R). Each lever is a comptime knob echoed in the
bench JSON; kept only on a disjoint-range win of the stated floor, at most
two builds per lever.

| lever | knob | prediction @4096^3 | keep floor | fail rule |
|---|---|---|---|---|
| 1.1 PRIO | `PRIO=1`: s_setprio 3 around the per-K-step mma block, 0 after | +0-3% | >= +1% disjoint | two builds < +1% -> rejected, knob stays only if zero-cost |
| 1.3b NT B operand | `TB=1`: B given as [N][K], B loader becomes A's b128 path, vendor arm `amarbaro_gemm_f16_nt` (HIPBLAS_OP_T) | +4-8% vs abl0; >= +2% at every size >= 1536^3; NT vendor within 3% of NN vendor | >= +2% disjoint at 4096^3 and 2048^3 | ours NT < abl0 or vendor NT > +5% over vendor NN (then the vendor NT arm, not ours, is the story) |

**Round 6 result, step 0.4** (2026-09-04, `.work/r6/ablation-0.4.log`, 3 probed
samples each, 2*4096^3 = 137.4 GFLOP per launch):

| arm | median GFLOP/s | ms | sclk_med | FLOP/clk/CU | power | receipt |
|---|---|---|---|---|---|---|
| abl0 full | 88923 | 1.545 | 2606-2630 | 354-357 | 279-325 W | 188 VGPR, 0 spills |
| abl1 no mma | 205955 | 0.667 | 2672-2704 | (n/a) | 285-349 W | 112, 0 |
| abl2 no post-prologue global loads | 102625 | 1.339 | 2860-2901 | 369-371 | 254-315 W | 192, 7 spills |
| abl3 + no LDS fragment reads | 81819 | 1.679 | 2827-2869 | 297-303 | 278-323 W | 192, 69 spills: DISCARDED |

Predictions: abl1 0.667 in range (0.7-0.9, just under); abl0 1.545 vs
1.40-1.50 (resident-idle, matches step 0.2); abl2 1.339 in range; abl3
unusable (spill-bound, slower than abl0). The stated fail rule fires on the
ms reading (t0 - t2 = 0.206 ms > 0.15) but the per-clock column says why,
and it is not latency: removing the global loads lifts abl0's 355 to only
370 FLOP/clk/CU (+4%) while the clock rises 2.61 -> 2.88 GHz (+11%). The
global-load traffic is not exposed, it is *paid in watts*. So Phase 3
(prefetch restructure) does not move ahead; the ordering stands, with the
target restated per column:

- clock gap: 2.61 GHz vs the 3.0 GHz the WMMA-only kernel holds = 13%,
  bought only by fewer joules per FLOP (fewer/wider LDS and VALU ops, fewer
  bytes moved: levers 1.3b, 1.5, 1.4);
- issue gap: 355 vs 436 FLOP/clk/CU = 19%, of which ~4% is global-load
  issue and the rest LDS/VALU/barrier issue (levers 1.1, 1.2, 1.3b, 2.1);
- memory skeleton 0.667 ms = 61% of full time, so overlap is far from the
  point where the loads would be the floor.

**Round 7 result, lever 1.1 PRIO** (2026-09-04, commit 0b7ec7b, 5 probed
rounds, `.work/r6/race-1.1-{4096,2048}.log`, all `correct: true`):

| size | abl0 median (min-max) | prio1 median (min-max) | delta |
|---|---|---|---|
| 4096^3 | 88950 (88822-89658) | 85792 (85275-85939) | -3.5%, disjoint |
| 2048^3 | 91673 (88062-91971) | 88712 (88632-89109) | -3.2% |

**Rejected on the first build** (floor was >= +1%; the range is disjoint on
the wrong side, so the second build allowed by the rule is not spent).
Raising wave priority around the mma block starves the sibling waves' loads
on the same SIMD: the K-step is issue-bound on LDS/VALU, not on mma
arbitration. The knob stays at PRIO=0 (ISA-identical to before, receipt
checked by lane A).

**Round 7, levers 1.5 and 1.2 -- frozen BEFORE building** (2026-09-04, from
`.work/isa/champion-4096/DIET-SGB.md`): per K-step the champion issues 42
VALU+SALU (excluding 9 s_waitcnt, 9 s_delay_alu); 16 of them are per-lane
loop-invariant swizzle/index math written inside the loop body and NOT
hoisted by the compiler, 8 are loop-carried pointer updates, 4 fragment
packing, 14 other.

| lever | knob | prediction @4096^3 | keep floor | fail rule |
|---|---|---|---|---|
| 1.5 address diet | `HOIST=1`: lane constants computed once above the K-loop | +1.5-4% (16 fewer ALU per 16 wmma; clock may rise) and ISA receipt shows class (a) = 0 | >= +1% disjoint AND receipt class (a) = 0 | receipt still shows the math in the loop -> compiler re-materialised, rejected |
| 1.2 SGB | `SGB=1`: schedule_group_barrier sequence from DIET-SGB.md §2 | +0-3% (hint only; cannot reduce instruction count) | >= +1% disjoint | two builds < +1% -> rejected |

**Round 7 result, lever 1.3b NT B operand** (2026-09-04, commits c72cc6c,
6f786db, dbd8324; 5 probed rounds; `.work/r6/race-1.3b-{4096,2048}.log`;
all arms `correct: true`; receipts: TB=1 189 VGPR / 0 spills, K-step
ds_store_b16 count 0, TB=0 ISA-identical to champion):

| size | abl0 | tb1 | vendor NN | vendor NT |
|---|---|---|---|---|
| 4096^3 | 90506 (88837-91700) @2.59 GHz, 357/clk | 83093 (81584-84904) @2.63 GHz, 325/clk | 81401 (75757-83478) | 78196 (76832-80790) |
| 2048^3 | 91576 (91470-91702) @2.72 GHz, 350/clk | 65623 (64751-66344) @3.00 GHz, 227/clk | 78630 (72284-79802) | 70077 (65594-73003) |

**Rejected on the first build**: -8% at 4096^3, -28% at 2048^3, both
disjoint. The store side got cheaper as predicted, but the [n][k] LDS tile
makes the B fragment reads the bottleneck: at 2048^3 the card idles up to
3.0 GHz while FLOP/clk/CU falls to 227, the signature of waves stalled on
LDS rather than of extra instructions. The vendor NT arm is also slower
than its NN arm (-4% / -11%), so hipBLASLt has no NT trick here either.
Prediction (+4-8%) was wrong because it counted stores and ignored the
read-side bank pattern of the transposed tile. The knob stays (TB=0 is
byte-identical); a second build would need a different LDS swizzle for
the [n][k] tile, which is a new lever (preregister separately if pursued).

**Round 7, lever 2.1 -- frozen BEFORE building** (2026-09-04). Champion
2x4 wave tile issues 24 ds_load_b128 per 16 wmma per wave per K-step; a
4x4 wave tile (WARPS 2x2, 4 waves on 128x128) needs 16 ds_load_b128 per
32 wmma, halving LDS reads per FLOP, which is both the issue gap and the
joule gap of step 0.4. The 2026-09-02 4x4 attempt was PGR1 and hit 91.3k;
this build is PGR2 with the B fragments streamed per tn (8 VGPR live
instead of 32) and launch bounds on.

| lever | knob | prediction @4096^3 | keep floor | fail rule |
|---|---|---|---|---|
| 2.1 4x4 PGR2 streamed-B | `WARPS_M=2 WARPS_N=2 WTILE_M=4 WTILE_N=4 PGR=2 LB=1 BSTREAM=1` | +5-10% vs abl0 if issue-bound; sclk may rise | >= +3% disjoint at 4096^3, >= 0 at 2048^3 | ISA receipt with any spill -> do not time, stop the lever; two builds < +3% -> rejected |

**Round 7 result, lever 2.1** (2026-09-04, branch `lane-b-4x4` in
`.work/wt-B`, BSTREAM knob; `.work/r6/race-2.1-4096.log`):

Receipts first: the preregistered arm (4x4, PGR2, LB=1, BSTREAM=1) compiles
to 256 VGPR with **76 spills** (scratch_load/store present) -> per the fail
rule it was not timed and the lever stops. The PGR1 control (same tile,
one-deep prefetch, BSTREAM=1) is 251 VGPR / 0 spills and was raced:

| arm | median GFLOP/s (min-max, 5) | sclk_med | FLOP/clk/CU |
|---|---|---|---|
| abl0 champion 2x4, 8 waves | 89283 (88419-90118) | 2604 MHz | 357 |
| 4x4 PGR1 streamed-B, 4 waves | 84060 (82906-84348) | 2748 MHz | 319 |

-5.9%, disjoint. Halving LDS reads per FLOP did raise the clock (+5.5%)
as the energy model predicted, but per-clock throughput fell 11%: with 4
waves per CU and one-deep prefetch there is not enough latency hiding, and
the two-deep variant that would fix it does not fit in 256 VGPR with a
128x128 block. Lever 2.1 closed: the 4x4 tile is register-bound on
gfx1100 at this block size. Not merged (branch kept for the record).
