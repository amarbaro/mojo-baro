# q4 weights protocol — frozen before first run

> Same rules as `bench/q8-protocol.md` and `bench/PROTOCOL-RULES.md` P1:
> every arm-defining parameter read back from the running system and
> recorded before a timed run. No receipt, no arm.

Frozen 2026-09-04 on the tree that carries the MTP loop and kernel registry
(`7805393` + README/prompt commits). Nothing below this line was measured
before it was written.

## Why q4 and not the KV cache

Byte model per no-spec token today: 10.7 GB of q8 weights, f32 KV cache
< 20 MB at 128 positions. KV quantization moves nothing at this context
length; weights are the only byte lever left. `bench/q8-protocol.md` line
73 already names int4 as the next one.

## Quant on the record

`Q4_0` (ggml): blocks of 32, 4-bit signed nibbles around a per-block fp16
scale, 18 bytes / 32 weights = 4.5 bits/weight vs Q8_0's 8.5. Chosen over
Q4_K_M because llama.cpp's Q4_0 is a single block format our pack can be
proven byte-equal to (Q0 stage below), which is what made the q8 claim
honest. Q4_K_M is a follow-up round, not this one.

## Stages (each gates the next; stop on first failure)

- **Q0 pack parity.** `tools/engine-pack.py --q4` emits Q4_0 blocks; a
  `tools/q4-check.py` proves them byte-equal to `llama-quantize Q4_0` of the
  same bf16 GGUF. Pass = 0 differing bytes across all 2D weights.
- **Q1 kernel.** `amar_matmul_skinny_q4row[UNROLL, MR]`, wave-per-row like
  q8row, nibble unpack in-register. Cold-cache receipt on the ffn shape
  (`bench/bench_coldcache_q8row.mojo` pattern).
- **Q2 engine.** q4 pack default only if Q3 lands; otherwise `BARO_PACK`
  opt-in. Reference = `tools/model-ref.py` run on the same Q4_0 tensors
  (numpy, fp32 accumulate), plus llama.cpp Q4_0 greedy on the same GGUF.
- **Q3 race.** 5 repeats, discard first, median of 4, both arms, bge-m3 and
  every llama-server stopped. Bar = llama.cpp Q4_0 no-spec and with MTP,
  measured first with `tools/llama-ref-run.sh` on the Q4_0 GGUF.

## Frozen predictions

| stage | prediction | land rule |
|---|---|---|
| Q0 | byte-equal, first try or after one scale-rounding fix | 0 diff bytes |
| Q1 ffn-shape stream | 100 MB-equiv in **34-42 us** (q8row 62.6 us; 0.53x bytes, ALU tax 0-20%) | <= 45 us AND parity vs numpy q4 dequant |
| Q2 identity | 64/64 vs llama.cpp Q4_0 on the 5-token prompt AND >= 18/20 prompts identical on `bench/mtp-prompts/` | recorded; < 16/20 = pack or kernel bug, not "quant noise" |
| Q3 no-spec tok/s_gen | **105-125** (roof at 0.53x bytes ~ 176; 60-71% of roof, q8 reached 74%) | >= 1.4x over 68.8 |
| Q3 MTP tok/s_gen | **170-215** (draft/verify acceptance may drop 5-15 pts on q4) | >= 1.3x over 128.0 |
| vs llama.cpp Q4_0 | no-spec 0.85-1.0x; MTP >= 1.1x | recorded |
| falsifier | Q1 > 50 us or Q3 no-spec < 1.25x: stop, diagnose, re-preregister | |

Draft head stays q8 in this round (0.2 GB/token, not worth a second quant
path); recorded as an asymmetry, not corrected.

## Result

Not yet run.
