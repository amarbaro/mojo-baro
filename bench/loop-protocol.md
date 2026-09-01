# Self-optimising loop protocol

Bound by `PROTOCOL-RULES.md`. The served model proposes rewrites of the engine
sources embedded in its own gguf; a mechanical gate decides; winners are
re-embedded into a NEW gguf with lineage. Claude reads survivors only.

## Tools

- `tools/gguf-closure.sh MODEL` -- build the engine from the gguf's own sources, gate 64/64. The floor.
- `BARO_PROFILE=1 .work/engine` -- per-sub-block GPU shares; picks the target region.
- `tools/loop-propose.py MODEL ITER --n N --start I` -- N identity framings, one diff each, into `.work/loop/ITER/`.
- `tools/loop-gate.sh ITER CHAMPION_TOKPS` -- scope -> compile -> identity@64 -> perf -> ISA. Receipt per candidate.
- `tools/loop-embed-winner.sh SRC.gguf ITER` -- embeds the COMMITTED repo sources into `<src>-loop-ITER.gguf`, adds `baro.kernel.parent`.

## Frozen acceptance rule (2026-09-01)

A candidate lands only if: touches only `baro.kernel.files`; compiles; 64/64
greedy tokens identical; median of 3 `tok/s_gen` read back from the engine's
own output >= champion + 2% with spread < 5%; no scratch and no spills in any
embedded code object. The candidate's own `PREDICT` line is its
preregistration and is recorded in the receipt. Server (port 8083) must be
stopped for stages 2-4; it is the proposer, not the instrument.

## GPU choreography

propose (server up) -> stop server by exact PID -> gate (engine owns GPU) ->
Claude commits winner -> embed -> restart server from
`~/Brain/mojo-baro/llama-server-cmdline.txt`, `curl /health`.

## Worth-it rule

After 5 iterations: < 1 accepted winner or < 2% aggregate tok/s gain =>
widen the region to whole-layer rewrites instead of more iterations.

## Receipts

| iter | region | identities | candidates | survivors | champion before -> after | gguf |
|---|---|---|---|---|---|---|
| 001 (2026-09-01, commit 7833260) | ffn (52%) | 18-skeptic, 19-builder, 20-stranger | 3 | 0 | 41.3 -> 41.3 | none |

Iteration 001 notes: prompt 25k chars (bindings + ffn region + elementwise +
matmul_skinny). All three failed before any timed run: no diff fence; patch
does not apply (invented `g_ffn`, `Wfg`, `g_ffn_fused`); compiles against an
invented `w_h_ffn2` and silently dropped the up and down projections. The
ladder held. Proposer quality, not the gate, is the bottleneck: next iteration
gives the model a symbol table of real binding names and a smaller region.
