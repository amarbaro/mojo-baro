# Milestone 5 — make the self-contained engine fast

Handoff plan, written 2026-09-01 at the close of the M1–M4 session.
Read `docs/BASELINE.md`, `docs/ENGINE-ROADMAP.md`, `docs/qwen35-ssm-notes.md`
(incl. §7b/7c) and `~/Brain/mojo-baro/2026-09-01-decode-shape-sweep.md`
before touching anything. Board: `~/Brain/mojo-baro/whiteboard.md`.

## State you inherit (all verified 2026-09-01)

- `serve/engine.mojo` decodes Qwythos-9B **token-identical to llama.cpp**
  (16/16 greedy, prompt "The capital of France is") at **25.5 tok/s**.
- Regression gate (run after EVERY change, non-negotiable):
  `./.work/engine` output must equal `.work/engine-pack/ref-tokens.txt`
  (`prompt-tokens.txt` beside it). Engine pack = `.work/engine-pack/`
  (built FROM the BARO gguf; `-orig` sibling from the original file).
- Self-describing model: `~/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/
  Qwythos-9B-Claude-Mythos-5-1M-MTP-BF16-BARO.gguf` embeds all kernel +
  engine sources (9 `baro.kernel.*` KVs). **Re-embed after any kernel/engine
  change** (`tools/gguf-embed.py`; delete the old BARO file first) or the
  model ships stale kernels — that bug already happened once.
- Key traps already paid for: gdn v-head h pairs with k-head **h % 16**
  (ggml_repeat tiles); q/k L2-norm not RMS; state decays BEFORE the delta
  error term; yarn ext_factor=1 validated; head_dim 256, contiguous q|gate.

## GPU sharing discipline (llama-server owns 22 GB VRAM)

- Server: PID via `pgrep -af '^/home/mario/llama.cpp'`, local port per
  `~/Brain/OS/local-service-ports.md` ("Local LLM (llama.cpp)" entry).
- The full restart command line is preserved byte-exact in
  `~/Brain/mojo-baro/llama-server-cmdline.txt` — restore from there only.
- Pattern: kill by exact PID (never pkill with a pattern containing
  'llama-server' — it self-matches your own shell; that bit twice), run
  engine, restart with setsid nohup, curl /health until ok. Always restore
  before ending a stint.

## Work items, in order

1. **Measure the bar**: llama-server GPU tok/s (its /metrics + a timed
   /completion, greedy, with and without MTP if feasible). Preregister the
   comparison protocol first (bench/coldcache-protocol.md style, frozen by
   commit) — every perf claim in this repo is preregistered or it is noise.
2. **Launch-count reduction** (biggest lever, target ~2x): the engine issues
   ~20 launches/layer at M=1. Fuse: reduce+cast pairs, gates_k into the
   alpha/beta reduce, rmsnorm+cast, split/norm/rope chains. Consider one
   fused "layer prologue" and "layer epilogue" kernel. Keep each fusion
   behind the token-identity gate.
3. **Prefill batching**: process the prompt with M=n_prompt GEMMs (skinny
   handles M<=8; batch larger prompts in chunks of 8). Attention prefill
   needs a causal-mask variant of attn_decode.
4. **q8b weight path** end-to-end (kernels exist, parity 9.2e-4): halves the
   bandwidth ceiling (~50 -> ~100 tok/s roof). Quantize the pack, add a
   `--q8` engine mode. Token identity may legitimately drift under quant —
   define and preregister an acceptance gate first (e.g. greedy match on
   >=90% tokens + a perplexity spot-check), never hand-wave it.
5. **MTP** (blk.32 nextn tensors, `nextn_predict_layers=1`): the model's
   built-in draft head; ~2x on top. Needs a llama.cpp source read for the
   eh_proj wiring (the sonnet research-agent pattern in the Brain notes
   worked well; file:line receipts mandatory).
6. **Cold-cache GEMM efficiency** (54% vs vendor 85% of HBM peak):
   multi-column/thread + wider loads on the B-layout skinny;
   `bench/bench_coldcache.mojo` is the instrument.

## Repo rules that bind you (CLAUDE.md)

Kernel files carry zero comments/docstrings. Perf numbers only from
preregistered protocols. Commit per concern with why-bodies, no model
attribution lines. Findings -> `~/Brain/mojo-baro/` dated notes; board
updated same turn work completes. Results also -> `~/AMDHQ/runs/`.
