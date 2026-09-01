# MTP / NextN draft head — llama.cpp implementation notes (qwen35, Qwythos-9B)

Model: `Qwythos-9B-Claude-Mythos-5-1M-MTP-BF16.gguf`, `general.architecture = qwen35`
(`qwen35.block_count = 33`, `qwen35.nextn_predict_layers = 1` → layers 0-31 are the
main trunk, layer 32 (`blk.32`) is the single NextN/MTP draft block).
Reference tree: `~/llama.cpp` (arch dispatch `src/models/qwen35.cpp`;
the MoE sibling `src/models/qwen3next.cpp` is structurally identical but not
what this model uses — this model's FFN and blk.32 FFN are dense, not MoE).

Hparams (from GGUF header, via `tools/gguf-extract.py`'s `parse()`):
`n_embd=4096, n_head=16, n_head_kv=4, head_dim(key/value)=256(?)*, n_ff=12288,
rope.dimension_count=64, rope.dimension_sections=[11,11,10,0] (mrope),
rope.freq_base=1e7, rope.scaling=yarn factor=4.0 orig_ctx=262144,
full_attention_interval=4, ssm.* (gated-delta-net params for the recurrent
trunk layers — NOT used by the MTP block, which is always full attention)`.
*(`attention.key_length`/`value_length` = 256 in the GGUF header, but this is
the historical Qwen3.5 metadata quirk carried over from qwen3next: the real
per-head dim used in the graph is `n_embd_head_v() = n_embd/n_head` after
`hparams.n_embd_head_v()`/`k()` resolution — verify against
`n_embd_head_k * n_head` used in tensor shapes below, which is 4096, i.e.
head_dim=256 IS the real per-head size with n_head=16.)*

## 1. blk.32 (NextN) tensor inventory

Extracted with `tools/gguf-extract.py`'s `parse()` against the real GGUF
(`gguf-extract.py:47` = `parse(path)`, GGUF v3 header/kv/tensor-info reader):

```
blk.32.attn_norm.weight            (4096,)          f32   (ttype 0)
blk.32.attn_q.weight               (4096, 8192)     bf16  (ttype 30)
blk.32.attn_q_norm.weight          (256,)           f32
blk.32.attn_k.weight               (4096, 1024)     bf16
blk.32.attn_k_norm.weight          (256,)           f32
blk.32.attn_v.weight               (4096, 1024)     bf16
blk.32.attn_output.weight          (4096, 4096)     bf16
blk.32.post_attention_norm.weight  (4096,)          f32
blk.32.ffn_gate.weight             (4096, 12288)    bf16
blk.32.ffn_up.weight               (4096, 12288)    bf16
blk.32.ffn_down.weight             (12288, 4096)    bf16
blk.32.nextn.eh_proj.weight        (8192, 4096)     bf16
blk.32.nextn.enorm.weight          (4096,)          f32
blk.32.nextn.hnorm.weight          (4096,)          f32
blk.32.nextn.shared_head_norm.weight (4096,)        f32
```

Notably **absent** from this GGUF (present as optional tensors in the arch
code but `TENSOR_NOT_REQUIRED` and not baked into this particular model):
`blk.32.nextn.embed_tokens.weight` and `blk.32.nextn.shared_head_head.weight`.
Their absence means the MTP block reuses the main model's `token_embd.weight`
for its token-embedding lookup and the main model's `output.weight` (LM head)
+ `output_norm.weight` for producing draft logits — see §2 fallback logic.

Tensor creation (dims as declared, before the loader applies any per-tensor
transpose/layout choice):
- `src/models/qwen35.cpp:114` `nextn.eh_proj` → `{2*n_embd, n_embd}` = `{8192,4096}`
- `src/models/qwen35.cpp:115` `nextn.enorm`   → `{n_embd}` = `{4096}`
- `src/models/qwen35.cpp:116` `nextn.hnorm`   → `{n_embd}` = `{4096}`
- `src/models/qwen35.cpp:117` `nextn.embed_tokens` → `{n_embd, n_vocab}`, `TENSOR_NOT_REQUIRED`
- `src/models/qwen35.cpp:118` `nextn.shared_head_head` → `{n_embd, n_vocab}`, `TENSOR_NOT_REQUIRED`
- `src/models/qwen35.cpp:119` `nextn.shared_head_norm` → `{n_embd}`, `TENSOR_NOT_REQUIRED`
- The rest of blk.32 (`attn_norm`, `attn_post_norm`, `wq/wk/wv/wo`,
  `attn_q_norm/attn_k_norm`, `ffn_gate/up/down`) is created identically to a
  main-trunk full-attention block by the same `load_block_mtp` lambda
  (`src/models/qwen35.cpp:97-120`), which literally duplicates the trunk's
  `create_tensor_qkv`/`ffn_*` calls (compare `load_block_trunk`,
  `src/models/qwen35.cpp:55-95`, lines 67-77 attn/norm, 92-94 ffn).
- `LLM_TENSOR_NEXTN_EH_PROJ` name mapping → `"blk.%d.nextn.eh_proj"`
  (`src/llama-arch.cpp:571`); analogous entries exist for enorm/hnorm/
  embed_tokens/shared_head_*  (`LLM_TENSOR_NEXTN_*` enum, `src/llama-arch.h:681-688`).
- GGUF metadata key for the layer count: `LLM_KV_NEXTN_PREDICT_LAYERS` read at
  `src/models/qwen35.cpp:16` into `hparams.n_layer_nextn` (must be `< n_layer_all`,
  asserted same line 17). For this model `n_layer_nextn=1`.
- Whether blk.32 attn is dense full-attention (not gated-delta-net) is forced
  structurally: `load_block_mtp` never checks `hparams.is_recr(il)` — it always
  calls `create_tensor_qkv`/`wo`/`attn_q_norm`/`attn_k_norm` (the non-recurrent
  branch of `load_block_trunk`), so the MTP block is always a full-attention
  block regardless of what layer index 32 would have been assigned in the
  interleave pattern (`src/models/qwen35.cpp:100-107` comment: "MTP block
  looks like a full-attention Qwen3.5 decoder block").
- `mtp_only` probe: if `blk.0.attn_norm.weight` is absent, this GGUF is treated
  as an MTP-only checkpoint and trunk tensors become `TENSOR_NOT_REQUIRED`
  (`src/models/qwen35.cpp:40-41`) — not the case for Qwythos-9B (full trunk present).
- `mtp_flags = !ml.load_mtp ? TENSOR_SKIP : 0` (`src/models/qwen35.cpp:42`):
  the MTP block's tensors are entirely skippable at load time via a
  loader-level `load_mtp` flag — i.e. llama.cpp can load the same GGUF with or
  without the draft head, controlled outside this file (loader option, not
  shown here — grep `ml.load_mtp` for the setter if wiring the equivalent flag
  in mojo-baro).

## 2. Draft-head forward pass (`graph_mtp`, one call per drafted token)

Entry point: `llama_model_qwen35::build_arch_graph` returns a `graph_mtp`
instance when `params.gtype == LLM_GRAPH_TYPE_DECODER_MTP`
(`src/models/qwen35.cpp:130-135`); `LLM_GRAPH_TYPE_DECODER_MTP` is selected by
`llama-context.cpp:30` (`LLAMA_CONTEXT_TYPE_MTP` → that graph type) — i.e. the
draft head runs as **a second llama_context** (`ctx_dft`) built over the same
GGUF, not inline in the main decode graph.

Full construction: `llama_model_qwen35::graph_mtp::graph_mtp`
(`src/models/qwen35.cpp:488-645`). Step by step:

1. Asserts `n_layer_nextn == 1` (`qwen35.cpp:491-492` — multi-MTP-layer
   "chain_heads" mode, used by other archs, is out of scope for this model).
   `il = hparams.n_layer()` (=32) selects `model.layers[32]`
   (`qwen35.cpp:499-500`).
2. Two graph inputs are declared per forward call (`qwen35.cpp:510-536`,
   type `llm_graph_input_embd_h`):
   - `inp->tokens` (i32, `n_tokens`) — the token id(s) being drafted/verified.
   - `inp->embd` (f32, `[n_embd_inp, n_tokens]`) — fallback raw embedding path
     (used only when `ubatch.token` is false, i.e. an embeddings-only batch).
   - `inp->h` (f32, `[n_embd, n_tokens]`, tensor name **`mtp_h_input`**,
     `qwen35.cpp:530-532`) — **the previous layer's post-output-norm hidden
     state** (`h_nextn`, defined below) for the position(s) being drafted.
     This is the tensor the harness (`common/speculative.cpp`) fills by
     `memcpy`ing rows out of `llama_get_embeddings_nextn[_ith]` from the
     **target** model's last forward pass (see §3).
3. Token embedding lookup (`qwen35.cpp:520-528`): if this is a token batch,
   `tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens :
   model.tok_embd` — since this GGUF has no `blk.32.nextn.embed_tokens.weight`,
   it falls back to `model.tok_embd` (the shared main-model token-embedding
   table). Row-gather via `ggml_get_rows`.
4. **enorm/hnorm + concat** (`qwen35.cpp:543-550`):
   `h_norm = RMSNorm(inp->h, layer.nextn.hnorm)` — normalizes the *incoming
   hidden state* (from the target model, or from the previous MTP head).
   `e_norm = RMSNorm(tok_embd, layer.nextn.enorm)` — normalizes the *token
   embedding* of the token being drafted next.
   `concat = ggml_concat(e_norm, h_norm, dim=0)` — **embedding first, hidden
   state second**, concatenated along the feature axis → shape `[2*n_embd, n_tokens]`.
5. **eh_proj** (`qwen35.cpp:552-553`): `cur = eh_proj @ concat` (via
   `build_lora_mm`, i.e. plain matmul with optional LoRA scale/adapter — none
   present in this GGUF) → `[n_embd, n_tokens]`. This is the "residual" input
   `inpSA` for the block's own attention (assigned right after, `qwen35.cpp:555`).
6. From here it is **structurally identical to one full-attention trunk decoder
   block** (compare `build_layer_attn`, `qwen35.cpp:258-...`, and
   `build_layer_ffn`/`build_ffn` calls at `qwen35.cpp:614-623` vs.
   trunk `qwen35.cpp:194-200`):
   - `attn_norm` RMSNorm (own `layer.attn_norm`, i.e. `blk.32.attn_norm`).
   - Joint Q+gate projection `wq` → view-split into `Qcur`/`gate` (same
     "joint QG" layout as the trunk attention, half is Q, half is a sigmoid
     gate multiplied in after attention) — `qwen35.cpp:560-577`.
   - `attn_q_norm`/`attn_k_norm` RMSNorm on Q/K per head (`qwen35.cpp:568,581`).
   - K, V projections (`wk`, `wv`).
   - RoPE: **`ggml_rope_multi`** (mrope, 4-section) using
     `hparams.rope_sections = [11,11,10,0]` (`qwen35.cpp:506-507,588-593`) —
     NOT plain `ggml_rope_ext`; this is the multi-axis RoPE Qwen3.5 inherits
     from its vision-model heritage even for pure-text decode.
   - `build_attn` full (non-causal-masked-by-default; masking/positions come
     from `inp_attn = build_attn_inp_kv()`, `qwen35.cpp:541`) attention with
     `kq_scale = hparams.f_attention_scale` or `1/sqrt(head_dim)`
     (`qwen35.cpp:595-596`).
   - Output gated by `sigmoid(gate)` (`qwen35.cpp:603`), then `wo` projection.
   - Residual add: `cur + inpSA` (the eh_proj output, NOT the raw h/embd —
     `qwen35.cpp:607`).
   - `attn_post_norm`, then **dense** SwiGLU FFN via `build_ffn` with
     `layer.ffn_up/gate/down` (`qwen35.cpp:614-620`) — LLM_FFN_SILU, PAR
     (parallel gate*up then down) — no MoE routing in this arch (dense-9B
     model; contrast `qwen3next.cpp` which is MoE and calls `build_moe_ffn`
     for its MTP block too).
   - Residual add with `ffn_residual` (the post-attention-residual value,
     `qwen35.cpp:610,622`).
7. **Output head** (`qwen35.cpp:625-644`):
   - `head_norm_w = layer.nextn.shared_head_norm ? … : model.output_norm` —
     this GGUF HAS `blk.32.nextn.shared_head_norm.weight`, so that tensor is
     used (own dedicated final norm, not the main model's `output_norm`).
   - RMSNorm → this result is exported as **`res->t_h_nextn`** (tensor name
     `"h_nextn"`, `qwen35.cpp:629-632`) — i.e. the MTP block ALSO produces its
     own `h_nextn` output, which becomes the `inp->h` input to a *second*
     chained MTP call if `n_layer_nextn > 1` (not applicable here, single head).
   - `cur = ggml_get_rows(cur, inp_out_ids)` (`qwen35.cpp:634`) — select only
     the output positions actually needed (draft-generation only cares about
     the last position typically).
   - `head_w = layer.nextn.shared_head_head ? … : model.output` — this GGUF has
     NO `blk.32.nextn.shared_head_head.weight`, so **falls back to the main
     model's `output.weight`** (shared LM head) for the final logits matmul
     (`qwen35.cpp:637-640`).
   - Result: `res->t_logits` (`qwen35.cpp:643`).

Summary of the fallback matrix actually in effect for Qwythos-9B:
| component            | dedicated MTP tensor present? | tensor actually used |
|-----------------------|:---:|---|
| token embedding        | no  | `model.tok_embd` (shared) |
| final pre-logit norm   | yes | `blk.32.nextn.shared_head_norm.weight` |
| LM head (logits)       | no  | `model.output.weight` (shared) |

## 3. Server / speculative-decode driver (`--spec-type draft-mtp`)

CLI flag registration & mapping to `COMMON_SPECULATIVE_TYPE_DRAFT_MTP`:
`common/speculative.cpp:37` (name table), option parsed generically at
`common/arg.cpp:4257` (`.set_spec()`), `spec_types_is_default`/download-mtp
wiring at `common/arg.cpp:357-396` (auto-selects MTP-capable draft weights
when `--spec-type draft-mtp` is requested and no explicit `--model-draft`).

Implementation: `struct common_speculative_impl_draft_mtp`
(`common/speculative.cpp:1364-1801`). It runs as one of three modes chosen in
the constructor (`common/speculative.cpp:1376-1382,1457-1458`); for a
single-MTP-layer dense qwen35 model neither `is_mem_shared` (gemma4, shared KV)
nor `chain_heads` (multi-head "step35") applies — Qwythos-9B uses the plain
single-trained-head path.

Key mechanics:
- `ctx_dft` is a **separate `llama_context`** built over the *same GGUF*
  (loader loads MTP tensors because `ml.load_mtp` is set for that context);
  `ctx_tgt` is the normal full-model context. `n_embd = llama_model_n_embd_out(...)`
  must match between them (`common/speculative.cpp:1408-1410`).
- `llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false)` and
  `llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true)`
  (`common/speculative.cpp:1454-1455`) — flips on `cparams.embeddings_nextn`
  (target: unmasked — every output row exports `h_nextn`; draft: masked —
  only `inp_out_ids`-selected rows do). This flag gates whether
  `t_h_nextn`/`res->get_h_nextn()` is populated at all
  (`src/llama-context.cpp:1478,1856`; the graph itself always computes
  `t_h_nextn`, `src/llama-graph.cpp:1325,1369` sets it as a `ggml_set_output`
  only if non-null) — i.e. the flag controls whether the *engine* bothers
  reading it back, not whether the graph computes it.
- **process()** (`common/speculative.cpp:1518-1634`, called on every target
  prompt/verify batch): shifts the target's exported `h_nextn` rows right by
  one token position (`memcpy(batch.embd + 1*n_embd, h_tgt, ...)`,
  `speculative.cpp:1566-1567` — comment explains why: MTP predicts token
  t+1 from (embedding of token t+1, hidden state after token t)), feeding the
  draft model's `batch.embd` (i.e. `inp->h` in the graph) with those rows;
  `batch.token` carries the actual token ids in parallel so `graph_mtp` can
  still do its own `enorm(tok_embd[token])`. It then `llama_decode(ctx_dft,
  batch)` once (single head, no `chain_heads` loop) to build the draft
  context's own KV over the verified prefix. After decode it copies
  `llama_get_embeddings_nextn_ith(ctx_tgt, ...)` into `verify_h[seq_id]`
  (rows = target's `h_nextn` for every verified/accepted position,
  `speculative.cpp:1615-1631`) and stashes the last row into `pending_h`
  for cross-call carryover (a single MTP call's last output row pairs with
  the FIRST token of the next process() call).
- **draft()** (`common/speculative.cpp:1636-1785`, the actual multi-token
  drafting loop): seeds `batch` with `dp.id_last` (last accepted/sampled
  token) plus its **paired `pending_h`** row as `batch.embd`
  (`speculative.cpp:1658-1659` — h from *before* this token, matching the
  eh_proj `concat(e_norm(this_token), h_norm(prior_hidden))` semantics), then
  loops up to `n_max` (`--spec-draft-n-max`) times: `llama_decode(ctx_dft,
  batch)` (single head each time here — grows KV incrementally, one new
  token per step since `chain_heads=false`), samples top-1 via a
  dedicated top-k=10 sampler chain (`speculative.cpp:1430-1435`,
  `1443-1449` optional backend-offloaded sampler), accepts only if
  `p >= params.p_min`, and for **each newly drafted token** re-reads
  `llama_get_embeddings_nextn_ith(ctx_dft, i_last[seq_id])` (the draft
  model's own `h_nextn` output for its last decoded position) as the `h_row`
  fed alongside that token for the NEXT draft step
  (`speculative.cpp:1706,1751-1758`) — i.e. the draft head recursively feeds
  its own exported hidden state back into itself for every additional
  speculative token beyond the first, matching the intent of `t_h_nextn`
  being produced both by the trunk graph and by `graph_mtp` itself.
- **accept()** (`common/speculative.cpp:1787-1800`): after the target verifies
  N of the drafted tokens, snaps `pending_h[seq_id]` to
  `verify_h[seq_id][min(n_accepted, n_rows-1)]` — i.e. re-syncs the carried
  hidden state to the TARGET's own `h_nextn` for the actually-accepted
  position (discarding whatever divergent hidden states the draft model
  computed for tokens that got rejected), so the next `draft()` call starts
  from ground truth again.
- Server integration note: `tools/server/server-context.cpp:3514` comment
  references mirroring `t_h_nextn` into `ctx_dft` as part of the streaming
  hook — the actual read/write plumbing (`llama_get_embeddings_nextn*`,
  `llama_set_embeddings_nextn`) lives in the public C API
  (`src/llama-context.cpp:1478,1545-1551,1856,1946-1954`).

## 4. Implications for mojo-baro's engine

mojo-baro (`serve/engine.mojo`) currently does full-model greedy decode,
one token at a time (milestone 4, see `docs/BASELINE.md` for current KV/
offset-table/B-layout-transposed conventions). To add draft-MTP:

- Need to run blk.32 as an *extra* decoder block, structurally = one more
  full-attention transformer layer (own QKVO, own Q/K RMSNorm, own gated
  SwiGLU FFN, own final RMSNorm) **plus** the eh_proj fusion step
  (§2 step 4-5) that mixes in the *previous* forward pass's post-output-norm
  hidden state (`h_nextn`) with the *current* token's embedding.
- `h_nextn` = the trunk's hidden state right after `output_norm` RMSNorm but
  **before** the final `output`/LM-head matmul (`qwen35.cpp:210-213`,
  `res->t_h_nextn = cur` — captured before `ggml_get_rows`/logits mm). mojo-baro's
  engine must export/retain this per-token vector from the main decode path.
- eh_proj concat order matters: `[enorm(token_embd), hnorm(h_nextn)]`, feature
  axis, embedding half first (§2 step 4) — swapping halves would silently
  produce wrong (but plausibly-shaped) results.
- RoPE for the MTP block uses the same mrope sections as the trunk's full-
  attention layers (`[11,11,10,0]`, `qwen35.cpp:506-507`), not a
  simplified single-section rope — reuse mojo-baro's existing trunk full-
  attention RoPE kernel rather than writing a new one.
- Since this checkpoint has no dedicated `nextn.embed_tokens` /
  `nextn.shared_head_head`, mojo-baro's draft head must reuse the SAME
  `token_embd.weight` and `output.weight` tensors already loaded for the
  trunk (no extra vocab-sized weights to load beyond what's already resident,
  besides `eh_proj` (8192x4096), `enorm`/`hnorm`/`shared_head_norm` (4096
  each), and the block's own attn/ffn weights — all already itemized in §1).
- Acceptance/verification loop (top-1 sample, `p_min` gate, re-sync hidden
  state to target's ground truth on every accept) is orthogonal to kernel
  work — it belongs in the serving loop, not the kernel; mirror
  `common/speculative.cpp`'s `process()`/`draft()`/`accept()` split
  (§3) rather than reinventing the protocol.

## 5. Resolved: mrope vs single-section YaRN for the draft head

§2 step 6 and §4 both say blk.32 uses `ggml_rope_multi` with sections
`[11,11,10,0]`, while also saying to reuse the trunk's existing RoPE kernel.
The engine has no mrope — `kernels/attn.mojo`'s `amar_rope_yarn` is
single-section partial-rotary NEOX YaRN over NROT=64 — so this reads as a
contradiction, and a reference implementation written against it has to guess.

It is not a guess. In llama.cpp the **trunk** full-attention layers call
`ggml_rope_multi` at `src/models/qwen35.cpp:303,309`, and the MTP block calls
the same function with the same `hparams.rope_sections` at `:588,591`. Our
engine substitutes single-section `amar_rope_yarn` for those trunk calls and
still reproduces llama.cpp's greedy output **bit-identically over 64 tokens**
(`tools/check-tokens.sh`). The trunk is therefore already the experiment: for
pure-text decode, where all four mrope position axes carry identical position
ids, mrope with these sections and single-section YaRN are numerically
equivalent — established by measurement, not assumed.

Because blk.32 invokes the identical function with the identical sections, the
equivalence transfers. **Use `amar_rope_yarn` for the draft head.** Do not port
mrope for it.

The bound on this claim: it holds only while positions are text-only and shared
across axes. A multimodal path feeding distinct per-axis position ids would
break it, and the 64-token gate would not catch that — it never exercises one.
