# Engine roadmap — what llama.cpp has that we don't

Written 2026-09-01, after `amar_matmul_skinny` beat tuned hipBLASLt 1.49x at the
decode shape (`46693f9`). Conclusion of that work: the GEMM layer is ahead of
everything around it. A faster matmul changes nothing until a forward pass
exists to put it in. This file is the gap inventory and the build order.

Target model: Qwythos-9B (embedding 4096, ffn 12288). fp32 weights are
~36 GB and do not fit in 24 GB — **fp16 is a prerequisite, not an
optimization**. The WMMA kernels already exist; the skinny kernel needs an
fp16 variant (halves B traffic, doubles its roof).

## Gap inventory vs llama.cpp

| Layer | llama.cpp | mojo-baro today |
|---|---|---|
| Weight loading | GGUF mmap + every quant format | nothing |
| Tokenizer | own BPE/SPM impl | nothing |
| Elementwise kernels | RMSNorm, RoPE, SwiGLU, softmax | nothing |
| Attention | flash-decode style, GQA | nothing |
| KV cache | paged, quantized cache | nothing |
| Forward pass | graph per arch family | nothing |
| Sampling | full menu | nothing |
| Server | OpenAI-compatible HTTP | empty `serve/src` |
| GEMM decode M=8 | ~roofline via quants | **1.49x over vendor fp32** |

## Build order — each milestone has a hard verify

Principle: shortest path to a token-level parity check, then speed. Parity
before performance at every step (see BASELINE's retracted-claim lesson).

1. **fp16 skinny + safetensors loader.** Load real Qwythos-9B weights
   (safetensors via mmap; GGUF can wait — HF is the reference anyway).
   Verify: one real weight matrix through skinny-fp16 matches fp32 host ref.
2. **Elementwise kernel pack.** RMSNorm, RoPE, SwiGLU, softmax, embedding
   lookup, argmax. All bandwidth-trivial at M=8; correctness is the work.
   Verify: each vs a NumPy reference on real shapes.
3. **Single transformer layer, contiguous KV cache.** Attention as
   skinny-GEMM + fused softmax first; flash-decode later.
   Verify: layer-0 output parity vs HF transformers on the same inputs.
4. **Full greedy decode.** Tokenizer via HF `tokenizers` through the Python
   layer (writing our own BPE is llama.cpp-envy, not value).
   Verify: N greedy tokens identical to HF reference. **This is the "we
   have an engine" line.**
5. **tokens/sec vs llama.cpp on this box, same weights.** Only scoreboard
   that counts. Expect to lose first round; profile, fix the top kernel.
6. **Quant decode (int8 → int4) dequant-in-kernel**, fused SwiGLU/bias
   epilogues, paged KV, sampling menu, Rust HTTP shell. Order by what the
   step-5 profile says, not by what is fun.

## Non-goals for now

- Multi-GPU, batching/scheduling (vLLM's turf — single-request decode first).
- Own tokenizer, GGUF writer, CPU inference path.
- Beating llama.cpp on formats; we beat it on this GPU or not at all.

## Amendment 2026-09-01 — milestones 1–2 done; milestone 3 rescoped

Done and verified this session:
1. **GGUF loader path + bf16 GEMM** — `tools/gguf-extract.py` (full v3
   parser incl. metadata values), `matmul_skinny_wt` consumes GGUF-native
   [out, in] weights, `test_gguf_gemm` passes on real blk.0.ffn_up.weight
   (max rel err 6e-4 vs numpy fp32).
2. **Elementwise pack** — amar_rmsnorm/amar_swiglu/rope/softmax/embed/argmax in
   `kernels/elementwise.mojo`, all ≤1e-6 rel err vs fp64 host refs.

Rescope: metadata shows qwen35 is a **hybrid SSM/attention** arch —
`ssm.conv_kernel=4, state_size=128, group_count=16, inner_size=4096`,
`full_attention_interval=4` (3 of 4 layers are linear-attention/SSM),
GQA 16 q / 4 kv heads at head_dim 256, YaRN rope (factor 4, sections
[11,11,10,0], freq_base 1e7), MTP block 32. Milestone 3 therefore needs
TWO layer types (SSM/linear-attention + gated full attention), not one —
llama.cpp's qwen3.5 implementation is the working reference. Reference
against llama.cpp layer dumps, since no HF transformers class may match
this checkpoint locally.
