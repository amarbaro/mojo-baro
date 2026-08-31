# Engine roadmap — what llama.cpp has that we don't

Written 2026-09-01, after `matmul_skinny` beat tuned hipBLASLt 1.49x at the
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
