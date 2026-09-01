# qwen35 SSM (gated-delta-net) block — exact per-token decode math

Source: `/home/mario/llama.cpp/src/models/qwen35.cpp` (`build_layer_attn_linear`,
`build_qkvz`, `build_norm_gated`) + `/home/mario/llama.cpp/src/models/delta-net-base.cpp`
(`build_conv_state`, `build_recurrent_attn`, `build_delta_net_autoregressive`) +
`ggml_compute_forward_ssm_conv_f32` in
`/home/mario/llama.cpp/ggml/src/ggml-cpu/ops.cpp:9564`. Re-verified against
source on 2026-09-01 (llama.cpp local checkout, no upstream fetch).

Dims (hidden=4096, from stated metadata): S=state_size=128, n_k_heads=group_count=16,
n_v_heads=time_step_rank=32, head_v_dim = d_inner/n_v_heads = 4096/32 = 128 = S.
key_dim = S*16 = 2048, value_dim = S*32 = 4096 = d_inner. conv_dim = key_dim*2+value_dim
= 2048+2048+4096 = 8192. conv_kernel=4. NOTE: given tensor shapes qkv[4096,8192] and
gate[4096,4096] match wqkv (out=conv_dim=8192, in agrees) and wqkv_gate (out=value_dim=4096).

## 1. Projections (per token, hidden state `cur` = RMSNorm(attn_norm) output, [4096])

```
qkv_mixed = cur @ ssm_qkv.weight      # [8192]  (attn_qkv.weight, no bias)
z         = cur @ ssm_gate.weight     # [4096]  (attn_gate.weight, no bias)  -- output gate
beta_raw  = cur @ ssm_beta.weight     # [32]    (one scalar per v-head)
alpha_raw = cur @ ssm_alpha.weight    # [32]
```
No activation applied to qkv_mixed/z/beta_raw/alpha_raw at this stage (plain linear, no bias).

## 2. beta, alpha -> decay gate `g` and write-strength `beta`

```
beta  = sigmoid(beta_raw)                        # [32], per v-head, in (0,1)  -- delta-rule write strength
a_sp  = softplus(alpha_raw + ssm_dt.bias)         # [32], softplus(x)=log(1+exp(x))
g     = a_sp * ssm_a                              # [32], ssm_a is the (already -exp(A_log)) per-head decay factor
                                                    #  (code comment: "-A_log.exp() * softplus"; ssm_a stored value IS -exp(A_log))
```
`g` is the log-decay applied this step (see step 4): `exp(g)` multiplies the recurrent state.

## 3. Conv1d (causal depthwise, per-channel, kernel=4) over concat(q,k,v)

`qkv_mixed` [8192] is NOT yet split; it's transposed into `[conv_channels=8192, 1]`,
concatenated with the last `conv_kernel-1=3` cached columns (per-channel sliding window state,
shape `[3, 8192]` from the recurrent cache) along the time axis, giving `conv_input`
`[3+n_t, 8192]`. New window state = last 3 columns of `conv_input`, cached back
(`ggml_cpy` into `conv_states_all`) for the next token.

`ggml_ssm_conv` (ggml-cpu ops.cpp:9564): for each channel `r` (0..8191) and output
position `t`: `y[r,t] = sum_{j=0..3} conv_input[j+t, r] * conv1d.weight[j, r]`
— i.e. a per-channel (depthwise) causal dot product of the 4-wide window against that
channel's 4 kernel taps (`ssm_conv1d.weight` is `[4, 8192]`, one 4-tap filter per channel, no bias).

```
conv_out = ggml_ssm_conv(conv_input, ssm_conv1d.weight)   # [8192]
conv_out = silu(conv_out)                                  # silu(x) = x * sigmoid(x)
```

## 4. Split conv_out into q, k, v and L2-normalize q, k

Layout of the 8192 conv channels is `[q(2048) | k(2048) | v(4096)]`, i.e. same order as
the qkv_mixed projection, reshaped as:
```
q = conv_out[0:2048]           -> [head_k_dim=128, n_k_heads=16]
k = conv_out[2048:4096]        -> [128, 16]
v = conv_out[4096:8192]        -> [head_v_dim=128, n_v_heads=32]
q = l2_normalize(q, eps=f_norm_rms_eps)      # per (head_k_dim) vector, L2 not RMS
k = l2_normalize(k, eps=f_norm_rms_eps)
```
Since n_v_heads(32) != n_k_heads(16), q and k are head-repeated 2x (`ggml_repeat_4d`) to
broadcast the 16 k/q-groups across the 32 v-heads (GQA-style, group size 2) — unless the
fused GDN kernel path is active, which broadcasts internally instead.

## 5. Delta-rule recurrent state update (single-token decode path)

This is `build_delta_net_autoregressive` (`n_tokens==1` branch of `build_delta_net`,
taken whenever not using the fused `ggml_gated_delta_net` op). Per v-head (32 heads,
state `S_head` is `[128,128]`, i.e. key_dim x value_dim per head):

```
q = q * (1/sqrt(S))                     # S=128, scale = 1/sqrt(128)

# decay the previous state
g_exp = exp(g)                          # [1,1] per head, scalar decay this step
S_head = S_head * g_exp                 # elementwise scale of entire [128,128] matrix

# delta-rule error / correction term
sk   = sum_row(S_head * k)              # (S_head @ k), sum over key dim -> [128] (predicted v from old state)
d    = (v - sk) * beta                  # [128]  -- beta is the scalar write strength for this head
# outer-product state update (delta rule):  S_new = S_old*g_exp + k ⊗ d
S_head = S_head + outer(k, d)           # k broadcast-repeated to [128,128], elementwise mul by d^T, add

# read out this token's output
o = sum_row(S_head * q)                 # [128], per head
```

Equivalently, in standard gated-delta-net notation per head h, token t:
```
S_t = S_{t-1} * exp(g_t)  +  k_t ⊗ ( beta_t * (v_t − S_{t-1}·k_t · exp(g_t)... ) )
```
but note the exact code order: decay is applied to S BEFORE computing `sk` (so `sk` already
uses the decayed state), matching: `S' = S*exp(g); d=(v - S'·k)*beta; S_new = S' + k⊗d; o = S_new·q`.

State is persisted per-sequence into the recurrent-memory cache (`ssm_states_all`) via
`ggml_cpy` after each token, shape `[S_v=128, S_v=128, H_v=32, n_seqs]`.

## 6. Output gated norm + projection

```
o        # [128, 32] per-head outputs, reshape to [head_v_dim=128, n_v_heads=32]
z_2d = reshape(z, [128, 32])
out  = RMSNorm(o, ssm_norm.weight, eps=f_norm_rms_eps) * silu(z_2d)   # build_norm_gated:
                                                                        #   normalized = rms_norm(o) * ssm_norm.weight
                                                                        #   gated = silu(z)
                                                                        #   out = normalized * gated
out  = reshape(out, [4096])
cur  = out @ ssm_out.weight            # [4096] -- attn_out projection back to hidden
```
`cur` is then added as the residual to the block input (outside this function).

## 7. Full-attention block (every 4th layer, `full_attention_interval=4`)

Not gated-delta-net — standard GQA attention with RoPE (multi-section/M-RoPE) and a
sigmoid output gate, in `build_layer_attn`:
```
Qcur_full = cur @ wq            # [ (128*2) * n_head ]  -- joint query+gate projection
Q    = Qcur_full[:, 0::2 interleave by head]  # first half of each 256-wide head slice -> [128, n_head]
gate = Qcur_full[:, second half of each head slice] -> [128, n_head] -> flattened [4096]
Q = RMSNorm(Q, attn_q_norm)
K = cur @ wk ; K = reshape(head_dim=128, n_head_kv) ; K = RMSNorm(K, attn_k_norm)
V = cur @ wv ; V = reshape(head_dim=128, n_head_kv)
Q, K = rope_multi(Q/K, sections, ...)     # M-RoPE, 4 sections
attn_out = softmax(Q·K^T * kq_scale) · V   # kq_scale = 1/sqrt(head_dim) unless f_attention_scale set
attn_out = attn_out * sigmoid(gate)        # elementwise output gate (sigmoid, not silu)
cur = attn_out @ wo                        # [4096]
```
So attn.qkv here is really a fused `wq`(head_dim*2*n_head) / `wk` / `wv` split (per-head
interleaved query,gate pairs), distinct from the linear-attn block's flat qkv concat
described above — the two [4096,8192]-shaped tensors in the prompt (attn_qkv vs the
SSM-block's own qkv/gate) are separate weight matrices per layer type; a full-attention
layer does NOT carry `ssm_*` tensors, and an SSM layer does NOT carry `wq/wk/wv`.

## 7b. Full-attention block — verified details

Source: `/home/mario/llama.cpp/src/models/qwen35.cpp` `build_layer_attn` (lines 258-337),
`ggml_rope_multi` impl in `/home/mario/llama.cpp/ggml/src/ggml-cpu/ops.cpp`
(`ggml_mrope_cache_init` 5858-5926, `rope_yarn` 5825-5840, dispatch ~5995-6100),
YaRN corr-dims in `/home/mario/llama.cpp/ggml/src/ggml.c` (`ggml_rope_yarn_corr_dim`
4406-4408, `ggml_rope_yarn_corr_dims` 4410-4418), hparams loading in
`/home/mario/llama.cpp/src/llama-model.cpp` (n_rot/key_length ~1326-1343, rope type
switch ~2951-2955, rope_freq_scale_train ~1301-1319).

**§7 was wrong on head_dim: it's 256, not 128.** `hparams.n_embd_head_k_full` /
`n_embd_head_v` default to `n_embd/n_head` but are overridden by
`attention.key_length`/`value_length` in the GGUF (qwen35.cpp:264-265 asserts
`n_embd_head_v() == n_embd_head_k()`). GGUF says `attention.key_length=256`,
`head_count=16` (q), `head_count_kv=4`, so `n_embd_head = 256`. wq is `4096->8192`
because it's a joint query+gate projection: `8192 = 256(head_dim) * 2(q+gate) * 16(n_head)`.
wk/wv are `4096->1024 = 256 * 4(n_head_kv)`, matching q/k norm weight shape `[256]`.

**1. Q/gate split (qwen35.cpp:270-297).** `Qcur_full = wq @ cur` → shape
`[8192, n_tokens]`, logically `[(head_dim*2)=512, n_head=16, n_tokens]` with per-head
stride 512 elems. Per-head slice is NOT interleaved element-pairwise — it's a
contiguous 512-wide block split into two contiguous halves:
- `Qcur` (line 273-275): `ggml_view_3d(Qcur_full, n_embd_head=256, n_head, n_tokens, nb1=elemsize*256*2, nb2=elemsize*256*2*16, offset=0)` — first 256 of each 512-wide head slice, contiguous.
- `gate` (line 293-297): same view shape/strides but `offset = elemsize*256` — second 256 of each head slice, contiguous. Then `ggml_cont_2d` flattens/materializes it to `[4096, n_tokens]` (line 297).

So per head: `[q(256 contiguous) | gate(256 contiguous)]`, not interleaved — §7's
"0::2 interleave" description was wrong; it's a plain contiguous half/half split via
strided view, same split shape for every head.

K/V have no gate: `Kcur = wk @ cur` reshaped directly to `[256, n_head_kv=4, n_tokens]`
(line 289), same for V (line 300).

**2. RMSNorm placement (qwen35.cpp:278-280 for Q, 288-291 for K).** Applied per-head
over the full 256-wide head_dim, BEFORE RoPE: `Qcur = build_norm(Qcur, attn_q_norm,
nullptr, LLM_NORM_RMS, il)` right after the view (line 279), and
`Kcur = build_norm(Kcur, attn_k_norm, ...)` right after K's reshape (line 290) — both
precede the `ggml_rope_multi` calls at lines 303-313. `build_norm`
(`llama-graph.cpp:1580-1613`) dispatches `LLM_NORM_RMS` to
`ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps)` — eps = whatever
`attention.layer_norm_rms_epsilon` sets in the GGUF (loaded qwen35.cpp:5,
`load_arch_hparams`), no separate q/k-norm eps key. Weight `attn_q_norm`/`attn_k_norm`
is `[256]` = one scale per head_dim element, broadcast identically across all heads
(shared weight, not per-head).

**3. rope_type = IMROPE, not NEOX.** `llama-model.cpp` rope-type switch (~2951-2955):
`LLM_ARCH_QWEN35` (and QWEN35MOE/QWEN3VL/QWEN4EXP/QWEN3TTS) → `LLAMA_ROPE_TYPE_IMROPE`
(`GGML_ROPE_TYPE_IMROPE = 40`, `ggml.h:254`), i.e. interleaved M-RoPE — comment at
`ggml.h:1871`: `n_dims=16 --> [ttyxttyxttyxttyx00]` (per-dim-group interleave of
t/h/w/e sections, cos/sin still applied in NEOX pairing/ordering — NOT plain NEOX,
NOT classic non-interleaved MRoPE). Dispatch confirms `is_imrope = (mode ==
GGML_ROPE_TYPE_IMROPE)` (`ops.cpp:6015`), `mrope_used = mode & GGML_ROPE_TYPE_MROPE`
also true for IMROPE since 40 & 8 != 0 (`ops.cpp:6016`), so it goes through
`ggml_mrope_cache_init` (`ops.cpp:5858-5926`), not the plain `ggml_rope_cache_init`.

**n_dims / rotated width.** `n_rot` passed into `ggml_rope_multi` (qwen35.cpp:305) is
`hparams.n_rot(il)`, loaded from `LLM_KV_ROPE_DIMENSION_COUNT`
(`rope.dimension_count`) if present, else defaulting to `n_embd_head_k_full`
(`llama-model.cpp:1335-1338`). If the GGUF sets `rope.dimension_count=64`, only the
first 64 of the 256 head_dim elements are rotated by RoPE; the remaining 192 dims
pass through `ggml_rope_multi` unrotated (standard llama.cpp partial-rotary behavior
— rope ops only touch `[0, n_dims)`, dims `[n_dims, head_dim)` are copied through
unchanged). This is architecturally identical to Qwen2-VL/Qwen3-VL's partial-rotary
M-RoPE, just with head_dim widened to 256 and Q carrying a fused gate.

**sections [11,11,10,0] / position source.** `sections` = `hparams.rope_sections`,
loaded via `LLM_KV_ROPE_DIMENSION_SECTIONS` with 4 required entries
(qwen35.cpp:6, `get_key_or_arr(..., 4, true)`); copied into a local `int sections[4]`
before the rope call (qwen35.cpp:507/144). `11+11+10+0 = 32` half-dim sections
(each section counts in units of rotated-dim/2 = pairs; `2*32=64` matches
`n_rot=64` exactly — confirms n_dims=64 is the rotated width, not 256).
`ggml_mrope_cache_init` (`ops.cpp:5858-5926`) walks `i0` in `[0, n_dims)` step 2,
computes `sector` from the running pair index mod `sect_dims`, and for IMROPE
(`is_imrope=true`, `indep_sects=false` branch) picks `theta_t/theta_h/theta_w/theta_e`
in a `sector % 3` interleave pattern (t/h/w cycling every 3, `e` used once sector
exceeds `3*sections[i]` bound) — i.e. within each 3-slot group one slot is time,
one height, one width, cycling, with `sections[3]=0` meaning the `e` (extra/vision)
position id is essentially unused for text-only decode (no vision tokens ⇒
`theta_e` branch never reached, or degenerates to the same value as time since
`inp_pos` supplies only one position stream per token here — qwen35 is text-only,
so t/h/w collapse to the same scalar position per token; the section split still
runs, but with identical `theta_t=theta_h=theta_w` since `build_layer_attn`'s
`inp_pos` is a single 1-D position array, not the 4-stream position tensor
Qwen-VL's mrope preprocessing builds). Confirm from call site: qwen35.cpp:303-313
passes a single `inp_pos` (not a 4-row multi-pos tensor) into `ggml_rope_multi`,
consistent with §7's "text-only position" hypothesis — all sections rotate against
the same token position, the section split is structurally present but numerically
inert (t=h=w=e all equal) for pure-text sequences.

**4. YaRN.** `freq_base=10000000` (`rope.freq_base`), `ext_factor`/`attn_factor` come
from `hparams.rope_ext_factor`/`rope_attn_factor` (loaded via
`LLM_KV_ROPE_SCALING_ATTN_FACTOR` etc., `llama-model.cpp` ~1300), `freq_scale =
hparams.rope_freq_scale_train = 1/ropescale` where `ropescale` is
`LLM_KV_ROPE_SCALING_FACTOR` (`llama-model.cpp:1315-1319`) — with `scaling.factor=4.0`
this gives `freq_scale = 0.25`. `n_ctx_orig` = `hparams.n_ctx_orig_yarn`, loaded from
`LLM_KV_ROPE_SCALING_ORIG_CTX_LEN` = `original_context_length=262144`
(`llama-model.cpp:1301`). `beta_fast`/`beta_slow` are the standard llama.cpp YaRN
defaults (32.0 / 1.0) unless the GGUF overrides them (no qwen35-specific override
found in `load_arch_hparams`).

Corrected-frequency math (`ggml.c:4404-4418`, `ops.cpp:5825-5840`), pseudocode:
```
corr_dim(n_rot, base) = n_dims * ln(n_ctx_orig / (n_rot * 2π)) / (2 * ln(base))
low  = floor(corr_dim(beta_fast, freq_base))     # more rotations/faster-changing dims
high = ceil (corr_dim(beta_slow, freq_base))      # fewer rotations/slower dims
low, high = clamp(low, 0, n_dims-1), clamp(high, 0, n_dims-1)

# per rotated dim pair i0 (0..n_dims step 2), per theta stream (t/h/w/e per IMROPE section):
theta_extrap = theta_base * (freq_base^(-i0/n_dims))    # via running theta *= theta_scale each step
theta_interp = freq_scale * theta_extrap                 # freq_scale = 1/yarn_factor = 0.25
ramp = clamp((i0/2 - low) / max(high-low, 0.001), 0, 1) * ext_factor   # rope_yarn_ramp
theta = theta_interp * (1 - ramp) + theta_extrap * ramp
mscale = attn_factor * (1 + 0.1 * ln(1/freq_scale))       # magnitude correction, only if ext_factor != 0
cos, sin = cos(theta) * mscale, sin(theta) * mscale
```
i.e. low-frequency (slowly rotating) dims get full NTK interpolation (`theta_interp`,
scaled down by 0.25 for the 4x context extension), high-frequency dims stay
extrapolated (`theta_extrap`, unscaled), with a smooth ramp between `low`/`high`
correction-dim bounds — standard YaRN, applied independently within each of the
`n_dims=64` rotated positions, replicated across the 4 sections (each section's
`theta_t/h/w/e` stream uses the same `corr_dims`/`freq_scale`/`ext_factor`, only the
starting `theta_base_*` differs per position stream — moot here since t=h=w=e for
text-only decode per point 3 above).

## Summary of activation functions used (SSM block)
- beta: sigmoid
- alpha path: softplus(alpha + dt.bias), then multiplied by stored -exp(A_log) (`ssm_a`) to form log-decay `g`
- conv1d output: silu
- q,k post-conv: L2 normalize (not RMS), eps = attention rms eps
- gated output norm: RMSNorm(o) * silu(z)  (build_norm_gated)
- decay applied to state: exp(g) (elementwise scalar per head)

## Metadata used
ssm.conv_kernel=4, ssm.state_size=128, ssm.group_count=16, ssm.time_step_rank=32,
ssm.inner_size=4096, full_attention_interval=4 (every 4th layer, 1-indexed, is full attn:
`(i+1) % 4 != 0` marks recurrent layers, i.e. layers 0,1,2 recurrent, layer 3 full-attn, repeat).
