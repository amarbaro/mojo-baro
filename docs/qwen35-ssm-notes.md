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
