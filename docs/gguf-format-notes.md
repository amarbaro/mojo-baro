# GGUF format notes (v3) — minimal-reader reference

Source: github.com/ggml-org/ggml/blob/master/docs/gguf.md (GGUF v3 spec).
Verified against `/home/mario/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-BF16.gguf`.

## 1. Header layout

All little-endian by default (big-endian variant exists but is rare; no marker to detect it — assume LE unless told otherwise).

```
offset  size  field
0       4     magic        = ASCII "GGUF" = bytes 0x47 0x47 0x55 0x46 (u32 LE = 0x46554747)
4       4     version      uint32, must be 3 for this spec (v3 adds big-endian support)
8       8     tensor_count uint64, number of tensors
16      8     metadata_kv_count uint64, number of metadata KV pairs
24      ...   metadata_kv[metadata_kv_count]   (see §2)
...     ...   tensor_info[tensor_count]        (see §3)
...     ...   padding to align_offset(cur_pos) with general.alignment
...     ...   tensor_data[]  (raw tensor bytes, each tensor's offset within this
                              region must be a multiple of alignment)
```

Fields/arrays are packed sequentially, no per-field alignment except where noted (tensor_data region and each tensor's start offset).

## 2. Metadata KV encoding (enough to SKIP all of it)

Each KV pair:
```
gguf_string_t key    // see below
uint32_t      value_type   // enum, 4 bytes
<value>                    // encoding depends on value_type
```

`gguf_string_t` (used for the key, for STRING values, and array element strings):
```
uint64_t len      // length in BYTES
char     string[len]   // UTF-8, NOT null-terminated
```

`value_type` enum (GGUF_METADATA_VALUE_TYPE_*, all as uint32 tag):
```
0  UINT8    1 byte
1  INT8     1 byte
2  UINT16   2 bytes LE
3  INT16    2 bytes LE
4  UINT32   4 bytes LE
5  INT32    4 bytes LE
6  FLOAT32  4 bytes LE, IEEE754
7  BOOL     1 byte (0 or 1; anything else = invalid)
8  STRING   gguf_string_t (len-prefixed, see above)
9  ARRAY    uint32 element_type, then uint64 array_len, then array_len values
            of element_type back-to-back (length = element COUNT not bytes;
            arrays can nest, e.g. an array of arrays)
10 UINT64   8 bytes LE
11 INT64    8 bytes LE
12 FLOAT64  8 bytes LE, IEEE754
```

To skip a KV: read key (u64 len + bytes), read value_type (u32), then skip the
value per the table above — for ARRAY, recurse using element_type and
array_len. No other structure to worry about; metadata never needs semantic
interpretation to advance the cursor.

## 3. Tensor-info record layout

Immediately follows the metadata_kv array, one record per tensor
(tensor_count records, in file order — this is also the tensor's implicit
index):

```
gguf_string_t name        // <=64 bytes long by spec convention
uint32_t      n_dimensions // currently <=4
uint64_t      dimensions[n_dimensions]   // see ordering note below
uint32_t      type          // ggml_type enum, see §4
uint64_t      offset        // byte offset of tensor data, RELATIVE to the
                             // start of tensor_data[] (not file start);
                             // must satisfy offset % ALIGNMENT == 0
```

**Dims order**: GGUF stores `dimensions[]` in the tensor's native (row-major,
i.e. "ne" / ggml) order, which is the REVERSE of typical PyTorch shape
convention. E.g. a PyTorch `Linear(in=4096, out=12288)` weight of shape
`[12288, 4096]` is stored in GGUF as `dims=[4096, 12288]` (fastest-varying /
innermost dim first). Confirmed empirically below (ffn_gate/up: torch shape
[12288,4096] out×in → GGUF dims [4096,12288] in-first).

**Alignment**: `general.alignment` (uint32 metadata key) sets ALIGNMENT
globally; if absent, ALIGNMENT = 32 (Qwythos file has no such key → 32
applies). `align_offset(x) = x + (ALIGNMENT - x % ALIGNMENT) % ALIGNMENT`.
The start of `tensor_data[]` itself is padded to ALIGNMENT from the end of
the tensor_info array; each individual tensor's `offset` (relative to that
start) must also be a multiple of ALIGNMENT, so gaps between tensors may
contain `0x00` padding bytes.

## 4. Relevant ggml_type enum ids (uint32)

```
0  F32    4 bytes/elem
1  F16    2 bytes/elem (IEEE754 half)
8  Q8_0   block=32 elems: 1x f16 scale + 32x int8            = 34 bytes/block  (1.0625 B/elem)
14 Q6_K   block=256 elems: 128B ql + 64B qh + 16B scales(int8) + 2B f16 d = 210 bytes/block (~0.82 B/elem)
30 BF16   2 bytes/elem (bfloat16: sign1/exp8/mantissa7)
```
(Full enum has 40 entries 0–39; only the above are relevant to this file/task.)

## 5. Qwythos file — verified contents

Read via a Python script that opens the file and reads only header +
metadata + tensor-info bytes (~11 MB region), never the 18 GB tensor_data —
confirmed by `f.tell()` after parsing == 10,968,660 bytes.

- magic = `GGUF`, version = 3, tensor_count = 442, metadata_kv_count = 36
- No `general.alignment` key present → alignment = 32 (default)
- `general.architecture` = `qwen35`; `general.file_type` = 32 (custom/mixed,
  since tensors are mostly BF16 with some F32 norms)

First 10 tensor records as stored in the file (file order, NOT sorted by
name — happens to start mid-way through block 32 / the MTP block):
```
blk.32.nextn.eh_proj.weight        dims=[8192,4096]   BF16  offset=0
blk.32.attn_norm.weight            dims=[4096]        F32   offset=67108864
blk.32.ffn_down.weight             dims=[12288,4096]  BF16  offset=67125248
blk.32.ffn_gate.weight             dims=[4096,12288]  BF16  offset=167788544
blk.32.ffn_up.weight               dims=[4096,12288]  BF16  offset=268451840
blk.32.post_attention_norm.weight  dims=[4096]        F32   offset=369115136
blk.32.attn_k_norm.weight          dims=[256]         F32   offset=369131520
blk.32.attn_k.weight               dims=[4096,1024]   BF16  offset=369132544
blk.32.attn_output.weight          dims=[4096,4096]   BF16  offset=377521152
blk.32.attn_q_norm.weight          dims=[256]         F32   offset=411075584
```
(blk.32 = a separate MTP/speculative-decoding head, block_count metadata=33
i.e. blocks 0-31 are the main stack, block 32 is the MTP extra block —
consistent with the `mtp` sidecar naming convention in the spec and the
`nextn.eh_proj` / `nextn_predict_layers=1` metadata key.)

Requested named tensors (searched across all 442 records):
```
token_embd.weight       dims=[4096,248320]  BF16  offset=2520860672
blk.0.attn_qkv.weight   dims=[4096,8192]    BF16  offset=4555770112   (fused QKV, no separate q/k/v)
blk.0.ffn_down.weight   dims=[12288,4096]   BF16  offset=4689988352
blk.0.ffn_gate.weight   dims=[4096,12288]   BF16  offset=4790651648
blk.0.ffn_up.weight     dims=[4096,12288]   BF16  offset=4891314944
output_norm.weight      dims=[4096]         F32   offset=18396336128
output.weight           dims=[4096,248320]  BF16  offset=486623232
```
Notes:
- This architecture (qwen35) fuses attention Q/K/V into a single
  `blk.N.attn_qkv.weight` tensor (dims [4096,8192] = embd × (q+k+v concat));
  there is no separate `attn_q`/`attn_k`/`attn_v`/`attn_output` split at
  block 0 the way the generic spec's naming convention shows for `blk.32`
  (which DOES have separate `attn_k`/`attn_q_norm` — MTP block differs from
  main-stack blocks in this checkpoint).
- `token_embd.weight` and `output.weight` share the same shape
  `[4096,248320]` (embedding dim × vocab 248320) — tied-shape but distinct
  offsets (not confirmed weight-tied, just same tensor spec).
- offsets above are RELATIVE to start of `tensor_data[]`, not absolute file
  offsets — must add the aligned end-of-tensor-info position to get an
  absolute file byte offset when actually reading tensor bytes.

## 6. Gotchas

- **BF16 conversion**: bfloat16 = the top 16 bits of an IEEE754 float32
  (sign + 8-bit exponent + 7-bit mantissa, truncated/rounded mantissa vs
  F32's 23 bits). To widen to fp32: `f32_bits = uint32(bf16_bits) << 16`,
  then reinterpret as float32. No exponent rebias needed (BF16 keeps F32's
  exponent range/bias exactly). Byte order: within the 2-byte BF16 value
  itself, treat as a plain little-endian uint16 when the file is LE (as
  here); the "<<16" happens after reading it as a 16-bit LE integer, not as
  a byte-swap trick.
- Tensor `offset` in tensor_info is relative to `tensor_data[]` start, NOT
  file start — a naive reader that seeks to `offset` from byte 0 will read
  garbage. Compute absolute = `align_offset(end_of_tensor_info) + offset`.
- `name` strings are length-prefixed (u64) and NOT null-terminated — do not
  scan for a `\0`.
- Metadata array values give element COUNT in the length field, not byte
  count — don't multiply by element size when skipping.
- `general.alignment`, when absent, defaults to 32, not 1 — padding gaps
  between tensor_info end and tensor_data start (and between tensors) are
  real and must be honored or offsets will be misread.
- Dims are stored innermost-first (reverse of the usual PyTorch
  `[out_features, in_features]` convention) — swap before assuming
  row-major-from-the-left semantics.
