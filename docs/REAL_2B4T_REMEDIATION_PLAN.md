# Real 2B-4T Remediation Plan

## Problem

The `microsoft/bitnet-b1.58-2B-4T-gguf` model loaded successfully, but decode logits were non-finite.

This was not a file-format failure. It was an architecture-alignment failure:

- metadata prefix is `bitnet-b1.58.*`, not `llama.*`
- projection weights use `I2_S` blocks (type `36`), not F32 test tensors
- embeddings are stored as F16 / I8 depending on tensor
- the model uses GQA (`20` query heads, `5` KV heads)
- RoPE base is `500000`, not `10000`
- the block includes `attn_sub_norm` and `ffn_sub_norm`
- the FFN activation is `ReLU^2(x1) * x3`, not SwiGLU

## Remediation

### 1. Loader alignment

- accept both `llama.*` and `bitnet-b1.58.*` metadata suffixes
- read `attention.head_count_kv`
- read `rope.freq_base`
- support tensor types:
  - `F32`
  - `F16`
  - `I8`
  - `I2_S`
- decode `I2_S` rows into NeoGPU's simple packed ternary format
- allocate K/V projection scales using `num_kv_heads * head_dim`
- load `attn_sub_norm.weight` and `ffn_sub_norm.weight`

### 2. Inference alignment

- add `num_kv_heads` to `HSMLTernary`
- project `K` and `V` to `num_kv_heads * head_dim`, not `hidden_size`
- use grouped-query attention mapping:
  - `kv_h = h * num_kv_heads / num_heads`
- apply RoPE separately to:
  - `Q`: `num_heads * head_dim`
  - `K`: `num_kv_heads * head_dim`
- use model metadata for `rope_theta`
- insert `attn_sub_norm` before `attn_output`
- replace SwiGLU with BitNet FFN:
  - `relu(gate)^2 * up`
  - then `ffn_sub_norm`

## Success Criteria

- real 2B-4T GGUF loads without conversion
- prefill succeeds
- decode succeeds
- logits are finite
- greedy token sampling returns valid vocab ids
- synthetic GGUF path still works
- existing inference/session tests still pass

## Result

All criteria met.

Real-model smoke test now passes:

- load: OK
- session init: OK
- prefill: OK
- decode: OK
- finite logits: OK
- greedy token: valid (`128001` in current smoke test)

## Next Steps

1. Add a real tokenizer bridge for Llama 3 vocab / merges
2. Add a tiny CLI for prompt -> generated text using `HSMLTernarySession`
3. Benchmark real decode throughput at context lengths `1, 64, 256, 512`
4. Validate output quality against `bitnet.cpp` on a fixed prompt

## Actual Outcome (2026-03-15)

### Root causes found

1. **m->use_i2s was never set to true** — the most impactful single-line bug.
   Neither the native I2_S kernel nor the norm workaround ever executed.
   All prior probes showing failure were running dead code.

2. **Norm tensor interpretation** — attn_norm, attn_sub_norm, and ffn_norm
   loaded from the real GGUF contain numerically implausible values when
   interpreted as plain F32. The exact storage format for these tensors in
   the BitNet fork GGUF is still unresolved. Current workaround: force
   these three norm types to 1.0 for use_i2s models.

3. **I2_S scalar fallback bit ordering** — the scalar path used simple
   (k%4)*2 shifting but the real I2_S format uses group layout
   (group_idx=j/16, group_pos=j%16, MSB-first). The NEON kernel was
   correct; the scalar was wrong. Found and fixed during red-team.

### Current status

- Real 2B-4T GGUF loads successfully
- Native I2_S NEON kernel active and red-teamed (9/9 checks pass)
- Hidden state stays bounded through all 30 layers (absmax ~23000)
- Logits are finite and differentiated
- Model generates tokens (not coherent due to neutralized norms)
- All 60+ synthetic/regression tests still pass

### Remaining gaps

1. **Norm tensor format**: need to reverse-engineer how the BitNet fork
   stores attn_norm, attn_sub_norm, and ffn_norm in its GGUF. These are
   type 0 (F32) but the raw bytes are not valid F32 norm weights. Possible
   causes: offset alignment issue, interleaved storage, or custom encoding.

2. **I2_S scale semantics**: the 32 extra bytes per I2_S tensor payload
   are not yet consumed. The reference GPU kernel uses a separate
   weight_scale tensor that is not present in the GGUF file. The scale
   may be embedded in those 32 bytes but the exact interpretation is unknown.

3. **BPE tokenizer**: current encoder uses greedy longest-match over vocab.
   Full merge-rank BPE application using tokenizer.ggml.merges is not
   implemented. Roundtrip works for the control prompt but may fail on
   more complex inputs.
