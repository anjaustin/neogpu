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
