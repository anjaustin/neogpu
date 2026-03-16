#!/usr/bin/env python3
"""
Convert BitNet b1.58-2B-4T from HuggingFace safetensors to clean GGUF.

Reads the packed uint8 weights from microsoft/bitnet-b1.58-2B-4T,
unpacks them to ternary {-1, 0, +1}, and writes a GGUF file where:
  - Projection weights: simple 2-bit packed (4 weights per byte, low bits first)
    Code mapping: 0=-1, 1=0, 2=+1
  - Norms: F32
  - Embeddings: F16
  - GGML type for projections: type 36 (I2_S slot, repurposed for simple 2-bit)

Usage:
  python3 tools/convert_bitnet_to_gguf.py [--model microsoft/bitnet-b1.58-2B-4T] [--output models/bitnet-2b4t-i2s.gguf]
"""

import struct
import sys
import os
import argparse
import numpy as np


def write_gguf_string(f, s):
    """Write a GGUF string (u64 length + bytes)."""
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def write_kv_string(f, key, value):
    write_gguf_string(f, key)
    f.write(struct.pack("<I", 8))  # GGUF_TYPE_STRING
    write_gguf_string(f, value)


def write_kv_u32(f, key, value):
    write_gguf_string(f, key)
    f.write(struct.pack("<I", 4))  # GGUF_TYPE_UINT32
    f.write(struct.pack("<I", value))


def write_kv_f32(f, key, value):
    write_gguf_string(f, key)
    f.write(struct.pack("<I", 6))  # GGUF_TYPE_FLOAT32
    f.write(struct.pack("<f", value))


def write_kv_bool(f, key, value):
    write_gguf_string(f, key)
    f.write(struct.pack("<I", 7))  # GGUF_TYPE_BOOL
    f.write(struct.pack("<B", 1 if value else 0))


def write_kv_string_array(f, key, strings):
    write_gguf_string(f, key)
    f.write(struct.pack("<I", 9))  # GGUF_TYPE_ARRAY
    f.write(struct.pack("<I", 8))  # element type = STRING
    f.write(struct.pack("<Q", len(strings)))
    for s in strings:
        write_gguf_string(f, s)


def write_kv_f32_array(f, key, values):
    write_gguf_string(f, key)
    f.write(struct.pack("<I", 9))  # GGUF_TYPE_ARRAY
    f.write(struct.pack("<I", 6))  # element type = FLOAT32
    f.write(struct.pack("<Q", len(values)))
    for v in values:
        f.write(struct.pack("<f", v))


def write_kv_i32_array(f, key, values):
    write_gguf_string(f, key)
    f.write(struct.pack("<I", 9))  # GGUF_TYPE_ARRAY
    f.write(struct.pack("<I", 5))  # element type = INT32
    f.write(struct.pack("<Q", len(values)))
    for v in values:
        f.write(struct.pack("<i", v))


GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_I2_S = 36  # We use this slot for simple 2-bit packed ternary

ALIGNMENT = 32


def pack_ternary_simple(weights_f32, N, K):
    """Pack ternary weights {-1, 0, +1} into simple 2-bit format.

    4 weights per byte, low bits first:
      bits[1:0] = w0, bits[3:2] = w1, bits[5:4] = w2, bits[7:6] = w3
    Code: 0 = -1, 1 = 0, 2 = +1

    Returns: uint8 array of shape [N, K//4]
    """
    w = weights_f32.reshape(N, K)
    # Map: -1 → 0, 0 → 1, +1 → 2
    codes = np.round(w).astype(np.int8)
    codes = np.where(codes == -1, 0, np.where(codes == 0, 1, 2)).astype(np.uint8)

    # Pack 4 codes per byte
    K4 = K // 4
    packed = np.zeros((N, K4), dtype=np.uint8)
    for b in range(4):
        packed |= codes[:, b::4] << (b * 2)

    return packed


def unpack_hf_weights(packed_uint8, scale, N, K):
    """Unpack HuggingFace packed uint8 weights to ternary float32.

    HF packing: shape [N//4, K], dtype uint8
    Each byte holds 4 codes: bits[1:0]=c0, [3:2]=c1, [5:4]=c2, [7:6]=c3
    Code - 1 gives ternary: 0→-1, 1→0, 2→+1
    Result divided by scale.
    """
    assert packed_uint8.shape == (N // 4, K), (
        f"Expected ({N // 4}, {K}), got {packed_uint8.shape}"
    )

    # Expand: each byte → 4 codes
    expanded = np.zeros((N // 4, 4, K), dtype=np.float32)
    for i, shift in enumerate([0, 2, 4, 6]):
        codes = (packed_uint8.astype(np.uint32) >> shift) & 3
        expanded[:, i, :] = codes.astype(np.float32) - 1.0

    # Reshape to [N, K]
    weights = expanded.transpose(0, 1, 2).reshape(N, K)

    # Apply inverse scale
    weights = weights / float(scale)

    return weights


def bf16_to_f32_array(bf16_data):
    """Convert bfloat16 raw bytes to float32 numpy array."""
    u16 = np.frombuffer(bf16_data, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Convert BitNet HF model to clean GGUF"
    )
    parser.add_argument(
        "--model", default="microsoft/bitnet-b1.58-2B-4T", help="HuggingFace model ID"
    )
    parser.add_argument(
        "--output", default="models/bitnet-2b4t-i2s.gguf", help="Output GGUF path"
    )
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")

    # Import torch for safetensors bf16 support
    import torch
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download

    # Download safetensors
    st_path = hf_hub_download(args.model, "model.safetensors")
    print(f"Safetensors: {st_path} ({os.path.getsize(st_path) / 1e6:.1f} MB)")

    # Also download tokenizer
    tok_path = hf_hub_download(args.model, "tokenizer.json")

    # Load config
    import json

    cfg_path = hf_hub_download(args.model, "config.json")
    with open(cfg_path) as f:
        config = json.load(f)

    hidden_size = config["hidden_size"]
    num_layers = config["num_hidden_layers"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    ffn_size = config["intermediate_size"]
    vocab_size = config["vocab_size"]
    max_context = config["max_position_embeddings"]
    rope_theta = config.get("rope_theta", 500000.0)
    rms_eps = config.get("rms_norm_eps", 1e-5)
    head_dim = hidden_size // num_heads
    kv_size = num_kv_heads * head_dim

    print(
        f"Config: hidden={hidden_size} layers={num_layers} heads={num_heads} "
        f"kv_heads={num_kv_heads} ffn={ffn_size} vocab={vocab_size}"
    )

    # Load tokenizer vocab
    with open(tok_path) as f:
        tok_json = json.load(f)

    # Extract vocab from tokenizer.json
    vocab_list = [""] * vocab_size
    if "model" in tok_json and "vocab" in tok_json["model"]:
        for token, idx in tok_json["model"]["vocab"].items():
            if idx < vocab_size:
                vocab_list[idx] = token

    # Also handle added_tokens
    if "added_tokens" in tok_json:
        for entry in tok_json["added_tokens"]:
            idx = entry["id"]
            if idx < vocab_size:
                vocab_list[idx] = entry["content"]

    # Extract merges
    merges = []
    if "model" in tok_json and "merges" in tok_json["model"]:
        merges = tok_json["model"]["merges"]

    print(
        f"Tokenizer: {sum(1 for v in vocab_list if v)} vocab entries, {len(merges)} merges"
    )

    # Open safetensors
    st = safe_open(st_path, framework="pt")
    tensor_names = sorted(list(st.keys()))

    # ── Prepare tensor data ──
    # We'll collect all tensors, then write GGUF

    tensors = []  # list of (name, ggml_type, shape, data_bytes)

    def add_f16_tensor(name, data_torch):
        """Add a tensor stored as F16."""
        data = data_torch.to(torch.float16).numpy()
        shape = list(data.shape)
        tensors.append((name, GGML_TYPE_F16, shape, data.tobytes()))

    def add_f32_tensor(name, data_torch):
        """Add a tensor stored as F32."""
        data = data_torch.to(torch.float32).numpy()
        shape = list(data.shape)
        tensors.append((name, GGML_TYPE_F32, shape, data.tobytes()))

    def add_i2s_tensor(name, ternary_f32, N, K, weight_scale=1.0):
        """Add a ternary weight tensor in simple 2-bit packed format.
        Also adds a companion _scale tensor (single F32 scalar).
        The inference engine should divide projection output by weight_scale.
        """
        packed = pack_ternary_simple(ternary_f32, N, K)
        shape = [K, N]  # GGUF convention: [cols, rows]
        tensors.append((name, GGML_TYPE_I2_S, shape, packed.tobytes()))
        # Add per-tensor scale as F32 scalar
        scale_data = np.array([weight_scale], dtype=np.float32)
        tensors.append((name + "_scale", GGML_TYPE_F32, [1], scale_data.tobytes()))

    # ── Load and convert tensors ──

    # Embedding
    print("Loading embedding...")
    emb = st.get_tensor("model.embed_tokens.weight")
    add_f16_tensor("token_embd.weight", emb)

    # Final norm
    print("Loading final norm...")
    fn = st.get_tensor("model.norm.weight")
    add_f32_tensor("output_norm.weight", fn)

    # Note: no output.weight (weight tying) - lm_head uses embedding

    # Per-layer weights
    PROJ_MAP = {
        "self_attn.q_proj": ("attn_q", hidden_size),
        "self_attn.k_proj": ("attn_k", kv_size),
        "self_attn.v_proj": ("attn_v", kv_size),
        "self_attn.o_proj": ("attn_output", hidden_size),
        "mlp.gate_proj": ("ffn_gate", ffn_size),
        "mlp.up_proj": ("ffn_up", ffn_size),
        "mlp.down_proj": ("ffn_down", hidden_size),
    }

    # Input dims for each projection
    PROJ_INPUT = {
        "self_attn.q_proj": hidden_size,
        "self_attn.k_proj": hidden_size,
        "self_attn.v_proj": hidden_size,
        "self_attn.o_proj": hidden_size,
        "mlp.gate_proj": hidden_size,
        "mlp.up_proj": hidden_size,
        "mlp.down_proj": ffn_size,
    }

    NORM_MAP = {
        "input_layernorm": "attn_norm",
        "self_attn.attn_sub_norm": "attn_sub_norm",
        "post_attention_layernorm": "ffn_norm",
        "mlp.ffn_sub_norm": "ffn_sub_norm",
    }

    for layer in range(num_layers):
        if layer % 5 == 0 or layer == num_layers - 1:
            print(f"Loading layer {layer + 1}/{num_layers}...")

        # Norms (BF16 in safetensors → F32 in GGUF)
        for hf_suffix, gguf_suffix in NORM_MAP.items():
            hf_name = f"model.layers.{layer}.{hf_suffix}.weight"
            gguf_name = f"blk.{layer}.{gguf_suffix}.weight"
            data = st.get_tensor(hf_name)
            add_f32_tensor(gguf_name, data)

        # Projections (packed uint8 → unpack → repack as simple 2-bit)
        for hf_suffix, (gguf_suffix, out_dim) in PROJ_MAP.items():
            hf_name = f"model.layers.{layer}.{hf_suffix}.weight"
            scale_name = f"model.layers.{layer}.{hf_suffix}.weight_scale"
            gguf_name = f"blk.{layer}.{gguf_suffix}.weight"

            packed = st.get_tensor(hf_name)  # [N//4, K] uint8
            scale = st.get_tensor(scale_name)  # [1] bf16

            K_in = PROJ_INPUT[hf_suffix]
            N_out = out_dim

            # Unpack: HF format → ternary float32
            p_np = packed.numpy()
            s_val = scale.to(torch.float32).item()

            # Expand 4 codes per byte
            w = np.zeros((N_out, K_in), dtype=np.float32)
            for i, shift in enumerate([0, 2, 4, 6]):
                codes = (p_np.astype(np.uint32) >> shift) & 3
                row_slice = slice(i * (N_out // 4), (i + 1) * (N_out // 4))
                w[row_slice, :] = codes.astype(np.float32) - 1.0

            # The weights are raw ternary {-1, 0, +1}
            # The weight_scale is needed for dequantization during inference:
            #   output = ternary_proj(input, w) / weight_scale
            # We store the scale as a companion tensor
            w_ternary = w  # Already {-1, 0, +1}

            add_i2s_tensor(gguf_name, w_ternary, N_out, K_in, weight_scale=s_val)

    # ── Write GGUF ──
    print(f"\nWriting GGUF to {args.output}...")

    # Count KV pairs
    kv_count = 14 + 3  # metadata + tokenizer arrays

    with open(args.output, "wb") as f:
        # Header
        f.write(struct.pack("<I", 0x46554747))  # GGUF magic
        f.write(struct.pack("<I", 3))  # version 3
        f.write(struct.pack("<Q", len(tensors)))  # tensor count
        f.write(struct.pack("<Q", kv_count))  # kv count

        # Metadata KV
        write_kv_string(f, "general.architecture", "bitnet-b1.58")
        write_kv_string(f, "general.name", "bitnet-b1.58-2B-4T")
        write_kv_u32(f, "bitnet-b1.58.vocab_size", vocab_size)
        write_kv_u32(f, "bitnet-b1.58.context_length", max_context)
        write_kv_u32(f, "bitnet-b1.58.embedding_length", hidden_size)
        write_kv_u32(f, "bitnet-b1.58.block_count", num_layers)
        write_kv_u32(f, "bitnet-b1.58.feed_forward_length", ffn_size)
        write_kv_u32(f, "bitnet-b1.58.attention.head_count", num_heads)
        write_kv_u32(f, "bitnet-b1.58.attention.head_count_kv", num_kv_heads)
        write_kv_f32(f, "bitnet-b1.58.rope.freq_base", rope_theta)
        write_kv_f32(f, "bitnet-b1.58.attention.layer_norm_rms_epsilon", rms_eps)
        write_kv_string(f, "tokenizer.ggml.model", "gpt2")
        write_kv_u32(f, "tokenizer.ggml.bos_token_id", 128000)
        write_kv_u32(f, "tokenizer.ggml.eos_token_id", 128001)

        # Tokenizer vocab (string array)
        write_kv_string_array(f, "tokenizer.ggml.tokens", vocab_list)

        # Tokenizer scores (float array - all 0 for BPE)
        write_kv_f32_array(f, "tokenizer.ggml.scores", [0.0] * vocab_size)

        # Tokenizer merges (string array)
        write_kv_string_array(f, "tokenizer.ggml.merges", merges)

        # Tensor infos
        data_offset = 0
        for name, ggml_type, shape, data in tensors:
            write_gguf_string(f, name)
            f.write(struct.pack("<I", len(shape)))  # n_dims
            for dim in shape:
                f.write(struct.pack("<Q", dim))
            f.write(struct.pack("<I", ggml_type))
            f.write(struct.pack("<Q", data_offset))

            # Compute aligned size
            size = len(data)
            aligned_size = (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1)
            data_offset += aligned_size

        # Align to ALIGNMENT before tensor data
        pos = f.tell()
        aligned_pos = (pos + ALIGNMENT - 1) & ~(ALIGNMENT - 1)
        f.write(b"\x00" * (aligned_pos - pos))

        # Write tensor data
        for name, ggml_type, shape, data in tensors:
            f.write(data)
            # Pad to alignment
            size = len(data)
            aligned_size = (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1)
            if aligned_size > size:
                f.write(b"\x00" * (aligned_size - size))

        total_size = f.tell()

    print(f"Done! Written {total_size / 1e6:.1f} MB ({len(tensors)} tensors)")
    print(f"  Projections: simple 2-bit packed (type {GGML_TYPE_I2_S})")
    print(f"  Embeddings: F16")
    print(f"  Norms: F32")


if __name__ == "__main__":
    main()
