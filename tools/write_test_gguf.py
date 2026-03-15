#!/usr/bin/env python3
"""
Write a minimal valid GGUF file with random ternary weights.
No external dependencies — pure Python struct only.

Produces a GGUF that hs_mlt_load_gguf() can read and load into HSMLTernary.

Weight format: F32 (converted to 2-bit packed ternary by the loader).
Scale format:  F32 per-row, stored as companion tensor <name>_scale.

Usage:
    python3 tools/write_test_gguf.py --out /tmp/test.gguf --size tiny
    python3 tools/write_test_gguf.py --out /tmp/test.gguf --size small
"""

import struct
import random
import argparse
import math
import os

# ── GGUF constants ──────────────────────────────────────────────────────────
GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3
ALIGNMENT = 32

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9

GGML_TYPE_F32 = 0
GGML_TYPE_I8 = 24  # we use this for our 2-bit packed ternary rows

# Model configs
CONFIGS = {
    "tiny": dict(vocab=256, hidden=64, n_layers=2, n_heads=4, ffn=128, ctx=64),
    "small": dict(vocab=512, hidden=128, n_layers=2, n_heads=4, ffn=256, ctx=128),
    "med": dict(vocab=512, hidden=256, n_layers=4, n_heads=8, ffn=512, ctx=256),
}


# ── Binary helpers ───────────────────────────────────────────────────────────


def pack_u8(v):
    return struct.pack("<B", v)


def pack_u32(v):
    return struct.pack("<I", v)


def pack_u64(v):
    return struct.pack("<Q", v)


def pack_i32(v):
    return struct.pack("<i", v)


def pack_f32(v):
    return struct.pack("<f", v)


def pack_string(s):
    b = s.encode("utf-8")
    return pack_u64(len(b)) + b


def pack_kv_uint32(key, val):
    return pack_string(key) + pack_u32(GGUF_TYPE_UINT32) + pack_u32(val)


def pack_kv_int32(key, val):
    return pack_string(key) + pack_u32(GGUF_TYPE_INT32) + pack_i32(val)


def pack_kv_float32(key, val):
    return pack_string(key) + pack_u32(GGUF_TYPE_FLOAT32) + pack_f32(val)


def pack_kv_string(key, val):
    return pack_string(key) + pack_u32(GGUF_TYPE_STRING) + pack_string(val)


def pack_kv_array_string(key, vals):
    body = pack_u32(GGUF_TYPE_STRING) + pack_u64(len(vals))
    for v in vals:
        body += pack_string(v)
    return pack_string(key) + pack_u32(GGUF_TYPE_ARRAY) + body


def pack_kv_array_f32(key, vals):
    body = pack_u32(GGUF_TYPE_FLOAT32) + pack_u64(len(vals))
    for v in vals:
        body += pack_f32(v)
    return pack_string(key) + pack_u32(GGUF_TYPE_ARRAY) + body


def pack_kv_array_i32(key, vals):
    body = pack_u32(GGUF_TYPE_INT32) + pack_u64(len(vals))
    for v in vals:
        body += pack_i32(v)
    return pack_string(key) + pack_u32(GGUF_TYPE_ARRAY) + body


def align_up(x, a=ALIGNMENT):
    return (x + a - 1) & ~(a - 1)


def pad_to(data, length):
    return data + b"\x00" * (length - len(data))


# ── Ternary weight generation ────────────────────────────────────────────────


def make_ternary_f32(rows, cols, rng):
    """Return (data_bytes, scale_bytes) for a ternary weight matrix.
    data: F32 array [rows*cols] with values in {-1, 0, +1}
    scale: F32 array [rows] — max abs per row (= 1.0 for ternary ±1)
    """
    data = bytearray()
    scales = bytearray()
    for _ in range(rows):
        row_max = 0.0
        row_vals = []
        for _ in range(cols):
            r = rng.randint(0, 3)
            v = 1.0 if r == 0 else (-1.0 if r == 1 else 0.0)
            row_vals.append(v)
            if abs(v) > row_max:
                row_max = abs(v)
        for v in row_vals:
            data += pack_f32(v)
        scales += pack_f32(row_max if row_max > 0 else 1.0)
    return bytes(data), bytes(scales)


def make_f32_tensor(size, rng, scale=0.02):
    """Float32 tensor with small random values."""
    data = bytearray()
    for _ in range(size):
        v = (rng.random() * 2 - 1) * scale
        data += pack_f32(v)
    return bytes(data)


def make_ones_f32(size):
    data = bytearray()
    for _ in range(size):
        data += pack_f32(1.0)
    return bytes(data)


def make_vocab_strings(vocab_size):
    """Generate simple token strings."""
    tokens = []
    for i in range(vocab_size):
        if i == 0:
            tokens.append("<unk>")
        elif i == 1:
            tokens.append("<s>")
        elif i == 2:
            tokens.append("</s>")
        else:
            tokens.append(f"t{i}")
    return tokens


def make_token_scores(vocab_size):
    return [-float(i) * 0.01 for i in range(vocab_size)]


# ── Tensor info packing ──────────────────────────────────────────────────────


def pack_tensor_info(name, shape, ggml_type, offset):
    """Pack a gguf_tensor_info_t."""
    b = pack_string(name)
    b += pack_u32(len(shape))
    for dim in shape:
        b += pack_u64(dim)
    b += pack_u32(ggml_type)
    b += pack_u64(offset)
    return b


# ── Main writer ──────────────────────────────────────────────────────────────


def write_gguf(path, cfg, seed=42):
    rng = random.Random(seed)
    V = cfg["vocab"]
    H = cfg["hidden"]
    L = cfg["n_layers"]
    NH = cfg["n_heads"]
    HD = H // NH
    F = cfg["ffn"]
    CTX = cfg["ctx"]

    print(f"Writing GGUF: vocab={V} hidden={H} layers={L} heads={NH} ffn={F} ctx={CTX}")

    # ── Metadata KV pairs ────────────────────────────────────────────────────
    kv = bytearray()
    kv += pack_kv_string("general.architecture", "llama")
    kv += pack_kv_string("general.name", "neogpu-test-ternary")
    kv += pack_kv_uint32(
        "general.file_type", 1
    )  # mostly F16 (we use F32 but close enough)
    kv += pack_kv_uint32("llama.context_length", CTX)
    kv += pack_kv_uint32("llama.embedding_length", H)
    kv += pack_kv_uint32("llama.block_count", L)
    kv += pack_kv_uint32("llama.feed_forward_length", F)
    kv += pack_kv_uint32("llama.attention.head_count", NH)
    kv += pack_kv_uint32("llama.attention.head_count_kv", NH)
    kv += pack_kv_uint32("llama.rope.dimension_count", HD)
    kv += pack_kv_float32("llama.attention.layer_norm_rms_epsilon", 1e-5)

    # Tokenizer
    tokens = make_vocab_strings(V)
    scores = make_token_scores(V)
    token_types = [1] * V
    token_types[0] = 2  # unknown
    token_types[1] = 3  # BOS
    token_types[2] = 3  # EOS
    kv += pack_kv_string("tokenizer.ggml.model", "llama")
    kv += pack_kv_array_string("tokenizer.ggml.tokens", tokens)
    kv += pack_kv_array_f32("tokenizer.ggml.scores", scores)
    kv += pack_kv_array_i32("tokenizer.ggml.token_type", token_types)
    kv += pack_kv_uint32("tokenizer.ggml.bos_token_id", 1)
    kv += pack_kv_uint32("tokenizer.ggml.eos_token_id", 2)
    kv += pack_kv_uint32("tokenizer.ggml.unknown_token_id", 0)
    kv_count = 18

    # ── Tensor definitions ───────────────────────────────────────────────────
    # Name, shape (GGUF uses col-major: [K, N] for weight [N, K] in row-major)
    # We store everything as F32 for simplicity; loader converts to packed ternary.
    #
    # Naming follows GGUF standard (blk.N.attn_q.weight etc.)
    tensors = []  # (name, shape_gguf, ggml_type, data_fn)

    # Embedding and head
    tensors.append(
        (
            "token_embd.weight",
            [H, V],
            GGML_TYPE_F32,
            lambda: make_f32_tensor(V * H, rng),
        )
    )
    tensors.append(
        ("output.weight", [H, V], GGML_TYPE_F32, lambda: make_f32_tensor(V * H, rng))
    )
    tensors.append(("output_norm.weight", [H], GGML_TYPE_F32, lambda: make_ones_f32(H)))

    for l in range(L):
        # Attention norms
        tensors.append(
            (f"blk.{l}.attn_norm.weight", [H], GGML_TYPE_F32, lambda: make_ones_f32(H))
        )
        tensors.append(
            (f"blk.{l}.ffn_norm.weight", [H], GGML_TYPE_F32, lambda: make_ones_f32(H))
        )

        # Ternary projection weights (F32, loader converts)
        # Q, K, V, O: shape [H, H] row-major = GGUF [H, H]
        # Gate, Up:   shape [F, H] row-major = GGUF [H, F]
        # Down:       shape [H, F] row-major = GGUF [F, H]
        for wname, rows, cols in [
            (f"blk.{l}.attn_q.weight", H, H),
            (f"blk.{l}.attn_k.weight", H, H),
            (f"blk.{l}.attn_v.weight", H, H),
            (f"blk.{l}.attn_output.weight", H, H),
            (f"blk.{l}.ffn_gate.weight", F, H),
            (f"blk.{l}.ffn_up.weight", F, H),
            (f"blk.{l}.ffn_down.weight", H, F),
        ]:
            r, c = rows, cols
            data_fn, scale_fn = (
                lambda r=r, c=c: make_ternary_f32(r, c, rng)[0],
                lambda r=r, c=c: make_ternary_f32(r, c, rng)[1],
            )
            # Store scale alongside weight as <name>_scale
            # GGUF shape: [cols, rows] (transposed from row-major)
            tensors.append((wname, [c, r], GGML_TYPE_F32, data_fn))
            tensors.append(
                (wname + "_scale", [r], GGML_TYPE_F32, lambda r=r: make_ones_f32(r))
            )  # scale=1.0 (random ternary ±1)

    # ── Generate tensor data ─────────────────────────────────────────────────
    print(f"  Generating {len(tensors)} tensors...")
    tensor_data_list = []
    for name, shape_gguf, ggml_type, data_fn in tensors:
        data = data_fn()
        tensor_data_list.append(data)

    # ── Compute offsets (relative to tensor_data start, ALIGNMENT-aligned) ──
    offsets = []
    cur = 0
    for data in tensor_data_list:
        offsets.append(cur)
        cur = align_up(cur + len(data))

    # ── Pack tensor infos ────────────────────────────────────────────────────
    tensor_infos = bytearray()
    for i, (name, shape_gguf, ggml_type, _) in enumerate(tensors):
        tensor_infos += pack_tensor_info(name, shape_gguf, ggml_type, offsets[i])

    # ── File layout ──────────────────────────────────────────────────────────
    # header: magic(4) + version(4) + tensor_count(8) + kv_count(8) + kv_data
    header = (
        pack_u32(GGUF_MAGIC)
        + pack_u32(GGUF_VERSION)
        + pack_u64(len(tensors))
        + pack_u64(kv_count)
        + bytes(kv)
    )

    # tensor_infos follow header immediately
    pre_data = bytes(header) + bytes(tensor_infos)

    # Pad pre_data to ALIGNMENT boundary — that's where tensor_data starts
    data_start = align_up(len(pre_data))
    pre_data = pad_to(pre_data, data_start)

    # Concatenate all tensor data blocks with alignment padding
    tensor_blob = bytearray()
    for i, data in enumerate(tensor_data_list):
        assert len(tensor_blob) == offsets[i], f"offset mismatch at tensor {i}"
        tensor_blob += data
        padded_size = align_up(len(tensor_blob))
        tensor_blob = bytearray(pad_to(bytes(tensor_blob), padded_size))

    total = len(pre_data) + len(tensor_blob)
    print(f"  File size: {total:,} bytes ({total / 1024 / 1024:.2f} MB)")
    print(f"  Tensors: {len(tensors)}")

    with open(path, "wb") as f:
        f.write(pre_data)
        f.write(tensor_blob)

    print(f"  Written: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Write test GGUF for NeoGPU ML loader")
    parser.add_argument("--out", default="/tmp/test_neogpu.gguf", help="Output path")
    parser.add_argument("--size", default="tiny", choices=CONFIGS.keys())
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = CONFIGS[args.size]
    write_gguf(args.out, cfg, seed=args.seed)


if __name__ == "__main__":
    main()
