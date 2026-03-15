#!/usr/bin/env python3
"""
Patch hs_ml_loader_ternary.c to read norm tensors as F16 pairs.

BitNet 2B-4T GGUF stores norm tensors as F16 in F32-typed slots:
  attn_norm slot [2560]:   first 2560 F16 = attn_norm, next 2560 F16 = attn_sub_norm
  ffn_sub_norm slot [6912]: first 6912 F16 = ffn_sub_norm
  ffn_norm slot [2560]:    first 2560 F16 attempted as ffn_norm
  output_norm slot [2560]: first 2560 F16 attempted as output_norm
"""

from pathlib import Path
import sys

p = Path("src/hs_ml_loader_ternary.c")
if not p.exists():
    print("Run from neogpu root directory")
    sys.exit(1)

lines = p.read_text().splitlines()

# Find the norm override block
start = end = None
for i, l in enumerate(lines):
    if "Real 2B-4T GGUF currently exposes norm tensors" in l:
        start = i - 1
    if start is not None and "free(tis);" in l and i > start + 5:
        end = i
        break

if start is None or end is None:
    print("Could not find norm override block")
    sys.exit(1)

print(f"Replacing lines {start}..{end}")

# The key insight: ALL type-0 norm tensors in this GGUF are actually F16.
# attn_norm contains attn_norm + attn_sub_norm as F16 pair.
# ffn_sub_norm contains ffn_sub_norm as F16 (first half).
# ffn_norm and output_norm: try reading as F16 directly.
new_block = [
    "    /* BitNet 2B-4T GGUF: norm tensors stored as F16 in F32-typed slots.",
    "     * attn_norm slot: F16 pair (attn_norm + attn_sub_norm)",
    "     * ffn_sub_norm slot: F16 (first half)",
    "     * ffn_norm, output_norm: also read as F16 */",
    "    if (m->use_i2s) {",
    "        for (u32 l = 0; l < n_layers; l++) {",
    "            HSTernaryLayer *lay = &m->layers[l];",
    "            char wname[256];",
    "            int ni;",
    "",
    "            /* attn_norm slot -> attn_norm + attn_sub_norm as F16 pair */",
    '            snprintf(wname, sizeof(wname), "blk.%u.attn_norm.weight", l);',
    "            ni = find_tensor(tis, (int)tensor_count, wname);",
    "            if (ni >= 0) {",
    "                uint16_t *buf = malloc(hidden_size * 2 * sizeof(uint16_t));",
    "                if (buf) {",
    "                    fseek(f, data_start + (long)tis[ni].offset, SEEK_SET);",
    "                    if (fread(buf, 2, hidden_size * 2, f) == hidden_size * 2) {",
    "                        for (u32 i = 0; i < hidden_size; i++)",
    "                            lay->attn_norm[i] = fp16_to_f32(buf[i]);",
    "                        for (u32 i = 0; i < hidden_size; i++)",
    "                            lay->attn_sub_norm[i] = fp16_to_f32(buf[hidden_size + i]);",
    "                    }",
    "                    free(buf);",
    "                }",
    "            }",
    "",
    "            /* ffn_sub_norm slot -> ffn_sub_norm as F16 (first half) */",
    '            snprintf(wname, sizeof(wname), "blk.%u.ffn_sub_norm.weight", l);',
    "            ni = find_tensor(tis, (int)tensor_count, wname);",
    "            if (ni >= 0) {",
    "                uint16_t *buf = malloc(ffn_size * sizeof(uint16_t));",
    "                if (buf) {",
    "                    fseek(f, data_start + (long)tis[ni].offset, SEEK_SET);",
    "                    if (fread(buf, 2, ffn_size, f) == (size_t)ffn_size) {",
    "                        for (u32 i = 0; i < ffn_size; i++)",
    "                            lay->ffn_sub_norm[i] = fp16_to_f32(buf[i]);",
    "                    }",
    "                    free(buf);",
    "                }",
    "            }",
    "",
    "            /* ffn_norm slot -> try as F16 */",
    '            snprintf(wname, sizeof(wname), "blk.%u.ffn_norm.weight", l);',
    "            ni = find_tensor(tis, (int)tensor_count, wname);",
    "            if (ni >= 0) {",
    "                uint16_t *buf = malloc(hidden_size * sizeof(uint16_t));",
    "                if (buf) {",
    "                    fseek(f, data_start + (long)tis[ni].offset, SEEK_SET);",
    "                    if (fread(buf, 2, hidden_size, f) == hidden_size) {",
    "                        int sane = 1;",
    "                        for (u32 i = 0; i < hidden_size && sane; i++) {",
    "                            float v = fp16_to_f32(buf[i]);",
    "                            if (v != v || v > 100.0f || v < -100.0f) sane = 0;",
    "                        }",
    "                        if (sane) {",
    "                            for (u32 i = 0; i < hidden_size; i++)",
    "                                lay->ffn_norm[i] = fp16_to_f32(buf[i]);",
    "                        }",
    "                    }",
    "                    free(buf);",
    "                }",
    "            }",
    "        }",
    "",
    "        /* output_norm -> try as F16 */",
    "        {",
    '            int ni = find_tensor(tis, (int)tensor_count, "output_norm.weight");',
    "            if (ni >= 0) {",
    "                uint16_t *buf = malloc(hidden_size * sizeof(uint16_t));",
    "                if (buf) {",
    "                    fseek(f, data_start + (long)tis[ni].offset, SEEK_SET);",
    "                    if (fread(buf, 2, hidden_size, f) == hidden_size) {",
    "                        int sane = 1;",
    "                        for (u32 i = 0; i < hidden_size && sane; i++) {",
    "                            float v = fp16_to_f32(buf[i]);",
    "                            if (v != v || v > 100.0f || v < -100.0f) sane = 0;",
    "                        }",
    "                        if (sane) {",
    "                            for (u32 i = 0; i < hidden_size; i++)",
    "                                m->final_norm[i] = fp16_to_f32(buf[i]);",
    "                        }",
    "                    }",
    "                    free(buf);",
    "                }",
    "            }",
    "        }",
    "    }",
    "",
]

lines[start:end] = new_block
p.write_text("\n".join(lines) + "\n")
print("Patched successfully")
