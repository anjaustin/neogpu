#!/usr/bin/env python3
"""
Patch hs_ml_loader_ternary.c to load real BF16 norm weights from sidecar file.

The sidecar (models/norms_bf16.bin) contains 121 BF16 norm tensors extracted
from the original HuggingFace model.safetensors checkpoint.

Name mapping:
  model.layers.N.input_layernorm.weight     -> attn_norm
  model.layers.N.self_attn.attn_sub_norm.weight -> attn_sub_norm
  model.layers.N.post_attention_layernorm.weight -> ffn_norm
  model.layers.N.mlp.ffn_sub_norm.weight    -> ffn_sub_norm
  model.norm.weight                          -> final_norm
"""

from pathlib import Path
import sys

p = Path("src/hs_ml_loader_ternary.c")
if not p.exists():
    print("Run from neogpu root directory")
    sys.exit(1)

lines = p.read_text().splitlines()

# Find the norm override block (use_i2s block)
start = end = None
for i, l in enumerate(lines):
    if "BitNet 2B-4T GGUF" in l and "norm tensors" in l:
        start = i - 1
    if start is not None and "free(tis);" in l and i > start + 5:
        end = i
        break

if start is None or end is None:
    # Try alternate marker
    for i, l in enumerate(lines):
        if "if (m->use_i2s)" in l and i > 600:
            start = i - 2
        if start is not None and "free(tis);" in l and i > start + 5:
            end = i
            break

if start is None or end is None:
    print("Could not find norm override block")
    sys.exit(1)

print(f"Replacing lines {start}..{end}")

new_block = [
    "    /* Load real BF16 norms from sidecar file (extracted from HF checkpoint).",
    "     * The GGUF norm tensors are corrupt (upstream converter bug).",
    "     * Sidecar format: u32 header_len + JSON header + BF16 data */",
    "    if (m->use_i2s) {",
    '        const char *norm_path = "/home/ztflynn/001/neogpu/models/norms_bf16.bin";',
    '        FILE *nf = fopen(norm_path, "rb");',
    "        if (nf) {",
    "            uint32_t hlen;",
    "            if (fread(&hlen, 4, 1, nf) == 1 && hlen < 100000) {",
    "                char *hjson = malloc(hlen + 1);",
    "                if (hjson && fread(hjson, 1, hlen, nf) == hlen) {",
    "                    hjson[hlen] = 0;",
    "                    long norm_data_base = 4 + hlen;",
    "                    /* Parse JSON minimally: find tensor offsets by name */",
    "                    /* For each layer, load 4 norms; plus final_norm */",
    "                    char search[256];",
    "                    char *found;",
    "                    for (u32 l = 0; l < n_layers; l++) {",
    "                        HSTernaryLayer *lay = &m->layers[l];",
    "",
    '                        /* Helper macro: find "NAME":..."offset":N,"size":M and read BF16 */',
    "#define LOAD_NORM_BF16(field, field_size, hf_name) do { \\",
    '    snprintf(search, sizeof(search), "\\"%s\\"", hf_name); \\',
    "    found = strstr(hjson, search); \\",
    "    if (found) { \\",
    '        char *op = strstr(found, "\\"offset\\": "); \\',
    '        char *sp = strstr(found, "\\"size\\": "); \\',
    '        if (!op) op = strstr(found, "\\"offset\\":"); \\',
    '        if (!sp) sp = strstr(found, "\\"size\\":"); \\',
    "        if (op && sp) { \\",
    "            long off = atol(op + (strchr(op, ':') - op) + 1); \\",
    "            long sz = atol(sp + (strchr(sp, ':') - sp) + 1); \\",
    "            uint16_t *buf = malloc(sz); \\",
    "            if (buf) { \\",
    "                fseek(nf, norm_data_base + off, SEEK_SET); \\",
    "                if (fread(buf, 1, sz, nf) == (size_t)sz) { \\",
    "                    for (u32 _i = 0; _i < (u32)(field_size); _i++) { \\",
    "                        uint32_t bits = (uint32_t)buf[_i] << 16; \\",
    "                        float fv; memcpy(&fv, &bits, 4); \\",
    "                        (field)[_i] = fv; \\",
    "                    } \\",
    "                } \\",
    "                free(buf); \\",
    "            } \\",
    "        } \\",
    "    } \\",
    "} while(0)",
    "",
    '                        snprintf(search, sizeof(search), "model.layers.%u.input_layernorm.weight", l);',
    "                        LOAD_NORM_BF16(lay->attn_norm, hidden_size, search);",
    "",
    '                        snprintf(search, sizeof(search), "model.layers.%u.self_attn.attn_sub_norm.weight", l);',
    "                        LOAD_NORM_BF16(lay->attn_sub_norm, hidden_size, search);",
    "",
    '                        snprintf(search, sizeof(search), "model.layers.%u.post_attention_layernorm.weight", l);',
    "                        LOAD_NORM_BF16(lay->ffn_norm, hidden_size, search);",
    "",
    '                        snprintf(search, sizeof(search), "model.layers.%u.mlp.ffn_sub_norm.weight", l);',
    "                        LOAD_NORM_BF16(lay->ffn_sub_norm, ffn_size, search);",
    "                    }",
    '                    LOAD_NORM_BF16(m->final_norm, hidden_size, "model.norm.weight");',
    "#undef LOAD_NORM_BF16",
    '                    printf("loader: loaded real BF16 norms from %s\\n", norm_path);',
    "                }",
    "                free(hjson);",
    "            }",
    "            fclose(nf);",
    "        } else {",
    '            fprintf(stderr, "loader: norm sidecar not found at %s, using defaults\\n", norm_path);',
    "        }",
    "    }",
    "",
]

lines[start:end] = new_block
p.write_text("\n".join(lines) + "\n")
print("Patched successfully")
