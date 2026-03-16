/*
 * Correctness test: hs_ml_ternary_f32_proj matches scalar reference.
 * Tests all projection shapes for the 2B-4T model architecture.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "hs_ml_infer.h"

extern void hs_ml_ternary_f32_proj(float *out, const float *in,
                                    const uint8_t *W, u32 N, u32 K);

static int check(const float *kernel, const float *scalar, u32 N, const char *name) {
    float max_diff = 0;
    for (u32 i = 0; i < N; i++) {
        float d = fabsf(kernel[i] - scalar[i]);
        if (d > max_diff) max_diff = d;
    }
    int pass = max_diff < 1e-4f;
    printf("  %-20s max_diff=%.2e  %s\n", name, max_diff, pass ? "PASS" : "FAIL");
    return pass;
}

static void scalar_proj(float *out, const float *in, const u8 *W, u32 N, u32 K) {
    u32 rb = K / 4;
    for (u32 n = 0; n < N; n++) {
        float acc = 0;
        const u8 *row = W + (size_t)n * rb;
        for (u32 bi = 0; bi < rb; bi++) {
            u8 b = row[bi];
            for (int s = 0; s < 8; s += 2) {
                float w = (float)((int8_t)((b >> s) & 3) - 1);
                acc += w * in[bi * 4 + s / 2];
            }
        }
        out[n] = acc;
    }
}

int main(void) {
    HSMLTernary m;
    hs_mlt_init(&m);
    if (hs_mlt_load_gguf(&m, "/home/ztflynn/001/neogpu/models/ggml-model-i2_s.gguf")) return 1;

    u32 H = m.hidden_size, F = m.ffn_hidden_size;
    u32 kv = m.num_kv_heads * m.head_dim;

    /* Build a realistic test input */
    float *in_h = malloc(H * sizeof(float));
    float *in_f = malloc(F * sizeof(float));
    memcpy(in_h, m.embedding + (size_t)128000 * H, H * sizeof(float));
    /* Normalize */
    float ss = 0;
    for (u32 i = 0; i < H; i++) ss += in_h[i] * in_h[i];
    float sc = 1.0f / sqrtf(ss / H + 1e-5f);
    for (u32 i = 0; i < H; i++) in_h[i] *= sc * m.layers[0].attn_norm[i];
    for (u32 i = 0; i < F; i++) in_f[i] = (float)(i % 100) / 100.0f - 0.5f;

    float *ko = NULL, *so = NULL;
    int all_pass = 1;

    printf("Layer 0 projection correctness (kernel vs scalar):\n");

    /* Q: H x H */
    ko = malloc(H * sizeof(float)); so = malloc(H * sizeof(float));
    hs_ml_ternary_f32_proj(ko, in_h, m.layers[0].q_proj, H, H);
    scalar_proj(so, in_h, m.layers[0].q_proj, H, H);
    all_pass &= check(ko, so, H, "q_proj [2560x2560]");
    free(ko); free(so);

    /* K: kv x H */
    ko = malloc(kv * sizeof(float)); so = malloc(kv * sizeof(float));
    hs_ml_ternary_f32_proj(ko, in_h, m.layers[0].k_proj, kv, H);
    scalar_proj(so, in_h, m.layers[0].k_proj, kv, H);
    all_pass &= check(ko, so, kv, "k_proj [640x2560] ");
    free(ko); free(so);

    /* Gate: F x H */
    ko = malloc(F * sizeof(float)); so = malloc(F * sizeof(float));
    hs_ml_ternary_f32_proj(ko, in_h, m.layers[0].gate_proj, F, H);
    scalar_proj(so, in_h, m.layers[0].gate_proj, F, H);
    all_pass &= check(ko, so, F, "gate_proj [6912x2560]");
    free(ko); free(so);

    /* Down: H x F */
    ko = malloc(H * sizeof(float)); so = malloc(H * sizeof(float));
    hs_ml_ternary_f32_proj(ko, in_f, m.layers[0].down_proj, H, F);
    scalar_proj(so, in_f, m.layers[0].down_proj, H, F);
    all_pass &= check(ko, so, H, "down_proj [2560x6912]");
    free(ko); free(so);

    free(in_h); free(in_f);
    hs_mlt_free(&m);

    printf("\n%s\n", all_pass ? "ALL PASS" : "FAILURES DETECTED");
    return all_pass ? 0 : 1;
}
