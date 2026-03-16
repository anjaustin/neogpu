#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "hs_ml_infer.h"

/* Our new kernel */
extern void hs_ml_ternary_f32_proj(float *out, const float *in,
                                    const uint8_t *W, u32 N, u32 K);

/* Scalar reference - known correct */
static float scalar_dot(const float *in, const uint8_t *wrow, u32 K) {
    float acc = 0;
    u32 row_bytes = K / 4;
    for (u32 bi = 0; bi < row_bytes; bi++) {
        uint8_t b = wrow[bi];
        for (int s = 0; s < 8; s += 2) {
            float w = (float)((int8_t)((b >> s) & 3) - 1);
            acc += w * in[bi * 4 + s / 2];
        }
    }
    return acc;
}

int main(void) {
    HSMLTernary m; hs_mlt_init(&m);
    if (hs_mlt_load_gguf(&m, "/home/ztflynn/001/neogpu/models/ggml-model-i2_s.gguf")) return 1;
    u32 H = m.hidden_size;

    float *in = malloc(H * sizeof(float));
    memcpy(in, m.embedding + (size_t)128000 * H, H * sizeof(float));

    float ss = 0;
    for (u32 i = 0; i < H; i++) ss += in[i]*in[i];
    float sc = 1.0f / sqrtf(ss/H + 1e-5f);
    for (u32 i = 0; i < H; i++) in[i] *= sc * m.layers[0].attn_norm[i];

    /* Scalar reference for first 5 outputs */
    u32 row_bytes = H / 4;
    for (int r = 0; r < 5; r++) {
        float ref = scalar_dot(in, m.layers[0].q_proj + (size_t)r * row_bytes, H);
        printf("scalar row %d: %.6f\n", r, ref);
    }

    /* Kernel output for first 5 */
    float *out = malloc(H * sizeof(float));
    hs_ml_ternary_f32_proj(out, in, m.layers[0].q_proj, H, H);
    for (int r = 0; r < 5; r++) printf("kernel row %d: %.6f\n", r, out[r]);

    free(in); free(out); hs_mlt_free(&m); return 0;
}
