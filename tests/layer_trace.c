#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "hs_ml_infer.h"

static void statsf(const char *name, const float *x, u32 n) {
    float min = x[0], max = x[0], absmax = 0;
    double sum = 0;
    for (u32 i = 0; i < n; i++) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        sum += x[i];
        float a = fabsf(x[i]);
        if (a > absmax) absmax = a;
    }
    printf("%-24s min=%12.4g max=%12.4g mean=%12.4g absmax=%12.4g\n",
           name, min, max, sum / n, absmax);
}

int main(void) {
    const char *model_path = "/home/ztflynn/001/neogpu/models/ggml-model-i2_s.gguf";
    const char *norms_path = "/home/ztflynn/001/neogpu/models/norms_v2.bin";

    HSMLTernary m;
    hs_mlt_init(&m);
    if (hs_mlt_load_gguf(&m, model_path)) return 1;

    printf("use_i2s=%d rope_theta=%.0f\n", (int)m.use_i2s, m.rope_theta);

    /* Load real BF16 norms from sidecar */
    if (hs_mlt_load_norms_sidecar(&m, norms_path) == 0) {
        printf("norms loaded from sidecar\n");
    } else {
        printf("WARNING: sidecar not loaded — using GGUF norms (may be corrupt)\n");
    }

    /* Print a sample of norm magnitudes for sanity check */
    {
        HSTernaryLayer *l0 = &m.layers[0];
        float an_am = 0, asn_am = 0, fn_am = 0, fsn_am = 0;
        for (u32 i = 0; i < m.hidden_size; i++) {
            float a;
            a = fabsf(l0->attn_norm[i]);     if (a > an_am)  an_am  = a;
            a = fabsf(l0->attn_sub_norm[i]); if (a > asn_am) asn_am = a;
            a = fabsf(l0->ffn_norm[i]);      if (a > fn_am)  fn_am  = a;
        }
        for (u32 i = 0; i < m.ffn_hidden_size; i++) {
            float a = fabsf(l0->ffn_sub_norm[i]);
            if (a > fsn_am) fsn_am = a;
        }
        printf("layer0 norm absmax: attn=%.4f attn_sub=%.4f ffn=%.4f ffn_sub=%.4f\n",
               an_am, asn_am, fn_am, fsn_am);
    }

    u32 H = m.hidden_size;
    float *hidden = malloc(H * sizeof(float));
    u32 tok = 128000;
    memcpy(hidden, m.embedding + (size_t)tok * H, H * sizeof(float));
    statsf("embed", hidden, H);

    HSKVCache *caches = calloc(m.num_layers, sizeof(HSKVCache));
    for (u32 l = 0; l < m.num_layers; l++)
        hs_kv_cache_init(&caches[l], 64, m.num_heads, m.num_kv_heads, m.head_dim);

    for (u32 l = 0; l < m.num_layers; l++) {
        hs_mlt_layer_forward(hidden, l, &caches[l], &m);
        if (l < 5 || l % 5 == 4 || l == m.num_layers - 1) {
            char name[32];
            snprintf(name, sizeof(name), "after_layer_%u", l);
            statsf(name, hidden, H);
        }
        /* early exit if blown */
        float am = 0;
        for (u32 i = 0; i < H; i++) {
            float a = fabsf(hidden[i]);
            if (a > am) am = a;
        }
        if (am > 1e10f) {
            printf("EXPLOSION at layer %u, absmax=%g — stopping\n", l, am);
            break;
        }
    }

    for (u32 l = 0; l < m.num_layers; l++) hs_kv_cache_free(&caches[l]);
    free(caches);
    free(hidden);
    hs_mlt_free(&m);
    return 0;
}
