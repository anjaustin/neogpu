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
    HSMLTernary m;
    hs_mlt_init(&m);
    if (hs_mlt_load_gguf(&m, "/home/ztflynn/001/neogpu/models/ggml-model-i2_s.gguf"))
        return 1;

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
