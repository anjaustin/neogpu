#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "hs_ml_infer.h"
int main(void) {
    HSMLTernary m; hs_mlt_init(&m);
    if (hs_mlt_load_gguf(&m, "/home/ztflynn/001/neogpu/models/ggml-model-i2_s.gguf")) return 1;
    u32 H = m.hidden_size;
    float *in = malloc(H * sizeof(float));
    memcpy(in, m.embedding + (size_t)128000 * H, H * sizeof(float));
    printf("embed[0:5]: %.8f %.8f %.8f %.8f %.8f\n", in[0],in[1],in[2],in[3],in[4]);
    float ss = 0;
    for (u32 i = 0; i < H; i++) ss += in[i]*in[i];
    float sc = 1.0f / sqrtf(ss/H + 1e-5f);
    for (u32 i = 0; i < H; i++) in[i] *= sc * m.layers[0].attn_norm[i];
    printf("normed[0:5]: %.8f %.8f %.8f %.8f %.8f\n", in[0],in[1],in[2],in[3],in[4]);
    printf("normed absmax: ");
    float mx=0; for(u32 i=0;i<H;i++) if(fabsf(in[i])>mx) mx=fabsf(in[i]);
    printf("%.8f\n", mx);
    /* Expected from Python: normed absmax=0.081101 */
    free(in); hs_mlt_free(&m); return 0;
}
