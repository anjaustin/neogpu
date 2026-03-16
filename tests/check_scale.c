#include <stdio.h>
#include "hs_ml_infer.h"
int main(void) {
    HSMLTernary m; hs_mlt_init(&m);
    if (hs_mlt_load_gguf(&m, "/home/ztflynn/001/neogpu/models/ggml-model-i2_s.gguf")) return 1;
    printf("q_scale[0]=%f  k_scale[0]=%f\n", m.layers[0].q_scale[0], m.layers[0].k_scale[0]);
    hs_mlt_free(&m); return 0;
}
