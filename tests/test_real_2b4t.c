#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "hs_ml_infer.h"

static int finite_all(const float *x, u32 n) {
    for (u32 i = 0; i < n; i++) {
        if (!isfinite(x[i])) return 0;
    }
    return 1;
}

int main(void) {
    const char *path       = "/home/ztflynn/001/neogpu/models/ggml-model-i2_s.gguf";
    const char *norms_path = "/home/ztflynn/001/neogpu/models/norms_v2.bin";

    HSMLTernary m;
    hs_mlt_init(&m);

    int rc = hs_mlt_load_gguf(&m, path);
    printf("load_rc=%d loaded=%d use_i2s=%d\n", rc, m.loaded, (int)m.use_i2s);
    printf("hidden=%u layers=%u q_heads=%u kv_heads=%u ffn=%u vocab=%u ctx=%u rope=%.0f\n",
           m.hidden_size, m.num_layers, m.num_heads, m.num_kv_heads,
           m.ffn_hidden_size, m.vocab_size, m.max_context, m.rope_theta);
    if (rc != 0) return 1;

    /* Load real BF16 norms */
    if (hs_mlt_load_norms_sidecar(&m, norms_path) == 0)
        printf("norms sidecar loaded\n");
    else
        printf("WARNING: norms sidecar missing, using GGUF norms\n");

    HSMLTernarySession s;
    rc = hs_mlt_session_init(&s, &m);
    printf("session_rc=%d ready=%d\n", rc, s.ready);
    if (rc != 0) return 2;

    /* Llama 3 tokenizer special ids from model metadata */
    u32 prompt[] = {128000, 3923, 374, 527, 30};
    rc = hs_mlt_prefill(&s, prompt, 5);
    printf("prefill_rc=%d seq_len=%u\n", rc, s.seq_len);
    if (rc != 0) return 3;

    float *logits = malloc(m.vocab_size * sizeof(float));
    if (!logits) return 4;

    rc = hs_mlt_decode(&s, 128001, logits);
    printf("decode_rc=%d seq_len=%u finite=%d\n", rc, s.seq_len, finite_all(logits, m.vocab_size));
    if (rc == 0) {
        u32 tok = hs_mlt_sample_greedy(logits, m.vocab_size);
        printf("greedy_token=%u\n", tok);
    }

    free(logits);
    hs_mlt_session_free(&s);
    hs_mlt_free(&m);
    return 0;
}
