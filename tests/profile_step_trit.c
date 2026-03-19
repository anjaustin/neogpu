// Quick test: profile with ternary lm_head
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "hs_ml_infer.h"

int main(int argc, char **argv) {
    const char *model = argv[1] ? argv[1] : "models/bitnet-2b4t-i2s.gguf";
    const char *norms = argv[2] ? argv[2] : "models/norms_v2.bin";
    
    HSMLTernary m;
    hs_mlt_init(&m);
    hs_mlt_load_gguf(&m, model);
    hs_mlt_load_norms_sidecar(&m, norms);
    
    // Encode to ternary - THIS IS THE KEY
    hs_mlt_lmhead_encode(&m);
    printf("use_trit_lmhead=%d\n", m.use_trit_lmhead);
    
    // Profile decode
    uint32_t tokens[16] = {m.tokenizer_bos};
    tokens[1] = 39; // 'H'
    HSMLTernarySession s;
    hs_mlt_session_init(&s, &m);
    hs_mlt_prefill(&s, tokens, 2);
    
    float *logits = malloc(m.vocab_size * 4);
    
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    hs_mlt_decode(&s, 1100, logits);  // 'ello'
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("decode step with ternary lm_head: %.1f ms\n", ms);
    
    free(logits);
    hs_mlt_session_free(&s);
    hs_mlt_free(&m);
    return 0;
}
