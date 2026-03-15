/*
 * NeoGPU ML - GGUF loader end-to-end test
 *
 * 1. Write a test GGUF (via tools/write_test_gguf.py)
 * 2. Load it with hs_mlt_load_gguf
 * 3. Run prefill + decode
 * 4. Verify logits are finite and non-trivial
 * 5. Generate 8 tokens greedily
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "hs_ml_infer.h"

static int failures = 0;
#define CHECK(cond, msg) do {     if (!(cond)) { printf("  [FAIL] %s\n", (msg)); failures++; }     else         { printf("  [PASS] %s\n", (msg)); } } while(0)

static int logits_finite(const float* v, u32 n) {
    for (u32 i=0;i<n;i++) if (!isfinite(v[i])) return 0;
    return 1;
}
static int logits_nonzero(const float* v, u32 n) {
    for (u32 i=0;i<n;i++) if (v[i] != 0.0f) return 1;
    return 0;
}

int main(int argc, char* argv[]) {
    const char* gguf_path = argc > 1 ? argv[1] : "/tmp/test_neogpu.gguf";

    printf("NeoGPU ML - GGUF Loader Test\n");
    printf("=============================\n");
    printf("Loading: %s\n\n", gguf_path);

    /* ── Load model ── */
    HSMLTernary m;
    hs_mlt_init(&m);

    int rc = hs_mlt_load_gguf(&m, gguf_path);
    CHECK(rc == 0,   "hs_mlt_load_gguf returns 0");
    CHECK(m.loaded,  "model.loaded is true");
    if (rc != 0) { printf("Cannot continue without model.\n"); return 1; }

    printf("  Model: hidden=%u layers=%u heads=%u ffn=%u vocab=%u ctx=%u\n",
           m.hidden_size, m.num_layers, m.num_heads,
           m.ffn_hidden_size, m.vocab_size, m.max_context);

    CHECK(m.hidden_size > 0,     "hidden_size > 0");
    CHECK(m.num_layers > 0,      "num_layers > 0");
    CHECK(m.layers != NULL,      "layers allocated");
    CHECK(m.embedding != NULL,   "embedding allocated");
    CHECK(m.layers[0].q_proj != NULL, "q_proj allocated");

    /* ── Verify ternary packing correctness ── */
    /* Random ternary weights should produce only {00, 01, 10} nibbles */
    {
        u8* w = m.layers[0].q_proj;
        u32 K4 = m.hidden_size / 4;
        int bad = 0;
        for (u32 i = 0; i < K4; i++) {
            for (int b = 0; b < 4; b++) {
                u8 v = (w[i] >> (b*2)) & 3;
                if (v == 3) { bad++; break; }  /* 11 is invalid ternary */
            }
        }
        CHECK(bad == 0, "packed ternary has no invalid codes (no 11 nibbles)");
    }

    /* ── Session: prefill + decode ── */
    printf("\n=== Inference test ===\n");
    HSMLTernarySession sess;
    rc = hs_mlt_session_init(&sess, &m);
    CHECK(rc == 0, "session_init returns 0");

    u32 prompt[] = {1, 5, 12, 3};   /* BOS + 3 tokens */
    rc = hs_mlt_prefill(&sess, prompt, 4);
    CHECK(rc == 0, "prefill 4 tokens returns 0");
    CHECK(sess.seq_len == 4, "seq_len == 4 after prefill");

    float* logits = malloc(m.vocab_size * sizeof(float));
    rc = hs_mlt_decode(&sess, 7, logits);
    CHECK(rc == 0,                       "decode returns 0");
    CHECK(logits_finite(logits, m.vocab_size), "logits are finite");
    CHECK(logits_nonzero(logits, m.vocab_size), "logits are not all zero");
    CHECK(sess.seq_len == 5,             "seq_len == 5 after decode");

    /* ── Greedy generation ── */
    printf("\n=== Greedy generation (8 tokens) ===\n");
    hs_mlt_session_reset(&sess);
    hs_mlt_prefill(&sess, prompt, 4);

    printf("  Prompt tokens: ");
    for (int i=0;i<4;i++) printf("%u ", prompt[i]);
    printf("\n  Generated:     ");

    u32 tok = 7;
    int all_valid = 1;
    for (int step = 0; step < 8; step++) {
        hs_mlt_decode(&sess, tok, logits);
        tok = hs_mlt_sample_greedy(logits, m.vocab_size);
        if (tok >= m.vocab_size) { all_valid = 0; }
        printf("%u ", tok);
    }
    printf("\n");
    CHECK(all_valid, "all generated tokens in vocab range");

    free(logits);
    hs_mlt_session_free(&sess);
    hs_mlt_free(&m);

    printf("\n=============================\n");
    printf("%s\n", failures==0 ? "All tests passed." : "FAILURES DETECTED.");
    return failures;
}
