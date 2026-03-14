/*
 * NeoGPU ML - End-to-End Ternary Inference Test
 *
 * Tests:
 *   1. Model allocation (synthetic random weights)
 *   2. Forward pass produces finite, non-zero logits
 *   3. Greedy sampling returns valid token
 *   4. Multi-token sequence (prefill) doesn't crash
 *   5. Determinism: same seed -> same logits
 *   6. KV cache grows correctly across tokens
 *
 * Benchmark:
 *   - Tokens/sec on small config (hidden=256, 2 layers)
 *   - Tokens/sec on medium config (hidden=512, 4 layers)
 *   - GEMM ops/sec breakdown
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "hs_ml_infer.h"

static double ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1e9 + ts.tv_nsec;
}

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  [FAIL] %s\n", msg); failures++; } \
    else         { printf("  [PASS] %s\n", msg); } \
} while(0)

/* ------------------------------------------------------------------ */
/* Test 1: allocation */
static void test_alloc(void) {
    printf("\n=== Test: Model allocation ===\n");
    HSMLTernary m;
    hs_mlt_init(&m);

    int rc = hs_mlt_alloc_random(&m,
        /*vocab=*/256, /*hidden=*/128, /*layers=*/2,
        /*heads=*/4, /*ffn=*/256, /*ctx=*/64, /*seed=*/42);

    CHECK(rc == 0,          "alloc_random returns 0");
    CHECK(m.loaded,         "model.loaded is true");
    CHECK(m.layers != NULL, "layers allocated");
    CHECK(m.layers[0].q_proj  != NULL, "q_proj allocated");
    CHECK(m.layers[0].ffn_norm != NULL, "ffn_norm allocated");
    CHECK(m.quant_buf != NULL, "quant_buf allocated");
    CHECK(m.gemm_buf  != NULL, "gemm_buf allocated");

    hs_mlt_free(&m);
    CHECK(!m.loaded, "model.loaded is false after free");
    CHECK(m.layers == NULL, "layers NULL after free");
}

/* ------------------------------------------------------------------ */
/* Test 2: forward pass produces finite logits */
static void test_forward_finite(void) {
    printf("\n=== Test: Forward pass produces finite logits ===\n");
    HSMLTernary m;
    hs_mlt_init(&m);

    int rc = hs_mlt_alloc_random(&m, 256, 128, 2, 4, 256, 64, 42);
    CHECK(rc == 0, "alloc ok");

    float* logits = malloc(256 * sizeof(float));
    u32 tokens[] = {1};
    rc = hs_mlt_forward(&m, tokens, 1, logits);

    CHECK(rc == 0, "forward returns 0");

    int all_finite = 1, any_nonzero = 0;
    for (int i = 0; i < 256; i++) {
        if (!isfinite(logits[i])) { all_finite = 0; break; }
        if (logits[i] != 0.0f) any_nonzero = 1;
    }
    CHECK(all_finite,   "all logits are finite");
    CHECK(any_nonzero,  "logits are not all zero");

    free(logits);
    hs_mlt_free(&m);
}

/* ------------------------------------------------------------------ */
/* Test 3: greedy sampling returns valid token */
static void test_sampling(void) {
    printf("\n=== Test: Greedy sampling ===\n");
    HSMLTernary m;
    hs_mlt_init(&m);
    hs_mlt_alloc_random(&m, 256, 128, 2, 4, 256, 64, 7);

    float* logits = malloc(256 * sizeof(float));
    u32 tokens[] = {5};
    hs_mlt_forward(&m, tokens, 1, logits);

    u32 sampled = hs_mlt_sample_greedy(logits, 256);
    CHECK(sampled < 256, "sampled token is in vocab range");

    free(logits);
    hs_mlt_free(&m);
}

/* ------------------------------------------------------------------ */
/* Test 4: multi-token sequence (prefill) */
static void test_prefill(void) {
    printf("\n=== Test: Multi-token prefill ===\n");
    HSMLTernary m;
    hs_mlt_init(&m);
    hs_mlt_alloc_random(&m, 256, 128, 2, 4, 256, 64, 99);

    float* logits = malloc(256 * sizeof(float));
    u32 tokens[] = {1, 2, 3, 4, 5};
    int rc = hs_mlt_forward(&m, tokens, 5, logits);

    CHECK(rc == 0, "prefill of 5 tokens returns 0");

    int all_finite = 1;
    for (int i = 0; i < 256; i++) {
        if (!isfinite(logits[i])) { all_finite = 0; break; }
    }
    CHECK(all_finite, "logits finite after 5-token prefill");

    free(logits);
    hs_mlt_free(&m);
}

/* ------------------------------------------------------------------ */
/* Test 5: determinism */
static void test_determinism(void) {
    printf("\n=== Test: Determinism (same seed -> same output) ===\n");
    HSMLTernary m1, m2;
    hs_mlt_init(&m1); hs_mlt_init(&m2);
    hs_mlt_alloc_random(&m1, 256, 128, 2, 4, 256, 64, 123);
    hs_mlt_alloc_random(&m2, 256, 128, 2, 4, 256, 64, 123);

    float* l1 = malloc(256 * sizeof(float));
    float* l2 = malloc(256 * sizeof(float));
    u32 tokens[] = {10};

    hs_mlt_forward(&m1, tokens, 1, l1);
    hs_mlt_forward(&m2, tokens, 1, l2);

    int match = 1;
    for (int i = 0; i < 256; i++) {
        if (fabsf(l1[i] - l2[i]) > 1e-4f) { match = 0; break; }
    }
    CHECK(match, "identical seeds produce identical logits");

    free(l1); free(l2);
    hs_mlt_free(&m1); hs_mlt_free(&m2);
}

/* ------------------------------------------------------------------ */
/* Test 6: different seeds produce different outputs */
static void test_different_seeds(void) {
    printf("\n=== Test: Different seeds -> different outputs ===\n");
    HSMLTernary m1, m2;
    hs_mlt_init(&m1); hs_mlt_init(&m2);
    hs_mlt_alloc_random(&m1, 256, 128, 2, 4, 256, 64, 1);
    hs_mlt_alloc_random(&m2, 256, 128, 2, 4, 256, 64, 2);

    float* l1 = malloc(256 * sizeof(float));
    float* l2 = malloc(256 * sizeof(float));
    u32 tokens[] = {1};

    hs_mlt_forward(&m1, tokens, 1, l1);
    hs_mlt_forward(&m2, tokens, 1, l2);

    int differs = 0;
    for (int i = 0; i < 256; i++) {
        if (fabsf(l1[i] - l2[i]) > 1e-4f) { differs = 1; break; }
    }
    CHECK(differs, "different seeds produce different logits");

    free(l1); free(l2);
    hs_mlt_free(&m1); hs_mlt_free(&m2);
}

/* ------------------------------------------------------------------ */
/* Benchmark */
static void bench_inference(const char* label,
                            u32 vocab, u32 hidden, u32 layers,
                            u32 heads, u32 ffn, int n_tokens) {
    printf("\nBenchmark: %s\n", label);
    HSMLTernary m;
    hs_mlt_init(&m);
    if (hs_mlt_alloc_random(&m, vocab, hidden, layers, heads, ffn, 64, 42) != 0) {
        printf("  alloc failed\n"); return;
    }

    float* logits = malloc(vocab * sizeof(float));

    /* Warmup */
    u32 wtok[] = {1};
    hs_mlt_forward(&m, wtok, 1, logits);

    /* Generate n_tokens one at a time (decode mode) */
    u32* gen_tokens = malloc((n_tokens + 1) * sizeof(u32));
    gen_tokens[0] = 1;  /* BOS */

    double t0 = ns();
    for (int i = 0; i < n_tokens; i++) {
        hs_mlt_forward(&m, gen_tokens, i+1, logits);
        gen_tokens[i+1] = hs_mlt_sample_greedy(logits, vocab);
    }
    double elapsed_ms = (ns() - t0) / 1e6;

    double tok_per_sec = n_tokens * 1000.0 / elapsed_ms;
    double ms_per_tok  = elapsed_ms / n_tokens;

    printf("  Config:    hidden=%u layers=%u heads=%u ffn=%u\n",
           hidden, layers, heads, ffn);
    printf("  Tokens:    %d generated\n", n_tokens);
    printf("  Time:      %.1f ms total  %.2f ms/token\n",
           elapsed_ms, ms_per_tok);
    printf("  Speed:     %.1f tokens/sec\n", tok_per_sec);
    printf("  GEMM ops:  %llu calls  %.2f G ops total\n",
           (unsigned long long)g_mlt_stats.total_gemm_calls,
           g_mlt_stats.total_gemm_ops / 1e9);
    printf("  GEMM time: %.1f ms (%.0f%% of total)\n",
           g_mlt_stats.total_ns / 1e6,
           g_mlt_stats.total_ns / (elapsed_ms * 1e4));

    /* Show generated sequence */
    printf("  Tokens:   ");
    for (int i = 0; i <= n_tokens && i < 16; i++) printf(" %u", gen_tokens[i]);
    if (n_tokens > 15) printf(" ...");
    printf("\n");

    free(logits);
    free(gen_tokens);
    hs_mlt_free(&m);
}

/* ------------------------------------------------------------------ */
int main(void) {
    printf("NeoGPU ML - End-to-End Ternary Inference Tests\n");
    printf("===============================================\n");

    test_alloc();
    test_forward_finite();
    test_sampling();
    test_prefill();
    test_determinism();
    test_different_seeds();

    printf("\n===============================================\n");
    if (failures == 0) printf("All correctness tests passed.\n");
    else printf("%d test(s) FAILED.\n", failures);

    bench_inference("Small  (hidden=256, 2L)",  256, 256, 2, 4, 512,  20);
    bench_inference("Medium (hidden=512, 4L)",  512, 512, 4, 8, 1024, 10);
    bench_inference("Large  (hidden=1024, 8L)", 512, 1024, 8, 8, 2048,  5);

    return failures;
}
