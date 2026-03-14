/*
 * NeoGPU ML - Stateful Decode Session Tests
 *
 * Correctness:
 *   1. Session init/free lifecycle
 *   2. Single decode step produces finite logits
 *   3. Prefill + decode matches hs_mlt_forward (same seed, same tokens)
 *   4. Session reset clears state, second run matches first
 *   5. Multi-step decode: each step advances seq_len
 *
 * Benchmark:
 *   - Decode tokens/sec at ctx=1, 64, 256, 512
 *   - Prefill throughput (tokens/sec)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "hs_ml_infer.h"

static double get_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  [FAIL] %s\n", (msg)); failures++; } \
    else         { printf("  [PASS] %s\n", (msg)); } \
} while(0)

static int logits_close(const float* a, const float* b, u32 n, float tol) {
    for (u32 i = 0; i < n; i++)
        if (fabsf(a[i] - b[i]) > tol) return 0;
    return 1;
}

static int logits_finite(const float* v, u32 n) {
    for (u32 i = 0; i < n; i++) if (!isfinite(v[i])) return 0;
    return 1;
}

/* ------------------------------------------------------------------ */
static void test_lifecycle(void) {
    printf("\n=== Test: Session lifecycle ===\n");
    HSMLTernary m;
    hs_mlt_init(&m);
    hs_mlt_alloc_random(&m, 64, 64, 2, 4, 128, 64, 1);

    HSMLTernarySession sess;
    int rc = hs_mlt_session_init(&sess, &m);
    CHECK(rc == 0,        "session_init returns 0");
    CHECK(sess.ready,     "session.ready is true");
    CHECK(sess.hidden != NULL, "hidden allocated");
    CHECK(sess.caches != NULL, "caches allocated");
    CHECK(sess.seq_len == 0,   "seq_len starts at 0");

    hs_mlt_session_free(&sess);
    CHECK(!sess.ready,       "session.ready false after free");
    CHECK(sess.hidden == NULL, "hidden NULL after free");
    CHECK(sess.caches == NULL, "caches NULL after free");

    hs_mlt_free(&m);
}

/* ------------------------------------------------------------------ */
static void test_decode_finite(void) {
    printf("\n=== Test: Decode produces finite logits ===\n");
    HSMLTernary m;
    hs_mlt_init(&m);
    hs_mlt_alloc_random(&m, 64, 64, 2, 4, 128, 64, 2);

    HSMLTernarySession sess;
    hs_mlt_session_init(&sess, &m);

    float* logits = malloc(64 * sizeof(float));
    int rc = hs_mlt_decode(&sess, 1, logits);

    CHECK(rc == 0,                    "decode returns 0");
    CHECK(logits_finite(logits, 64),  "logits are finite");
    CHECK(sess.seq_len == 1,          "seq_len == 1 after one decode");

    free(logits);
    hs_mlt_session_free(&sess);
    hs_mlt_free(&m);
}

/* ------------------------------------------------------------------ */
static void test_prefill_decode_vs_forward(void) {
    printf("\n=== Test: prefill+decode matches hs_mlt_forward ===\n");
    /*
     * hs_mlt_forward processes all tokens including the last one,
     * and the last hidden state drives logits.
     *
     * prefill(tokens[0..n-2]) + decode(tokens[n-1]) should give the
     * same logits as forward(tokens[0..n-1]).
     *
     * Note: hs_mlt_forward re-initialises KV caches from empty each call,
     * so the comparison is exact if both start from position 0.
     */
    u32 V = 64, H = 64, L = 2, nh = 4, F = 128, ctx = 32;
    HSMLTernary m;
    hs_mlt_init(&m);
    hs_mlt_alloc_random(&m, V, H, L, nh, F, ctx, 99);

    u32 seq_len = 5;
    u32 tokens[] = {1, 3, 7, 2, 5};

    /* Reference: hs_mlt_forward over all 5 tokens */
    float* ref_logits = malloc(V * sizeof(float));
    hs_mlt_forward(&m, tokens, seq_len, ref_logits);

    /* Session: prefill first 4, decode last */
    HSMLTernarySession sess;
    hs_mlt_session_init(&sess, &m);

    hs_mlt_prefill(&sess, tokens, seq_len - 1);
    CHECK(sess.seq_len == seq_len - 1, "seq_len after prefill correct");

    float* sess_logits = malloc(V * sizeof(float));
    hs_mlt_decode(&sess, tokens[seq_len - 1], sess_logits);
    CHECK(sess.seq_len == seq_len, "seq_len after decode correct");

    /* Logits should match (same computation, same weights) */
    CHECK(logits_close(sess_logits, ref_logits, V, 1e-3f),
          "session logits match hs_mlt_forward");

    free(ref_logits);
    free(sess_logits);
    hs_mlt_session_free(&sess);
    hs_mlt_free(&m);
}

/* ------------------------------------------------------------------ */
static void test_reset(void) {
    printf("\n=== Test: Session reset produces same output ===\n");
    u32 V = 64, H = 64, L = 2, nh = 4, F = 128;
    HSMLTernary m;
    hs_mlt_init(&m);
    hs_mlt_alloc_random(&m, V, H, L, nh, F, 64, 7);

    HSMLTernarySession sess;
    hs_mlt_session_init(&sess, &m);

    u32 tokens[] = {2, 4, 6};
    float* l1 = malloc(V * sizeof(float));
    float* l2 = malloc(V * sizeof(float));

    /* First run */
    hs_mlt_prefill(&sess, tokens, 2);
    hs_mlt_decode(&sess, tokens[2], l1);
    u32 sl1 = sess.seq_len;

    /* Reset + second run */
    hs_mlt_session_reset(&sess);
    CHECK(sess.seq_len == 0, "seq_len 0 after reset");
    hs_mlt_prefill(&sess, tokens, 2);
    hs_mlt_decode(&sess, tokens[2], l2);
    u32 sl2 = sess.seq_len;

    CHECK(sl1 == sl2, "seq_len matches after reset");
    CHECK(logits_close(l1, l2, V, 1e-4f), "logits identical after reset");

    free(l1); free(l2);
    hs_mlt_session_free(&sess);
    hs_mlt_free(&m);
}

/* ------------------------------------------------------------------ */
static void test_multi_step_decode(void) {
    printf("\n=== Test: Multi-step decode advances correctly ===\n");
    u32 V = 64, H = 64, L = 2, nh = 4, F = 128;
    HSMLTernary m;
    hs_mlt_init(&m);
    hs_mlt_alloc_random(&m, V, H, L, nh, F, 64, 42);

    HSMLTernarySession sess;
    hs_mlt_session_init(&sess, &m);

    float* logits = malloc(V * sizeof(float));
    u32 tok = 1;
    int all_finite = 1, all_valid = 1;

    for (int step = 0; step < 8; step++) {
        hs_mlt_decode(&sess, tok, logits);
        tok = hs_mlt_sample_greedy(logits, V);
        if (!logits_finite(logits, V)) all_finite = 0;
        if (tok >= V) all_valid = 0;
    }

    CHECK(sess.seq_len == 8,  "seq_len == 8 after 8 decode steps");
    CHECK(all_finite,          "all decode logits are finite");
    CHECK(all_valid,           "all sampled tokens in vocab range");

    free(logits);
    hs_mlt_session_free(&sess);
    hs_mlt_free(&m);
}

/* ------------------------------------------------------------------ */
static void bench_decode_speed(void) {
    printf("\n=== Benchmark: Decode tokens/sec by context depth ===\n");
    u32 V = 512, H = 512, L = 4, nh = 8, hd = 64, F = 1024;
    HSMLTernary m;
    hs_mlt_init(&m);
    hs_mlt_alloc_random(&m, V, H, L, nh, F, 2048, 42);

    u32 ctx_depths[] = {1, 16, 64, 128, 256, 512, 1024};
    int NC = 7;

    printf("  %-8s  %-10s  %-10s  %-10s\n",
           "ctx", "ms/token", "tok/sec", "GEMM%");

    for (int ci = 0; ci < NC; ci++) {
        u32 ctx = ctx_depths[ci];
        if (ctx >= m.max_context) continue;

        HSMLTernarySession sess;
        hs_mlt_session_init(&sess, &m);

        /* Build up context */
        u32* prompt = malloc(ctx * sizeof(u32));
        for (u32 i = 0; i < ctx; i++) prompt[i] = i % V;
        hs_mlt_prefill(&sess, prompt, ctx);
        free(prompt);

        float* logits = malloc(V * sizeof(float));

        /* Warmup */
        hs_mlt_decode(&sess, 0, logits);
        hs_mlt_session_reset(&sess);
        if (ctx > 0) {
            u32* p2 = malloc(ctx * sizeof(u32));
            for (u32 i = 0; i < ctx; i++) p2[i] = i % V;
            hs_mlt_prefill(&sess, p2, ctx);
            free(p2);
        }

        /* Benchmark: 10 decode steps */
        int STEPS = 10;
        hs_mlt_reset_stats(&m);
        double t0 = get_ns();
        u32 tok = 1;
        for (int s = 0; s < STEPS; s++) {
            hs_mlt_decode(&sess, tok, logits);
            tok = hs_mlt_sample_greedy(logits, V);
        }
        double elapsed_ms = (get_ns() - t0) / 1e6;
        double ms_per_tok = elapsed_ms / STEPS;
        double tok_per_sec = 1000.0 / ms_per_tok;
        double gemm_pct = g_mlt_stats.total_ns / (elapsed_ms * 1e4);

        printf("  %-8u  %-10.2f  %-10.1f  %-10.1f%%\n",
               ctx, ms_per_tok, tok_per_sec, gemm_pct);

        free(logits);
        hs_mlt_session_free(&sess);
    }

    hs_mlt_free(&m);
}

/* ------------------------------------------------------------------ */
static void bench_prefill_speed(void) {
    printf("\n=== Benchmark: Prefill throughput ===\n");
    u32 V = 512, H = 512, L = 4, nh = 8, F = 1024;
    HSMLTernary m;
    hs_mlt_init(&m);
    hs_mlt_alloc_random(&m, V, H, L, nh, F, 2048, 42);

    u32 lens[] = {1, 8, 32, 128, 512};
    printf("  %-8s  %-12s  %-12s\n", "tokens", "ms total", "tok/sec");
    for (int li = 0; li < 5; li++) {
        u32 n = lens[li];
        if (n >= m.max_context) continue;
        u32* toks = malloc(n * sizeof(u32));
        for (u32 i = 0; i < n; i++) toks[i] = i % V;

        HSMLTernarySession sess;
        hs_mlt_session_init(&sess, &m);

        /* Warmup */
        hs_mlt_prefill(&sess, toks, n);
        hs_mlt_session_reset(&sess);

        int REPS = 5;
        double t0 = get_ns();
        for (int r = 0; r < REPS; r++) {
            hs_mlt_prefill(&sess, toks, n);
            hs_mlt_session_reset(&sess);
        }
        double ms = (get_ns() - t0) / (REPS * 1e6);
        printf("  %-8u  %-12.2f  %-12.1f\n", n, ms, n * 1000.0 / ms);

        free(toks);
        hs_mlt_session_free(&sess);
    }
    hs_mlt_free(&m);
}

/* ------------------------------------------------------------------ */
int main(void) {
    printf("NeoGPU ML - Stateful Decode Session Tests\n");
    printf("==========================================\n");

    test_lifecycle();
    test_decode_finite();
    test_prefill_decode_vs_forward();
    test_reset();
    test_multi_step_decode();

    printf("\n==========================================\n");
    printf("%s\n", failures == 0 ? "All tests passed." : "FAILURES DETECTED.");

    bench_decode_speed();
    bench_prefill_speed();

    return failures;
}
