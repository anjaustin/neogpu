/*
 * Red-team + benchmark: MT7 NEON-optimized kernel vs original
 *
 * 1. Correctness: NEON output == original output (bit-for-bit on decode, near-exact on rmsnorm)
 * 2. Throughput: NEON vs original on real model sizes
 * 3. Cache behavior: sustained throughput over many iterations
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

/* Original MT7 (hs_ml_mt7.c) */
extern void mt7_encode_f32(uint8_t *out, const float *in, uint32_t n);
extern void mt7_decode_to_f32(float *out, const uint8_t *in, uint32_t n);
extern void mt7_rmsnorm(float *out, const float *in, const uint8_t *mt7_weights,
                        float eps, uint32_t n);
extern uint32_t mt7_storage_bytes(uint32_t n);

/* NEON-optimized MT7 (hs_ml_mt7_neon.c) */
extern void mt7_decode_neon(float *out, const uint8_t *in, uint32_t n);
extern void mt7_rmsnorm_neon(float *out, const float *in, const uint8_t *mt7_weights,
                             float eps, uint32_t n);

static double ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  [FAIL] %s\n", (msg)); failures++; } \
    else         { printf("  [PASS] %s\n", (msg)); } \
} while(0)

int main(void) {
    printf("Red-team: MT7 NEON kernel vs original\n");
    printf("======================================\n");

    uint32_t sizes[] = {256, 2560, 6912};
    int nsizes = 3;

    for (int s = 0; s < nsizes; s++) {
        uint32_t n = sizes[s];
        printf("\n--- n=%u ---\n", n);

        float *weights = malloc(n * sizeof(float));
        uint8_t *mt7 = malloc(mt7_storage_bytes(n));
        float *dec_orig = malloc(n * sizeof(float));
        float *dec_neon = malloc(n * sizeof(float));
        float *input = malloc(n * sizeof(float));
        float *norm_orig = malloc(n * sizeof(float));
        float *norm_neon = malloc(n * sizeof(float));

        srand(42 + s);
        for (uint32_t i = 0; i < n; i++) {
            weights[i] = ((float)(rand() % 2000) - 1000) / 500.0f;
            input[i] = ((float)(rand() % 2000) - 1000) / 500.0f;
        }
        mt7_encode_f32(mt7, weights, n);

        /* 1. Decode correctness */
        mt7_decode_to_f32(dec_orig, mt7, n);
        mt7_decode_neon(dec_neon, mt7, n);

        int dec_match = 1;
        for (uint32_t i = 0; i < n; i++) {
            if (dec_orig[i] != dec_neon[i]) { dec_match = 0; break; }
        }
        CHECK(dec_match, "decode: NEON == original (bit-exact)");

        /* 2. RMSNorm correctness */
        mt7_rmsnorm(norm_orig, input, mt7, 1e-5f, n);
        mt7_rmsnorm_neon(norm_neon, input, mt7, 1e-5f, n);

        float max_diff = 0;
        for (uint32_t i = 0; i < n; i++) {
            float d = fabsf(norm_orig[i] - norm_neon[i]);
            if (d > max_diff) max_diff = d;
        }
        CHECK(max_diff < 1e-5f, "rmsnorm: NEON matches original (max_diff < 1e-5)");
        if (max_diff > 0) printf("    max_diff = %e\n", max_diff);

        /* 3. Decode throughput */
        int iters = 100000;
        double t0, elapsed;

        /* warmup */
        for (int i = 0; i < 100; i++) mt7_decode_to_f32(dec_orig, mt7, n);
        for (int i = 0; i < 100; i++) mt7_decode_neon(dec_neon, mt7, n);

        t0 = ns();
        for (int i = 0; i < iters; i++) mt7_decode_to_f32(dec_orig, mt7, n);
        double us_orig_dec = (ns() - t0) / (iters * 1000.0);

        t0 = ns();
        for (int i = 0; i < iters; i++) mt7_decode_neon(dec_neon, mt7, n);
        double us_neon_dec = (ns() - t0) / (iters * 1000.0);

        printf("  decode: orig=%.3f us  neon=%.3f us  speedup=%.2fx\n",
               us_orig_dec, us_neon_dec, us_orig_dec / us_neon_dec);

        /* 4. RMSNorm throughput */
        for (int i = 0; i < 100; i++) mt7_rmsnorm(norm_orig, input, mt7, 1e-5f, n);
        for (int i = 0; i < 100; i++) mt7_rmsnorm_neon(norm_neon, input, mt7, 1e-5f, n);

        t0 = ns();
        for (int i = 0; i < iters; i++) mt7_rmsnorm(norm_orig, input, mt7, 1e-5f, n);
        double us_orig_norm = (ns() - t0) / (iters * 1000.0);

        t0 = ns();
        for (int i = 0; i < iters; i++) mt7_rmsnorm_neon(norm_neon, input, mt7, 1e-5f, n);
        double us_neon_norm = (ns() - t0) / (iters * 1000.0);

        printf("  rmsnorm: orig=%.3f us  neon=%.3f us  speedup=%.2fx\n",
               us_orig_norm, us_neon_norm, us_orig_norm / us_neon_norm);
        printf("  throughput: %.0f M values/s (neon)\n", n / us_neon_norm);

        free(weights); free(mt7); free(dec_orig); free(dec_neon);
        free(input); free(norm_orig); free(norm_neon);
    }

    /* 5. Sustained cache test: 1M iterations at n=2560 */
    printf("\n--- Sustained cache test (n=2560, 1M iters) ---\n");
    {
        uint32_t n = 2560;
        float *weights = malloc(n * sizeof(float));
        uint8_t *mt7 = malloc(mt7_storage_bytes(n));
        float *input = malloc(n * sizeof(float));
        float *output = malloc(n * sizeof(float));
        srand(77);
        for (uint32_t i = 0; i < n; i++) {
            weights[i] = ((float)(rand() % 2000) - 1000) / 500.0f;
            input[i] = ((float)(rand() % 2000) - 1000) / 500.0f;
        }
        mt7_encode_f32(mt7, weights, n);

        int iters = 1000000;
        /* Warmup */
        for (int i = 0; i < 1000; i++) mt7_rmsnorm_neon(output, input, mt7, 1e-5f, n);

        double t0 = ns();
        for (int i = 0; i < iters; i++) mt7_rmsnorm_neon(output, input, mt7, 1e-5f, n);
        double us = (ns() - t0) / (iters * 1000.0);

        printf("  sustained: %.3f us/call  %.0f M values/s\n", us, n / us);
        printf("  LUT cache behavior: %s\n",
               us < 10.0 ? "HOT (L1 resident)" : "COLD (cache thrashing)");

        free(weights); free(mt7); free(input); free(output);
    }

    printf("\n======================================\n");
    printf("%s\n", failures == 0 ? "All NEON MT7 checks passed." : "FAILURES.");
    return failures;
}
