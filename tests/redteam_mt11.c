/*
 * Red-team + benchmark: MT11 (17.43-bit multi-trit float)
 *
 * Compare MT11 vs MT7 vs F16 vs F32:
 *   1. Precision (MT11 should beat F16)
 *   2. Decode throughput (no LUT, pure arithmetic)
 *   3. RMSNorm throughput
 *   4. Storage
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

/* MT11 */
extern void mt11_encode_f32(uint8_t *out, const float *in, uint32_t n);
extern void mt11_decode_to_f32(float *out, const uint8_t *in, uint32_t n);
extern void mt11_decode_neon(float *out, const uint8_t *in, uint32_t n);
extern void mt11_rmsnorm_neon(float *out, const float *in, const uint8_t *w, float eps, uint32_t n);
extern uint32_t mt11_storage_bytes(uint32_t n);

/* MT7 */
extern void mt7_encode_f32(uint8_t *out, const float *in, uint32_t n);
extern void mt7_rmsnorm(float *out, const float *in, const uint8_t *w, float eps, uint32_t n);
extern uint32_t mt7_storage_bytes(uint32_t n);

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

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

static void rmsnorm_f32(float *out, const float *in, const float *w, float eps, uint32_t n) {
    float ss = 0;
#ifdef __ARM_NEON
    float32x4_t vss = vdupq_n_f32(0);
    uint32_t n4 = n & ~3u;
    for (uint32_t i = 0; i < n4; i += 4) { float32x4_t v = vld1q_f32(in+i); vss = vfmaq_f32(vss,v,v); }
    ss = vaddvq_f32(vss);
    for (uint32_t i = n4; i < n; i++) ss += in[i]*in[i];
#else
    for (uint32_t i = 0; i < n; i++) ss += in[i]*in[i];
#endif
    float sc = 1.0f / sqrtf(ss/n + eps);
    for (uint32_t i = 0; i < n; i++) out[i] = in[i] * sc * w[i];
}

int main(void) {
    printf("================================================================\n");
    printf("  MT11 vs MT7 vs F16 vs F32 — Red-team + Benchmark\n");
    printf("================================================================\n\n");

    uint32_t n = 2560;
    float *weights = malloc(n * sizeof(float));
    float *input = malloc(n * sizeof(float));
    float *ref = malloc(n * sizeof(float));
    float *out_mt7 = malloc(n * sizeof(float));
    float *out_mt11 = malloc(n * sizeof(float));
    float *decoded = malloc(n * sizeof(float));
    uint8_t *mt7_w = malloc(mt7_storage_bytes(n));
    uint8_t *mt11_w = malloc(mt11_storage_bytes(n));

    srand(42);
    for (uint32_t i = 0; i < n; i++) {
        weights[i] = ((float)(rand() % 4000) - 2000) / 1000.0f;
        input[i] = ((float)(rand() % 2000) - 1000) / 500.0f;
    }

    mt7_encode_f32(mt7_w, weights, n);
    mt11_encode_f32(mt11_w, weights, n);

    /* 1. Roundtrip precision */
    printf("=== Roundtrip precision (n=%u) ===\n\n", n);
    mt11_decode_to_f32(decoded, mt11_w, n);

    float mt11_max_err = 0, mt11_sum_err = 0;
    float mt11_max_rel = 0, mt11_sum_rel = 0;
    int mt11_rel_count = 0;
    for (uint32_t i = 0; i < n; i++) {
        float err = fabsf(weights[i] - decoded[i]);
        if (err > mt11_max_err) mt11_max_err = err;
        mt11_sum_err += err;
        if (fabsf(weights[i]) > 0.01f) {
            float rel = err / fabsf(weights[i]);
            if (rel > mt11_max_rel) mt11_max_rel = rel;
            mt11_sum_rel += rel;
            mt11_rel_count++;
        }
    }

    printf("  MT11:  max_abs=%.8f  mean_abs=%.8f  max_rel=%.4f%%  mean_rel=%.4f%%\n",
           mt11_max_err, mt11_sum_err / n, mt11_max_rel * 100, mt11_sum_rel / mt11_rel_count * 100);
    printf("  (F16 typical: max_rel ~0.05%%, mean_rel ~0.01%%)\n");
    printf("  (MT7 typical: max_rel ~1.8%%,  mean_rel ~0.5%%)\n");
    CHECK(mt11_max_rel < 0.005f, "MT11 max_rel < 0.5% (better than MT7)");
    CHECK(mt11_max_rel < 0.001f, "MT11 max_rel < 0.1% (approaching F16)");

    /* 2. NEON decode correctness */
    printf("\n=== NEON decode correctness ===\n\n");
    float *dec_scalar = malloc(n * sizeof(float));
    float *dec_neon = malloc(n * sizeof(float));
    mt11_decode_to_f32(dec_scalar, mt11_w, n);
    mt11_decode_neon(dec_neon, mt11_w, n);
    float max_diff = 0;
    for (uint32_t i = 0; i < n; i++) {
        float d = fabsf(dec_scalar[i] - dec_neon[i]);
        if (d > max_diff) max_diff = d;
    }
    CHECK(max_diff < 1e-6f, "NEON decode matches scalar (max_diff < 1e-6)");

    /* 3. RMSNorm accuracy */
    printf("\n=== RMSNorm accuracy vs F32 (n=%u) ===\n\n", n);
    rmsnorm_f32(ref, input, weights, 1e-5f, n);
    mt7_rmsnorm(out_mt7, input, mt7_w, 1e-5f, n);
    mt11_rmsnorm_neon(out_mt11, input, mt11_w, 1e-5f, n);

    float ref_absmax = 0;
    for (uint32_t i = 0; i < n; i++) { float a = fabsf(ref[i]); if (a > ref_absmax) ref_absmax = a; }

    float mt7_me = 0, mt11_me = 0, mt7_se = 0, mt11_se = 0;
    for (uint32_t i = 0; i < n; i++) {
        float e7 = fabsf(ref[i] - out_mt7[i]);
        float e11 = fabsf(ref[i] - out_mt11[i]);
        if (e7 > mt7_me) mt7_me = e7;
        if (e11 > mt11_me) mt11_me = e11;
        mt7_se += e7; mt11_se += e11;
    }

    printf("  ref absmax: %.4f\n", ref_absmax);
    printf("  MT7:   max=%.6f (%.3f%%)  mean=%.6f (%.3f%%)\n",
           mt7_me, mt7_me/ref_absmax*100, mt7_se/n, mt7_se/n/ref_absmax*100);
    printf("  MT11:  max=%.6f (%.3f%%)  mean=%.6f (%.3f%%)\n",
           mt11_me, mt11_me/ref_absmax*100, mt11_se/n, mt11_se/n/ref_absmax*100);
    CHECK(mt11_me < mt7_me, "MT11 RMSNorm more accurate than MT7");

    /* 4. Throughput */
    printf("\n=== Throughput (n=%u, 100K iters) ===\n\n", n);
    int iters = 100000;
    double t0;

    /* Decode */
    for (int i = 0; i < 100; i++) mt11_decode_neon(decoded, mt11_w, n);
    t0 = ns();
    for (int i = 0; i < iters; i++) mt11_decode_neon(decoded, mt11_w, n);
    double us_mt11_dec = (ns() - t0) / (iters * 1000.0);

    for (int i = 0; i < 100; i++) mt11_decode_to_f32(decoded, mt11_w, n);
    t0 = ns();
    for (int i = 0; i < iters; i++) mt11_decode_to_f32(decoded, mt11_w, n);
    double us_mt11_scalar = (ns() - t0) / (iters * 1000.0);

    printf("  MT11 decode scalar: %7.3f us  (%5.0f M/s)\n", us_mt11_scalar, n/us_mt11_scalar);
    printf("  MT11 decode NEON:   %7.3f us  (%5.0f M/s)  %.2fx speedup\n",
           us_mt11_dec, n/us_mt11_dec, us_mt11_scalar/us_mt11_dec);

    /* RMSNorm */
    for (int i = 0; i < 100; i++) rmsnorm_f32(ref, input, weights, 1e-5f, n);
    t0 = ns();
    for (int i = 0; i < iters; i++) rmsnorm_f32(ref, input, weights, 1e-5f, n);
    double us_f32 = (ns() - t0) / (iters * 1000.0);

    for (int i = 0; i < 100; i++) mt7_rmsnorm(out_mt7, input, mt7_w, 1e-5f, n);
    t0 = ns();
    for (int i = 0; i < iters; i++) mt7_rmsnorm(out_mt7, input, mt7_w, 1e-5f, n);
    double us_mt7 = (ns() - t0) / (iters * 1000.0);

    for (int i = 0; i < 100; i++) mt11_rmsnorm_neon(out_mt11, input, mt11_w, 1e-5f, n);
    t0 = ns();
    for (int i = 0; i < iters; i++) mt11_rmsnorm_neon(out_mt11, input, mt11_w, 1e-5f, n);
    double us_mt11 = (ns() - t0) / (iters * 1000.0);

    printf("\n  RMSNorm:\n");
    printf("    F32:   %7.3f us  (%5.0f M/s)\n", us_f32, n/us_f32);
    printf("    MT7:   %7.3f us  (%5.0f M/s)  %.2fx vs F32\n", us_mt7, n/us_mt7, us_f32/us_mt7);
    printf("    MT11:  %7.3f us  (%5.0f M/s)  %.2fx vs F32\n", us_mt11, n/us_mt11, us_f32/us_mt11);

    /* 5. Storage */
    printf("\n=== Storage (n=%u) ===\n\n", n);
    printf("  F32:   %6u bytes  (32 bits/val)\n", n * 4);
    printf("  F16:   %6u bytes  (16 bits/val)\n", n * 2);
    printf("  MT11:  %6u bytes  (24 bits/val, 17.43 bits info)\n", mt11_storage_bytes(n));
    printf("  MT7:   %6u bytes  (12 bits/val, 11.09 bits info)\n", mt7_storage_bytes(n));
    printf("\n");
    printf("  MT11 info density:  17.43 bits in 24 bits = %.0f%% efficient\n", 17.43/24*100);
    printf("  MT7  info density:  11.09 bits in 12 bits = %.0f%% efficient\n", 11.09/12*100);
    printf("  F16  info density:  16.00 bits in 16 bits = 100%% efficient\n");

    printf("\n================================================================\n");
    printf("%s\n", failures == 0 ? "All MT11 checks passed." : "FAILURES.");

    free(weights); free(input); free(ref); free(out_mt7); free(out_mt11);
    free(decoded); free(dec_scalar); free(dec_neon); free(mt7_w); free(mt11_w);
    return failures;
}
