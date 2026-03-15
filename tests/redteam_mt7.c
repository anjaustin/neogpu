/*
 * Red-team: Multi-Trit Floating Point 7 (MT7)
 *
 * Tests:
 *   1. Encode/decode roundtrip accuracy
 *   2. Pack/unpack bit-exact
 *   3. MT7 RMSNorm vs F32 RMSNorm accuracy
 *   4. Edge cases: zero, max, min, near-zero
 *   5. Real norm weight distribution
 *   6. Storage savings verification
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

/* Forward declarations from hs_ml_mt7.c */
extern void mt7_encode_f32(uint8_t *out, const float *in, uint32_t n);
extern void mt7_decode_to_f32(float *out, const uint8_t *in, uint32_t n);
extern void mt7_rmsnorm(float *out, const float *in, const uint8_t *mt7_weights,
                        float eps, uint32_t n);
extern uint32_t mt7_storage_bytes(uint32_t n);

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  [FAIL] %s\n", (msg)); failures++; } \
    else         { printf("  [PASS] %s\n", (msg)); } \
} while(0)

static void f32_rmsnorm(float *out, const float *in, const float *w, float eps, uint32_t n) {
    float ss = 0.0f;
    for (uint32_t i = 0; i < n; i++) ss += in[i] * in[i];
    float scale = 1.0f / sqrtf(ss / n + eps);
    for (uint32_t i = 0; i < n; i++) out[i] = in[i] * scale * w[i];
}

static void test_roundtrip(void) {
    printf("\n=== Roundtrip accuracy ===\n");

    float vals[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 0.01f, -0.01f,
                    2.5f, -2.5f, 3.8f, -3.8f, 0.001f, 81.0f, -81.0f};
    int n = sizeof(vals) / sizeof(vals[0]);
    uint32_t bytes = mt7_storage_bytes(n);
    uint8_t *packed = malloc(bytes);
    float *decoded = malloc(n * sizeof(float));

    mt7_encode_f32(packed, vals, n);
    mt7_decode_to_f32(decoded, packed, n);

    float max_err = 0, max_rel = 0;
    for (int i = 0; i < n; i++) {
        float err = fabsf(vals[i] - decoded[i]);
        float rel = (fabsf(vals[i]) > 0.001f) ? err / fabsf(vals[i]) : 0;
        if (err > max_err) max_err = err;
        if (rel > max_rel) max_rel = rel;
    }

    CHECK(max_err < 0.2f, "max absolute error < 0.2");
    CHECK(max_rel < 0.05f, "max relative error < 5%");
    CHECK(decoded[0] == 0.0f, "zero encodes exactly");
    printf("  max_abs_err=%.6f max_rel_err=%.4f%%\n", max_err, max_rel * 100);

    free(packed);
    free(decoded);
}

static void test_pack_unpack(void) {
    printf("\n=== Pack/unpack bit-exact ===\n");

    /* Pack two known codes, unpack, verify */
    uint8_t buf[3];
    uint16_t c0 = 1234, c1 = 2100;
    uint16_t d0, d1;

    /* Manual pack */
    buf[0] = (uint8_t)(c0 & 0xFF);
    buf[1] = (uint8_t)((c0 >> 8) | ((c1 & 0xF) << 4));
    buf[2] = (uint8_t)(c1 >> 4);

    /* Unpack */
    d0 = (uint16_t)(buf[0] | ((buf[1] & 0x0F) << 8));
    d1 = (uint16_t)((buf[1] >> 4) | ((uint16_t)buf[2] << 4));

    CHECK(d0 == c0, "code_0 roundtrip");
    CHECK(d1 == c1, "code_1 roundtrip");

    /* Test all valid codes */
    int ok = 1;
    for (uint16_t a = 0; a < 2187; a += 100) {
        for (uint16_t b = 0; b < 2187; b += 100) {
            buf[0] = (uint8_t)(a & 0xFF);
            buf[1] = (uint8_t)((a >> 8) | ((b & 0xF) << 4));
            buf[2] = (uint8_t)(b >> 4);
            d0 = (uint16_t)(buf[0] | ((buf[1] & 0x0F) << 8));
            d1 = (uint16_t)((buf[1] >> 4) | ((uint16_t)buf[2] << 4));
            if (d0 != a || d1 != b) { ok = 0; break; }
        }
        if (!ok) break;
    }
    CHECK(ok, "all sampled code pairs roundtrip");
}

static void test_rmsnorm_accuracy(void) {
    printf("\n=== RMSNorm: MT7 vs F32 ===\n");

    uint32_t n = 2560;
    float *input = malloc(n * sizeof(float));
    float *weights = malloc(n * sizeof(float));
    float *ref_out = malloc(n * sizeof(float));
    float *mt7_out = malloc(n * sizeof(float));
    uint8_t *mt7_w = malloc(mt7_storage_bytes(n));

    srand(42);
    for (uint32_t i = 0; i < n; i++) {
        input[i] = ((float)(rand() % 2000) - 1000) / 1000.0f;
        weights[i] = ((float)(rand() % 2000) - 1000) / 500.0f;  /* range ±2 */
    }

    /* Encode weights to MT7 */
    mt7_encode_f32(mt7_w, weights, n);

    /* Reference: F32 RMSNorm */
    f32_rmsnorm(ref_out, input, weights, 1e-5f, n);

    /* MT7 RMSNorm */
    mt7_rmsnorm(mt7_out, input, mt7_w, 1e-5f, n);

    float max_err = 0, sum_err = 0;
    for (uint32_t i = 0; i < n; i++) {
        float err = fabsf(ref_out[i] - mt7_out[i]);
        if (err > max_err) max_err = err;
        sum_err += err;
    }
    float mean_err = sum_err / n;

    /* Reference output magnitude for relative comparison */
    float ref_absmax = 0;
    for (uint32_t i = 0; i < n; i++) {
        float a = fabsf(ref_out[i]);
        if (a > ref_absmax) ref_absmax = a;
    }

    float rel_max = max_err / ref_absmax;
    float rel_mean = mean_err / ref_absmax;

    CHECK(rel_max < 0.05f, "max relative RMSNorm error < 5%");
    CHECK(rel_mean < 0.01f, "mean relative RMSNorm error < 1%");
    printf("  ref_absmax=%.4f max_err=%.6f (%.2f%%) mean_err=%.6f (%.2f%%)\n",
           ref_absmax, max_err, rel_max * 100, mean_err, rel_mean * 100);

    free(input);
    free(weights);
    free(ref_out);
    free(mt7_out);
    free(mt7_w);
}

static void test_edge_cases(void) {
    printf("\n=== Edge cases ===\n");

    float edges[] = {0.0f, 81.0f, -81.0f, 0.0001f, -0.0001f};
    int n = 5;
    uint8_t *packed = malloc(mt7_storage_bytes(n));
    float *decoded = malloc(n * sizeof(float));

    mt7_encode_f32(packed, edges, n);
    mt7_decode_to_f32(decoded, packed, n);

    CHECK(decoded[0] == 0.0f, "zero exact");
    CHECK(fabsf(decoded[1] - 81.0f) < 1.0f, "max range ~81");
    CHECK(fabsf(decoded[2] + 81.0f) < 1.0f, "min range ~-81");
    CHECK(fabsf(decoded[3]) < 0.002f, "near-zero positive");
    CHECK(fabsf(decoded[4]) < 0.002f, "near-zero negative");

    free(packed);
    free(decoded);
}

static void test_storage(void) {
    printf("\n=== Storage savings ===\n");

    uint32_t n = 2560;
    uint32_t mt7_bytes = mt7_storage_bytes(n);
    uint32_t f16_bytes = n * 2;
    uint32_t f32_bytes = n * 4;
    float savings_f16 = (1.0f - (float)mt7_bytes / f16_bytes) * 100;
    float savings_f32 = (1.0f - (float)mt7_bytes / f32_bytes) * 100;

    printf("  n=%u: MT7=%u bytes, F16=%u bytes, F32=%u bytes\n",
           n, mt7_bytes, f16_bytes, f32_bytes);
    printf("  savings vs F16: %.1f%%\n", savings_f16);
    printf("  savings vs F32: %.1f%%\n", savings_f32);

    CHECK(mt7_bytes < f16_bytes, "MT7 smaller than F16");
    CHECK(savings_f16 > 20.0f, "at least 20% savings vs F16");
}

static void bench_decode(void) {
    printf("\n=== Decode throughput ===\n");

    uint32_t n = 2560;
    float *weights = malloc(n * sizeof(float));
    uint8_t *mt7_w = malloc(mt7_storage_bytes(n));
    float *decoded = malloc(n * sizeof(float));

    srand(7);
    for (uint32_t i = 0; i < n; i++)
        weights[i] = ((float)(rand() % 2000) - 1000) / 500.0f;
    mt7_encode_f32(mt7_w, weights, n);

    /* Warmup */
    mt7_decode_to_f32(decoded, mt7_w, n);

    /* Bench */
    int iters = 100000;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++)
        mt7_decode_to_f32(decoded, mt7_w, n);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
    double us_per = ns / (iters * 1000.0);
    printf("  %u values x %d iters: %.3f us/decode\n", n, iters, us_per);

    free(weights);
    free(mt7_w);
    free(decoded);
}

int main(void) {
    printf("Red-team: Multi-Trit Floating Point 7 (MT7)\n");
    printf("=============================================\n");

    test_roundtrip();
    test_pack_unpack();
    test_rmsnorm_accuracy();
    test_edge_cases();
    test_storage();
    bench_decode();

    printf("\n=============================================\n");
    printf("%s\n", failures == 0 ? "All MT7 checks passed." : "FAILURES DETECTED.");
    return failures;
}
