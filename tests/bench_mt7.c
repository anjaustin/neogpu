/*
 * NeoGPU ML - MT7 Face-Melting Benchmarks
 *
 * Compares MT7 against F32, F16, and INT8 for norm operations
 * across the full NeoGPU ML stack on Pi4 Cortex-A72.
 *
 * Tests:
 *   1. Raw decode throughput: MT7 vs F16 vs F32
 *   2. RMSNorm throughput: MT7 vs F32 at real model sizes
 *   3. Full layer norm chain: all 5 norms per layer
 *   4. Memory footprint comparison for full 2B model
 *   5. Quantization error distribution
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* MT7 externals */
extern void mt7_encode_f32(uint8_t *out, const float *in, uint32_t n);
extern void mt7_decode_to_f32(float *out, const uint8_t *in, uint32_t n);
extern void mt7_rmsnorm(float *out, const float *in, const uint8_t *mt7_weights,
                        float eps, uint32_t n);
extern uint32_t mt7_storage_bytes(uint32_t n);

static double ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

/* F32 RMSNorm baseline */
static void rmsnorm_f32(float *out, const float *in, const float *w,
                        float eps, uint32_t n) {
    float ss = 0.0f;
#ifdef __ARM_NEON
    float32x4_t vss = vdupq_n_f32(0.0f);
    uint32_t n4 = n & ~3u;
    for (uint32_t i = 0; i < n4; i += 4) {
        float32x4_t v = vld1q_f32(in + i);
        vss = vfmaq_f32(vss, v, v);
    }
    ss = vaddvq_f32(vss);
    for (uint32_t i = n4; i < n; i++) ss += in[i] * in[i];
#else
    for (uint32_t i = 0; i < n; i++) ss += in[i] * in[i];
#endif
    float scale = 1.0f / sqrtf(ss / n + eps);
    for (uint32_t i = 0; i < n; i++) out[i] = in[i] * scale * w[i];
}

/* F16 decode baseline */
static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign;
        else {
            exp = 127 - 15 + 1;
            while ((mant & 0x0400) == 0) { mant <<= 1; exp--; }
            mant &= 0x03FF;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, sizeof(float));
    return out;
}

static void decode_f16_to_f32(float *out, const uint16_t *in, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) out[i] = fp16_to_f32(in[i]);
}

/* F16 RMSNorm: decode then apply */
static void rmsnorm_f16(float *out, const float *in, const uint16_t *w_f16,
                        float eps, uint32_t n) {
    float ss = 0.0f;
    for (uint32_t i = 0; i < n; i++) ss += in[i] * in[i];
    float scale = 1.0f / sqrtf(ss / n + eps);
    for (uint32_t i = 0; i < n; i++)
        out[i] = in[i] * scale * fp16_to_f32(w_f16[i]);
}

/* F32 to F16 encode */
static uint16_t f32_to_fp16(float v) {
    uint32_t f;
    memcpy(&f, &v, sizeof(float));
    uint32_t sign = (f >> 16) & 0x8000;
    int32_t exp = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (f >> 13) & 0x03FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | mant);
}

#define BEST_OF 5
#define ITERS 10000

static double bench(void (*fn)(void *), void *arg, int iters) {
    double best = 1e18;
    for (int r = 0; r < BEST_OF; r++) {
        double t0 = ns();
        for (int i = 0; i < iters; i++) fn(arg);
        double us = (ns() - t0) / (iters * 1000.0);
        if (us < best) best = us;
    }
    return best;
}

typedef struct {
    float *out, *in;
    float *w_f32;
    uint16_t *w_f16;
    uint8_t *w_mt7;
    float eps;
    uint32_t n;
} NormArgs;

static void run_f32(void *p)  { NormArgs *a = p; rmsnorm_f32(a->out, a->in, a->w_f32, a->eps, a->n); }
static void run_f16(void *p)  { NormArgs *a = p; rmsnorm_f16(a->out, a->in, a->w_f16, a->eps, a->n); }
static void run_mt7(void *p)  { NormArgs *a = p; mt7_rmsnorm(a->out, a->in, a->w_mt7, a->eps, a->n); }

typedef struct { float *out; uint8_t *mt7; uint32_t n; } DecArgs;
typedef struct { float *out; uint16_t *f16; uint32_t n; } F16Args;
typedef struct { float *out, *in; uint32_t n; } CopyArgs;
static void run_mt7_dec(void *p) { DecArgs *a = p; mt7_decode_to_f32(a->out, a->mt7, a->n); }
static void run_f16_dec(void *p) { F16Args *a = p; decode_f16_to_f32(a->out, a->f16, a->n); }
static void run_f32_copy(void *p) { CopyArgs *a = p; memcpy(a->out, a->in, a->n * sizeof(float)); }

int main(void) {
    printf("================================================================\n");
    printf("  NeoGPU ML — MT7 Benchmarks on Pi4 Cortex-A72\n");
    printf("================================================================\n\n");

    uint32_t sizes[] = {2560, 6912};
    const char *names[] = {"hidden (2560)", "ffn (6912)"};

    for (int s = 0; s < 2; s++) {
        uint32_t n = sizes[s];
        printf("--- %s ---\n\n", names[s]);

        float *input = aligned_alloc(64, n * sizeof(float));
        float *output = aligned_alloc(64, n * sizeof(float));
        float *w_f32 = aligned_alloc(64, n * sizeof(float));
        uint16_t *w_f16 = aligned_alloc(64, n * sizeof(uint16_t));
        uint8_t *w_mt7 = aligned_alloc(64, mt7_storage_bytes(n));
        float *dec_buf = aligned_alloc(64, n * sizeof(float));

        srand(42);
        for (uint32_t i = 0; i < n; i++) {
            input[i] = ((float)(rand() % 2000) - 1000) / 500.0f;
            w_f32[i] = ((float)(rand() % 2000) - 1000) / 500.0f;
            w_f16[i] = f32_to_fp16(w_f32[i]);
        }
        mt7_encode_f32(w_mt7, w_f32, n);

        /* 1. Raw decode throughput */
        DecArgs da = {dec_buf, w_mt7, n};
        F16Args fa = {dec_buf, w_f16, n};
        CopyArgs ca = {dec_buf, w_f32, n};

        double us_f32_copy = bench(run_f32_copy, &ca, ITERS);
        double us_f16_dec  = bench(run_f16_dec,  &fa, ITERS);
        double us_mt7_dec  = bench(run_mt7_dec,  &da, ITERS);

        printf("  Decode throughput (%dx%d iters, best of %d):\n", n, ITERS, BEST_OF);
        printf("    F32 memcpy:  %7.3f us  (%6.0f M values/s)\n",
               us_f32_copy, n / us_f32_copy);
        printf("    F16 decode:  %7.3f us  (%6.0f M values/s)\n",
               us_f16_dec, n / us_f16_dec);
        printf("    MT7 decode:  %7.3f us  (%6.0f M values/s)  %.1fx vs F16\n",
               us_mt7_dec, n / us_mt7_dec, us_f16_dec / us_mt7_dec);
        printf("\n");

        /* 2. RMSNorm throughput */
        NormArgs na_args = {output, input, w_f32, w_f16, w_mt7, 1e-5f, n};

        double us_f32 = bench(run_f32, &na_args, ITERS);
        double us_f16 = bench(run_f16, &na_args, ITERS);
        double us_mt7 = bench(run_mt7, &na_args, ITERS);

        printf("  RMSNorm throughput:\n");
        printf("    F32:  %7.3f us  (%6.0f M norms/s)\n", us_f32, n / us_f32);
        printf("    F16:  %7.3f us  (%6.0f M norms/s)  %.2fx vs F32\n",
               us_f16, n / us_f16, us_f32 / us_f16);
        printf("    MT7:  %7.3f us  (%6.0f M norms/s)  %.2fx vs F32\n",
               us_mt7, n / us_mt7, us_f32 / us_mt7);
        printf("\n");

        /* 3. RMSNorm accuracy */
        float *ref = malloc(n * sizeof(float));
        float *test_f16 = malloc(n * sizeof(float));
        float *test_mt7 = malloc(n * sizeof(float));
        rmsnorm_f32(ref, input, w_f32, 1e-5f, n);
        rmsnorm_f16(test_f16, input, w_f16, 1e-5f, n);
        mt7_rmsnorm(test_mt7, input, w_mt7, 1e-5f, n);

        float ref_absmax = 0;
        for (uint32_t i = 0; i < n; i++) {
            float a = fabsf(ref[i]);
            if (a > ref_absmax) ref_absmax = a;
        }
        float f16_max_err = 0, mt7_max_err = 0;
        float f16_sum_err = 0, mt7_sum_err = 0;
        for (uint32_t i = 0; i < n; i++) {
            float e16 = fabsf(ref[i] - test_f16[i]);
            float emt = fabsf(ref[i] - test_mt7[i]);
            if (e16 > f16_max_err) f16_max_err = e16;
            if (emt > mt7_max_err) mt7_max_err = emt;
            f16_sum_err += e16;
            mt7_sum_err += emt;
        }

        printf("  RMSNorm accuracy vs F32 reference (absmax=%.3f):\n", ref_absmax);
        printf("    F16:  max_err=%.6f (%.3f%%)  mean_err=%.6f (%.3f%%)\n",
               f16_max_err, f16_max_err / ref_absmax * 100,
               f16_sum_err / n, f16_sum_err / n / ref_absmax * 100);
        printf("    MT7:  max_err=%.6f (%.3f%%)  mean_err=%.6f (%.3f%%)\n",
               mt7_max_err, mt7_max_err / ref_absmax * 100,
               mt7_sum_err / n, mt7_sum_err / n / ref_absmax * 100);
        printf("\n");

        /* 4. Memory footprint */
        uint32_t b_f32 = n * 4;
        uint32_t b_f16 = n * 2;
        uint32_t b_mt7 = mt7_storage_bytes(n);
        printf("  Memory per norm vector:\n");
        printf("    F32:  %6u bytes\n", b_f32);
        printf("    F16:  %6u bytes  (%.0f%% of F32)\n", b_f16, 100.0f * b_f16 / b_f32);
        printf("    MT7:  %6u bytes  (%.0f%% of F32, %.0f%% of F16)\n",
               b_mt7, 100.0f * b_mt7 / b_f32, 100.0f * b_mt7 / b_f16);
        printf("\n");

        free(input); free(output); free(w_f32); free(w_f16); free(w_mt7);
        free(dec_buf); free(ref); free(test_f16); free(test_mt7);
    }

    /* 5. Full model memory comparison */
    printf("--- Full 2B-4T model norm memory ---\n\n");
    uint32_t H = 2560, F = 6912, L = 30;
    uint32_t per_layer_vals = H * 4 + F;  /* attn_norm, attn_sub_norm, ffn_norm, output_norm each [H], ffn_sub_norm [F] */
    /* Actually: attn_norm[H] + attn_sub_norm[H] + ffn_norm[H] + ffn_sub_norm[F] = 3H + F per layer
     * Plus output_norm[H] and final_norm[H] = 2H global */
    uint32_t total_vals = L * (3 * H + F) + 2 * H;
    uint32_t total_f32 = total_vals * 4;
    uint32_t total_f16 = total_vals * 2;
    uint32_t total_mt7 = mt7_storage_bytes(total_vals);
    float model_total_mb = 1190.0f;  /* ~1.19 GB for full I2_S model */

    printf("  Norm values:   %u\n", total_vals);
    printf("  F32:           %u bytes  (%.2f MB)  %.2f%% of model\n",
           total_f32, total_f32 / 1e6f, total_f32 / (model_total_mb * 1e6f) * 100);
    printf("  F16:           %u bytes  (%.2f MB)  %.2f%% of model\n",
           total_f16, total_f16 / 1e6f, total_f16 / (model_total_mb * 1e6f) * 100);
    printf("  MT7:           %u bytes  (%.2f MB)  %.2f%% of model\n",
           total_mt7, total_mt7 / 1e6f, total_mt7 / (model_total_mb * 1e6f) * 100);
    printf("  MT7 savings:   %u bytes vs F16  (%.1f%%)\n",
           total_f16 - total_mt7, (1.0f - (float)total_mt7 / total_f16) * 100);

    /* 6. Per-layer norm chain timing */
    printf("\n--- Per-layer norm chain (5 norms) ---\n\n");
    {
        float *buf = aligned_alloc(64, F * sizeof(float));
        float *out = aligned_alloc(64, F * sizeof(float));
        float *w1 = malloc(H * sizeof(float));
        float *w2 = malloc(H * sizeof(float));
        float *w3 = malloc(H * sizeof(float));
        float *w4 = malloc(F * sizeof(float));
        float *w5 = malloc(H * sizeof(float));
        uint8_t *m1 = malloc(mt7_storage_bytes(H));
        uint8_t *m2 = malloc(mt7_storage_bytes(H));
        uint8_t *m3 = malloc(mt7_storage_bytes(H));
        uint8_t *m4 = malloc(mt7_storage_bytes(F));
        uint8_t *m5 = malloc(mt7_storage_bytes(H));

        srand(99);
        for (uint32_t i = 0; i < F; i++) buf[i] = ((float)(rand() % 2000) - 1000) / 500.0f;
        for (uint32_t i = 0; i < H; i++) { w1[i] = w2[i] = w3[i] = w5[i] = ((float)(rand() % 2000) - 1000) / 500.0f; }
        for (uint32_t i = 0; i < F; i++) w4[i] = ((float)(rand() % 2000) - 1000) / 500.0f;
        mt7_encode_f32(m1, w1, H); mt7_encode_f32(m2, w2, H);
        mt7_encode_f32(m3, w3, H); mt7_encode_f32(m4, w4, F);
        mt7_encode_f32(m5, w5, H);

        /* F32 chain */
        int chain_iters = 50000;
        double t0 = ns();
        for (int i = 0; i < chain_iters; i++) {
            rmsnorm_f32(out, buf, w1, 1e-5f, H);
            rmsnorm_f32(out, buf, w2, 1e-5f, H);
            rmsnorm_f32(out, buf, w3, 1e-5f, H);
            rmsnorm_f32(out, buf, w4, 1e-5f, F);
            rmsnorm_f32(out, buf, w5, 1e-5f, H);
        }
        double us_f32_chain = (ns() - t0) / (chain_iters * 1000.0);

        /* MT7 chain */
        t0 = ns();
        for (int i = 0; i < chain_iters; i++) {
            mt7_rmsnorm(out, buf, m1, 1e-5f, H);
            mt7_rmsnorm(out, buf, m2, 1e-5f, H);
            mt7_rmsnorm(out, buf, m3, 1e-5f, H);
            mt7_rmsnorm(out, buf, m4, 1e-5f, F);
            mt7_rmsnorm(out, buf, m5, 1e-5f, H);
        }
        double us_mt7_chain = (ns() - t0) / (chain_iters * 1000.0);

        printf("  5-norm chain (%d iters):\n", chain_iters);
        printf("    F32:  %7.2f us/layer\n", us_f32_chain);
        printf("    MT7:  %7.2f us/layer  (%.2fx vs F32)\n",
               us_mt7_chain, us_f32_chain / us_mt7_chain);
        printf("    Per token (30 layers):\n");
        printf("      F32: %.2f us\n", us_f32_chain * 30);
        printf("      MT7: %.2f us\n", us_mt7_chain * 30);

        free(buf); free(out);
        free(w1); free(w2); free(w3); free(w4); free(w5);
        free(m1); free(m2); free(m3); free(m4); free(m5);
    }

    printf("\n================================================================\n");
    return 0;
}
