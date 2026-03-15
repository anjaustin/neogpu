/*
 * Red-team: Pure float32 ternary routing kernel
 *
 * 1. NEON matches scalar reference
 * 2. Fused norm+proj matches separate norm then proj
 * 3. Handles real model dimensions (2560, 640, 6912)
 * 4. Benchmark vs int8 quantize+project+dequantize path
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

extern void hs_ml_ternary_f32_proj(float *out, const float *in,
                                    const uint8_t *W, uint32_t N, uint32_t K);
extern void hs_ml_ternary_f32_norm_proj(float *proj_out,
                                         const float *hidden, const float *norm_w,
                                         const uint8_t *W, uint32_t N, uint32_t K,
                                         float eps);

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

/* Scalar reference */
#define I2S_QK 64u
static void ref_proj(float *out, const float *in, const uint8_t *W,
                     uint32_t N, uint32_t K) {
    uint32_t row_bytes = K / 4;
    uint32_t nblk = K / I2S_QK;
    for (uint32_t n = 0; n < N; n++) {
        const uint8_t *wrow = W + n * row_bytes;
        float acc = 0.0f;
        for (uint32_t bi = 0; bi < nblk; bi++) {
            const uint8_t *block = wrow + bi * 16;
            for (uint32_t j = 0; j < I2S_QK; j++) {
                uint32_t k = bi * I2S_QK + j;
                uint32_t gi = j / 16, gp = j % 16;
                uint8_t raw = (block[gp] >> (6 - 2 * gi)) & 3;
                if (raw == 2) acc += in[k];
                else if (raw == 0) acc -= in[k];
            }
        }
        out[n] = acc;
    }
}

static void ref_rmsnorm(float *out, const float *in, const float *w,
                        float eps, uint32_t n) {
    float ss = 0;
    for (uint32_t i = 0; i < n; i++) ss += in[i] * in[i];
    float sc = 1.0f / sqrtf(ss / n + eps);
    for (uint32_t i = 0; i < n; i++) out[i] = in[i] * sc * w[i];
}

int main(void) {
    printf("Red-team: Pure float32 ternary routing\n");
    printf("=======================================\n");

    uint32_t sizes[][2] = {{64, 64}, {256, 256}, {2560, 2560}, {640, 2560}, {6912, 2560}};
    int nsizes = 5;

    /* 1. NEON vs scalar correctness */
    printf("\n=== NEON vs scalar correctness ===\n");
    for (int s = 0; s < nsizes; s++) {
        uint32_t N = sizes[s][0], K = sizes[s][1];
        uint32_t row_bytes = K / 4;
        uint8_t *W = aligned_alloc(64, (size_t)N * row_bytes);
        float *in = aligned_alloc(64, K * sizeof(float));
        float *out_ref = malloc(N * sizeof(float));
        float *out_neon = malloc(N * sizeof(float));

        srand(42 + s);
        for (size_t i = 0; i < (size_t)N * row_bytes; i++) W[i] = rand();
        for (uint32_t i = 0; i < K; i++) in[i] = ((float)(rand() % 2000) - 1000) / 500.0f;

        ref_proj(out_ref, in, W, N, K);
        hs_ml_ternary_f32_proj(out_neon, in, W, N, K);

        float max_diff = 0;
        for (uint32_t i = 0; i < N; i++) {
            float d = fabsf(out_ref[i] - out_neon[i]);
            if (d > max_diff) max_diff = d;
        }

        char msg[64];
        snprintf(msg, sizeof(msg), "N=%u K=%u max_diff=%e", N, K, max_diff);
        CHECK(max_diff < 1e-3f, msg);

        free(W); free(in); free(out_ref); free(out_neon);
    }

    /* 2. Fused norm+proj correctness */
    printf("\n=== Fused norm+proj correctness ===\n");
    {
        uint32_t N = 2560, K = 2560;
        uint32_t row_bytes = K / 4;
        uint8_t *W = aligned_alloc(64, (size_t)N * row_bytes);
        float *hidden = aligned_alloc(64, K * sizeof(float));
        float *norm_w = malloc(K * sizeof(float));
        float *normed = malloc(K * sizeof(float));
        float *out_sep = malloc(N * sizeof(float));
        float *out_fused = malloc(N * sizeof(float));

        srand(77);
        for (size_t i = 0; i < (size_t)N * row_bytes; i++) W[i] = rand();
        for (uint32_t i = 0; i < K; i++) {
            hidden[i] = ((float)(rand() % 2000) - 1000) / 500.0f;
            norm_w[i] = ((float)(rand() % 200) - 100) / 100.0f;  /* small norms like real model */
        }

        /* Separate: norm then proj */
        ref_rmsnorm(normed, hidden, norm_w, 1e-5f, K);
        ref_proj(out_sep, normed, W, N, K);

        /* Fused */
        hs_ml_ternary_f32_norm_proj(out_fused, hidden, norm_w, W, N, K, 1e-5f);

        float max_diff = 0;
        for (uint32_t i = 0; i < N; i++) {
            float d = fabsf(out_sep[i] - out_fused[i]);
            if (d > max_diff) max_diff = d;
        }
        char msg[64];
        snprintf(msg, sizeof(msg), "fused matches separate (max_diff=%e)", max_diff);
        CHECK(max_diff < 1e-2f, msg);

        free(W); free(hidden); free(norm_w); free(normed); free(out_sep); free(out_fused);
    }

    /* 3. Small norm weights (real model scenario) */
    printf("\n=== Small norm weights (absmax=0.06) ===\n");
    {
        uint32_t N = 2560, K = 2560;
        uint32_t row_bytes = K / 4;
        uint8_t *W = aligned_alloc(64, (size_t)N * row_bytes);
        float *hidden = aligned_alloc(64, K * sizeof(float));
        float *norm_w = malloc(K * sizeof(float));
        float *out = malloc(N * sizeof(float));

        srand(42);
        for (size_t i = 0; i < (size_t)N * row_bytes; i++) W[i] = rand();
        for (uint32_t i = 0; i < K; i++) {
            hidden[i] = ((float)(rand() % 2000) - 1000) / 500.0f;
            norm_w[i] = 0.02f + 0.04f * (float)(rand() % 100) / 100.0f;  /* 0.02..0.06 like real attn_norm */
        }

        hs_ml_ternary_f32_norm_proj(out, hidden, norm_w, W, N, K, 1e-5f);

        int all_finite = 1;
        float absmax = 0;
        for (uint32_t i = 0; i < N; i++) {
            if (!isfinite(out[i])) { all_finite = 0; break; }
            float a = fabsf(out[i]);
            if (a > absmax) absmax = a;
        }
        CHECK(all_finite, "output finite with small norms");
        CHECK(absmax < 1000.0f, "output bounded with small norms");
        printf("    absmax=%f\n", absmax);

        free(W); free(hidden); free(norm_w); free(out);
    }

    /* 4. Benchmark */
    printf("\n=== Throughput (N=2560 K=2560) ===\n");
    {
        uint32_t N = 2560, K = 2560;
        uint32_t row_bytes = K / 4;
        uint8_t *W = aligned_alloc(64, (size_t)N * row_bytes);
        float *in = aligned_alloc(64, K * sizeof(float));
        float *norm_w = malloc(K * sizeof(float));
        float *out = malloc(N * sizeof(float));

        srand(42);
        for (size_t i = 0; i < (size_t)N * row_bytes; i++) W[i] = rand();
        for (uint32_t i = 0; i < K; i++) {
            in[i] = ((float)(rand() % 2000) - 1000) / 500.0f;
            norm_w[i] = 0.03f;
        }

        int iters = 1000;
        /* warmup */
        for (int i = 0; i < 10; i++) hs_ml_ternary_f32_proj(out, in, W, N, K);

        double t0 = ns();
        for (int i = 0; i < iters; i++) hs_ml_ternary_f32_proj(out, in, W, N, K);
        double us_proj = (ns() - t0) / (iters * 1000.0);

        for (int i = 0; i < 10; i++) hs_ml_ternary_f32_norm_proj(out, in, norm_w, W, N, K, 1e-5f);
        t0 = ns();
        for (int i = 0; i < iters; i++) hs_ml_ternary_f32_norm_proj(out, in, norm_w, W, N, K, 1e-5f);
        double us_fused = (ns() - t0) / (iters * 1000.0);

        double ops = (double)N * K;
        printf("    f32 proj:       %7.2f ms  %5.1f GOPS\n", us_proj / 1000, ops / (us_proj * 1e3));
        printf("    f32 norm+proj:  %7.2f ms  %5.1f GOPS\n", us_fused / 1000, ops / (us_fused * 1e3));

        free(W); free(in); free(norm_w); free(out);
    }

    printf("\n=======================================\n");
    printf("%s\n", failures == 0 ? "All float32 ternary checks passed." : "FAILURES.");
    return failures;
}
