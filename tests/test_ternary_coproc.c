/*
 * Test: Ternary Coprocessor (GLES 3.1 Compute)
 *
 * Verifies the GPU ternary GEMM implementation produces correct results.
 *
 * Build:
 *   gcc -O3 -march=armv8-a+simd -mtune=cortex-a72 -DHAS_GLES_COMPUTE \
 *       -Iinclude tests/test_ternary_coproc.c \
 *       src/hs_ml_ternary_coproc.c src/hs_ml_ternary_f32.c \
 *       -o /tmp/test_ternary_coproc -lGLESv2 -lEGL -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "hs_ml_ternary_coproc.h"

/* Reference CPU implementation */
extern void hs_ml_ternary_f32_proj(float *out, const float *in,
                                    const uint8_t *W, uint32_t N, uint32_t K);

static uint64_t ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/* Pack ternary weights using I2_S encoding: -1=0, 0=1, +1=2 */
static void pack_ternary(uint8_t* out, const int8_t* in, uint32_t count) {
    for (uint32_t i = 0; i < count; i += 4) {
        uint8_t b = 0;
        for (int j = 0; j < 4 && (i + j) < count; j++) {
            int8_t w = in[i + j];
            /* I2_S encoding: 0=-1, 1=0, 2=+1, 3=0 */
            uint8_t code = (w == 1) ? 2 : (w == -1) ? 0 : 1;
            b |= (code << (j * 2));
        }
        out[i / 4] = b;
    }
}

static int test_small(void) {
    printf("=== Test: Small matrix (64x64) ===\n");
    
    uint32_t N = 64, K = 64;
    
    float* input = malloc(K * sizeof(float));
    float* output_gpu = malloc(N * sizeof(float));
    float* output_cpu = malloc(N * sizeof(float));
    int8_t* weights_raw = malloc(N * K);
    uint8_t* weights_packed = malloc(N * K / 4);
    
    /* Initialize random data */
    srand(42);
    for (uint32_t i = 0; i < K; i++) {
        input[i] = (float)(rand() % 200 - 100) / 10.0f;
    }
    for (uint32_t i = 0; i < N * K; i++) {
        int r = rand() % 3;
        weights_raw[i] = (r == 0) ? -1 : (r == 1) ? 0 : 1;
    }
    pack_ternary(weights_packed, weights_raw, N * K);
    
    /* CPU reference */
    hs_ml_ternary_f32_proj(output_cpu, input, weights_packed, N, K);
    
    /* GPU */
    TernaryProj proj = {
        .output = output_gpu,
        .input = input,
        .weights = weights_packed,
        .N = N,
        .K = K
    };
    
    int ret = ternary_coproc_run_batch(&proj, 1);
    if (ret != 0) {
        printf("  GPU not available (ret=%d), skipping GPU test\n", ret);
        free(input); free(output_gpu); free(output_cpu);
        free(weights_raw); free(weights_packed);
        return 0;
    }
    
    /* Compare */
    int errors = 0;
    float max_err = 0.0f;
    for (uint32_t i = 0; i < N; i++) {
        float err = fabsf(output_gpu[i] - output_cpu[i]);
        if (err > max_err) max_err = err;
        if (err > 0.001f) {
            errors++;
            if (errors <= 3) {
                printf("  ERROR at %u: GPU=%f, CPU=%f\n", i, output_gpu[i], output_cpu[i]);
            }
        }
    }
    
    if (errors == 0) {
        printf("  PASS: All %u outputs match (max_err=%.6f)\n", N, max_err);
    } else {
        printf("  FAIL: %d errors (max_err=%.6f)\n", errors, max_err);
    }
    
    free(input); free(output_gpu); free(output_cpu);
    free(weights_raw); free(weights_packed);
    return errors;
}

static int test_large(void) {
    printf("\n=== Test: Large matrix (2560x2560) ===\n");
    
    uint32_t N = 2560, K = 2560;
    
    float* input = malloc(K * sizeof(float));
    float* output_gpu = malloc(N * sizeof(float));
    float* output_cpu = malloc(N * sizeof(float));
    uint8_t* weights_packed = malloc(N * K / 4);
    
    /* Initialize */
    srand(123);
    for (uint32_t i = 0; i < K; i++) {
        input[i] = (float)(rand() % 200 - 100) / 100.0f;
    }
    for (uint32_t i = 0; i < N * K / 4; i++) {
        weights_packed[i] = rand() & 0xFF;
    }
    
    /* CPU timing */
    uint64_t t0 = ns_now();
    hs_ml_ternary_f32_proj(output_cpu, input, weights_packed, N, K);
    uint64_t t1 = ns_now();
    double cpu_ms = (t1 - t0) / 1000000.0;
    printf("  CPU: %.2f ms\n", cpu_ms);
    
    /* GPU timing */
    TernaryProj proj = {
        .output = output_gpu,
        .input = input,
        .weights = weights_packed,
        .N = N,
        .K = K
    };
    
    t0 = ns_now();
    int ret = ternary_coproc_run_batch(&proj, 1);
    t1 = ns_now();
    
    if (ret != 0) {
        printf("  GPU not available\n");
        free(input); free(output_gpu); free(output_cpu); free(weights_packed);
        return 0;
    }
    
    double gpu_ms = (t1 - t0) / 1000000.0;
    printf("  GPU: %.2f ms (%.1fx %s)\n", gpu_ms, 
           cpu_ms / gpu_ms, gpu_ms < cpu_ms ? "faster" : "slower");
    
    /* Verify */
    int errors = 0;
    for (uint32_t i = 0; i < N; i++) {
        float err = fabsf(output_gpu[i] - output_cpu[i]);
        if (err > 0.01f) {
            errors++;
            if (errors <= 3) {
                printf("  ERROR at %u: GPU=%f, CPU=%f\n", i, output_gpu[i], output_cpu[i]);
            }
        }
    }
    
    if (errors == 0) {
        printf("  PASS: All %u outputs correct\n", N);
    } else {
        printf("  FAIL: %d errors\n", errors);
    }
    
    free(input); free(output_gpu); free(output_cpu); free(weights_packed);
    return errors;
}

int main(void) {
    printf("=== Ternary Coprocessor Test ===\n\n");
    
    /* Initialize */
    if (ternary_coproc_init() != 0) {
        printf("Failed to initialize coprocessor\n");
        return 1;
    }
    
    printf("Coprocessor available: %s\n\n", 
           ternary_coproc_available() ? "GPU" : "CPU only");
    
    int total_errors = 0;
    total_errors += test_small();
    total_errors += test_large();
    
    /* Stats */
    TernaryCoprocStats stats;
    ternary_coproc_get_stats(&stats);
    printf("\n=== Stats ===\n");
    printf("  Projections: %u\n", stats.num_projections);
    printf("  Total time: %.2f ms\n", stats.total_time_ns / 1000000.0);
    printf("  QPUs used: %u\n", stats.num_qpus_used);
    
    ternary_coproc_shutdown();
    
    printf("\n=== %s ===\n", total_errors == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return total_errors;
}
