/*
 * Benchmark: NEON vs V3D GPU kernels
 * 
 * Build on Pi:
 *   gcc -O3 -march=armv8-a+simd+fp16 -mtune=cortex-a72 -ffast-math \
 *       -funroll-loops -DNDEBUG -Iinclude \
 *       src/hs_ml_v3d_gpu.c -o /tmp/bench_v3d -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#include "hs_ml_v3d.h"

static uint64_t ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

#define N_ITERS 100

int main(void) {
    printf("=== NEON Kernel Benchmarks ===\n\n");

    /* Softmax benchmark */
    {
        uint32_t N = 4096;
        float* in = malloc(N * sizeof(float));
        float* out = malloc(N * sizeof(float));
        for (uint32_t i = 0; i < N; i++) in[i] = (float)(rand() % 1000) / 100.0f;

        uint64_t t0 = ns_now();
        for (int i = 0; i < N_ITERS; i++) {
            neon_softmax(out, in, N);
        }
        uint64_t t1 = ns_now();
        double ms = (t1 - t0) / 1000000.0 / N_ITERS;
        printf("softmax(%u):   %.3f ms  (%.1f M elem/s)\n", 
               N, ms, N / ms / 1000.0);
        
        free(in); free(out);
    }

    /* Add benchmark */
    {
        uint32_t N = 8192;
        float* a = malloc(N * sizeof(float));
        float* b = malloc(N * sizeof(float));
        float* out = malloc(N * sizeof(float));
        for (uint32_t i = 0; i < N; i++) {
            a[i] = (float)(rand() % 1000) / 100.0f;
            b[i] = (float)(rand() % 1000) / 100.0f;
        }

        uint64_t t0 = ns_now();
        for (int i = 0; i < N_ITERS; i++) {
            neon_add(out, a, b, N);
        }
        uint64_t t1 = ns_now();
        double ms = (t1 - t0) / 1000000.0 / N_ITERS;
        printf("add(%u):       %.3f ms  (%.1f M elem/s)\n", 
               N, ms, N / ms / 1000.0);
        
        free(a); free(b); free(out);
    }

    /* Activate (relu2*up) benchmark */
    {
        uint32_t N = 8192;
        float* gate = malloc(N * sizeof(float));
        float* up = malloc(N * sizeof(float));
        float* out = malloc(N * sizeof(float));
        for (uint32_t i = 0; i < N; i++) {
            gate[i] = (float)(rand() % 1000) / 100.0f - 5.0f;
            up[i] = (float)(rand() % 1000) / 100.0f;
        }

        uint64_t t0 = ns_now();
        for (int i = 0; i < N_ITERS; i++) {
            neon_activate(out, gate, up, N);
        }
        uint64_t t1 = ns_now();
        double ms = (t1 - t0) / 1000000.0 / N_ITERS;
        printf("activate(%u):  %.3f ms  (%.1f M elem/s)\n", 
               N, ms, N / ms / 1000.0);
        
        free(gate); free(up); free(out);
    }

    /* RoPE benchmark */
    {
        uint32_t nh = 20, hd = 128;
        uint32_t N = nh * hd;
        float* x = malloc(N * sizeof(float));
        for (uint32_t i = 0; i < N; i++) x[i] = (float)(rand() % 1000) / 100.0f;

        uint64_t t0 = ns_now();
        for (int i = 0; i < N_ITERS; i++) {
            neon_rope(x, nh, hd, i, 500000.0f);
        }
        uint64_t t1 = ns_now();
        double ms = (t1 - t0) / 1000000.0 / N_ITERS;
        printf("rope(%u x %u): %.3f ms  (%.1f M elem/s)\n", 
               nh, hd, ms, N / ms / 1000.0);
        
        free(x);
    }

    printf("\n=== GPU Availability ===\n");
    printf("V3D GPU: %s\n", v3d_available() ? "available" : "not available");
    v3d_init();
    printf("V3D after init: %s\n", v3d_available() ? "available" : "not available");
    v3d_shutdown();

    printf("\n=== Decode Step Estimation ===\n");
    printf("With GPU offload (ROPE, SOFTMAX, ACTIVATE, ADD):\n");
    printf("  Current:  ~1170 ms\n");
    printf("  Projections: 493 ms (stay on CPU, 4-thread NEON)\n");
    printf("  lm_head (F16): 670 ms (needs ternary path)\n");
    printf("  GPU-eligible: ~5 ms (ROPE=3ms, ADD=~2ms, etc)\n");
    printf("  Potential speedup: minimal (GPU ops are tiny)\n");
    
    return 0;
}
