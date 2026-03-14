/*
 * Test and benchmark for true ternary GEMM kernel
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include "hs_ml.h"

/* Declare the new ternary kernels */
void hs_ml_gemm_ternary_neon(int32_t* C, const int8_t* A, const u8* B_ternary, u32 M, u32 N, u32 K);
void hs_ml_gemm_ternary_neon_4col(int32_t* C, const int8_t* A, const u8* B_ternary, u32 M, u32 N, u32 K);

/* Reference scalar implementation */
void ternary_gemm_scalar(int32_t* C, const int8_t* A, const u8* B_ternary, u32 M, u32 N, u32 K) {
    memset(C, 0, M * N * sizeof(int32_t));
    
    for (u32 m = 0; m < M; m++) {
        for (u32 n = 0; n < N; n++) {
            int32_t sum = 0;
            const u8* B_col = B_ternary + n * (K / 4);
            
            for (u32 k = 0; k < K; k++) {
                u8 byte = B_col[k / 4];
                u8 bits = (byte >> ((k % 4) * 2)) & 0x03;
                int8_t a = A[m * K + k];
                
                if (bits == 1) sum += a;
                else if (bits == 2) sum -= a;
            }
            
            C[m * N + n] = sum;
        }
    }
}

static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int test_correctness(u32 M, u32 N, u32 K) {
    printf("Testing correctness: M=%u, N=%u, K=%u\n", M, N, K);
    
    int8_t* A = malloc(M * K);
    u8* B = malloc(N * (K / 4));
    int32_t* C_ref = malloc(M * N * sizeof(int32_t));
    int32_t* C_neon = malloc(M * N * sizeof(int32_t));
    int32_t* C_4col = malloc(M * N * sizeof(int32_t));
    
    /* Initialize with random data */
    srand(42);
    for (u32 i = 0; i < M * K; i++) {
        A[i] = (rand() % 21) - 10;  /* -10 to +10 */
    }
    
    /* Random ternary weights */
    memset(B, 0, N * (K / 4));
    for (u32 n = 0; n < N; n++) {
        for (u32 k = 0; k < K; k++) {
            u8 bits = rand() % 3;  /* 0, 1, or 2 */
            u32 byte_idx = n * (K / 4) + k / 4;
            u32 bit_shift = (k % 4) * 2;
            B[byte_idx] |= (bits << bit_shift);
        }
    }
    
    /* Compute reference */
    ternary_gemm_scalar(C_ref, A, B, M, N, K);
    
    /* Compute with NEON kernels */
    hs_ml_gemm_ternary_neon(C_neon, A, B, M, N, K);
    hs_ml_gemm_ternary_neon_4col(C_4col, A, B, M, N, K);
    
    /* Verify */
    int passed = 1;
    for (u32 i = 0; i < M * N; i++) {
        if (C_neon[i] != C_ref[i]) {
            printf("  FAIL neon: C[%u] = %d, expected %d\n", i, C_neon[i], C_ref[i]);
            passed = 0;
            break;
        }
        if (C_4col[i] != C_ref[i]) {
            printf("  FAIL 4col: C[%u] = %d, expected %d\n", i, C_4col[i], C_ref[i]);
            passed = 0;
            break;
        }
    }
    
    if (passed) printf("  PASSED\n");
    
    free(A);
    free(B);
    free(C_ref);
    free(C_neon);
    free(C_4col);
    
    return passed;
}

void benchmark(u32 M, u32 N, u32 K, int iterations) {
    printf("\nBenchmark: M=%u, N=%u, K=%u, %d iterations\n", M, N, K, iterations);
    
    int8_t* A = malloc(M * K);
    u8* B = malloc(N * (K / 4));
    int32_t* C = malloc(M * N * sizeof(int32_t));
    float* scales = malloc(N * sizeof(float));
    
    /* Initialize */
    srand(123);
    for (u32 i = 0; i < M * K; i++) A[i] = rand() % 21 - 10;
    memset(B, 0x55, N * (K / 4));  /* All +1 weights */
    for (u32 i = 0; i < N; i++) scales[i] = 1.0f;
    
    uint64_t t0, t1;
    double ns_per_call;
    double gops;  /* giga-operations per second: M*N*K ops per call */
    double ops_per_call = (double)M * N * K;
    
    /* Warm up */
    for (int i = 0; i < 10; i++) {
        hs_ml_gemm_ternary_neon(C, A, B, M, N, K);
    }
    
    /* Benchmark scalar (original fallback) */
    t0 = get_time_ns();
    for (int i = 0; i < iterations; i++) {
        hs_ml_gemm_int8(C, A, B, scales, M, N, K);
    }
    t1 = get_time_ns();
    ns_per_call = (double)(t1 - t0) / iterations;
    gops = ops_per_call / ns_per_call;
    printf("  Original fallback: %.2f us/call, %.2f GOPS\n", ns_per_call / 1000.0, gops);
    
    /* Benchmark new NEON */
    t0 = get_time_ns();
    for (int i = 0; i < iterations; i++) {
        hs_ml_gemm_ternary_neon(C, A, B, M, N, K);
    }
    t1 = get_time_ns();
    ns_per_call = (double)(t1 - t0) / iterations;
    gops = ops_per_call / ns_per_call;
    printf("  NEON ternary:      %.2f us/call, %.2f GOPS\n", ns_per_call / 1000.0, gops);
    
    /* Benchmark 4-column NEON */
    t0 = get_time_ns();
    for (int i = 0; i < iterations; i++) {
        hs_ml_gemm_ternary_neon_4col(C, A, B, M, N, K);
    }
    t1 = get_time_ns();
    ns_per_call = (double)(t1 - t0) / iterations;
    gops = ops_per_call / ns_per_call;
    printf("  NEON 4-col:        %.2f us/call, %.2f GOPS\n", ns_per_call / 1000.0, gops);
    
    free(A);
    free(B);
    free(C);
    free(scales);
}

int main(void) {
    printf("NeoGPU ML - Ternary GEMM Tests\n");
    printf("==============================\n");
    
    /* Correctness tests */
    int passed = 1;
    passed &= test_correctness(1, 1, 64);
    passed &= test_correctness(4, 4, 64);
    passed &= test_correctness(16, 16, 64);
    passed &= test_correctness(16, 16, 128);
    passed &= test_correctness(1, 64, 256);
    passed &= test_correctness(8, 32, 512);
    passed &= test_correctness(1, 4096, 4096);  /* LLM-sized single token */
    
    if (!passed) {
        printf("\nSome tests FAILED\n");
        return 1;
    }
    
    printf("\nAll correctness tests PASSED\n");
    
    /* Benchmarks */
    benchmark(1, 64, 256, 10000);      /* Small */
    benchmark(1, 256, 1024, 1000);     /* Medium */
    benchmark(1, 4096, 4096, 100);     /* LLM FFN size */
    benchmark(4, 4096, 4096, 100);     /* Batched */
    
    return 0;
}

