/*
 * Test binary routing implementation
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/* Declarations from hs_ml_binary.c */
void hs_ml_route_binary(int32_t* C, const uint8_t* A, const uint8_t* B,
                        uint32_t M, uint32_t N, uint32_t K);
void hs_ml_route_binary_opt(int32_t* C, const uint8_t* A, const uint8_t* B,
                            uint32_t M, uint32_t N, uint32_t K);
void hs_ml_route_ternary_x_ternary(int32_t* C,
                                   const uint8_t* A_active, const uint8_t* A_sign,
                                   const uint8_t* B_active, const uint8_t* B_sign,
                                   uint32_t M, uint32_t N, uint32_t K);

static double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000.0 + ts.tv_nsec / 1000.0;
}

/* Reference binary dot product */
static int32_t binary_dot_ref(const uint8_t* a, const uint8_t* b, uint32_t K) {
    int32_t sum = 0;
    uint32_t K_bytes = K / 8;
    for (uint32_t i = 0; i < K_bytes; i++) {
        uint8_t xnor = ~(a[i] ^ b[i]);
        int matches = __builtin_popcount(xnor);
        sum += 2 * matches - 8;
    }
    return sum;
}

int main(void) {
    printf("Binary Routing Tests\n");
    printf("====================\n\n");
    
    /* Test correctness */
    printf("Correctness tests:\n");
    
    uint32_t M = 4, N = 64, K = 256;
    uint32_t K_bytes = K / 8;
    
    uint8_t* A = aligned_alloc(64, M * K_bytes);
    uint8_t* B = aligned_alloc(64, N * K_bytes);
    int32_t* C = aligned_alloc(64, M * N * sizeof(int32_t));
    int32_t* C_ref = aligned_alloc(64, M * N * sizeof(int32_t));
    
    /* Random data */
    srand(42);
    for (size_t i = 0; i < M * K_bytes; i++) A[i] = rand() & 0xFF;
    for (size_t i = 0; i < N * K_bytes; i++) B[i] = rand() & 0xFF;
    
    /* Compute reference */
    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t n = 0; n < N; n++) {
            C_ref[m * N + n] = binary_dot_ref(A + m * K_bytes, B + n * K_bytes, K);
        }
    }
    
    /* Test basic version */
    memset(C, 0, M * N * sizeof(int32_t));
    hs_ml_route_binary(C, A, B, M, N, K);
    
    int errors = 0;
    for (uint32_t i = 0; i < M * N; i++) {
        if (C[i] != C_ref[i]) {
            if (errors < 5) {
                printf("  [FAIL] basic[%u]: got %d, expected %d\n", i, C[i], C_ref[i]);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("  [PASS] hs_ml_route_binary\n");
    } else {
        printf("  [FAIL] hs_ml_route_binary: %d errors\n", errors);
    }
    
    /* Test optimized version */
    memset(C, 0, M * N * sizeof(int32_t));
    hs_ml_route_binary_opt(C, A, B, M, N, K);
    
    errors = 0;
    for (uint32_t i = 0; i < M * N; i++) {
        if (C[i] != C_ref[i]) {
            if (errors < 5) {
                printf("  [FAIL] opt[%u]: got %d, expected %d\n", i, C[i], C_ref[i]);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("  [PASS] hs_ml_route_binary_opt\n");
    } else {
        printf("  [FAIL] hs_ml_route_binary_opt: %d errors\n", errors);
    }
    
    free(A); free(B); free(C); free(C_ref);
    
    /* Performance benchmark */
    printf("\nPerformance benchmark (M=4, N=4096, K=4096):\n");
    
    M = 4; N = 4096; K = 4096;
    K_bytes = K / 8;
    
    A = aligned_alloc(64, M * K_bytes);
    B = aligned_alloc(64, N * K_bytes);
    C = aligned_alloc(64, M * N * sizeof(int32_t));
    
    for (size_t i = 0; i < M * K_bytes; i++) A[i] = rand() & 0xFF;
    for (size_t i = 0; i < N * K_bytes; i++) B[i] = rand() & 0xFF;
    
    /* Warmup */
    hs_ml_route_binary_opt(C, A, B, M, N, K);
    
    /* Benchmark */
    int iters = 20;
    double t0 = get_time_us();
    for (int i = 0; i < iters; i++) {
        hs_ml_route_binary_opt(C, A, B, M, N, K);
    }
    double us = get_time_us() - t0;
    double ms = us / (iters * 1000.0);
    double ops = (double)M * N * K;
    double gops = ops / (ms * 1e6);
    
    printf("  Binary GEMM: %.2f ms, %.1f GOPS\n", ms, gops);
    printf("  Memory: A=%.1f KB, B=%.1f MB\n", 
           M * K_bytes / 1024.0, N * K_bytes / (1024.0 * 1024.0));
    
    free(A); free(B); free(C);
    
    /* Ternary x Ternary */
    printf("\nTernary x Ternary benchmark (M=4, N=4096, K=4096):\n");
    
    uint8_t* A_act = aligned_alloc(64, M * K_bytes);
    uint8_t* A_sgn = aligned_alloc(64, M * K_bytes);
    uint8_t* B_act = aligned_alloc(64, N * K_bytes);
    uint8_t* B_sgn = aligned_alloc(64, N * K_bytes);
    C = aligned_alloc(64, M * N * sizeof(int32_t));
    
    for (size_t i = 0; i < M * K_bytes; i++) {
        A_act[i] = rand() & 0xFF;
        A_sgn[i] = rand() & A_act[i];
    }
    for (size_t i = 0; i < N * K_bytes; i++) {
        B_act[i] = rand() & 0xFF;
        B_sgn[i] = rand() & B_act[i];
    }
    
    /* Warmup */
    hs_ml_route_ternary_x_ternary(C, A_act, A_sgn, B_act, B_sgn, M, N, K);
    
    /* Benchmark */
    t0 = get_time_us();
    for (int i = 0; i < iters; i++) {
        hs_ml_route_ternary_x_ternary(C, A_act, A_sgn, B_act, B_sgn, M, N, K);
    }
    us = get_time_us() - t0;
    ms = us / (iters * 1000.0);
    gops = ops / (ms * 1e6);
    
    printf("  TernaryxTernary GEMM: %.2f ms, %.1f GOPS\n", ms, gops);
    printf("  Memory: A=%.1f KB (x2), B=%.1f MB (x2)\n",
           M * K_bytes * 2 / 1024.0, N * K_bytes * 2 / (1024.0 * 1024.0));
    
    free(A_act); free(A_sgn);
    free(B_act); free(B_sgn);
    free(C);
    
    printf("\nDone.\n");
    return 0;
}
