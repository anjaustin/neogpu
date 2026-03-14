#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include "hs_ml.h"

void hs_ml_gemm_ternary_v18(int32_t* C, const int8_t* A, const u8* B_ternary, u32 M, u32 N, u32 K);
void hs_ml_gemm_ternary_v21(int32_t* C, const int8_t* A, const u8* B_ternary, u32 M, u32 N, u32 K);

static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main(void) {
    printf("NeoGPU Ternary GEMM Benchmark - FINAL\n");
    printf("=====================================\n\n");
    
    u32 sizes[][3] = {{1, 256, 1024}, {1, 1024, 1024}, {1, 4096, 4096}, {4, 4096, 4096}};
    
    for (int s = 0; s < 4; s++) {
        u32 M = sizes[s][0], N = sizes[s][1], K = sizes[s][2];
        int8_t* A = malloc(M * K);
        u8* B = malloc(N * (K / 4));
        int32_t* C = malloc(M * N * sizeof(int32_t));
        
        srand(42);
        for (u32 i = 0; i < M * K; i++) A[i] = rand() % 21 - 10;
        memset(B, 0x55, N * (K / 4));
        
        int iters = (M * N * K < 10000000) ? 1000 : 100;
        printf("M=%u, N=%u, K=%u (%d iters):\n", M, N, K, iters);
        
        /* V18 baseline */
        for (int i = 0; i < 5; i++) hs_ml_gemm_ternary_v18(C, A, B, M, N, K);
        uint64_t t0 = get_time_ns();
        for (int i = 0; i < iters; i++) hs_ml_gemm_ternary_v18(C, A, B, M, N, K);
        uint64_t t1 = get_time_ns();
        double ns = (double)(t1 - t0) / iters;
        printf("  V18:  %8.2f us, %5.2f GOPS\n", ns/1000, (double)M*N*K/ns);
        
        /* V21 kernel */
        for (int i = 0; i < 5; i++) hs_ml_gemm_ternary_v21(C, A, B, M, N, K);
        t0 = get_time_ns();
        for (int i = 0; i < iters; i++) hs_ml_gemm_ternary_v21(C, A, B, M, N, K);
        t1 = get_time_ns();
        ns = (double)(t1 - t0) / iters;
        printf("  V21:  %8.2f us, %5.2f GOPS\n\n", ns/1000, (double)M*N*K/ns);
        
        free(A); free(B); free(C);
    }
    return 0;
}
