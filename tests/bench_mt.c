#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include "hs_ml.h"

void hs_ml_gemm_ternary_mt(int32_t* C, const int8_t* A, const u8* B_ternary,
                           u32 M, u32 N, u32 K, int num_threads);

static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

void gemm_ref(int32_t* C, const int8_t* A, const u8* B, u32 M, u32 N, u32 K) {
    memset(C, 0, M * N * sizeof(int32_t));
    for (u32 m = 0; m < M; m++) {
        for (u32 n = 0; n < N; n++) {
            for (u32 k = 0; k < K; k++) {
                u8 byte = B[n * (K/4) + k/4];
                u8 bits = (byte >> ((k % 4) * 2)) & 0x03;
                if (bits == 1) C[m*N+n] += A[m*K+k];
                else if (bits == 2) C[m*N+n] -= A[m*K+k];
            }
        }
    }
}

int main(void) {
    printf("Multi-threaded Ternary GEMM Benchmark\n");
    printf("=====================================\n\n");
    
    u32 sizes[][3] = {{1, 256, 1024}, {1, 1024, 1024}, {1, 4096, 4096}, {4, 4096, 4096}};
    
    for (int s = 0; s < 4; s++) {
        u32 M = sizes[s][0], N = sizes[s][1], K = sizes[s][2];
        int8_t* A = aligned_alloc(64, M * K);
        u8* B = aligned_alloc(64, N * (K / 4));
        int32_t* C = aligned_alloc(64, M * N * sizeof(int32_t));
        int32_t* C_ref = malloc(M * N * sizeof(int32_t));
        
        srand(42);
        for (u32 i = 0; i < M * K; i++) A[i] = rand() % 21 - 10;
        for (u32 i = 0; i < N * (K/4); i++) B[i] = rand() % 256;
        
        /* Correctness check */
        hs_ml_gemm_ternary_mt(C, A, B, M, N, K, 4);
        gemm_ref(C_ref, A, B, M, N, K);
        int correct = 1;
        for (u32 i = 0; i < M * N; i++) {
            if (C[i] != C_ref[i]) { correct = 0; break; }
        }
        
        int iters = (M * N * K < 10000000) ? 500 : 50;
        printf("M=%u, N=%u, K=%u (correct=%s):\n", M, N, K, correct ? "yes" : "NO");
        
        for (int threads = 1; threads <= 4; threads++) {
            /* Warmup */
            for (int i = 0; i < 3; i++) 
                hs_ml_gemm_ternary_mt(C, A, B, M, N, K, threads);
            
            uint64_t t0 = get_time_ns();
            for (int i = 0; i < iters; i++)
                hs_ml_gemm_ternary_mt(C, A, B, M, N, K, threads);
            uint64_t t1 = get_time_ns();
            
            double ns = (double)(t1 - t0) / iters;
            double gops = (double)M * N * K / ns;
            printf("  %d thread(s): %8.2f us, %6.2f GOPS\n", threads, ns/1000, gops);
        }
        printf("\n");
        
        free(A); free(B); free(C); free(C_ref);
    }
    
    return 0;
}
