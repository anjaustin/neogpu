#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../include/hs_ml.h"

extern void hs_ml_gemm_ternary_mt(int32_t* C, const int8_t* A, const u8* B, u32 M, u32 N, u32 K, int threads);

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int main(void) {
    srand(42);
    
    u32 M = 4, N = 4096, K = 4096;
    
    int8_t* A = aligned_alloc(64, M * K);
    u8* B = aligned_alloc(64, N * K / 4);
    int32_t* C = aligned_alloc(64, M * N * sizeof(int32_t));
    
    for (u32 i = 0; i < M * K; i++) A[i] = rand() % 256 - 128;
    for (u32 i = 0; i < N * K / 4; i++) B[i] = rand();
    
    printf("Stable MT Benchmark: M=%u N=%u K=%u\n", M, N, K);
    printf("Running 50 iterations each, taking best of 5 runs...\n\n");
    
    for (int threads = 1; threads <= 4; threads++) {
        double best_gops = 0;
        
        for (int run = 0; run < 5; run++) {
            /* Warmup */
            for (int i = 0; i < 5; i++) {
                hs_ml_gemm_ternary_mt(C, A, B, M, N, K, threads);
            }
            
            int iters = 50;
            double t0 = get_time_ms();
            for (int i = 0; i < iters; i++) {
                hs_ml_gemm_ternary_mt(C, A, B, M, N, K, threads);
            }
            double ms = (get_time_ms() - t0) / iters;
            double ops = 2.0 * M * N * K;
            double gops = ops / (ms * 1e6);
            
            if (gops > best_gops) best_gops = gops;
        }
        
        printf("%d thread(s): best %.2f GOPS\n", threads, best_gops);
    }
    
    free(A); free(B); free(C);
    return 0;
}
