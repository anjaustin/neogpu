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

static void bench(u32 M, u32 N, u32 K) {
    int8_t* A = aligned_alloc(64, M * K);
    u8* B = aligned_alloc(64, N * K / 4);
    int32_t* C = aligned_alloc(64, M * N * sizeof(int32_t));
    
    for (u32 i = 0; i < M * K; i++) A[i] = rand() % 256 - 128;
    for (u32 i = 0; i < N * K / 4; i++) B[i] = rand();
    
    printf("\nM=%u N=%u K=%u:\n", M, N, K);
    
    for (int threads = 1; threads <= 4; threads++) {
        /* Warmup */
        for (int i = 0; i < 3; i++) {
            hs_ml_gemm_ternary_mt(C, A, B, M, N, K, threads);
        }
        
        int iters = 20;
        double t0 = get_time_ms();
        for (int i = 0; i < iters; i++) {
            hs_ml_gemm_ternary_mt(C, A, B, M, N, K, threads);
        }
        double ms = (get_time_ms() - t0) / iters;
        double ops = 2.0 * M * N * K;
        double gops = ops / (ms * 1e6);
        
        printf("  %d threads: %.2f ms, %.2f GOPS\n", threads, ms, gops);
    }
    
    free(A); free(B); free(C);
}

int main(void) {
    srand(42);
    printf("Multi-Size MT Benchmark\n");
    
    /* Llama 7B typical sizes */
    bench(1, 4096, 4096);   /* Single token, 4K model */
    bench(4, 4096, 4096);   /* 4 tokens */
    bench(1, 11008, 4096);  /* FFN up projection */
    bench(1, 4096, 11008);  /* FFN down projection */
    bench(4, 11008, 4096);  /* 4 tokens, FFN up */
    bench(4, 4096, 11008);  /* 4 tokens, FFN down */
    
    return 0;
}
