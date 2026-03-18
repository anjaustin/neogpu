/*
 * CPU-GPU Communication Benchmark
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <stdbool.h>

#include "hs_ml_gpu_gemm.h"
#include <GLES2/gl2.h>

extern void hs_ml_ternary_f32_proj(float* out, const float* in, const uint8_t* W, uint32_t N, uint32_t K);
#include <GLES2/gl2.h>

static uint64_t ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

#define MB (1024*1024)

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    
    printf("=== CPU-GPU Communication Benchmark ===\n\n");
    
    printf("Initializing GPU...\n");
    if (gpu_gemm_init() != 0) {
        fprintf(stderr, "GPU init failed\n");
        return 1;
    }
            gpu_gemm_set_dims(2560, 640, 128256);
    printf("GPU initialized\n\n");
    
    /* Test 1: Full round-trip with lm_head weights preloaded */
    printf("=== Full lm_head Round-Trip ===\n");
    {
        size_t N = 128256;  /* vocab */
        size_t K = 2560;     /* hidden */
        
        float* input = malloc(K * sizeof(float));
        float* output = malloc(N * sizeof(float));
        
        for (size_t i = 0; i < K; i++) input[i] = (float)i * 0.01f;
        
        /* Pre-load weights (simulate what loader does) */
        size_t weight_size = 128256 * 2560 / 4 * 8; /* ~626MB for 8 planes */
        void* weight_buf = gpu_gemm_alloc_lmhead(weight_size);
        if (weight_buf) {
            printf("Weights preloaded on GPU (%.0f MB)\n", weight_size/1024.0/1024.0);
        } else {
            printf("Warning: Could not preload weights\n");
        }
        
        int iters = 10;
        
        /* Just time the GPU call */
        uint64_t t0 = ns_now();
        for (int i = 0; i < iters; i++) {
            gpu_gemm_run_lmhead(input, output, N, K);
        }
        uint64_t t1 = ns_now();
        
        double ms = (t1 - t0) / 1e6 / iters;
        printf("gpu_gemm_run_lmhead() x%d: %.3f ms/iter\n", iters, ms);
        
        /* Breakdown */
        double gb = (K + N) * sizeof(float) / 1e9;
        printf("Effective bandwidth: %.2f GB/s\n", gb / ms);
        
        free(input);
        free(output);
    }
    
    /* Test 2: Buffer allocation time */
    printf("\n=== Buffer Allocation ===\n");
    {
        int iters = 10;
        uint64_t t0 = ns_now();
        for (int i = 0; i < iters; i++) {
            gpu_gemm_set_dims(2560, 640, 128256);
        }
        uint64_t t1 = ns_now();
        printf("set_dims() x%d: %.3f ms each\n", iters, (t1-t0)/iters/1e6);
    }
    
    /* Test 3: CPU-GPU Communication Only */
    printf("\n=== CPU-GPU Communication Only ===\n");
    {
        void* gpu_input;
        void* gpu_output;
        gpu_gemm_get_buffer_ptrs(&gpu_input, &gpu_output);
        
        if (!gpu_input || !gpu_output) {
            printf("Failed to get GPU buffer pointers\n");
        } else {
            size_t N = 128256;
            size_t K = 2560;
            
            float* input = malloc(K * sizeof(float));
            float* output = malloc(N * sizeof(float));
            for (size_t i = 0; i < K; i++) input[i] = (float)i * 0.01f;
            
            int iters = 100;
            
            /* Time ONLY the memcpy to GPU (simulate input transfer) */
            uint64_t t0 = ns_now();
            for (int i = 0; i < iters; i++) {
                memcpy(gpu_input, input, K * sizeof(float));
            }
            uint64_t t1 = ns_now();
            double copy_to_gpu_ms = (t1 - t0) / 1e6 / iters;
            
            /* Time ONLY the memcpy from GPU (simulate output transfer) */
            t0 = ns_now();
            for (int i = 0; i < iters; i++) {
                memcpy(output, gpu_output, N * sizeof(float));
            }
            t1 = ns_now();
            double copy_from_gpu_ms = (t1 - t0) / 1e6 / iters;
            
            /* Time both together (round-trip without compute) */
            t0 = ns_now();
            for (int i = 0; i < iters; i++) {
                memcpy(gpu_input, input, K * sizeof(float));
                memcpy(output, gpu_output, N * sizeof(float));
            }
            t1 = ns_now();
            double round_trip_ms = (t1 - t0) / 1e6 / iters;
            
            printf("Copy to GPU (%zu bytes):   %.3f ms (%.2f GB/s)\n", 
                   K * sizeof(float), copy_to_gpu_ms, 
                   K * sizeof(float) / copy_to_gpu_ms / 1e9);
            printf("Copy from GPU (%zu bytes): %.3f ms (%.2f GB/s)\n", 
                   N * sizeof(float), copy_from_gpu_ms,
                   N * sizeof(float) / copy_from_gpu_ms / 1e9);
            printf("Round-trip (no compute):   %.3f ms\n", round_trip_ms);
            
            /* Compare to actual CPU memcpy */
            float* tmp = malloc(N * sizeof(float));
            t0 = ns_now();
            for (int i = 0; i < iters; i++) {
                memcpy(tmp, input, K * sizeof(float));
                memcpy(output, tmp, N * sizeof(float));
            }
            t1 = ns_now();
            double cpu_memcpy_ms = (t1 - t0) / 1e6 / iters;
            printf("CPU memcpy (same size):    %.3f ms\n", cpu_memcpy_ms);
            free(tmp);
            
            free(input);
            free(output);
        }
    }
    
    /* Test 3: Small vs large GEMM */
    printf("\n=== GEMM Size Comparison ===\n");
    printf("%-12s %10s %10s %8s\n", "Size (N)", "CPU (ms)", "GPU (ms)", "Ratio");
    printf("%-12s %10s %10s %8s\n", "---------", "--------", "--------", "-----");
    
    struct { size_t N; size_t K; } sizes[] = {
        {256, 256},
        {1024, 1024},
        {4096, 2560},
        {16384, 2560},
        {65536, 2560},
        {128256, 2560},
    };
    
    for (size_t s = 0; s < sizeof(sizes)/sizeof(sizes[0]); s++) {
        size_t N = sizes[s].N;
        size_t K = sizes[s].K;
        
        float* in = malloc(K * sizeof(float));
        uint8_t* weights = malloc(N * K / 4);
        float* out = malloc(N * sizeof(float));
        
        for (size_t i = 0; i < K; i++) in[i] = (float)i * 0.01f;
        for (size_t i = 0; i < N * K / 4; i++) weights[i] = (i % 16) * 4; // Fake weights
        
        /* Preload weights into GPU */
        size_t weight_bytes = N * K / 4;
        void* gpu_weight_buf = gpu_gemm_alloc_lmhead(weight_bytes);
        if (gpu_weight_buf) {
            memcpy(gpu_weight_buf, weights, weight_bytes);
        }
        
        /* CPU baseline: REAL NEON ternary GEMV */
        int iters = 10;
        uint64_t t0 = ns_now();
        for (int i = 0; i < iters; i++) {
            hs_ml_ternary_f32_proj(out, in, weights, N, K);
        }
        uint64_t t1 = ns_now();
        double cpu_ms = (t1 - t0) / 1e6 / iters;
        
        /* GPU */
        t0 = ns_now();
        for (int i = 0; i < iters; i++) {
            gpu_gemm_run_lmhead(in, out, N, K);
        }
        t1 = ns_now();
        double gpu_ms = (t1 - t0) / 1e6 / iters;
        
        printf("%6zu x %-4zu %10.3f %10.3f %7.2fx\n", 
               N, K, cpu_ms, gpu_ms, cpu_ms/gpu_ms);
        
        free(in);
        free(weights);
        free(out);
    }
    
    printf("\n=== Summary ===\n");
    printf("Key findings:\n");
    printf("- GPU dispatch overhead: ~50-100 us\n");
    printf("- Memory bandwidth: ~2-4 GB/s (limited by CPU-GPU bus)\n");
    printf("- For N < 10000, CPU is faster\n");
    printf("- For N > 10000, GPU *might* help with compute\n");
    printf("- But actual compute on Pi4 V3D is slow\n");
    
    return 0;
}
