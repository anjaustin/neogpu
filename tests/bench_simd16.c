/*
 * NeoGPU - Ternary GEMM with GPU-style SIMD16 Architecture
 *
 * This implements the same algorithm that would run on the V3D QPUs,
 * but using ARM NEON. This can run immediately for benchmarking,
 * and the algorithm mirrors what the QPU shader will do.
 *
 * Architecture:
 *   - 4 QPUs x 16 SIMD = 64 parallel rows
 *   - Each "thread" processes one output row
 *   - Ternary decode via LUT (same as CPU path)
 *   - FMA for multiply-accumulate
 */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* 256-entry LUT for ternary decode */
static float ternary_lut[256][4] __attribute__((aligned(64)));
static int lut_built = 0;

static void build_lut(void) {
    if (lut_built) return;
    for (int b = 0; b < 256; b++) {
        ternary_lut[b][0] = (float)((int)((b >> 0) & 3) - 1);
        ternary_lut[b][1] = (float)((int)((b >> 2) & 3) - 1);
        ternary_lut[b][2] = (float)((int)((b >> 4) & 3) - 1);
        ternary_lut[b][3] = (float)((int)((b >> 6) & 3) - 1);
    }
    lut_built = 1;
}

/*============================================================================
 * SIMD16-style Ternary GEMM
 * 
 * This mimics what the QPU would do:
 *   - Process multiple rows in parallel (SIMD)
 *   - Use LUT for ternary decode
 *   - FMA for accumulation
 *============================================================================*/

#ifdef __ARM_NEON

/* Process 4 rows at once using NEON */
static void gemm_rows_4x(float* out0, float* out1, float* out2, float* out3,
                        const float* in,
                        const uint8_t* w0, const uint8_t* w1, const uint8_t* w2, const uint8_t* w3,
                        uint32_t K) {
    uint32_t row_bytes = K / 4;
    
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    
    uint32_t b4 = row_bytes & ~3u;
    
    for (uint32_t bi = 0; bi < b4; bi += 4) {
        /* Prefetch */
        __builtin_prefetch(w0 + bi + 64, 0, 1);
        __builtin_prefetch(in + (bi + 8) * 4, 0, 0);
        
        /* Load weights */
        float32x4_t lut0 = vld1q_f32(ternary_lut[w0[bi  ]]);
        float32x4_t lut1 = vld1q_f32(ternary_lut[w1[bi  ]]);
        float32x4_t lut2 = vld1q_f32(ternary_lut[w2[bi  ]]);
        float32x4_t lut3 = vld1q_f32(ternary_lut[w3[bi  ]]);
        
        /* Load activations */
        float32x4_t in0 = vld1q_f32(in + bi*4);
        float32x4_t in1 = vld1q_f32(in + (bi+1)*4);
        float32x4_t in2 = vld1q_f32(in + (bi+2)*4);
        float32x4_t in3 = vld1q_f32(in + (bi+3)*4);
        
        /* FMA */
        acc0 = vfmaq_f32(acc0, lut0, in0);
        acc1 = vfmaq_f32(acc1, lut1, in1);
        acc2 = vfmaq_f32(acc2, lut2, in2);
        acc3 = vfmaq_f32(acc3, lut3, in3);
    }
    
    /* Tail */
    for (uint32_t bi = b4; bi < row_bytes; bi++) {
        float w0_val = ternary_lut[w0[bi]][0];
        float w1_val = ternary_lut[w1[bi]][0];
        float w2_val = ternary_lut[w2[bi]][0];
        float w3_val = ternary_lut[w3[bi]][0];
        
        for (uint32_t d = 0; d < 4; d++) {
            out0[bi*4+d] += in[bi*4+d] * w0_val;
            out1[bi*4+d] += in[bi*4+d] * w1_val;
            out2[bi*4+d] += in[bi*4+d] * w2_val;
            out3[bi*4+d] += in[bi*4+d] * w3_val;
        }
    }
    
    /* Reduce */
    float sum0 = vaddvq_f32(acc0);
    float sum1 = vaddvq_f32(acc1);
    float sum2 = vaddvq_f32(acc2);
    float sum3 = vaddvq_f32(acc3);
    
    out0[0] = sum0; out1[0] = sum1; out2[0] = sum2; out3[0] = sum3;
}

#endif /* __ARM_NEON */

/*============================================================================
 * Batch GEMM (mimics QPU batch processing)
 *============================================================================*/

static void simd16_ternary_gemm(float* output, const float* input,
                                const uint8_t** weights, uint32_t N, uint32_t K,
                                int num_threads) {
#ifdef __ARM_NEON
    build_lut();
    
    /* Process 4 rows at a time */
    uint32_t row = 0;
    while (row + 4 <= N) {
        gemm_rows_4x(output + row, output + row + 1, output + row + 2, output + row + 3,
                    input,
                    weights[row], weights[row+1], weights[row+2], weights[row+3],
                    K);
        row += 4;
    }
    
    /* Tail */
    while (row < N) {
        /* Single row */
        const uint8_t* w = weights[row];
        uint32_t row_bytes = K / 4;
        float acc = 0;
        
        for (uint32_t bi = 0; bi < row_bytes; bi++) {
            float* lut = ternary_lut[w[bi]];
            float* in_ptr = (float*)(input + bi*4);
            acc += in_ptr[0] * lut[0] + in_ptr[1] * lut[1] + 
                   in_ptr[2] * lut[2] + in_ptr[3] * lut[3];
        }
        output[row] = acc;
        row++;
    }
#else
    /* Scalar fallback */
    extern void hs_ml_ternary_f32_proj(float *out, const float *in,
                                        const uint8_t *W, uint32_t N, uint32_t K);
    hs_ml_ternary_f32_proj(output, input, weights[0], N, K);
#endif
}

/*============================================================================
 * Benchmark
 *============================================================================*/

static uint64_t ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main(void) {
    printf("=== SIMD16 Ternary GEMM Benchmark ===\n\n");
    
    build_lut();
    
    uint32_t N = 2560, K = 2560;
    
    /* Allocate */
    float* output = malloc(N * sizeof(float));
    float* input = malloc(K * sizeof(float));
    uint8_t** weights = malloc(N * sizeof(uint8_t*));
    for (uint32_t i = 0; i < N; i++) {
        weights[i] = malloc(K / 4);
    }
    
    /* Initialize */
    srand(42);
    for (uint32_t i = 0; i < K; i++) input[i] = (float)(rand() % 100) / 10.0f;
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t j = 0; j < K/4; j++) {
            weights[i][j] = rand() % 256;
        }
    }
    
    /* Warmup */
    printf("Warming up...\n");
    simd16_ternary_gemm(output, input, (const uint8_t**)weights, N, K, 4);
    
    /* Benchmark SIMD16 path */
    printf("Running SIMD16 GEMM (%u x %u)...\n", N, K);
    uint64_t t0 = ns_now();
    for (int iter = 0; iter < 10; iter++) {
        simd16_ternary_gemm(output, input, (const uint8_t**)weights, N, K, 4);
    }
    uint64_t t1 = ns_now();
    double ms = (t1 - t0) / 1000000.0 / 10.0;
    double gops = (double)N * K / ms / 1e6;
    
    printf("SIMD16: %.2f ms  (%.2f GOPS)\n", ms, gops);
    printf("Output[0] = %f\n", output[0]);
    
    /* Compare with original CPU path */
    printf("\nComparing with original CPU path...\n");
    extern void hs_ml_ternary_f32_proj(float *out, const float *in,
                                        const uint8_t *W, uint32_t N, uint32_t K);
    
    t0 = ns_now();
    for (int iter = 0; iter < 10; iter++) {
        hs_ml_ternary_f32_proj(output, input, weights[0], N, K);
    }
    t1 = ns_now();
    double ms_orig = (t1 - t0) / 1000000.0 / 10.0;
    double gops_orig = (double)N * K / ms_orig / 1e6;
    
    printf("Original: %.2f ms  (%.2f GOPS)\n", ms_orig, gops_orig);
    printf("Speedup: %.2fx\n", ms_orig / ms);
    
    /* Cleanup */
    free(output);
    free(input);
    for (uint32_t i = 0; i < N; i++) free(weights[i]);
    free(weights);
    
    return 0;
}
