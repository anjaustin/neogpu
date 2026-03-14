/*
 * Test FFN activation and RMSNorm functions
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "hs_ml.h"

#define TEST_N 16

void print_vector_f32(const char* name, float* v, u32 n) {
    printf("%s (%u): ", name, n);
    for (u32 i = 0; i < n && i < 8; i++) {
        printf("%.4f ", v[i]);
    }
    if (n > 8) printf("...");
    printf("\n");
}

int test_ffn_activate_gate(void) {
    printf("\n=== Test: FFN Activate Gate ===\n");
    
    float* input = malloc(2 * TEST_N * sizeof(float));
    float* output = malloc(TEST_N * sizeof(float));
    
    // Test case: input[0..N-1] = [2, -3, 4, -5, ...]
    //           input[N..2N-1] = [1, 1, 1, 1, ...] (gate = 1)
    // Expected: squared_relu = [4, 0, 16, 0, ...] * gate = same
    for (u32 i = 0; i < TEST_N; i++) {
        input[i] = (i % 2 == 0) ? (float)(i + 2) : -(float)(i + 3);  // 2, -3, 4, -5, 6, -7...
        input[i + TEST_N] = 1.0f;  // gate = 1
    }
    
    hs_ml_ffn_activate_gate(input, output, TEST_N);
    
    // Verify: output[i] = (max(0, input[i]))^2 * gate
    int passed = 1;
    for (u32 i = 0; i < TEST_N; i++) {
        float x = input[i];
        float gate = input[i + TEST_N];
        float expected = (x > 0.0f) ? x : 0.0f;
        expected = expected * expected * gate;
        
        if (output[i] != expected) {
            printf("  FAIL: output[%u] = %.4f, expected %.4f (x=%.4f, gate=%.4f)\n", 
                   i, output[i], expected, x, gate);
            passed = 0;
        }
    }
    
    if (passed) printf("  PASSED: FFN activate gate correct\n");
    
    free(input);
    free(output);
    return passed;
}

int test_rmsnorm(void) {
    printf("\n=== Test: RMSNorm ===\n");
    
    float* input = malloc(TEST_N * sizeof(float));
    float* output = malloc(TEST_N * sizeof(float));
    const float epsilon = 1e-5f;
    
    // Test case: all ones
    // mean_squares = 1.0, inv_std = 1 / sqrt(1 + eps) ≈ 1 - eps/2
    // output ≈ 1 * (1 - eps/2) ≈ 0.99999
    for (u32 i = 0; i < TEST_N; i++) {
        input[i] = 1.0f;
    }
    
    hs_ml_rmsnorm(input, output, epsilon, TEST_N);
    
    // Verify RMSNorm
    float sum_squares = 0.0f;
    for (u32 i = 0; i < TEST_N; i++) {
        sum_squares += input[i] * input[i];
    }
    float mean_squares = sum_squares / TEST_N;
    float inv_std = 1.0f / sqrtf(mean_squares + epsilon);
    
    int passed = 1;
    for (u32 i = 0; i < TEST_N; i++) {
        float expected = input[i] * inv_std;
        if (fabs(output[i] - expected) > 1e-6f) {
            printf("  FAIL: output[%u] = %.8f, expected %.8f\n", i, output[i], expected);
            passed = 0;
        }
    }
    
    if (passed) printf("  PASSED: RMSNorm correct (all ones)\n");
    
    // Test case: [1, 2, 3, ..., 16]
    for (u32 i = 0; i < TEST_N; i++) {
        input[i] = (float)(i + 1);
    }
    
    hs_ml_rmsnorm(input, output, epsilon, TEST_N);
    
    sum_squares = 0.0f;
    for (u32 i = 0; i < TEST_N; i++) {
        sum_squares += input[i] * input[i];
    }
    mean_squares = sum_squares / TEST_N;
    inv_std = 1.0f / sqrtf(mean_squares + epsilon);
    
    for (u32 i = 0; i < TEST_N; i++) {
        float expected = input[i] * inv_std;
        if (fabs(output[i] - expected) > 1e-6f) {
            printf("  FAIL: output[%u] = %.8f, expected %.8f (ramp test)\n", i, output[i], expected);
            passed = 0;
        }
    }
    
    if (passed) printf("  PASSED: RMSNorm correct (ramp)\n");
    
    free(input);
    free(output);
    return passed;
}

int test_ffn_combined(void) {
    printf("\n=== Test: Combined FFN Block ===\n");
    
    // Simulate: x -> GEMM1 -> split -> activate_gate -> rmsnorm -> GEMM2
    // We'll test the middle part: activate_gate + rmsnorm
    
    const u32 N = 8;
    float* gemm1_output = malloc(2 * N * sizeof(float));  // [x1, x3] concatenated
    float* activated = malloc(N * sizeof(float));
    float* normalized = malloc(N * sizeof(float));
    float* final_output = malloc(N * sizeof(float));
    
    // Set up test data
    // x1 = [2, -1, 3, -2, 4, -3, 5, -4]
    // x3 = [1, 2, 3, 4, 5, 6, 7, 8]  (gate)
    for (u32 i = 0; i < N; i++) {
        gemm1_output[i] = (i % 2 == 0) ? (float)(i + 2) : -(float)(i + 1);  // 2, -1, 3, -2...
        gemm1_output[i + N] = (float)(i + 1);  // 1, 2, 3, 4, 5, 6, 7, 8
    }
    
    // Apply activate_gate
    hs_ml_ffn_activate_gate(gemm1_output, activated, N);
    
    // Apply RMSNorm
    hs_ml_rmsnorm(activated, normalized, 1e-5f, N);
    
    // Manual verification for first few elements
    printf("  Sample values:\n");
    for (u32 i = 0; i < 4 && i < N; i++) {
        float x = gemm1_output[i];
        float gate = gemm1_output[i + N];
        float activated_val = (x > 0.0f) ? x : 0.0f;
        activated_val = activated_val * activated_val;
        activated_val = activated_val * gate;
        printf("    i=%u: x=%.1f, gate=%.1f, activated=%.4f\n", i, x, gate, activated_val);
    }
    
    // Basic sanity check: output should not be NaN or extreme
    int passed = 1;
    for (u32 i = 0; i < N; i++) {
        if (!isfinite(normalized[i])) {
            printf("  FAIL: non-finite output at [%u]\n", i);
            passed = 0;
        }
        if (fabs(normalized[i]) > 1000.0f) {  // arbitrary large threshold
            printf("  FAIL: excessively large output at [%u]: %.4f\n", i, normalized[i]);
            passed = 0;
        }
    }
    
    if (passed) printf("  PASSED: Combined FFN block sanity check\n");
    
    free(gemm1_output);
    free(activated);
    free(normalized);
    free(final_output);
    return passed;
}

int main(void) {
    printf("NeoGPU ML - FFN Tests\n");
    printf("=====================\n");
    
    int total = 0;
    int passed = 0;
    
    total++; passed += test_ffn_activate_gate();
    total++; passed += test_rmsnorm();
    total++; passed += test_ffn_combined();
    
    printf("\n=====================\n");
    printf("Results: %d/%d tests passed\n", passed, total);
    
    return (passed == total) ? 0 : 1;
}