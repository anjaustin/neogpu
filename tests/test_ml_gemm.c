/*
 * NeoGPU ML - GEMM Test
 * Tests INT8/ternary GEMM correctness
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <arm_neon.h>

#include "hs_ml.h"

#define TEST_M 16
#define TEST_N 16
#define TEST_K 64

void print_matrix_f32(const char* name, float* m, u32 rows, u32 cols) {
    printf("%s (%ux%u):\n", name, rows, cols);
    for (u32 i = 0; i < rows && i < 4; i++) {
        printf("  row%u: ", i);
        for (u32 j = 0; j < cols && j < 4; j++) {
            printf("%.2f ", m[i * cols + j]);
        }
        if (cols > 4) printf("...");
        printf("\n");
    }
}

void print_matrix_s8(const char* name, int8_t* m, u32 rows, u32 cols) {
    printf("%s (%ux%u):\n", name, rows, cols);
    for (u32 i = 0; i < rows && i < 4; i++) {
        printf("  row%u: ", i);
        for (u32 j = 0; j < cols && j < 8; j++) {
            printf("%d ", m[i * cols + j]);
        }
        if (cols > 8) printf("...");
        printf("\n");
    }
}

void print_matrix_ternary(const char* name, u8* m, u32 rows, u32 cols) {
    printf("%s (%ux%u, packed):\n", name, rows, cols);
    for (u32 i = 0; i < rows && i < 4; i++) {
        printf("  row%u: ", i);
        for (u32 j = 0; j < cols && j < 8; j++) {
            u8 packed = m[i * (cols / 4) + j];
            for (int b = 0; b < 4; b++) {
                u8 bits = (packed >> (b * 2)) & 0x03;
                if (bits == 1) printf("+1 ");
                else if (bits == 2) printf("-1 ");
                else printf(" 0 ");
            }
        }
        if (cols > 8) printf("...");
        printf("\n");
    }
}

int test_gemm_basic(void) {
    printf("\n=== Test: Basic GEMM ===\n");
    
    /* Simple test: A = all 1s, B = all 1s, expect C = K */
    int8_t* A = malloc(TEST_M * TEST_K * sizeof(int8_t));
    u8* B = malloc(TEST_K * TEST_N / 4 * sizeof(u8));  /* packed */
    int32_t* C = malloc(TEST_M * TEST_N * sizeof(int32_t));
    float* scales = malloc(TEST_N * sizeof(float));
    
    /* A = all 1s */
    for (u32 i = 0; i < TEST_M * TEST_K; i++) {
        A[i] = 1;
    }
    
    /* B = all +1 (bits = 01) */
    memset(B, 0x55, TEST_K * TEST_N / 4);  /* 01010101 = all +1 */
    
    /* scales = 1.0 */
    for (u32 i = 0; i < TEST_N; i++) {
        scales[i] = 1.0f;
    }
    
    hs_ml_gemm_int8(C, A, B, scales, TEST_M, TEST_N, TEST_K);
    
    /* Expected: each output = sum of K 1*1 = K */
    int passed = 1;
    for (u32 m = 0; m < TEST_M; m++) {
        for (u32 n = 0; n < TEST_N; n++) {
            int32_t expected = TEST_K;
            if (C[m * TEST_N + n] != expected) {
                printf("  FAIL: C[%u][%u] = %d, expected %d\n", m, n, C[m * TEST_N + n], expected);
                passed = 0;
            }
        }
    }
    
    if (passed) printf("  PASSED: All outputs = %d\n", TEST_K);
    
    free(A);
    free(B);
    free(C);
    free(scales);
    
    return passed;
}

int test_gemm_negative(void) {
    printf("\n=== Test: Negative weights ===\n");
    
    int8_t* A = malloc(TEST_M * TEST_K * sizeof(int8_t));
    u8* B = malloc(TEST_K * TEST_N / 4 * sizeof(u8));
    int32_t* C = malloc(TEST_M * TEST_N * sizeof(int32_t));
    float* scales = malloc(TEST_N * sizeof(float));
    
    /* A = all 1s */
    for (u32 i = 0; i < TEST_M * TEST_K; i++) {
        A[i] = 1;
    }
    
    /* B = all -1 (bits = 10 = 0xAA) */
    memset(B, 0xAA, TEST_K * TEST_N / 4);
    
    for (u32 i = 0; i < TEST_N; i++) scales[i] = 1.0f;
    
    hs_ml_gemm_int8(C, A, B, scales, TEST_M, TEST_N, TEST_K);
    
    /* Expected: each output = sum of K * (1 * -1) = -K */
    int passed = 1;
    for (u32 m = 0; m < TEST_M; m++) {
        for (u32 n = 0; n < TEST_N; n++) {
            int32_t expected = -TEST_K;
            if (C[m * TEST_N + n] != expected) {
                printf("  FAIL: C[%u][%u] = %d, expected %d\n", m, n, C[m * TEST_N + n], expected);
                passed = 0;
            }
        }
    }
    
    if (passed) printf("  PASSED: All outputs = %d\n", -TEST_K);
    
    free(A);
    free(B);
    free(C);
    free(scales);
    
    return passed;
}

int test_gemm_mixed(void) {
    printf("\n=== Test: Mixed weights ===\n");
    
    /* A = [1, 2, 3, 4, ...] */
    /* B first column = [+1, -1, 0, +1, ...] */
    
    int8_t* A = malloc(TEST_M * TEST_K * sizeof(int8_t));
    u8* B = malloc(TEST_K * TEST_N / 4 * sizeof(u8));
    int32_t* C = malloc(TEST_M * TEST_N * sizeof(int32_t));
    float* scales = malloc(TEST_N * sizeof(float));
    
    for (u32 i = 0; i < TEST_M * TEST_K; i++) {
        A[i] = (i % 16) + 1;  /* 1, 2, 3, ... 16, 1, 2, ... */
    }
    
    /* B: column 0 = [+1, -1, 0, +1, -1, 0, ...] */
    /* 01 = +1, 10 = -1, 00 = 0 */
    memset(B, 0, TEST_K * TEST_N / 4);
    for (u32 k = 0; k < TEST_K; k++) {
        u8 bits;
        switch (k % 3) {
            case 0: bits = 0x01; break;  /* +1 */
            case 1: bits = 0x02; break;  /* -1 */
            default: bits = 0x00; break; /* 0 */
        }
        /* Pack into correct byte position */
        u32 byte_idx = k / 4;
        u32 bit_shift = (k % 4) * 2;
        B[byte_idx] |= (bits << bit_shift);
    }
    
    for (u32 i = 0; i < TEST_N; i++) scales[i] = 1.0f;
    
    hs_ml_gemm_int8(C, A, B, scales, TEST_M, TEST_N, TEST_K);
    
    /* Verify first column: sum of (A_row[k] * B[k]) */
    /* B[0..63] pattern: [+1, -1, 0, +1, -1, 0, ...] */
    /* A[0..63] pattern: [1, 2, 3, 4, 5, 6, ...] */
    /* Expected for row 0: 1*(+1) + 2*(-1) + 3*0 + 4*(+1) + 5*(-1) + ... */
    /* = 1 - 2 + 0 + 4 - 5 + 0 + 7 - 8 + ... */
    
    int passed = 1;
    for (u32 m = 0; m < TEST_M; m++) {
        int32_t expected = 0;
        for (u32 k = 0; k < TEST_K; k++) {
            int8_t a = A[m * TEST_K + k];
            int w;
            switch (k % 3) {
                case 0: w = 1; break;
                case 1: w = -1; break;
                default: w = 0;
            }
            expected += a * w;
        }
        
        if (C[m * TEST_N + 0] != expected) {
            printf("  FAIL: C[%u][0] = %d, expected %d\n", m, C[m * TEST_N + 0], expected);
            passed = 0;
        }
    }
    
    if (passed) printf("  PASSED: Mixed weights correct\n");
    
    free(A);
    free(B);
    free(C);
    free(scales);
    
    return passed;
}

int test_gemm_random(void) {
    printf("\n=== Test: Random data (100 iterations) ===\n");
    
    srand(42);  /* reproducible */
    
    int passed = 1;
    
    for (int iter = 0; iter < 100; iter++) {
        u32 M = 1 + rand() % 8;
        u32 N = 1 + rand() % 8;
        u32 K = 64;  /* Must be multiple of 64 for ternary */
        
        int8_t* A = malloc(M * K * sizeof(int8_t));
        u8* B = malloc(K * N / 4 * sizeof(u8));
        int32_t* C = malloc(M * N * sizeof(int32_t));
        float* scales = malloc(N * sizeof(float));
        
        /* Random A values in range [-10, 10] */
        for (u32 i = 0; i < M * K; i++) {
            A[i] = (rand() % 21) - 10;
        }
        
        /* Random ternary B */
        memset(B, 0, K * N / 4);
        for (u32 k = 0; k < K; k++) {
            for (u32 n = 0; n < N; n++) {
                u8 bits = rand() % 3;  /* 0, 1, 2 */
                if (bits == 3) bits = 0;  /* no 3 */
                
                u32 byte_idx = (n * K + k) / 4;
                u32 bit_shift = ((n * K + k) % 4) * 2;
                B[byte_idx] |= (bits << bit_shift);
            }
        }
        
        for (u32 i = 0; i < N; i++) scales[i] = 1.0f;
        
        hs_ml_gemm_int8(C, A, B, scales, M, N, K);
        
        /* Verify against scalar computation */
        for (u32 m = 0; m < M && passed; m++) {
            for (u32 n = 0; n < N && passed; n++) {
                int32_t expected = 0;
                for (u32 k = 0; k < K; k++) {
                    u32 byte_idx = (n * K + k) / 4;
                    u32 bit_shift = ((n * K + k) % 4) * 2;
                    u8 bits = (B[byte_idx] >> bit_shift) & 0x03;
                    
                    int w;
                    if (bits == 1) w = 1;
                    else if (bits == 2) w = -1;
                    else w = 0;
                    
                    expected += (int32_t)A[m * K + k] * w;
                }
                
                if (C[m * N + n] != expected) {
                    printf("  FAIL iter %d: C[%u][%u] = %d, expected %d\n", 
                           iter, m, n, C[m * N + n], expected);
                    passed = 0;
                }
            }
        }
        
        free(A);
        free(B);
        free(C);
        free(scales);
        
        if (!passed) break;
    }
    
    if (passed) printf("  PASSED: All 100 random tests\n");
    
    return passed;
}

int test_quantize(void) {
    printf("\n=== Test: Ternary quantization ===\n");
    
    /* Test the quantization function */
    const u32 N = 128;
    float* input = malloc(N * sizeof(float));
    u8* output = malloc(N / 4 * sizeof(u8));
    float* scales = malloc((N / 64) * sizeof(float));
    
    /* Input: random values */
    srand(123);
    for (u32 i = 0; i < N; i++) {
        input[i] = ((float)(rand() % 100) - 50.0f) / 10.0f;
    }
    
    hs_ml_quantize_ternary(input, output, scales, N);
    
    /* Verify: for each non-zero quantized value, check scale */
    printf("  Scales computed: ");
    u32 num_blocks = (N + 63) / 64;
    for (u32 b = 0; b < num_blocks; b++) {
        printf("%.2f ", scales[b]);
    }
    printf("\n");
    
    /* Check that quantized values are -1, 0, or +1 */
    int passed = 1;
    for (u32 k = 0; k < N; k++) {
        u32 byte_idx = k / 4;
        u32 bit_shift = (k % 4) * 2;
        u8 bits = (output[byte_idx] >> bit_shift) & 0x03;
        
        if (bits > 2) {
            printf("  FAIL: Invalid ternary value %u at position %u\n", bits, k);
            passed = 0;
        }
    }
    
    if (passed) printf("  PASSED: Quantization produces valid ternary\n");
    
    free(input);
    free(output);
    free(scales);
    
    return passed;
}

int test_dequantize(void) {
    printf("\n=== Test: Dequantization ===\n");
    
    const u32 M = 4;
    const u32 N = 8;
    
    int32_t* C = malloc(M * N * sizeof(int32_t));
    float* scales = malloc(M * sizeof(float));
    float* output = malloc(M * N * sizeof(float));
    
    /* Setup: C values and scales */
    for (u32 m = 0; m < M; m++) {
        scales[m] = 0.5f;
        for (u32 n = 0; n < N; n++) {
            C[m * N + n] = (m * N + n) * 2;  /* 0, 2, 4, 6, ... */
        }
    }
    
    hs_ml_dequantize(C, scales, output, M, N);
    
    /* Verify: output = C * scale */
    int passed = 1;
    for (u32 m = 0; m < M; m++) {
        for (u32 n = 0; n < N; n++) {
            float expected = (float)C[m * N + n] * scales[m];
            if (output[m * N + n] != expected) {
                printf("  FAIL: output[%u][%u] = %f, expected %f\n", 
                       m, n, output[m * N + n], expected);
                passed = 0;
            }
        }
    }
    
    if (passed) printf("  PASSED: Dequantization correct\n");
    
    free(C);
    free(scales);
    free(output);
    
    return passed;
}

int main(void) {
    printf("NeoGPU ML - GEMM Tests\n");
    printf("======================\n");
    
    int total = 0;
    int passed = 0;
    
    total++; passed += test_gemm_basic();
    total++; passed += test_gemm_negative();
    total++; passed += test_gemm_mixed();
    total++; passed += test_gemm_random();
    total++; passed += test_quantize();
    total++; passed += test_dequantize();
    
    printf("\n======================\n");
    printf("Results: %d/%d tests passed\n", passed, total);
    
    return (passed == total) ? 0 : 1;
}
