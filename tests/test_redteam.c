/*
 * Red-team tests for ternary GEMM kernel - FIXED
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "hs_ml.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("Testing: %s... ", name)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)

/* Reference scalar implementation */
void gemm_reference(int32_t* C, const int8_t* A, const u8* B, u32 M, u32 N, u32 K) {
    memset(C, 0, M * N * sizeof(int32_t));
    for (u32 m = 0; m < M; m++) {
        for (u32 n = 0; n < N; n++) {
            int32_t sum = 0;
            for (u32 k = 0; k < K; k++) {
                u8 byte = B[n * (K/4) + k/4];
                u8 bits = (byte >> ((k % 4) * 2)) & 0x03;
                int8_t a = A[m * K + k];
                if (bits == 1) sum += a;
                else if (bits == 2) sum -= a;
            }
            C[m * N + n] = sum;
        }
    }
}

void test_k_remainder(void) {
    TEST("K not divisible by 64");
    u32 M = 1, N = 4, K = 100;
    int8_t* A = malloc(M * K);
    u8* B = malloc(N * (K / 4));
    int32_t* C = malloc(M * N * sizeof(int32_t));
    int32_t* C_ref = malloc(M * N * sizeof(int32_t));
    float scales[4] = {1,1,1,1};
    for (u32 i = 0; i < M * K; i++) A[i] = (i % 11) - 5;
    for (u32 i = 0; i < N * (K/4); i++) B[i] = 0x55;
    hs_ml_gemm_int8(C, A, B, scales, M, N, K);
    gemm_reference(C_ref, A, B, M, N, K);
    int ok = 1;
    for (u32 i = 0; i < M * N; i++) if (C[i] != C_ref[i]) { ok = 0; break; }
    if (ok) PASS(); else FAIL("Mismatch");
    free(A); free(B); free(C); free(C_ref);
}

void test_n_remainder(void) {
    TEST("N not divisible by 4");
    u32 M = 1, N = 7, K = 64;
    int8_t* A = malloc(M * K);
    u8* B = malloc(N * (K / 4));
    int32_t* C = malloc(M * N * sizeof(int32_t));
    int32_t* C_ref = malloc(M * N * sizeof(int32_t));
    float scales[8] = {1,1,1,1,1,1,1,1};
    for (u32 i = 0; i < M * K; i++) A[i] = (i % 7) - 3;
    for (u32 i = 0; i < N * (K/4); i++) B[i] = 0x55;
    hs_ml_gemm_int8(C, A, B, scales, M, N, K);
    gemm_reference(C_ref, A, B, M, N, K);
    int ok = 1;
    for (u32 i = 0; i < M * N; i++) if (C[i] != C_ref[i]) { ok = 0; break; }
    if (ok) PASS(); else FAIL("Mismatch");
    free(A); free(B); free(C); free(C_ref);
}

void test_small_dims(void) {
    TEST("Small dimensions (M=1, N=1, K=4)");
    int8_t A[4] = {1, 2, 3, 4};
    u8 B[1] = {0x55};
    int32_t C[1];
    float scales[1] = {1};
    hs_ml_gemm_int8(C, A, B, scales, 1, 1, 4);
    if (C[0] == 10) PASS();
    else { printf("FAIL: got %d, expected 10\n", C[0]); tests_failed++; }
}

void test_all_zeros(void) {
    TEST("All zero weights");
    u32 M = 2, N = 4, K = 64;
    int8_t* A = malloc(M * K);
    u8* B = malloc(N * (K / 4));
    int32_t* C = malloc(M * N * sizeof(int32_t));
    float scales[4] = {1,1,1,1};
    for (u32 i = 0; i < M * K; i++) A[i] = 127;
    memset(B, 0x00, N * (K / 4));
    hs_ml_gemm_int8(C, A, B, scales, M, N, K);
    int ok = 1;
    for (u32 i = 0; i < M * N; i++) if (C[i] != 0) { ok = 0; break; }
    if (ok) PASS(); else FAIL("Expected zeros");
    free(A); free(B); free(C);
}

void test_all_minus1(void) {
    TEST("All -1 weights");
    u32 K = 64;
    int8_t* A = malloc(K);
    u8* B = malloc(K / 4);
    int32_t C[1];
    float scales[1] = {1};
    for (u32 i = 0; i < K; i++) A[i] = 1;
    memset(B, 0xAA, K / 4);
    hs_ml_gemm_int8(C, A, B, scales, 1, 1, K);
    if (C[0] == -64) PASS();
    else { printf("FAIL: got %d, expected -64\n", C[0]); tests_failed++; }
    free(A); free(B);
}

void test_mixed_pattern(void) {
    TEST("Mixed weight pattern (FIXED)");
    int8_t A[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    /* Correct encoding: 
     * B[0]: w0=+1(01), w1=-1(10), w2=0(00), w3=+1(01) = 01_00_10_01 = 0x49
     * B[1]: w4=0(00), w5=+1(01), w6=-1(10), w7=0(00) = 00_10_01_00 = 0x24
     */
    u8 B[2] = {0x49, 0x24};
    int32_t C[1];
    float scales[1] = {1};
    hs_ml_gemm_int8(C, A, B, scales, 1, 1, 8);
    int32_t expected = 10 - 20 + 0 + 40 + 0 + 60 - 70 + 0;  /* = 20 */
    if (C[0] == expected) PASS();
    else { printf("FAIL: got %d, expected %d\n", C[0], expected); tests_failed++; }
}

void test_extreme_activations(void) {
    TEST("Extreme activations (INT8 min/max)");
    u32 K = 64;
    int8_t* A = malloc(K);
    u8* B = malloc(K / 4);
    int32_t C[1];
    float scales[1] = {1};
    for (u32 i = 0; i < K; i++) A[i] = (i % 2) ? 127 : -128;
    memset(B, 0x55, K / 4);
    hs_ml_gemm_int8(C, A, B, scales, 1, 1, K);
    int32_t expected = 32 * (-128) + 32 * 127;
    if (C[0] == expected) PASS();
    else { printf("FAIL: got %d, expected %d\n", C[0], expected); tests_failed++; }
    free(A); free(B);
}

void test_large_k_accumulation(void) {
    TEST("Large K accumulation (K=4096)");
    u32 K = 4096;
    int8_t* A = malloc(K);
    u8* B = malloc(K / 4);
    int32_t C[1], C_ref[1];
    float scales[1] = {1};
    for (u32 i = 0; i < K; i++) A[i] = 100;
    memset(B, 0x55, K / 4);
    hs_ml_gemm_int8(C, A, B, scales, 1, 1, K);
    gemm_reference(C_ref, A, B, 1, 1, K);
    if (C[0] == C_ref[0] && C[0] == 409600) PASS();
    else { printf("FAIL: got %d, expected 409600\n", C[0]); tests_failed++; }
    free(A); free(B);
}

void test_weight_0x03(void) {
    TEST("Weight encoding 0x03 treated as zero");
    int8_t A[4] = {10, 20, 30, 40};
    u8 B[1] = {0xFF};
    int32_t C[1];
    float scales[1] = {1};
    hs_ml_gemm_int8(C, A, B, scales, 1, 1, 4);
    if (C[0] == 0) PASS();
    else { printf("FAIL: got %d, expected 0\n", C[0]); tests_failed++; }
}

void test_multiple_rows(void) {
    TEST("Multiple rows (M=4)");
    u32 M = 4, N = 4, K = 64;
    int8_t* A = malloc(M * K);
    u8* B = malloc(N * (K / 4));
    int32_t* C = malloc(M * N * sizeof(int32_t));
    int32_t* C_ref = malloc(M * N * sizeof(int32_t));
    float scales[4] = {1,1,1,1};
    for (u32 i = 0; i < M * K; i++) A[i] = (i % 13) - 6;
    for (u32 i = 0; i < N * (K/4); i++) B[i] = 0x55 ^ (i & 0xFF);
    hs_ml_gemm_int8(C, A, B, scales, M, N, K);
    gemm_reference(C_ref, A, B, M, N, K);
    int ok = 1;
    for (u32 i = 0; i < M * N; i++) if (C[i] != C_ref[i]) { ok = 0; break; }
    if (ok) PASS(); else FAIL("Mismatch");
    free(A); free(B); free(C); free(C_ref);
}

void test_unaligned_pointers(void) {
    TEST("Unaligned pointers");
    u32 M = 1, N = 4, K = 128;
    int8_t* A_base = malloc(M * K + 16);
    u8* B_base = malloc(N * (K / 4) + 16);
    int32_t* C = malloc(M * N * sizeof(int32_t));
    int32_t* C_ref = malloc(M * N * sizeof(int32_t));
    float scales[4] = {1,1,1,1};
    int8_t* A = A_base + 1;
    u8* B = B_base + 1;
    for (u32 i = 0; i < M * K; i++) A[i] = i % 10;
    for (u32 i = 0; i < N * (K/4); i++) B[i] = 0x55;
    hs_ml_gemm_int8(C, A, B, scales, M, N, K);
    gemm_reference(C_ref, A, B, M, N, K);
    int ok = 1;
    for (u32 i = 0; i < M * N; i++) if (C[i] != C_ref[i]) { ok = 0; break; }
    if (ok) PASS(); else FAIL("Mismatch");
    free(A_base); free(B_base); free(C); free(C_ref);
}

void test_random_stress(void) {
    TEST("Random stress (1000 iters)");
    int failures = 0;
    for (int iter = 0; iter < 1000; iter++) {
        u32 M = 1 + (rand() % 4);
        u32 N = 4 + (rand() % 60);
        u32 K = 64 + (rand() % 500);
        K = (K / 4) * 4;
        int8_t* A = malloc(M * K);
        u8* B = malloc(N * (K / 4));
        int32_t* C = malloc(M * N * sizeof(int32_t));
        int32_t* C_ref = malloc(M * N * sizeof(int32_t));
        float* scales = malloc(N * sizeof(float));
        for (u32 i = 0; i < M * K; i++) A[i] = rand() % 256 - 128;
        for (u32 i = 0; i < N * (K/4); i++) B[i] = rand() % 256;
        for (u32 i = 0; i < N; i++) scales[i] = 1.0f;
        hs_ml_gemm_int8(C, A, B, scales, M, N, K);
        gemm_reference(C_ref, A, B, M, N, K);
        for (u32 i = 0; i < M * N; i++) {
            if (C[i] != C_ref[i]) { failures++; break; }
        }
        free(A); free(B); free(C); free(C_ref); free(scales);
    }
    if (failures == 0) PASS();
    else { printf("FAIL: %d failures\n", failures); tests_failed++; }
}

/* Additional red-team tests */

void test_k_exactly_64(void) {
    TEST("K exactly 64 (boundary)");
    u32 M = 1, N = 4, K = 64;
    int8_t* A = malloc(M * K);
    u8* B = malloc(N * (K / 4));
    int32_t* C = malloc(M * N * sizeof(int32_t));
    int32_t* C_ref = malloc(M * N * sizeof(int32_t));
    float scales[4] = {1,1,1,1};
    for (u32 i = 0; i < M * K; i++) A[i] = i - 32;
    for (u32 i = 0; i < N * (K/4); i++) B[i] = 0x69;  /* mixed pattern */
    hs_ml_gemm_int8(C, A, B, scales, M, N, K);
    gemm_reference(C_ref, A, B, M, N, K);
    int ok = 1;
    for (u32 i = 0; i < M * N; i++) if (C[i] != C_ref[i]) { ok = 0; break; }
    if (ok) PASS(); else FAIL("Mismatch");
    free(A); free(B); free(C); free(C_ref);
}

void test_n_exactly_4(void) {
    TEST("N exactly 4 (no remainder)");
    u32 M = 2, N = 4, K = 128;
    int8_t* A = malloc(M * K);
    u8* B = malloc(N * (K / 4));
    int32_t* C = malloc(M * N * sizeof(int32_t));
    int32_t* C_ref = malloc(M * N * sizeof(int32_t));
    float scales[4] = {1,1,1,1};
    for (u32 i = 0; i < M * K; i++) A[i] = (i * 7) % 256 - 128;
    for (u32 i = 0; i < N * (K/4); i++) B[i] = (i * 13) % 256;
    hs_ml_gemm_int8(C, A, B, scales, M, N, K);
    gemm_reference(C_ref, A, B, M, N, K);
    int ok = 1;
    for (u32 i = 0; i < M * N; i++) if (C[i] != C_ref[i]) { ok = 0; break; }
    if (ok) PASS(); else FAIL("Mismatch");
    free(A); free(B); free(C); free(C_ref);
}

void test_n_equals_1(void) {
    TEST("N equals 1 (single column)");
    u32 M = 2, N = 1, K = 256;
    int8_t* A = malloc(M * K);
    u8* B = malloc(N * (K / 4));
    int32_t* C = malloc(M * N * sizeof(int32_t));
    int32_t* C_ref = malloc(M * N * sizeof(int32_t));
    float scales[1] = {1};
    for (u32 i = 0; i < M * K; i++) A[i] = (i % 50) - 25;
    for (u32 i = 0; i < N * (K/4); i++) B[i] = 0x55;
    hs_ml_gemm_int8(C, A, B, scales, M, N, K);
    gemm_reference(C_ref, A, B, M, N, K);
    int ok = 1;
    for (u32 i = 0; i < M * N; i++) if (C[i] != C_ref[i]) { ok = 0; break; }
    if (ok) PASS(); else FAIL("Mismatch");
    free(A); free(B); free(C); free(C_ref);
}

void test_overflow_potential(void) {
    TEST("INT16 overflow potential (K=512, all +1, A=127)");
    /* 512 * 127 = 65024, which fits in int16 but is close to overflow */
    u32 K = 512;
    int8_t* A = malloc(K);
    u8* B = malloc(K / 4);
    int32_t C[1], C_ref[1];
    float scales[1] = {1};
    for (u32 i = 0; i < K; i++) A[i] = 127;
    memset(B, 0x55, K / 4);
    hs_ml_gemm_int8(C, A, B, scales, 1, 1, K);
    gemm_reference(C_ref, A, B, 1, 1, K);
    if (C[0] == C_ref[0] && C[0] == 65024) PASS();
    else { printf("FAIL: got %d, expected 65024\n", C[0]); tests_failed++; }
    free(A); free(B);
}

int main(void) {
    printf("\n========================================\n");
    printf("Red-Team Tests for Ternary GEMM Kernel\n");
    printf("========================================\n\n");
    srand(12345);
    
    test_k_remainder();
    test_n_remainder();
    test_small_dims();
    test_all_zeros();
    test_all_minus1();
    test_mixed_pattern();
    test_extreme_activations();
    test_large_k_accumulation();
    test_weight_0x03();
    test_multiple_rows();
    test_unaligned_pointers();
    test_random_stress();
    test_k_exactly_64();
    test_n_exactly_4();
    test_n_equals_1();
    test_overflow_potential();
    
    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");
    
    return tests_failed > 0 ? 1 : 0;
}
