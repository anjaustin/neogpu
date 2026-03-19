/*
 * NeoGPU ML - Ternary GEMM v18 - Optimized nibble LUT
 * 
 * Based on V17 which achieved 5.2 GOPS
 * Improvements:
 * - Better LUT design - single lookup per nibble
 * - Process 6 columns to better utilize registers
 * - Aggressive prefetching
 */

#include "hs_ml.h"
#include <arm_neon.h>
#include <string.h>

void hs_ml_gemm_ternary_v18(int32_t* C,
                            const int8_t* A,
                            const u8* B_ternary,
                            u32 M, u32 N, u32 K) {
    
    memset(C, 0, M * N * sizeof(int32_t));
    
    const u32 K64 = K & ~63u;
    const u32 N4 = N & ~3u;
    const u32 Kstride = K / 4;
    
    /* 
     * Nibble LUT: 4-bit index -> 2 signed weights
     * Index bits [1:0] -> w0, bits [3:2] -> w1  
     * Encoding: 00=0, 01=+1, 10=-1, 11=0
     */
    static const int8_t nibble_w0[16] __attribute__((aligned(16))) = {
        0, 1, -1, 0,  0, 1, -1, 0,  0, 1, -1, 0,  0, 1, -1, 0
    };
    static const int8_t nibble_w1[16] __attribute__((aligned(16))) = {
        0, 0, 0, 0,  1, 1, 1, 1,  -1, -1, -1, -1,  0, 0, 0, 0
    };
    
    int8x16_t lut_w0 = vld1q_s8(nibble_w0);
    int8x16_t lut_w1 = vld1q_s8(nibble_w1);
    
    for (u32 m = 0; m < M; m++) {
        const int8_t* A_row = A + m * K;
        int32_t* C_row = C + m * N;
        
        for (u32 n = 0; n < N4; n += 4) {
            const u8* B0 = B_ternary + (n + 0) * Kstride;
            const u8* B1 = B_ternary + (n + 1) * Kstride;
            const u8* B2 = B_ternary + (n + 2) * Kstride;
            const u8* B3 = B_ternary + (n + 3) * Kstride;
            
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);
            
            for (u32 k = 0; k < K64; k += 64) {
                __builtin_prefetch(A_row + k + 128, 0, 3);
                __builtin_prefetch(B0 + (k + 128) / 4, 0, 3);
                __builtin_prefetch(B1 + (k + 128) / 4, 0, 3);
                
                /* Load activations */
                int8x16_t a0 = vld1q_s8(A_row + k + 0);
                int8x16_t a1 = vld1q_s8(A_row + k + 16);
                int8x16_t a2 = vld1q_s8(A_row + k + 32);
                int8x16_t a3 = vld1q_s8(A_row + k + 48);
                
                /* Deinterleave to get stride-4 groups */
                int8x16x2_t d0 = vuzpq_s8(a0, a1);
                int8x16x2_t d1 = vuzpq_s8(a2, a3);
                int8x16x2_t g01 = vuzpq_s8(d0.val[0], d1.val[0]);
                int8x16x2_t g23 = vuzpq_s8(d0.val[1], d1.val[1]);
                
                int8x16_t ag0 = g01.val[0];
                int8x16_t ag2 = g01.val[1];
                int8x16_t ag1 = g23.val[0];
                int8x16_t ag3 = g23.val[1];
                
                u32 ko = k / 4;
                
                #define PROC_COL_V18(Bptr, acc) do { \
                    uint8x16_t wb = vld1q_u8(Bptr + ko); \
                    \
                    /* Extract nibbles */ \
                    uint8x16_t lo_nib = vandq_u8(wb, vdupq_n_u8(0x0F)); \
                    uint8x16_t hi_nib = vshrq_n_u8(wb, 4); \
                    \
                    /* Lookup weights for each nibble position */ \
                    int8x16_t w0 = vqtbl1q_s8(lut_w0, lo_nib); \
                    int8x16_t w1 = vqtbl1q_s8(lut_w1, lo_nib); \
                    int8x16_t w2 = vqtbl1q_s8(lut_w0, hi_nib); \
                    int8x16_t w3 = vqtbl1q_s8(lut_w1, hi_nib); \
                    \
                    /* Multiply-accumulate */ \
                    int16x8_t prod = vmull_s8(vget_low_s8(w0), vget_low_s8(ag0)); \
                    prod = vmlal_s8(prod, vget_high_s8(w0), vget_high_s8(ag0)); \
                    prod = vmlal_s8(prod, vget_low_s8(w1), vget_low_s8(ag1)); \
                    prod = vmlal_s8(prod, vget_high_s8(w1), vget_high_s8(ag1)); \
                    prod = vmlal_s8(prod, vget_low_s8(w2), vget_low_s8(ag2)); \
                    prod = vmlal_s8(prod, vget_high_s8(w2), vget_high_s8(ag2)); \
                    prod = vmlal_s8(prod, vget_low_s8(w3), vget_low_s8(ag3)); \
                    prod = vmlal_s8(prod, vget_high_s8(w3), vget_high_s8(ag3)); \
                    acc = vpadalq_s16(acc, prod); \
                } while(0)
                
                PROC_COL_V18(B0, acc0);
                PROC_COL_V18(B1, acc1);
                PROC_COL_V18(B2, acc2);
                PROC_COL_V18(B3, acc3);
                
                #undef PROC_COL_V18
            }
            
            C_row[n + 0] = vaddvq_s32(acc0);
            C_row[n + 1] = vaddvq_s32(acc1);
            C_row[n + 2] = vaddvq_s32(acc2);
            C_row[n + 3] = vaddvq_s32(acc3);
            
            /* Remainder K */
            for (u32 col = 0; col < 4; col++) {
                const u8* B_col = B_ternary + (n + col) * Kstride;
                for (u32 k = K64; k < K; k++) {
                    u8 byte = B_col[k / 4];
                    u8 bits = (byte >> ((k % 4) * 2)) & 0x03;
                    int8_t a = A_row[k];
                    if (bits == 1) C_row[n + col] += a;
                    else if (bits == 2) C_row[n + col] -= a;
                }
            }
        }
        
        /* Remainder N */
        for (u32 n = N4; n < N; n++) {
            const u8* B_col = B_ternary + n * Kstride;
            int32_t sum = 0;
            for (u32 k = 0; k < K; k++) {
                u8 byte = B_col[k / 4];
                u8 bits = (byte >> ((k % 4) * 2)) & 0x03;
                int8_t a = A_row[k];
                if (bits == 1) sum += a;
                else if (bits == 2) sum -= a;
            }
            C_row[n] = sum;
        }
    }
}

/*
 * V19: Process 128 elements at a time with unrolled inner loop
 */
void hs_ml_gemm_ternary_v19(int32_t* C,
                            const int8_t* A,
                            const u8* B_ternary,
                            u32 M, u32 N, u32 K) {
    
    memset(C, 0, M * N * sizeof(int32_t));
    
    const u32 K128 = K & ~127u;
    const u32 K64 = K & ~63u;
    const u32 N4 = N & ~3u;
    const u32 Kstride = K / 4;
    
    static const int8_t nibble_w0[16] __attribute__((aligned(16))) = {
        0, 1, -1, 0,  0, 1, -1, 0,  0, 1, -1, 0,  0, 1, -1, 0
    };
    static const int8_t nibble_w1[16] __attribute__((aligned(16))) = {
        0, 0, 0, 0,  1, 1, 1, 1,  -1, -1, -1, -1,  0, 0, 0, 0
    };
    
    int8x16_t lut_w0 = vld1q_s8(nibble_w0);
    int8x16_t lut_w1 = vld1q_s8(nibble_w1);
    
    for (u32 m = 0; m < M; m++) {
        const int8_t* A_row = A + m * K;
        int32_t* C_row = C + m * N;
        
        for (u32 n = 0; n < N4; n += 4) {
            const u8* B0 = B_ternary + (n + 0) * Kstride;
            const u8* B1 = B_ternary + (n + 1) * Kstride;
            const u8* B2 = B_ternary + (n + 2) * Kstride;
            const u8* B3 = B_ternary + (n + 3) * Kstride;
            
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);
            
            /* Process 128 elements at a time */
            for (u32 k = 0; k < K128; k += 128) {
                __builtin_prefetch(A_row + k + 192, 0, 3);
                
                /* First 64 elements */
                {
                    int8x16_t a0 = vld1q_s8(A_row + k + 0);
                    int8x16_t a1 = vld1q_s8(A_row + k + 16);
                    int8x16_t a2 = vld1q_s8(A_row + k + 32);
                    int8x16_t a3 = vld1q_s8(A_row + k + 48);
                    
                    int8x16x2_t d0 = vuzpq_s8(a0, a1);
                    int8x16x2_t d1 = vuzpq_s8(a2, a3);
                    int8x16x2_t g01 = vuzpq_s8(d0.val[0], d1.val[0]);
                    int8x16x2_t g23 = vuzpq_s8(d0.val[1], d1.val[1]);
                    
                    int8x16_t ag0 = g01.val[0], ag1 = g23.val[0];
                    int8x16_t ag2 = g01.val[1], ag3 = g23.val[1];
                    
                    u32 ko = k / 4;
                    
                    #define PROC64(Bptr, acc) do { \
                        uint8x16_t wb = vld1q_u8(Bptr + ko); \
                        uint8x16_t lo_nib = vandq_u8(wb, vdupq_n_u8(0x0F)); \
                        uint8x16_t hi_nib = vshrq_n_u8(wb, 4); \
                        int8x16_t w0 = vqtbl1q_s8(lut_w0, lo_nib); \
                        int8x16_t w1 = vqtbl1q_s8(lut_w1, lo_nib); \
                        int8x16_t w2 = vqtbl1q_s8(lut_w0, hi_nib); \
                        int8x16_t w3 = vqtbl1q_s8(lut_w1, hi_nib); \
                        int16x8_t prod = vmull_s8(vget_low_s8(w0), vget_low_s8(ag0)); \
                        prod = vmlal_s8(prod, vget_high_s8(w0), vget_high_s8(ag0)); \
                        prod = vmlal_s8(prod, vget_low_s8(w1), vget_low_s8(ag1)); \
                        prod = vmlal_s8(prod, vget_high_s8(w1), vget_high_s8(ag1)); \
                        prod = vmlal_s8(prod, vget_low_s8(w2), vget_low_s8(ag2)); \
                        prod = vmlal_s8(prod, vget_high_s8(w2), vget_high_s8(ag2)); \
                        prod = vmlal_s8(prod, vget_low_s8(w3), vget_low_s8(ag3)); \
                        prod = vmlal_s8(prod, vget_high_s8(w3), vget_high_s8(ag3)); \
                        acc = vpadalq_s16(acc, prod); \
                    } while(0)
                    
                    PROC64(B0, acc0);
                    PROC64(B1, acc1);
                    PROC64(B2, acc2);
                    PROC64(B3, acc3);
                    #undef PROC64
                }
                
                /* Second 64 elements */
                {
                    int8x16_t a0 = vld1q_s8(A_row + k + 64);
                    int8x16_t a1 = vld1q_s8(A_row + k + 80);
                    int8x16_t a2 = vld1q_s8(A_row + k + 96);
                    int8x16_t a3 = vld1q_s8(A_row + k + 112);
                    
                    int8x16x2_t d0 = vuzpq_s8(a0, a1);
                    int8x16x2_t d1 = vuzpq_s8(a2, a3);
                    int8x16x2_t g01 = vuzpq_s8(d0.val[0], d1.val[0]);
                    int8x16x2_t g23 = vuzpq_s8(d0.val[1], d1.val[1]);
                    
                    int8x16_t ag0 = g01.val[0], ag1 = g23.val[0];
                    int8x16_t ag2 = g01.val[1], ag3 = g23.val[1];
                    
                    u32 ko = (k + 64) / 4;
                    
                    #define PROC64(Bptr, acc) do { \
                        uint8x16_t wb = vld1q_u8(Bptr + ko); \
                        uint8x16_t lo_nib = vandq_u8(wb, vdupq_n_u8(0x0F)); \
                        uint8x16_t hi_nib = vshrq_n_u8(wb, 4); \
                        int8x16_t w0 = vqtbl1q_s8(lut_w0, lo_nib); \
                        int8x16_t w1 = vqtbl1q_s8(lut_w1, lo_nib); \
                        int8x16_t w2 = vqtbl1q_s8(lut_w0, hi_nib); \
                        int8x16_t w3 = vqtbl1q_s8(lut_w1, hi_nib); \
                        int16x8_t prod = vmull_s8(vget_low_s8(w0), vget_low_s8(ag0)); \
                        prod = vmlal_s8(prod, vget_high_s8(w0), vget_high_s8(ag0)); \
                        prod = vmlal_s8(prod, vget_low_s8(w1), vget_low_s8(ag1)); \
                        prod = vmlal_s8(prod, vget_high_s8(w1), vget_high_s8(ag1)); \
                        prod = vmlal_s8(prod, vget_low_s8(w2), vget_low_s8(ag2)); \
                        prod = vmlal_s8(prod, vget_high_s8(w2), vget_high_s8(ag2)); \
                        prod = vmlal_s8(prod, vget_low_s8(w3), vget_low_s8(ag3)); \
                        prod = vmlal_s8(prod, vget_high_s8(w3), vget_high_s8(ag3)); \
                        acc = vpadalq_s16(acc, prod); \
                    } while(0)
                    
                    PROC64(B0, acc0);
                    PROC64(B1, acc1);
                    PROC64(B2, acc2);
                    PROC64(B3, acc3);
                    #undef PROC64
                }
            }
            
            /* Remaining 64-element blocks */
            for (u32 k = K128; k < K64; k += 64) {
                int8x16_t a0 = vld1q_s8(A_row + k + 0);
                int8x16_t a1 = vld1q_s8(A_row + k + 16);
                int8x16_t a2 = vld1q_s8(A_row + k + 32);
                int8x16_t a3 = vld1q_s8(A_row + k + 48);
                
                int8x16x2_t d0 = vuzpq_s8(a0, a1);
                int8x16x2_t d1 = vuzpq_s8(a2, a3);
                int8x16x2_t g01 = vuzpq_s8(d0.val[0], d1.val[0]);
                int8x16x2_t g23 = vuzpq_s8(d0.val[1], d1.val[1]);
                
                int8x16_t ag0 = g01.val[0], ag1 = g23.val[0];
                int8x16_t ag2 = g01.val[1], ag3 = g23.val[1];
                
                u32 ko = k / 4;
                
                #define PROC64(Bptr, acc) do { \
                    uint8x16_t wb = vld1q_u8(Bptr + ko); \
                    uint8x16_t lo_nib = vandq_u8(wb, vdupq_n_u8(0x0F)); \
                    uint8x16_t hi_nib = vshrq_n_u8(wb, 4); \
                    int8x16_t w0 = vqtbl1q_s8(lut_w0, lo_nib); \
                    int8x16_t w1 = vqtbl1q_s8(lut_w1, lo_nib); \
                    int8x16_t w2 = vqtbl1q_s8(lut_w0, hi_nib); \
                    int8x16_t w3 = vqtbl1q_s8(lut_w1, hi_nib); \
                    int16x8_t prod = vmull_s8(vget_low_s8(w0), vget_low_s8(ag0)); \
                    prod = vmlal_s8(prod, vget_high_s8(w0), vget_high_s8(ag0)); \
                    prod = vmlal_s8(prod, vget_low_s8(w1), vget_low_s8(ag1)); \
                    prod = vmlal_s8(prod, vget_high_s8(w1), vget_high_s8(ag1)); \
                    prod = vmlal_s8(prod, vget_low_s8(w2), vget_low_s8(ag2)); \
                    prod = vmlal_s8(prod, vget_high_s8(w2), vget_high_s8(ag2)); \
                    prod = vmlal_s8(prod, vget_low_s8(w3), vget_low_s8(ag3)); \
                    prod = vmlal_s8(prod, vget_high_s8(w3), vget_high_s8(ag3)); \
                    acc = vpadalq_s16(acc, prod); \
                } while(0)
                
                PROC64(B0, acc0);
                PROC64(B1, acc1);
                PROC64(B2, acc2);
                PROC64(B3, acc3);
                #undef PROC64
            }
            
            C_row[n + 0] = vaddvq_s32(acc0);
            C_row[n + 1] = vaddvq_s32(acc1);
            C_row[n + 2] = vaddvq_s32(acc2);
            C_row[n + 3] = vaddvq_s32(acc3);
            
            /* Remainder K */
            for (u32 col = 0; col < 4; col++) {
                const u8* B_col = B_ternary + (n + col) * Kstride;
                for (u32 k = K64; k < K; k++) {
                    u8 byte = B_col[k / 4];
                    u8 bits = (byte >> ((k % 4) * 2)) & 0x03;
                    int8_t a = A_row[k];
                    if (bits == 1) C_row[n + col] += a;
                    else if (bits == 2) C_row[n + col] -= a;
                }
            }
        }
        
        /* Remainder N */
        for (u32 n = N4; n < N; n++) {
            const u8* B_col = B_ternary + n * Kstride;
            int32_t sum = 0;
            for (u32 k = 0; k < K; k++) {
                u8 byte = B_col[k / 4];
                u8 bits = (byte >> ((k % 4) * 2)) & 0x03;
                int8_t a = A_row[k];
                if (bits == 1) sum += a;
                else if (bits == 2) sum -= a;
            }
            C_row[n] = sum;
        }
    }
}
