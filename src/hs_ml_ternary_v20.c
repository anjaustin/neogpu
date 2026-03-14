/*
 * NeoGPU ML - Ternary GEMM v20 - Final push for 6 GOPS
 * 
 * Based on V18 (5.91 GOPS)
 * Try 6 columns to better amortize activation deinterleaving
 */

#include "hs_ml.h"
#include <arm_neon.h>
#include <string.h>

void hs_ml_gemm_ternary_v20(int32_t* C,
                            const int8_t* A,
                            const u8* B_ternary,
                            u32 M, u32 N, u32 K) {
    
    memset(C, 0, M * N * sizeof(int32_t));
    
    const u32 K64 = K & ~63u;
    const u32 N6 = (N / 6) * 6;
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
        
        /* Process 6 columns at a time */
        for (u32 n = 0; n < N6; n += 6) {
            const u8* B0 = B_ternary + (n + 0) * Kstride;
            const u8* B1 = B_ternary + (n + 1) * Kstride;
            const u8* B2 = B_ternary + (n + 2) * Kstride;
            const u8* B3 = B_ternary + (n + 3) * Kstride;
            const u8* B4 = B_ternary + (n + 4) * Kstride;
            const u8* B5 = B_ternary + (n + 5) * Kstride;
            
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);
            int32x4_t acc4 = vdupq_n_s32(0);
            int32x4_t acc5 = vdupq_n_s32(0);
            
            for (u32 k = 0; k < K64; k += 64) {
                __builtin_prefetch(A_row + k + 128, 0, 3);
                
                int8x16_t a0 = vld1q_s8(A_row + k + 0);
                int8x16_t a1 = vld1q_s8(A_row + k + 16);
                int8x16_t a2 = vld1q_s8(A_row + k + 32);
                int8x16_t a3 = vld1q_s8(A_row + k + 48);
                
                int8x16x2_t d0 = vuzpq_s8(a0, a1);
                int8x16x2_t d1 = vuzpq_s8(a2, a3);
                int8x16x2_t g01 = vuzpq_s8(d0.val[0], d1.val[0]);
                int8x16x2_t g23 = vuzpq_s8(d0.val[1], d1.val[1]);
                
                int8x16_t ag0 = g01.val[0];
                int8x16_t ag2 = g01.val[1];
                int8x16_t ag1 = g23.val[0];
                int8x16_t ag3 = g23.val[1];
                
                u32 ko = k / 4;
                
                #define PROC_COL(Bptr, acc) do { \
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
                
                PROC_COL(B0, acc0);
                PROC_COL(B1, acc1);
                PROC_COL(B2, acc2);
                PROC_COL(B3, acc3);
                PROC_COL(B4, acc4);
                PROC_COL(B5, acc5);
                
                #undef PROC_COL
            }
            
            C_row[n + 0] = vaddvq_s32(acc0);
            C_row[n + 1] = vaddvq_s32(acc1);
            C_row[n + 2] = vaddvq_s32(acc2);
            C_row[n + 3] = vaddvq_s32(acc3);
            C_row[n + 4] = vaddvq_s32(acc4);
            C_row[n + 5] = vaddvq_s32(acc5);
            
            for (u32 col = 0; col < 6; col++) {
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
        
        /* Remainder N - process 1 column at a time */
        for (u32 n = N6; n < N; n++) {
            const u8* B_col = B_ternary + n * Kstride;
            int32x4_t acc = vdupq_n_s32(0);
            
            for (u32 k = 0; k < K64; k += 64) {
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
                
                uint8x16_t wb = vld1q_u8(B_col + k / 4);
                uint8x16_t lo_nib = vandq_u8(wb, vdupq_n_u8(0x0F));
                uint8x16_t hi_nib = vshrq_n_u8(wb, 4);
                int8x16_t w0 = vqtbl1q_s8(lut_w0, lo_nib);
                int8x16_t w1 = vqtbl1q_s8(lut_w1, lo_nib);
                int8x16_t w2 = vqtbl1q_s8(lut_w0, hi_nib);
                int8x16_t w3 = vqtbl1q_s8(lut_w1, hi_nib);
                
                int16x8_t prod = vmull_s8(vget_low_s8(w0), vget_low_s8(ag0));
                prod = vmlal_s8(prod, vget_high_s8(w0), vget_high_s8(ag0));
                prod = vmlal_s8(prod, vget_low_s8(w1), vget_low_s8(ag1));
                prod = vmlal_s8(prod, vget_high_s8(w1), vget_high_s8(ag1));
                prod = vmlal_s8(prod, vget_low_s8(w2), vget_low_s8(ag2));
                prod = vmlal_s8(prod, vget_high_s8(w2), vget_high_s8(ag2));
                prod = vmlal_s8(prod, vget_low_s8(w3), vget_low_s8(ag3));
                prod = vmlal_s8(prod, vget_high_s8(w3), vget_high_s8(ag3));
                acc = vpadalq_s16(acc, prod);
            }
            
            int32_t sum = vaddvq_s32(acc);
            for (u32 k = K64; k < K; k++) {
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
