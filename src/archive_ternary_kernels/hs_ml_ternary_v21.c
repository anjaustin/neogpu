/*
 * NeoGPU ML - Ternary GEMM v21 - Final optimizations
 * 
 * Based on V18 (5.91 GOPS best)
 * Optimizations:
 * - Smarter prefetch (farther ahead for large K)
 * - Register hints
 * - Slight loop restructuring
 */

#include "hs_ml.h"
#include <arm_neon.h>
#include <string.h>

void hs_ml_gemm_ternary_v21(int32_t* C,
                            const int8_t* A,
                            const u8* B_ternary,
                            u32 M, u32 N, u32 K) {
    
    memset(C, 0, M * N * sizeof(int32_t));
    
    const u32 K64 = K & ~63u;
    const u32 N4 = N & ~3u;
    const u32 Kstride = K / 4;
    
    static const int8_t nibble_w0[16] __attribute__((aligned(16))) = {
        0, 1, -1, 0,  0, 1, -1, 0,  0, 1, -1, 0,  0, 1, -1, 0
    };
    static const int8_t nibble_w1[16] __attribute__((aligned(16))) = {
        0, 0, 0, 0,  1, 1, 1, 1,  -1, -1, -1, -1,  0, 0, 0, 0
    };
    
    register int8x16_t lut_w0 asm("v28") = vld1q_s8(nibble_w0);
    register int8x16_t lut_w1 asm("v29") = vld1q_s8(nibble_w1);
    
    for (u32 m = 0; m < M; m++) {
        const int8_t* __restrict A_row = A + m * K;
        int32_t* __restrict C_row = C + m * N;
        
        for (u32 n = 0; n < N4; n += 4) {
            const u8* __restrict B0 = B_ternary + (n + 0) * Kstride;
            const u8* __restrict B1 = B_ternary + (n + 1) * Kstride;
            const u8* __restrict B2 = B_ternary + (n + 2) * Kstride;
            const u8* __restrict B3 = B_ternary + (n + 3) * Kstride;
            
            register int32x4_t acc0 asm("v24") = vdupq_n_s32(0);
            register int32x4_t acc1 asm("v25") = vdupq_n_s32(0);
            register int32x4_t acc2 asm("v26") = vdupq_n_s32(0);
            register int32x4_t acc3 asm("v27") = vdupq_n_s32(0);
            
            /* Prefetch first blocks */
            __builtin_prefetch(A_row, 0, 3);
            __builtin_prefetch(B0, 0, 3);
            __builtin_prefetch(B1, 0, 3);
            __builtin_prefetch(B2, 0, 3);
            __builtin_prefetch(B3, 0, 3);
            
            for (u32 k = 0; k < K64; k += 64) {
                /* Prefetch ahead - distance based on K size */
                const u32 pf_dist = (K > 2048) ? 256 : 128;
                if (k + pf_dist < K64) {
                    __builtin_prefetch(A_row + k + pf_dist, 0, 1);
                    __builtin_prefetch(B0 + (k + pf_dist) / 4, 0, 1);
                    __builtin_prefetch(B1 + (k + pf_dist) / 4, 0, 1);
                }
                
                int8x16_t a0 = vld1q_s8(A_row + k + 0);
                int8x16_t a1 = vld1q_s8(A_row + k + 16);
                int8x16_t a2 = vld1q_s8(A_row + k + 32);
                int8x16_t a3 = vld1q_s8(A_row + k + 48);
                
                int8x16x2_t d0 = vuzpq_s8(a0, a1);
                int8x16x2_t d1 = vuzpq_s8(a2, a3);
                int8x16x2_t g01 = vuzpq_s8(d0.val[0], d1.val[0]);
                int8x16x2_t g23 = vuzpq_s8(d0.val[1], d1.val[1]);
                
                register int8x16_t ag0 = g01.val[0];
                register int8x16_t ag2 = g01.val[1];
                register int8x16_t ag1 = g23.val[0];
                register int8x16_t ag3 = g23.val[1];
                
                u32 ko = k / 4;
                
                /* Column 0 */
                {
                    uint8x16_t wb = vld1q_u8(B0 + ko);
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
                    acc0 = vpadalq_s16(acc0, prod);
                }
                /* Column 1 */
                {
                    uint8x16_t wb = vld1q_u8(B1 + ko);
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
                    acc1 = vpadalq_s16(acc1, prod);
                }
                /* Column 2 */
                {
                    uint8x16_t wb = vld1q_u8(B2 + ko);
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
                    acc2 = vpadalq_s16(acc2, prod);
                }
                /* Column 3 */
                {
                    uint8x16_t wb = vld1q_u8(B3 + ko);
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
                    acc3 = vpadalq_s16(acc3, prod);
                }
            }
            
            C_row[n + 0] = vaddvq_s32(acc0);
            C_row[n + 1] = vaddvq_s32(acc1);
            C_row[n + 2] = vaddvq_s32(acc2);
            C_row[n + 3] = vaddvq_s32(acc3);
            
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
