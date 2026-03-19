/*
 * NeoGPU ML - Ternary GEMM v9 - Targeting 6 GOPS
 * 
 * Key optimizations over v8:
 * - Process 8 columns at a time (vs 4)
 * - Inline weight decoding more aggressively
 * - Use separate accumulator strategy to reduce dependency chains
 */

#include "hs_ml.h"
#include <arm_neon.h>
#include <string.h>

/* Decode 16 weight bytes to 4 signed weight vectors */
#define DECODE_WEIGHTS(wb, w0, w1, w2, w3) do { \
    uint8x16_t _b0 = vandq_u8(wb, vdupq_n_u8(0x03)); \
    uint8x16_t _b1 = vandq_u8(vshrq_n_u8(wb, 2), vdupq_n_u8(0x03)); \
    uint8x16_t _b2 = vandq_u8(vshrq_n_u8(wb, 4), vdupq_n_u8(0x03)); \
    uint8x16_t _b3 = vshrq_n_u8(wb, 6); \
    w0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(_b0, vdupq_n_u8(1))), \
                  vreinterpretq_s8_u8(vshrq_n_u8(_b0, 1))); \
    w1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(_b1, vdupq_n_u8(1))), \
                  vreinterpretq_s8_u8(vshrq_n_u8(_b1, 1))); \
    w2 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(_b2, vdupq_n_u8(1))), \
                  vreinterpretq_s8_u8(vshrq_n_u8(_b2, 1))); \
    w3 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(_b3, vdupq_n_u8(1))), \
                  vreinterpretq_s8_u8(vshrq_n_u8(_b3, 1))); \
} while(0)

/* Multiply-accumulate one column's worth into acc */
#define MAC_COL(acc, w0, w1, w2, w3, ag0, ag1, ag2, ag3) do { \
    int16x8_t _p = vmull_s8(vget_low_s8(w0), vget_low_s8(ag0)); \
    _p = vmlal_s8(_p, vget_high_s8(w0), vget_high_s8(ag0)); \
    _p = vmlal_s8(_p, vget_low_s8(w1), vget_low_s8(ag1)); \
    _p = vmlal_s8(_p, vget_high_s8(w1), vget_high_s8(ag1)); \
    _p = vmlal_s8(_p, vget_low_s8(w2), vget_low_s8(ag2)); \
    _p = vmlal_s8(_p, vget_high_s8(w2), vget_high_s8(ag2)); \
    _p = vmlal_s8(_p, vget_low_s8(w3), vget_low_s8(ag3)); \
    _p = vmlal_s8(_p, vget_high_s8(w3), vget_high_s8(ag3)); \
    acc = vpadalq_s16(acc, _p); \
} while(0)

void hs_ml_gemm_ternary_v9(int32_t* C,
                           const int8_t* A,
                           const u8* B_ternary,
                           u32 M, u32 N, u32 K) {
    
    memset(C, 0, M * N * sizeof(int32_t));
    
    const u32 K64 = K & ~63u;
    const u32 N8 = N & ~7u;
    const u32 Kstride = K / 4;
    
    for (u32 m = 0; m < M; m++) {
        const int8_t* A_row = A + m * K;
        int32_t* C_row = C + m * N;
        
        /* Process 8 columns at a time */
        for (u32 n = 0; n < N8; n += 8) {
            const u8* B0 = B_ternary + (n + 0) * Kstride;
            const u8* B1 = B_ternary + (n + 1) * Kstride;
            const u8* B2 = B_ternary + (n + 2) * Kstride;
            const u8* B3 = B_ternary + (n + 3) * Kstride;
            const u8* B4 = B_ternary + (n + 4) * Kstride;
            const u8* B5 = B_ternary + (n + 5) * Kstride;
            const u8* B6 = B_ternary + (n + 6) * Kstride;
            const u8* B7 = B_ternary + (n + 7) * Kstride;
            
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);
            int32x4_t acc4 = vdupq_n_s32(0);
            int32x4_t acc5 = vdupq_n_s32(0);
            int32x4_t acc6 = vdupq_n_s32(0);
            int32x4_t acc7 = vdupq_n_s32(0);
            
            for (u32 k = 0; k < K64; k += 64) {
                __builtin_prefetch(A_row + k + 128, 0, 1);
                __builtin_prefetch(B0 + (k + 128) / 4, 0, 0);
                __builtin_prefetch(B4 + (k + 128) / 4, 0, 0);
                
                /* Load and deinterleave activations */
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
                
                /* Column 0 */
                {
                    uint8x16_t wb = vld1q_u8(B0 + ko);
                    int8x16_t w0, w1, w2, w3;
                    DECODE_WEIGHTS(wb, w0, w1, w2, w3);
                    MAC_COL(acc0, w0, w1, w2, w3, ag0, ag1, ag2, ag3);
                }
                /* Column 1 */
                {
                    uint8x16_t wb = vld1q_u8(B1 + ko);
                    int8x16_t w0, w1, w2, w3;
                    DECODE_WEIGHTS(wb, w0, w1, w2, w3);
                    MAC_COL(acc1, w0, w1, w2, w3, ag0, ag1, ag2, ag3);
                }
                /* Column 2 */
                {
                    uint8x16_t wb = vld1q_u8(B2 + ko);
                    int8x16_t w0, w1, w2, w3;
                    DECODE_WEIGHTS(wb, w0, w1, w2, w3);
                    MAC_COL(acc2, w0, w1, w2, w3, ag0, ag1, ag2, ag3);
                }
                /* Column 3 */
                {
                    uint8x16_t wb = vld1q_u8(B3 + ko);
                    int8x16_t w0, w1, w2, w3;
                    DECODE_WEIGHTS(wb, w0, w1, w2, w3);
                    MAC_COL(acc3, w0, w1, w2, w3, ag0, ag1, ag2, ag3);
                }
                /* Column 4 */
                {
                    uint8x16_t wb = vld1q_u8(B4 + ko);
                    int8x16_t w0, w1, w2, w3;
                    DECODE_WEIGHTS(wb, w0, w1, w2, w3);
                    MAC_COL(acc4, w0, w1, w2, w3, ag0, ag1, ag2, ag3);
                }
                /* Column 5 */
                {
                    uint8x16_t wb = vld1q_u8(B5 + ko);
                    int8x16_t w0, w1, w2, w3;
                    DECODE_WEIGHTS(wb, w0, w1, w2, w3);
                    MAC_COL(acc5, w0, w1, w2, w3, ag0, ag1, ag2, ag3);
                }
                /* Column 6 */
                {
                    uint8x16_t wb = vld1q_u8(B6 + ko);
                    int8x16_t w0, w1, w2, w3;
                    DECODE_WEIGHTS(wb, w0, w1, w2, w3);
                    MAC_COL(acc6, w0, w1, w2, w3, ag0, ag1, ag2, ag3);
                }
                /* Column 7 */
                {
                    uint8x16_t wb = vld1q_u8(B7 + ko);
                    int8x16_t w0, w1, w2, w3;
                    DECODE_WEIGHTS(wb, w0, w1, w2, w3);
                    MAC_COL(acc7, w0, w1, w2, w3, ag0, ag1, ag2, ag3);
                }
            }
            
            C_row[n + 0] = vaddvq_s32(acc0);
            C_row[n + 1] = vaddvq_s32(acc1);
            C_row[n + 2] = vaddvq_s32(acc2);
            C_row[n + 3] = vaddvq_s32(acc3);
            C_row[n + 4] = vaddvq_s32(acc4);
            C_row[n + 5] = vaddvq_s32(acc5);
            C_row[n + 6] = vaddvq_s32(acc6);
            C_row[n + 7] = vaddvq_s32(acc7);
            
            /* Remainder K */
            for (u32 col = 0; col < 8; col++) {
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
        
        /* Remainder N (columns) */
        for (u32 n = N8; n < N; n++) {
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
 * v10: Try different accumulation strategy - keep int16 accumulators longer
 * and only widen to int32 at the end of chunks
 */
void hs_ml_gemm_ternary_v10(int32_t* C,
                            const int8_t* A,
                            const u8* B_ternary,
                            u32 M, u32 N, u32 K) {
    
    memset(C, 0, M * N * sizeof(int32_t));
    
    /* Process in chunks of 128 to avoid int16 overflow (max 127*128=16256 < 32767) */
    const u32 CHUNK = 128;
    const u32 Kstride = K / 4;
    
    for (u32 m = 0; m < M; m++) {
        const int8_t* A_row = A + m * K;
        int32_t* C_row = C + m * N;
        
        for (u32 n = 0; n < N; n++) {
            const u8* B_col = B_ternary + n * Kstride;
            int32x4_t acc32 = vdupq_n_s32(0);
            
            u32 k = 0;
            for (; k + CHUNK <= K; k += CHUNK) {
                int16x8_t acc16 = vdupq_n_s16(0);
                
                /* Process 128 elements staying in int16 */
                for (u32 kk = 0; kk < CHUNK; kk += 64) {
                    u32 kabs = k + kk;
                    __builtin_prefetch(A_row + kabs + 128, 0, 1);
                    
                    int8x16_t a0 = vld1q_s8(A_row + kabs + 0);
                    int8x16_t a1 = vld1q_s8(A_row + kabs + 16);
                    int8x16_t a2 = vld1q_s8(A_row + kabs + 32);
                    int8x16_t a3 = vld1q_s8(A_row + kabs + 48);
                    
                    int8x16x2_t d0 = vuzpq_s8(a0, a1);
                    int8x16x2_t d1 = vuzpq_s8(a2, a3);
                    int8x16x2_t g01 = vuzpq_s8(d0.val[0], d1.val[0]);
                    int8x16x2_t g23 = vuzpq_s8(d0.val[1], d1.val[1]);
                    
                    int8x16_t ag0 = g01.val[0];
                    int8x16_t ag2 = g01.val[1];
                    int8x16_t ag1 = g23.val[0];
                    int8x16_t ag3 = g23.val[1];
                    
                    uint8x16_t wb = vld1q_u8(B_col + kabs / 4);
                    int8x16_t w0, w1, w2, w3;
                    DECODE_WEIGHTS(wb, w0, w1, w2, w3);
                    
                    /* Accumulate in int16 */
                    acc16 = vmlal_s8(acc16, vget_low_s8(w0), vget_low_s8(ag0));
                    acc16 = vmlal_s8(acc16, vget_high_s8(w0), vget_high_s8(ag0));
                    acc16 = vmlal_s8(acc16, vget_low_s8(w1), vget_low_s8(ag1));
                    acc16 = vmlal_s8(acc16, vget_high_s8(w1), vget_high_s8(ag1));
                    acc16 = vmlal_s8(acc16, vget_low_s8(w2), vget_low_s8(ag2));
                    acc16 = vmlal_s8(acc16, vget_high_s8(w2), vget_high_s8(ag2));
                    acc16 = vmlal_s8(acc16, vget_low_s8(w3), vget_low_s8(ag3));
                    acc16 = vmlal_s8(acc16, vget_high_s8(w3), vget_high_s8(ag3));
                }
                
                /* Widen to int32 at chunk boundary */
                acc32 = vpadalq_s16(acc32, acc16);
            }
            
            /* Handle remaining elements */
            for (; k + 64 <= K; k += 64) {
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
                
                uint8x16_t wb = vld1q_u8(B_col + k / 4);
                int8x16_t w0, w1, w2, w3;
                DECODE_WEIGHTS(wb, w0, w1, w2, w3);
                
                int16x8_t prod = vmull_s8(vget_low_s8(w0), vget_low_s8(ag0));
                prod = vmlal_s8(prod, vget_high_s8(w0), vget_high_s8(ag0));
                prod = vmlal_s8(prod, vget_low_s8(w1), vget_low_s8(ag1));
                prod = vmlal_s8(prod, vget_high_s8(w1), vget_high_s8(ag1));
                prod = vmlal_s8(prod, vget_low_s8(w2), vget_low_s8(ag2));
                prod = vmlal_s8(prod, vget_high_s8(w2), vget_high_s8(ag2));
                prod = vmlal_s8(prod, vget_low_s8(w3), vget_low_s8(ag3));
                prod = vmlal_s8(prod, vget_high_s8(w3), vget_high_s8(ag3));
                
                acc32 = vpadalq_s16(acc32, prod);
            }
            
            int32_t result = vaddvq_s32(acc32);
            
            /* Scalar remainder */
            for (; k < K; k++) {
                u8 byte = B_col[k / 4];
                u8 bits = (byte >> ((k % 4) * 2)) & 0x03;
                int8_t a = A_row[k];
                if (bits == 1) result += a;
                else if (bits == 2) result -= a;
            }
            
            C_row[n] = result;
        }
    }
}

