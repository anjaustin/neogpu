/*
 * NeoGPU ML - Ternary GEMM v15 - Software pipelining
 * 
 * Key insight: The inner loop is limited by:
 * 1. Load latency (activations + weights)
 * 2. vuzp deinterleave latency
 * 3. Weight decode latency
 * 4. vmlal dependency chain
 *
 * Strategy: Overlap loads/decodes for iteration N+1 while computing iteration N
 */

#include "hs_ml.h"
#include <arm_neon.h>
#include <string.h>

/* Decode weights macro */
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

void hs_ml_gemm_ternary_v15(int32_t* C,
                            const int8_t* A,
                            const u8* B_ternary,
                            u32 M, u32 N, u32 K) {
    
    memset(C, 0, M * N * sizeof(int32_t));
    
    const u32 K64 = K & ~63u;
    const u32 N4 = N & ~3u;
    const u32 Kstride = K / 4;
    
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
            
            if (K64 >= 64) {
                /* Prologue: Load and prepare first iteration */
                int8x16_t a0_cur = vld1q_s8(A_row + 0);
                int8x16_t a1_cur = vld1q_s8(A_row + 16);
                int8x16_t a2_cur = vld1q_s8(A_row + 32);
                int8x16_t a3_cur = vld1q_s8(A_row + 48);
                
                uint8x16_t wb0_cur = vld1q_u8(B0);
                uint8x16_t wb1_cur = vld1q_u8(B1);
                uint8x16_t wb2_cur = vld1q_u8(B2);
                uint8x16_t wb3_cur = vld1q_u8(B3);
                
                for (u32 k = 0; k < K64 - 64; k += 64) {
                    /* Start loading next iteration while processing current */
                    int8x16_t a0_nxt = vld1q_s8(A_row + k + 64);
                    int8x16_t a1_nxt = vld1q_s8(A_row + k + 80);
                    
                    /* Deinterleave current activations */
                    int8x16x2_t d0 = vuzpq_s8(a0_cur, a1_cur);
                    int8x16x2_t d1 = vuzpq_s8(a2_cur, a3_cur);
                    
                    int8x16_t a2_nxt = vld1q_s8(A_row + k + 96);
                    int8x16_t a3_nxt = vld1q_s8(A_row + k + 112);
                    
                    int8x16x2_t g01 = vuzpq_s8(d0.val[0], d1.val[0]);
                    int8x16x2_t g23 = vuzpq_s8(d0.val[1], d1.val[1]);
                    
                    u32 ko_nxt = (k + 64) / 4;
                    uint8x16_t wb0_nxt = vld1q_u8(B0 + ko_nxt);
                    uint8x16_t wb1_nxt = vld1q_u8(B1 + ko_nxt);
                    
                    int8x16_t ag0 = g01.val[0];
                    int8x16_t ag2 = g01.val[1];
                    int8x16_t ag1 = g23.val[0];
                    int8x16_t ag3 = g23.val[1];
                    
                    uint8x16_t wb2_nxt = vld1q_u8(B2 + ko_nxt);
                    uint8x16_t wb3_nxt = vld1q_u8(B3 + ko_nxt);
                    
                    /* Decode and MAC column 0 */
                    {
                        int8x16_t w0, w1, w2, w3;
                        DECODE_WEIGHTS(wb0_cur, w0, w1, w2, w3);
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
                    /* Decode and MAC column 1 */
                    {
                        int8x16_t w0, w1, w2, w3;
                        DECODE_WEIGHTS(wb1_cur, w0, w1, w2, w3);
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
                    /* Decode and MAC column 2 */
                    {
                        int8x16_t w0, w1, w2, w3;
                        DECODE_WEIGHTS(wb2_cur, w0, w1, w2, w3);
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
                    /* Decode and MAC column 3 */
                    {
                        int8x16_t w0, w1, w2, w3;
                        DECODE_WEIGHTS(wb3_cur, w0, w1, w2, w3);
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
                    
                    /* Move next to current */
                    a0_cur = a0_nxt;
                    a1_cur = a1_nxt;
                    a2_cur = a2_nxt;
                    a3_cur = a3_nxt;
                    wb0_cur = wb0_nxt;
                    wb1_cur = wb1_nxt;
                    wb2_cur = wb2_nxt;
                    wb3_cur = wb3_nxt;
                }
                
                /* Epilogue: Process last iteration */
                {
                    int8x16x2_t d0 = vuzpq_s8(a0_cur, a1_cur);
                    int8x16x2_t d1 = vuzpq_s8(a2_cur, a3_cur);
                    int8x16x2_t g01 = vuzpq_s8(d0.val[0], d1.val[0]);
                    int8x16x2_t g23 = vuzpq_s8(d0.val[1], d1.val[1]);
                    
                    int8x16_t ag0 = g01.val[0];
                    int8x16_t ag2 = g01.val[1];
                    int8x16_t ag1 = g23.val[0];
                    int8x16_t ag3 = g23.val[1];
                    
                    #define PROC_LAST(wbcur, acc) do { \
                        int8x16_t w0, w1, w2, w3; \
                        DECODE_WEIGHTS(wbcur, w0, w1, w2, w3); \
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
                    
                    PROC_LAST(wb0_cur, acc0);
                    PROC_LAST(wb1_cur, acc1);
                    PROC_LAST(wb2_cur, acc2);
                    PROC_LAST(wb3_cur, acc3);
                    #undef PROC_LAST
                }
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

