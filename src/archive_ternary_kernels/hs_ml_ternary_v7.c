/*
 * NeoGPU ML - Ternary GEMM v7 - Targeting 6 GOPS
 * 
 * Use vmlal_s8 with properly interleaved weights
 * Process 16 weight bytes at a time (64 activations)
 */

#include "hs_ml.h"
#include <arm_neon.h>
#include <string.h>

void hs_ml_gemm_ternary_v7(int32_t* C,
                           const int8_t* A,
                           const u8* B_ternary,
                           u32 M, u32 N, u32 K) {
    
    memset(C, 0, M * N * sizeof(int32_t));
    
    const u32 K64 = K & ~63u;
    
    for (u32 m = 0; m < M; m++) {
        const int8_t* A_row = A + m * K;
        int32_t* C_row = C + m * N;
        
        for (u32 n = 0; n < N; n++) {
            const u8* B_col = B_ternary + n * (K / 4);
            
            int32x4_t acc = vdupq_n_s32(0);
            
            for (u32 k = 0; k < K64; k += 64) {
                __builtin_prefetch(A_row + k + 128, 0, 1);
                __builtin_prefetch(B_col + (k + 128) / 4, 0, 1);
                
                /* Load 16 weight bytes (64 weights) */
                uint8x16_t wb = vld1q_u8(B_col + k / 4);
                
                /* Load 64 activations */
                int8x16_t a0 = vld1q_s8(A_row + k + 0);
                int8x16_t a1 = vld1q_s8(A_row + k + 16);
                int8x16_t a2 = vld1q_s8(A_row + k + 32);
                int8x16_t a3 = vld1q_s8(A_row + k + 48);
                
                /*
                 * Weight layout: wb[i] has weights for a[i*4+0..3]
                 * Bit layout: [w3:w2:w1:w0] for positions +3,+2,+1,+0
                 * 
                 * Strategy: Deinterleave activations into 4 groups
                 * a_g0 = a[0,4,8,12,...,60] - positions 4n+0
                 * a_g1 = a[1,5,9,13,...,61] - positions 4n+1  
                 * a_g2 = a[2,6,10,14,...,62] - positions 4n+2
                 * a_g3 = a[3,7,11,15,...,63] - positions 4n+3
                 * 
                 * Then wb bits [1:0] match a_g0, bits [3:2] match a_g1, etc.
                 * 
                 * Use vuzp to deinterleave by 2, twice to get stride-4
                 */
                
                /* First deinterleave by 2: separate even/odd positions */
                int8x16x2_t d0 = vuzpq_s8(a0, a1);  /* d0.val[0]=[a0,a2,a4,...], d0.val[1]=[a1,a3,a5,...] */
                int8x16x2_t d1 = vuzpq_s8(a2, a3);
                
                /* Second deinterleave: separate by 4 */
                int8x16x2_t g01 = vuzpq_s8(d0.val[0], d1.val[0]); /* positions 0,4,8,... and 2,6,10,... */
                int8x16x2_t g23 = vuzpq_s8(d0.val[1], d1.val[1]); /* positions 1,5,9,... and 3,7,11,... */
                
                int8x16_t a_g0 = g01.val[0];  /* a[0,4,8,12,...] */
                int8x16_t a_g2 = g01.val[1];  /* a[2,6,10,14,...] */
                int8x16_t a_g1 = g23.val[0];  /* a[1,5,9,13,...] */
                int8x16_t a_g3 = g23.val[1];  /* a[3,7,11,15,...] */
                
                /* Extract weight groups and convert to signed */
                uint8x16_t w_bits0 = vandq_u8(wb, vdupq_n_u8(0x03));
                uint8x16_t w_bits1 = vandq_u8(vshrq_n_u8(wb, 2), vdupq_n_u8(0x03));
                uint8x16_t w_bits2 = vandq_u8(vshrq_n_u8(wb, 4), vdupq_n_u8(0x03));
                uint8x16_t w_bits3 = vshrq_n_u8(wb, 6);
                
                /* Convert to signed: (bits & 1) - (bits >> 1) */
                int8x16_t w_s0 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(w_bits0, vdupq_n_u8(1))),
                    vreinterpretq_s8_u8(vshrq_n_u8(w_bits0, 1))
                );
                int8x16_t w_s1 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(w_bits1, vdupq_n_u8(1))),
                    vreinterpretq_s8_u8(vshrq_n_u8(w_bits1, 1))
                );
                int8x16_t w_s2 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(w_bits2, vdupq_n_u8(1))),
                    vreinterpretq_s8_u8(vshrq_n_u8(w_bits2, 1))
                );
                int8x16_t w_s3 = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(w_bits3, vdupq_n_u8(1))),
                    vreinterpretq_s8_u8(vshrq_n_u8(w_bits3, 1))
                );
                
                /* Now multiply and accumulate using vmlal */
                /* w_s0 matches a_g0, w_s1 matches a_g1, etc */
                int16x8_t prod = vmull_s8(vget_low_s8(w_s0), vget_low_s8(a_g0));
                prod = vmlal_s8(prod, vget_high_s8(w_s0), vget_high_s8(a_g0));
                prod = vmlal_s8(prod, vget_low_s8(w_s1), vget_low_s8(a_g1));
                prod = vmlal_s8(prod, vget_high_s8(w_s1), vget_high_s8(a_g1));
                prod = vmlal_s8(prod, vget_low_s8(w_s2), vget_low_s8(a_g2));
                prod = vmlal_s8(prod, vget_high_s8(w_s2), vget_high_s8(a_g2));
                prod = vmlal_s8(prod, vget_low_s8(w_s3), vget_low_s8(a_g3));
                prod = vmlal_s8(prod, vget_high_s8(w_s3), vget_high_s8(a_g3));
                
                /* Accumulate into int32 */
                acc = vpadalq_s16(acc, prod);
            }
            
            /* Horizontal sum */
            int32_t result = vaddvq_s32(acc);
            
            /* Remainder */
            for (u32 k = K64; k < K; k++) {
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

/*
 * v8: v7 but processing 4 columns at a time for better activation reuse
 */
void hs_ml_gemm_ternary_v8(int32_t* C,
                           const int8_t* A,
                           const u8* B_ternary,
                           u32 M, u32 N, u32 K) {
    
    memset(C, 0, M * N * sizeof(int32_t));
    
    const u32 K64 = K & ~63u;
    const u32 N4 = N & ~3u;
    
    for (u32 m = 0; m < M; m++) {
        const int8_t* A_row = A + m * K;
        int32_t* C_row = C + m * N;
        
        for (u32 n = 0; n < N4; n += 4) {
            const u8* B0 = B_ternary + (n + 0) * (K / 4);
            const u8* B1 = B_ternary + (n + 1) * (K / 4);
            const u8* B2 = B_ternary + (n + 2) * (K / 4);
            const u8* B3 = B_ternary + (n + 3) * (K / 4);
            
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);
            
            for (u32 k = 0; k < K64; k += 64) {
                __builtin_prefetch(A_row + k + 128, 0, 1);
                
                /* Load and deinterleave activations (shared across columns) */
                int8x16_t a0 = vld1q_s8(A_row + k + 0);
                int8x16_t a1 = vld1q_s8(A_row + k + 16);
                int8x16_t a2 = vld1q_s8(A_row + k + 32);
                int8x16_t a3 = vld1q_s8(A_row + k + 48);
                
                int8x16x2_t d0 = vuzpq_s8(a0, a1);
                int8x16x2_t d1 = vuzpq_s8(a2, a3);
                int8x16x2_t g01 = vuzpq_s8(d0.val[0], d1.val[0]);
                int8x16x2_t g23 = vuzpq_s8(d0.val[1], d1.val[1]);
                
                int8x16_t a_g0 = g01.val[0];
                int8x16_t a_g2 = g01.val[1];
                int8x16_t a_g1 = g23.val[0];
                int8x16_t a_g3 = g23.val[1];
                
                /* Process column 0 */
                #define PROCESS_COL_V8(Bptr, accvar) do { \
                    uint8x16_t wb = vld1q_u8(Bptr + k / 4); \
                    \
                    uint8x16_t w_bits0 = vandq_u8(wb, vdupq_n_u8(0x03)); \
                    uint8x16_t w_bits1 = vandq_u8(vshrq_n_u8(wb, 2), vdupq_n_u8(0x03)); \
                    uint8x16_t w_bits2 = vandq_u8(vshrq_n_u8(wb, 4), vdupq_n_u8(0x03)); \
                    uint8x16_t w_bits3 = vshrq_n_u8(wb, 6); \
                    \
                    int8x16_t w_s0 = vsubq_s8( \
                        vreinterpretq_s8_u8(vandq_u8(w_bits0, vdupq_n_u8(1))), \
                        vreinterpretq_s8_u8(vshrq_n_u8(w_bits0, 1))); \
                    int8x16_t w_s1 = vsubq_s8( \
                        vreinterpretq_s8_u8(vandq_u8(w_bits1, vdupq_n_u8(1))), \
                        vreinterpretq_s8_u8(vshrq_n_u8(w_bits1, 1))); \
                    int8x16_t w_s2 = vsubq_s8( \
                        vreinterpretq_s8_u8(vandq_u8(w_bits2, vdupq_n_u8(1))), \
                        vreinterpretq_s8_u8(vshrq_n_u8(w_bits2, 1))); \
                    int8x16_t w_s3 = vsubq_s8( \
                        vreinterpretq_s8_u8(vandq_u8(w_bits3, vdupq_n_u8(1))), \
                        vreinterpretq_s8_u8(vshrq_n_u8(w_bits3, 1))); \
                    \
                    int16x8_t prod = vmull_s8(vget_low_s8(w_s0), vget_low_s8(a_g0)); \
                    prod = vmlal_s8(prod, vget_high_s8(w_s0), vget_high_s8(a_g0)); \
                    prod = vmlal_s8(prod, vget_low_s8(w_s1), vget_low_s8(a_g1)); \
                    prod = vmlal_s8(prod, vget_high_s8(w_s1), vget_high_s8(a_g1)); \
                    prod = vmlal_s8(prod, vget_low_s8(w_s2), vget_low_s8(a_g2)); \
                    prod = vmlal_s8(prod, vget_high_s8(w_s2), vget_high_s8(a_g2)); \
                    prod = vmlal_s8(prod, vget_low_s8(w_s3), vget_low_s8(a_g3)); \
                    prod = vmlal_s8(prod, vget_high_s8(w_s3), vget_high_s8(a_g3)); \
                    \
                    accvar = vpadalq_s16(accvar, prod); \
                } while(0)
                
                PROCESS_COL_V8(B0, acc0);
                PROCESS_COL_V8(B1, acc1);
                PROCESS_COL_V8(B2, acc2);
                PROCESS_COL_V8(B3, acc3);
                
                #undef PROCESS_COL_V8
            }
            
            C_row[n + 0] = vaddvq_s32(acc0);
            C_row[n + 1] = vaddvq_s32(acc1);
            C_row[n + 2] = vaddvq_s32(acc2);
            C_row[n + 3] = vaddvq_s32(acc3);
            
            /* Remainder K */
            for (u32 col = 0; col < 4; col++) {
                const u8* B_col = B_ternary + (n + col) * (K / 4);
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
            const u8* B_col = B_ternary + n * (K / 4);
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

