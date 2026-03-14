/*
 * NeoGPU ML - Ternary GEMM v11 - Targeting 6 GOPS
 * 
 * Different approach: Instead of deinterleaving activations,
 * reorganize weights at load time to match activation layout.
 * 
 * Original: weight byte has [w3:w2:w1:w0] for a[4n+3,4n+2,4n+1,4n+0]
 * We load activations sequentially and need to replicate/shuffle weights
 * 
 * Key insight: Each weight byte controls 4 consecutive activations.
 * If we expand weight byte to 4 signed bytes, we can use standard vmlal.
 */

#include "hs_ml.h"
#include <arm_neon.h>
#include <string.h>

void hs_ml_gemm_ternary_v11(int32_t* C,
                            const int8_t* A,
                            const u8* B_ternary,
                            u32 M, u32 N, u32 K) {
    
    memset(C, 0, M * N * sizeof(int32_t));
    
    const u32 K64 = K & ~63u;
    const u32 N4 = N & ~3u;
    const u32 Kstride = K / 4;
    
    /* Lookup table to expand 2-bit weight to signed byte */
    /* Index 0->0, 1->+1, 2->-1, 3->0 */
    static const int8_t expand_lut[4] = {0, 1, -1, 0};
    
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
                __builtin_prefetch(A_row + k + 128, 0, 1);
                
                /* Load activations directly - no deinterleaving! */
                int8x16_t a0 = vld1q_s8(A_row + k + 0);
                int8x16_t a1 = vld1q_s8(A_row + k + 16);
                int8x16_t a2 = vld1q_s8(A_row + k + 32);
                int8x16_t a3 = vld1q_s8(A_row + k + 48);
                
                u32 ko = k / 4;
                
                /* For each column, expand weights to match activation layout */
                /* Each weight byte wb[i] needs to become 4 signed bytes */
                #define PROCESS_COL_V11(Bptr, accvar) do { \
                    /* Load 16 weight bytes = 64 weights packed */ \
                    uint8x16_t wb = vld1q_u8(Bptr + ko); \
                    \
                    /* Extract and expand each 2-bit field to a full lane */ \
                    /* w0_raw[i] = wb[i] & 0x03 -> expand_lut[w0_raw[i]] */ \
                    /* Need to create 4 vectors of 16 weights each */ \
                    \
                    /* Bits [1:0] -> weights for positions 4n+0 */ \
                    uint8x16_t b0 = vandq_u8(wb, vdupq_n_u8(0x03)); \
                    /* Bits [3:2] -> weights for positions 4n+1 */ \
                    uint8x16_t b1 = vandq_u8(vshrq_n_u8(wb, 2), vdupq_n_u8(0x03)); \
                    /* Bits [5:4] -> weights for positions 4n+2 */ \
                    uint8x16_t b2 = vandq_u8(vshrq_n_u8(wb, 4), vdupq_n_u8(0x03)); \
                    /* Bits [7:6] -> weights for positions 4n+3 */ \
                    uint8x16_t b3 = vshrq_n_u8(wb, 6); \
                    \
                    /* Convert to signed: (bits & 1) - (bits >> 1) */ \
                    int8x16_t w0_s = vsubq_s8( \
                        vreinterpretq_s8_u8(vandq_u8(b0, vdupq_n_u8(1))), \
                        vreinterpretq_s8_u8(vshrq_n_u8(b0, 1))); \
                    int8x16_t w1_s = vsubq_s8( \
                        vreinterpretq_s8_u8(vandq_u8(b1, vdupq_n_u8(1))), \
                        vreinterpretq_s8_u8(vshrq_n_u8(b1, 1))); \
                    int8x16_t w2_s = vsubq_s8( \
                        vreinterpretq_s8_u8(vandq_u8(b2, vdupq_n_u8(1))), \
                        vreinterpretq_s8_u8(vshrq_n_u8(b2, 1))); \
                    int8x16_t w3_s = vsubq_s8( \
                        vreinterpretq_s8_u8(vandq_u8(b3, vdupq_n_u8(1))), \
                        vreinterpretq_s8_u8(vshrq_n_u8(b3, 1))); \
                    \
                    /* Now interleave weights to match activation order */ \
                    /* w0_s[i] matches a[4i+0], w1_s[i] matches a[4i+1], etc */ \
                    /* We need: w_full[0..3] = w0_s[0],w1_s[0],w2_s[0],w3_s[0] */ \
                    /*          w_full[4..7] = w0_s[1],w1_s[1],w2_s[1],w3_s[1] */ \
                    \
                    /* Use vzip to interleave */ \
                    int8x16x2_t z01 = vzipq_s8(w0_s, w1_s); /* [w0[0],w1[0],w0[1],w1[1],...] */ \
                    int8x16x2_t z23 = vzipq_s8(w2_s, w3_s); /* [w2[0],w3[0],w2[1],w3[1],...] */ \
                    \
                    /* Now interleave the pairs */ \
                    int16x8x2_t zfinal0 = vzipq_s16( \
                        vreinterpretq_s16_s8(z01.val[0]), \
                        vreinterpretq_s16_s8(z23.val[0])); \
                    int16x8x2_t zfinal1 = vzipq_s16( \
                        vreinterpretq_s16_s8(z01.val[1]), \
                        vreinterpretq_s16_s8(z23.val[1])); \
                    \
                    int8x16_t wexp0 = vreinterpretq_s8_s16(zfinal0.val[0]); \
                    int8x16_t wexp1 = vreinterpretq_s8_s16(zfinal0.val[1]); \
                    int8x16_t wexp2 = vreinterpretq_s8_s16(zfinal1.val[0]); \
                    int8x16_t wexp3 = vreinterpretq_s8_s16(zfinal1.val[1]); \
                    \
                    /* Now multiply directly - wexp matches activation order */ \
                    int16x8_t prod = vmull_s8(vget_low_s8(wexp0), vget_low_s8(a0)); \
                    prod = vmlal_s8(prod, vget_high_s8(wexp0), vget_high_s8(a0)); \
                    prod = vmlal_s8(prod, vget_low_s8(wexp1), vget_low_s8(a1)); \
                    prod = vmlal_s8(prod, vget_high_s8(wexp1), vget_high_s8(a1)); \
                    prod = vmlal_s8(prod, vget_low_s8(wexp2), vget_low_s8(a2)); \
                    prod = vmlal_s8(prod, vget_high_s8(wexp2), vget_high_s8(a2)); \
                    prod = vmlal_s8(prod, vget_low_s8(wexp3), vget_low_s8(a3)); \
                    prod = vmlal_s8(prod, vget_high_s8(wexp3), vget_high_s8(a3)); \
                    \
                    accvar = vpadalq_s16(accvar, prod); \
                } while(0)
                
                PROCESS_COL_V11(B0, acc0);
                PROCESS_COL_V11(B1, acc1);
                PROCESS_COL_V11(B2, acc2);
                PROCESS_COL_V11(B3, acc3);
                
                #undef PROCESS_COL_V11
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
 * v12: Different approach - use conditional add/sub instead of multiply
 * For ternary weights, we only need to add or subtract activations
 * This avoids the multiply entirely
 */
void hs_ml_gemm_ternary_v12(int32_t* C,
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
            
            for (u32 k = 0; k < K64; k += 64) {
                __builtin_prefetch(A_row + k + 128, 0, 1);
                
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
                
                /* Widen activations to int16 for accumulation */
                int16x8_t ag0_lo = vmovl_s8(vget_low_s8(ag0));
                int16x8_t ag0_hi = vmovl_s8(vget_high_s8(ag0));
                int16x8_t ag1_lo = vmovl_s8(vget_low_s8(ag1));
                int16x8_t ag1_hi = vmovl_s8(vget_high_s8(ag1));
                int16x8_t ag2_lo = vmovl_s8(vget_low_s8(ag2));
                int16x8_t ag2_hi = vmovl_s8(vget_high_s8(ag2));
                int16x8_t ag3_lo = vmovl_s8(vget_low_s8(ag3));
                int16x8_t ag3_hi = vmovl_s8(vget_high_s8(ag3));
                
                u32 ko = k / 4;
                
                #define PROCESS_COL_V12(Bptr, accvar) do { \
                    uint8x16_t wb = vld1q_u8(Bptr + ko); \
                    \
                    /* Extract 2-bit fields */ \
                    uint8x16_t b0 = vandq_u8(wb, vdupq_n_u8(0x03)); \
                    uint8x16_t b1 = vandq_u8(vshrq_n_u8(wb, 2), vdupq_n_u8(0x03)); \
                    uint8x16_t b2 = vandq_u8(vshrq_n_u8(wb, 4), vdupq_n_u8(0x03)); \
                    uint8x16_t b3 = vshrq_n_u8(wb, 6); \
                    \
                    /* Create masks: is_plus = (bits == 1), is_minus = (bits == 2) */ \
                    uint8x16_t one = vdupq_n_u8(1); \
                    uint8x16_t two = vdupq_n_u8(2); \
                    \
                    /* For group 0 */ \
                    uint8x16_t plus0 = vceqq_u8(b0, one); \
                    uint8x16_t minus0 = vceqq_u8(b0, two); \
                    /* Use masks to select: result = (plus ? a : 0) - (minus ? a : 0) */ \
                    int8x16_t masked_plus0 = vandq_s8(ag0, vreinterpretq_s8_u8(plus0)); \
                    int8x16_t masked_minus0 = vandq_s8(ag0, vreinterpretq_s8_u8(minus0)); \
                    int8x16_t contrib0 = vsubq_s8(masked_plus0, masked_minus0); \
                    \
                    uint8x16_t plus1 = vceqq_u8(b1, one); \
                    uint8x16_t minus1 = vceqq_u8(b1, two); \
                    int8x16_t masked_plus1 = vandq_s8(ag1, vreinterpretq_s8_u8(plus1)); \
                    int8x16_t masked_minus1 = vandq_s8(ag1, vreinterpretq_s8_u8(minus1)); \
                    int8x16_t contrib1 = vsubq_s8(masked_plus1, masked_minus1); \
                    \
                    uint8x16_t plus2 = vceqq_u8(b2, one); \
                    uint8x16_t minus2 = vceqq_u8(b2, two); \
                    int8x16_t masked_plus2 = vandq_s8(ag2, vreinterpretq_s8_u8(plus2)); \
                    int8x16_t masked_minus2 = vandq_s8(ag2, vreinterpretq_s8_u8(minus2)); \
                    int8x16_t contrib2 = vsubq_s8(masked_plus2, masked_minus2); \
                    \
                    uint8x16_t plus3 = vceqq_u8(b3, one); \
                    uint8x16_t minus3 = vceqq_u8(b3, two); \
                    int8x16_t masked_plus3 = vandq_s8(ag3, vreinterpretq_s8_u8(plus3)); \
                    int8x16_t masked_minus3 = vandq_s8(ag3, vreinterpretq_s8_u8(minus3)); \
                    int8x16_t contrib3 = vsubq_s8(masked_plus3, masked_minus3); \
                    \
                    /* Sum contributions */ \
                    int8x16_t sum = vaddq_s8(vaddq_s8(contrib0, contrib1), \
                                            vaddq_s8(contrib2, contrib3)); \
                    \
                    /* Widen and accumulate */ \
                    int16x8_t sum16 = vpaddlq_s8(sum); \
                    accvar = vpadalq_s16(accvar, sum16); \
                } while(0)
                
                PROCESS_COL_V12(B0, acc0);
                PROCESS_COL_V12(B1, acc1);
                PROCESS_COL_V12(B2, acc2);
                PROCESS_COL_V12(B3, acc3);
                
                #undef PROCESS_COL_V12
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

