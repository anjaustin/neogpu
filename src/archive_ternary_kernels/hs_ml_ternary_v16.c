/*
 * NeoGPU ML - Ternary GEMM v16 - LUT-based weight expansion
 * 
 * Key insight: Weight decoding takes ~12 instructions per 16 bytes.
 * Use a 256-entry LUT to expand one byte to 4 signed bytes at once.
 * This trades memory for compute - LUT is 1KB which fits in L1.
 */

#include "hs_ml.h"
#include <arm_neon.h>
#include <string.h>

/* LUT: index is weight byte, output is 4 signed weights as int32 (packed) */
/* Byte layout: [w3:w2:w1:w0] each 2-bit -> {0,+1,-1} */
static int8_t weight_lut[256][4] __attribute__((aligned(64)));
static int weight_lut_initialized = 0;

static void init_weight_lut(void) {
    if (weight_lut_initialized) return;
    for (int b = 0; b < 256; b++) {
        for (int i = 0; i < 4; i++) {
            int bits = (b >> (i * 2)) & 0x03;
            /* 00=0, 01=+1, 10=-1, 11=0 */
            weight_lut[b][i] = (bits == 1) ? 1 : ((bits == 2) ? -1 : 0);
        }
    }
    weight_lut_initialized = 1;
}

void hs_ml_gemm_ternary_v16(int32_t* C,
                            const int8_t* A,
                            const u8* B_ternary,
                            u32 M, u32 N, u32 K) {
    
    init_weight_lut();
    memset(C, 0, M * N * sizeof(int32_t));
    
    const u32 K64 = K & ~63u;
    const u32 N4 = N & ~3u;
    const u32 Kstride = K / 4;
    
    /* Create NEON LUT (use table lookup) */
    /* We need 256 entries * 4 bytes = 1KB, too big for vqtbl */
    /* Instead, process weights byte-by-byte using scalar LUT */
    
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
                /* Load activations */
                int8x16_t a0 = vld1q_s8(A_row + k + 0);
                int8x16_t a1 = vld1q_s8(A_row + k + 16);
                int8x16_t a2 = vld1q_s8(A_row + k + 32);
                int8x16_t a3 = vld1q_s8(A_row + k + 48);
                
                /* Deinterleave activations */
                int8x16x2_t d0 = vuzpq_s8(a0, a1);
                int8x16x2_t d1 = vuzpq_s8(a2, a3);
                int8x16x2_t g01 = vuzpq_s8(d0.val[0], d1.val[0]);
                int8x16x2_t g23 = vuzpq_s8(d0.val[1], d1.val[1]);
                
                int8x16_t ag0 = g01.val[0];
                int8x16_t ag2 = g01.val[1];
                int8x16_t ag1 = g23.val[0];
                int8x16_t ag3 = g23.val[1];
                
                u32 ko = k / 4;
                
                /* Process column using LUT to expand weights */
                /* Load 16 weight bytes, expand each using LUT, then vmlal */
                #define PROC_COL_LUT(Bptr, acc) do { \
                    /* Load weight bytes */ \
                    const u8* wp = Bptr + ko; \
                    /* Expand using LUT - scalar loads but vectorized after */ \
                    int8_t wbuf[64]; \
                    for (int i = 0; i < 16; i++) { \
                        const int8_t* lut_entry = weight_lut[wp[i]]; \
                        wbuf[i*4+0] = lut_entry[0]; \
                        wbuf[i*4+1] = lut_entry[1]; \
                        wbuf[i*4+2] = lut_entry[2]; \
                        wbuf[i*4+3] = lut_entry[3]; \
                    } \
                    \
                    /* Now weights are expanded to match activation layout */ \
                    /* But we have them interleaved, need to deinterleave */ \
                    int8x16_t w0_raw = vld1q_s8(wbuf + 0); \
                    int8x16_t w1_raw = vld1q_s8(wbuf + 16); \
                    int8x16_t w2_raw = vld1q_s8(wbuf + 32); \
                    int8x16_t w3_raw = vld1q_s8(wbuf + 48); \
                    \
                    int8x16x2_t wd0 = vuzpq_s8(w0_raw, w1_raw); \
                    int8x16x2_t wd1 = vuzpq_s8(w2_raw, w3_raw); \
                    int8x16x2_t wg01 = vuzpq_s8(wd0.val[0], wd1.val[0]); \
                    int8x16x2_t wg23 = vuzpq_s8(wd0.val[1], wd1.val[1]); \
                    \
                    int8x16_t w0 = wg01.val[0]; \
                    int8x16_t w2 = wg01.val[1]; \
                    int8x16_t w1 = wg23.val[0]; \
                    int8x16_t w3 = wg23.val[1]; \
                    \
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
                
                PROC_COL_LUT(B0, acc0);
                PROC_COL_LUT(B1, acc1);
                PROC_COL_LUT(B2, acc2);
                PROC_COL_LUT(B3, acc3);
                
                #undef PROC_COL_LUT
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
 * v17: Use smaller LUT (16 entries for 4-bit nibble) and process in two passes
 * This allows using vqtbl1q for NEON-native table lookup
 */
void hs_ml_gemm_ternary_v17(int32_t* C,
                            const int8_t* A,
                            const u8* B_ternary,
                            u32 M, u32 N, u32 K) {
    
    memset(C, 0, M * N * sizeof(int32_t));
    
    const u32 K64 = K & ~63u;
    const u32 N4 = N & ~3u;
    const u32 Kstride = K / 4;
    
    /* 16-entry LUT for nibble expansion: 4 bits -> 2 signed weights packed as int16 */
    /* Bits [1:0] -> w0, bits [3:2] -> w1 */
    /* Pack as (w1 << 8) | (w0 & 0xFF) */
    static const uint8_t nibble_lut[32] __attribute__((aligned(16))) = {
        /* Index 0x0: w0=0, w1=0 */   0, 0,
        /* Index 0x1: w0=1, w1=0 */   1, 0,
        /* Index 0x2: w0=-1, w1=0 */ 255, 0,  /* -1 as int8 */
        /* Index 0x3: w0=0, w1=0 */   0, 0,
        /* Index 0x4: w0=0, w1=1 */   0, 1,
        /* Index 0x5: w0=1, w1=1 */   1, 1,
        /* Index 0x6: w0=-1, w1=1 */ 255, 1,
        /* Index 0x7: w0=0, w1=1 */   0, 1,
        /* Index 0x8: w0=0, w1=-1 */  0, 255,
        /* Index 0x9: w0=1, w1=-1 */  1, 255,
        /* Index 0xA: w0=-1, w1=-1 */ 255, 255,
        /* Index 0xB: w0=0, w1=-1 */  0, 255,
        /* Index 0xC: w0=0, w1=0 */   0, 0,
        /* Index 0xD: w0=1, w1=0 */   1, 0,
        /* Index 0xE: w0=-1, w1=0 */ 255, 0,
        /* Index 0xF: w0=0, w1=0 */   0, 0,
    };
    
    uint8x16_t lut_lo = vld1q_u8(nibble_lut);
    uint8x16_t lut_hi = vld1q_u8(nibble_lut + 16);
    
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
                /* Load activations */
                int8x16_t a0 = vld1q_s8(A_row + k + 0);
                int8x16_t a1 = vld1q_s8(A_row + k + 16);
                int8x16_t a2 = vld1q_s8(A_row + k + 32);
                int8x16_t a3 = vld1q_s8(A_row + k + 48);
                
                /* Deinterleave activations */
                int8x16x2_t d0 = vuzpq_s8(a0, a1);
                int8x16x2_t d1 = vuzpq_s8(a2, a3);
                int8x16x2_t g01 = vuzpq_s8(d0.val[0], d1.val[0]);
                int8x16x2_t g23 = vuzpq_s8(d0.val[1], d1.val[1]);
                
                int8x16_t ag0 = g01.val[0];
                int8x16_t ag2 = g01.val[1];
                int8x16_t ag1 = g23.val[0];
                int8x16_t ag3 = g23.val[1];
                
                u32 ko = k / 4;
                
                #define PROC_COL_V17(Bptr, acc) do { \
                    uint8x16_t wb = vld1q_u8(Bptr + ko); \
                    \
                    /* Split into nibbles */ \
                    uint8x16_t lo_nib = vandq_u8(wb, vdupq_n_u8(0x0F)); \
                    uint8x16_t hi_nib = vshrq_n_u8(wb, 4); \
                    \
                    /* Double indices for 2-byte entries */ \
                    lo_nib = vaddq_u8(lo_nib, lo_nib); \
                    hi_nib = vaddq_u8(hi_nib, hi_nib); \
                    \
                    /* Look up - each entry is 2 bytes (w0, w1) */ \
                    /* Low nibble lookup */ \
                    uint8x16_t lo_exp_lo = vqtbl1q_u8(lut_lo, lo_nib); \
                    uint8x16_t lo_exp_hi = vqtbl1q_u8(lut_lo, vaddq_u8(lo_nib, vdupq_n_u8(1))); \
                    \
                    /* High nibble lookup */ \
                    uint8x16_t hi_exp_lo = vqtbl1q_u8(lut_lo, hi_nib); \
                    uint8x16_t hi_exp_hi = vqtbl1q_u8(lut_lo, vaddq_u8(hi_nib, vdupq_n_u8(1))); \
                    \
                    /* Combine: lo_nib gives weights for positions 0,1, hi_nib for 2,3 */ \
                    /* w0 = lo_exp_lo[i], w1 = lo_exp_hi[i], w2 = hi_exp_lo[i], w3 = hi_exp_hi[i] */ \
                    int8x16_t w0 = vreinterpretq_s8_u8(lo_exp_lo); \
                    int8x16_t w1 = vreinterpretq_s8_u8(lo_exp_hi); \
                    int8x16_t w2 = vreinterpretq_s8_u8(hi_exp_lo); \
                    int8x16_t w3 = vreinterpretq_s8_u8(hi_exp_hi); \
                    \
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
                
                PROC_COL_V17(B0, acc0);
                PROC_COL_V17(B1, acc1);
                PROC_COL_V17(B2, acc2);
                PROC_COL_V17(B3, acc3);
                
                #undef PROC_COL_V17
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

