/*
 * NeoGPU ML - Ternary GEMM v2 - Targeting 6 GOPS
 * 
 * Optimizations:
 * 1. Eliminate vtbl4 gather by reordering the inner loop
 * 2. Process 64 weights per iteration (better instruction-level parallelism)
 * 3. Use prefetch hints for weight data
 * 4. Batch multiple rows (M) when possible
 */

#include "hs_ml.h"
#include <arm_neon.h>
#include <string.h>

/*
 * Key insight: Instead of gathering activations to match weights,
 * we can process weights in the order they're packed and accumulate
 * to the correct position.
 * 
 * Weight byte layout: [w3:w2:w1:w0] where each w is 2 bits
 * These correspond to activations at positions i*4+0, i*4+1, i*4+2, i*4+3
 * 
 * New approach: Load 16 bytes of activations (contiguous), then extract
 * the corresponding weight bits. No gather needed.
 */

void hs_ml_gemm_ternary_v2(int32_t* C,
                           const int8_t* A,
                           const u8* B_ternary,
                           u32 M, u32 N, u32 K) {
    
    memset(C, 0, M * N * sizeof(int32_t));
    
    /* Process 64 weights per iteration (16 packed bytes) */
    const u32 K64 = K & ~63u;
    const u32 N4 = N & ~3u;
    
    const uint8x16_t mask_03 = vdupq_n_u8(0x03);
    const uint8x16_t val_01  = vdupq_n_u8(0x01);
    const uint8x16_t val_02  = vdupq_n_u8(0x02);
    
    for (u32 m = 0; m < M; m++) {
        const int8_t* A_row = A + m * K;
        int32_t* C_row = C + m * N;
        
        /* Process 4 output columns at a time */
        for (u32 n = 0; n < N4; n += 4) {
            const u8* B_col0 = B_ternary + (n + 0) * (K / 4);
            const u8* B_col1 = B_ternary + (n + 1) * (K / 4);
            const u8* B_col2 = B_ternary + (n + 2) * (K / 4);
            const u8* B_col3 = B_ternary + (n + 3) * (K / 4);
            
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);
            
            for (u32 k = 0; k < K64; k += 64) {
                /* Prefetch next weight blocks */
                __builtin_prefetch(B_col0 + k/4 + 64, 0, 3);
                __builtin_prefetch(B_col1 + k/4 + 64, 0, 3);
                __builtin_prefetch(B_col2 + k/4 + 64, 0, 3);
                __builtin_prefetch(B_col3 + k/4 + 64, 0, 3);
                __builtin_prefetch(A_row + k + 128, 0, 3);
                
                /* Load 64 activations (4 x 16) */
                int8x16_t a0 = vld1q_s8(A_row + k + 0);
                int8x16_t a1 = vld1q_s8(A_row + k + 16);
                int8x16_t a2 = vld1q_s8(A_row + k + 32);
                int8x16_t a3 = vld1q_s8(A_row + k + 48);
                
                /* 
                 * For activations a[0..15], weights are in packed bytes 0..3
                 * packed[0] = weights for a[0,1,2,3]
                 * packed[1] = weights for a[4,5,6,7]
                 * packed[2] = weights for a[8,9,10,11]
                 * packed[3] = weights for a[12,13,14,15]
                 * 
                 * We need to expand each byte to match 4 activations.
                 * Use vzipq to interleave and create the right layout.
                 */
                
                /* Process column 0 */
                {
                    uint8x16_t packed = vld1q_u8(B_col0 + k / 4);
                    
                    /* 
                     * Expand packed weights: each byte becomes 4 weights
                     * We'll process in 4 groups of 16 activations each
                     */
                    
                    /* Group 0: activations 0-15 use packed bytes 0-3 */
                    /* Extract bits for each activation position */
                    
                    /* For a[i], weight is at packed[i/4], bits (i%4)*2 + 1 : (i%4)*2 */
                    
                    /* Expand byte 0-3 to match a0 */
                    /* packed[0] → a[0,1,2,3], packed[1] → a[4,5,6,7], etc */
                    
                    /* Use TBL to expand: create 16 indices that map each activation
                       to its corresponding packed byte */
                    static const uint8_t expand_idx[16] = {0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3};
                    uint8x16_t vidx = vld1q_u8(expand_idx);
                    
                    /* Get low 4 bytes of packed (weights for a0..a15) */
                    uint8x16_t p_lo = packed;  /* bytes 0-15 have weights for k..k+63 */
                    
                    /* Gather: each element becomes the byte containing its weight */
                    uint8x16_t w_bytes0 = vqtbl1q_u8(p_lo, vidx);
                    
                    /* Now extract the correct 2-bit field for each position */
                    /* Position within byte: 0,1,2,3,0,1,2,3,... */
                    static const uint8_t shifts[16] = {0,2,4,6, 0,2,4,6, 0,2,4,6, 0,2,4,6};
                    int8x16_t vshifts = vld1q_s8((const int8_t*)shifts);
                    
                    /* Shift each byte right by its position-specific amount */
                    /* NEON doesn't have per-lane variable shift for u8, so we use a trick */
                    /* Multiply by power of 2 then shift right by 6 to get top 2 bits */
                    /* Actually simpler: use vshl with negative shifts */
                    uint8x16_t w_shifted0 = vshlq_u8(w_bytes0, vnegq_s8(vshifts));
                    uint8x16_t w_bits0 = vandq_u8(w_shifted0, mask_03);
                    
                    /* Create masks */
                    uint8x16_t plus0  = vceqq_u8(w_bits0, val_01);
                    uint8x16_t minus0 = vceqq_u8(w_bits0, val_02);
                    
                    /* Select activations */
                    int8x16_t p0 = vandq_s8(a0, vreinterpretq_s8_u8(plus0));
                    int8x16_t m0 = vandq_s8(a0, vreinterpretq_s8_u8(minus0));
                    
                    /* Similarly for a1, a2, a3 */
                    static const uint8_t expand_idx1[16] = {4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7};
                    static const uint8_t expand_idx2[16] = {8,8,8,8, 9,9,9,9, 10,10,10,10, 11,11,11,11};
                    static const uint8_t expand_idx3[16] = {12,12,12,12, 13,13,13,13, 14,14,14,14, 15,15,15,15};
                    
                    uint8x16_t vidx1 = vld1q_u8(expand_idx1);
                    uint8x16_t vidx2 = vld1q_u8(expand_idx2);
                    uint8x16_t vidx3 = vld1q_u8(expand_idx3);
                    
                    uint8x16_t w_bytes1 = vqtbl1q_u8(p_lo, vidx1);
                    uint8x16_t w_bytes2 = vqtbl1q_u8(p_lo, vidx2);
                    uint8x16_t w_bytes3 = vqtbl1q_u8(p_lo, vidx3);
                    
                    uint8x16_t w_shifted1 = vshlq_u8(w_bytes1, vnegq_s8(vshifts));
                    uint8x16_t w_shifted2 = vshlq_u8(w_bytes2, vnegq_s8(vshifts));
                    uint8x16_t w_shifted3 = vshlq_u8(w_bytes3, vnegq_s8(vshifts));
                    
                    uint8x16_t w_bits1 = vandq_u8(w_shifted1, mask_03);
                    uint8x16_t w_bits2 = vandq_u8(w_shifted2, mask_03);
                    uint8x16_t w_bits3 = vandq_u8(w_shifted3, mask_03);
                    
                    uint8x16_t plus1  = vceqq_u8(w_bits1, val_01);
                    uint8x16_t minus1 = vceqq_u8(w_bits1, val_02);
                    uint8x16_t plus2  = vceqq_u8(w_bits2, val_01);
                    uint8x16_t minus2 = vceqq_u8(w_bits2, val_02);
                    uint8x16_t plus3  = vceqq_u8(w_bits3, val_01);
                    uint8x16_t minus3 = vceqq_u8(w_bits3, val_02);
                    
                    int8x16_t p1 = vandq_s8(a1, vreinterpretq_s8_u8(plus1));
                    int8x16_t m1 = vandq_s8(a1, vreinterpretq_s8_u8(minus1));
                    int8x16_t p2 = vandq_s8(a2, vreinterpretq_s8_u8(plus2));
                    int8x16_t m2 = vandq_s8(a2, vreinterpretq_s8_u8(minus2));
                    int8x16_t p3 = vandq_s8(a3, vreinterpretq_s8_u8(plus3));
                    int8x16_t m3 = vandq_s8(a3, vreinterpretq_s8_u8(minus3));
                    
                    /* Accumulate using pairwise add */
                    int16x8_t sum_plus = vpaddlq_s8(p0);
                    sum_plus = vpadalq_s8(sum_plus, p1);
                    sum_plus = vpadalq_s8(sum_plus, p2);
                    sum_plus = vpadalq_s8(sum_plus, p3);
                    
                    int16x8_t sum_minus = vpaddlq_s8(m0);
                    sum_minus = vpadalq_s8(sum_minus, m1);
                    sum_minus = vpadalq_s8(sum_minus, m2);
                    sum_minus = vpadalq_s8(sum_minus, m3);
                    
                    acc0 = vpadalq_s16(acc0, sum_plus);
                    acc0 = vsubq_s32(acc0, vpaddlq_s16(sum_minus));
                }
                
                /* Repeat for columns 1, 2, 3 (macro to avoid repetition) */
                #define PROCESS_COL_V2(B_col, acc) do { \
                    uint8x16_t packed = vld1q_u8(B_col + k / 4); \
                    static const uint8_t expand_idx_[16] = {0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3}; \
                    static const uint8_t expand_idx1_[16] = {4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7}; \
                    static const uint8_t expand_idx2_[16] = {8,8,8,8, 9,9,9,9, 10,10,10,10, 11,11,11,11}; \
                    static const uint8_t expand_idx3_[16] = {12,12,12,12, 13,13,13,13, 14,14,14,14, 15,15,15,15}; \
                    static const uint8_t shifts_[16] = {0,2,4,6, 0,2,4,6, 0,2,4,6, 0,2,4,6}; \
                    uint8x16_t vidx_ = vld1q_u8(expand_idx_); \
                    uint8x16_t vidx1_ = vld1q_u8(expand_idx1_); \
                    uint8x16_t vidx2_ = vld1q_u8(expand_idx2_); \
                    uint8x16_t vidx3_ = vld1q_u8(expand_idx3_); \
                    int8x16_t vshifts_ = vld1q_s8((const int8_t*)shifts_); \
                    \
                    uint8x16_t w_bytes0_ = vqtbl1q_u8(packed, vidx_); \
                    uint8x16_t w_bytes1_ = vqtbl1q_u8(packed, vidx1_); \
                    uint8x16_t w_bytes2_ = vqtbl1q_u8(packed, vidx2_); \
                    uint8x16_t w_bytes3_ = vqtbl1q_u8(packed, vidx3_); \
                    \
                    uint8x16_t w_shifted0_ = vshlq_u8(w_bytes0_, vnegq_s8(vshifts_)); \
                    uint8x16_t w_shifted1_ = vshlq_u8(w_bytes1_, vnegq_s8(vshifts_)); \
                    uint8x16_t w_shifted2_ = vshlq_u8(w_bytes2_, vnegq_s8(vshifts_)); \
                    uint8x16_t w_shifted3_ = vshlq_u8(w_bytes3_, vnegq_s8(vshifts_)); \
                    \
                    uint8x16_t w_bits0_ = vandq_u8(w_shifted0_, mask_03); \
                    uint8x16_t w_bits1_ = vandq_u8(w_shifted1_, mask_03); \
                    uint8x16_t w_bits2_ = vandq_u8(w_shifted2_, mask_03); \
                    uint8x16_t w_bits3_ = vandq_u8(w_shifted3_, mask_03); \
                    \
                    uint8x16_t plus0_  = vceqq_u8(w_bits0_, val_01); \
                    uint8x16_t minus0_ = vceqq_u8(w_bits0_, val_02); \
                    uint8x16_t plus1_  = vceqq_u8(w_bits1_, val_01); \
                    uint8x16_t minus1_ = vceqq_u8(w_bits1_, val_02); \
                    uint8x16_t plus2_  = vceqq_u8(w_bits2_, val_01); \
                    uint8x16_t minus2_ = vceqq_u8(w_bits2_, val_02); \
                    uint8x16_t plus3_  = vceqq_u8(w_bits3_, val_01); \
                    uint8x16_t minus3_ = vceqq_u8(w_bits3_, val_02); \
                    \
                    int8x16_t p0_ = vandq_s8(a0, vreinterpretq_s8_u8(plus0_)); \
                    int8x16_t _m0_ = vandq_s8(a0, vreinterpretq_s8_u8(minus0_)); \
                    int8x16_t p1_ = vandq_s8(a1, vreinterpretq_s8_u8(plus1_)); \
                    int8x16_t _m1_ = vandq_s8(a1, vreinterpretq_s8_u8(minus1_)); \
                    int8x16_t p2_ = vandq_s8(a2, vreinterpretq_s8_u8(plus2_)); \
                    int8x16_t _m2_ = vandq_s8(a2, vreinterpretq_s8_u8(minus2_)); \
                    int8x16_t p3_ = vandq_s8(a3, vreinterpretq_s8_u8(plus3_)); \
                    int8x16_t _m3_ = vandq_s8(a3, vreinterpretq_s8_u8(minus3_)); \
                    \
                    int16x8_t sum_plus_ = vpaddlq_s8(p0_); \
                    sum_plus_ = vpadalq_s8(sum_plus_, p1_); \
                    sum_plus_ = vpadalq_s8(sum_plus_, p2_); \
                    sum_plus_ = vpadalq_s8(sum_plus_, p3_); \
                    \
                    int16x8_t sum_minus_ = vpaddlq_s8(_m0_); \
                    sum_minus_ = vpadalq_s8(sum_minus_, _m1_); \
                    sum_minus_ = vpadalq_s8(sum_minus_, _m2_); \
                    sum_minus_ = vpadalq_s8(sum_minus_, _m3_); \
                    \
                    acc = vpadalq_s16(acc, sum_plus_); \
                    acc = vsubq_s32(acc, vpaddlq_s16(sum_minus_)); \
                } while(0)
                
                PROCESS_COL_V2(B_col1, acc1);
                PROCESS_COL_V2(B_col2, acc2);
                PROCESS_COL_V2(B_col3, acc3);
                
                #undef PROCESS_COL_V2
            }
            
            /* Horizontal sum and store */
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

