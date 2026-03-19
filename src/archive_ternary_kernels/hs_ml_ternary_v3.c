/*
 * NeoGPU ML - Ternary GEMM v3 - Targeting 6 GOPS
 * 
 * Key insight: Process 4 activations per weight byte directly.
 * No table lookup, no per-element shifts.
 * 
 * For each packed byte containing 4 weights:
 *   1. Extract all 4 weight values at once
 *   2. Process corresponding 4 activations 
 *   3. Accumulate
 */

#include "hs_ml.h"
#include <arm_neon.h>
#include <string.h>

void hs_ml_gemm_ternary_v3(int32_t* C,
                           const int8_t* A,
                           const u8* B_ternary,
                           u32 M, u32 N, u32 K) {
    
    memset(C, 0, M * N * sizeof(int32_t));
    
    /* Process in blocks of 64 activations (16 weight bytes) */
    const u32 K64 = K & ~63u;
    const u32 N4 = N & ~3u;
    
    for (u32 m = 0; m < M; m++) {
        const int8_t* A_row = A + m * K;
        int32_t* C_row = C + m * N;
        
        for (u32 n = 0; n < N4; n += 4) {
            const u8* B_col0 = B_ternary + (n + 0) * (K / 4);
            const u8* B_col1 = B_ternary + (n + 1) * (K / 4);
            const u8* B_col2 = B_ternary + (n + 2) * (K / 4);
            const u8* B_col3 = B_ternary + (n + 3) * (K / 4);
            
            int32_t sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
            
            /* 
             * Inner loop: process 16 weight bytes = 64 activations at a time
             * Use 128-bit loads for both weights and activations
             */
            for (u32 k = 0; k < K64; k += 64) {
                /* Prefetch */
                __builtin_prefetch(A_row + k + 128, 0, 0);
                __builtin_prefetch(B_col0 + (k + 128) / 4, 0, 0);
                
                /* Load 64 activations */
                int8x16_t a0 = vld1q_s8(A_row + k + 0);
                int8x16_t a1 = vld1q_s8(A_row + k + 16);
                int8x16_t a2 = vld1q_s8(A_row + k + 32);
                int8x16_t a3 = vld1q_s8(A_row + k + 48);
                
                /* Load 16 weight bytes for each column */
                uint8x16_t w0 = vld1q_u8(B_col0 + k / 4);
                uint8x16_t w1 = vld1q_u8(B_col1 + k / 4);
                uint8x16_t w2 = vld1q_u8(B_col2 + k / 4);
                uint8x16_t w3 = vld1q_u8(B_col3 + k / 4);
                
                /*
                 * Each weight byte has 4 x 2-bit weights.
                 * weight byte i corresponds to activations [i*4, i*4+1, i*4+2, i*4+3]
                 * 
                 * For w0[0..15], we need a[0..3], a[4..7], ..., a[60..63]
                 * 
                 * Extract bit pairs and create signed weights:
                 *   bits=01 (+1) → +1
                 *   bits=10 (-1) → -1  
                 *   bits=00 (0)  → 0
                 *   bits=11      → treat as 0 (unused)
                 * 
                 * Trick: (bits & 1) - ((bits >> 1) & 1) gives us -1, 0, or +1
                 */
                
                /* Process column 0 */
                {
                    /* Extract 4 weight groups from w0 */
                    /* Group 0: bits [1:0], Group 1: bits [3:2], etc */
                    uint8x16_t g0 = vandq_u8(w0, vdupq_n_u8(0x03));
                    uint8x16_t g1 = vandq_u8(vshrq_n_u8(w0, 2), vdupq_n_u8(0x03));
                    uint8x16_t g2 = vandq_u8(vshrq_n_u8(w0, 4), vdupq_n_u8(0x03));
                    uint8x16_t g3 = vshrq_n_u8(w0, 6);
                    
                    /* Convert to signed: low_bit - high_bit */
                    /* For bits=01: 1-0=+1, bits=10: 0-1=-1, bits=00: 0-0=0 */
                    int8x16_t s0 = vsubq_s8(
                        vreinterpretq_s8_u8(vandq_u8(g0, vdupq_n_u8(1))),
                        vreinterpretq_s8_u8(vshrq_n_u8(g0, 1))
                    );
                    int8x16_t s1 = vsubq_s8(
                        vreinterpretq_s8_u8(vandq_u8(g1, vdupq_n_u8(1))),
                        vreinterpretq_s8_u8(vshrq_n_u8(g1, 1))
                    );
                    int8x16_t s2 = vsubq_s8(
                        vreinterpretq_s8_u8(vandq_u8(g2, vdupq_n_u8(1))),
                        vreinterpretq_s8_u8(vshrq_n_u8(g2, 1))
                    );
                    int8x16_t s3 = vsubq_s8(
                        vreinterpretq_s8_u8(vandq_u8(g3, vdupq_n_u8(1))),
                        vreinterpretq_s8_u8(vshrq_n_u8(g3, 1))
                    );
                    
                    /*
                     * Now s0[i] is the weight for activation a[i*4+0]
                     * But a0 contains activations [0,1,2,3,4,5,...,15]
                     * So s0[0] matches a0[0], s0[1] matches a0[4], etc
                     * 
                     * We need to interleave/deinterleave somehow...
                     * 
                     * Actually, let's use multiply since we have signed weights now!
                     * This is simpler and might be fast enough.
                     */
                    
                    /* 
                     * Alternative: use vmull to multiply s * a and accumulate
                     * vmull_s8: int8x8 * int8x8 → int16x8
                     */
                    
                    /* But we need to match weights to activations properly */
                    /* s0 has 16 weights, each for groups of 4 activations */
                    /* s0[i] = weight for a[i*4+0] */
                    /* s1[i] = weight for a[i*4+1] */
                    /* s2[i] = weight for a[i*4+2] */
                    /* s3[i] = weight for a[i*4+3] */
                    
                    /* Interleave activations to match weights */
                    /* a0 = [a0, a1, a2, a3, a4, a5, a6, a7, a8, ...] */
                    /* We need [a0, a4, a8, a12, ...] to match s0 */
                    
                    /* Use deinterleave: vuzp extracts even/odd elements */
                    /* But we need stride-4, not stride-2 */
                    
                    /* Actually simpler: just process 4 activations per weight byte */
                    /* Use scalar loop structure but vectorize across weight bytes */
                    
                    /* TODO: This is getting complex. Let's try a different approach */
                    /* For now, use the multiply approach even if not optimal */
                    
                    /* Simple approach: sum += weight * activation per byte */
                    /* Process byte-by-byte but with manual unrolling */
                    
                    int32_t local_sum = 0;
                    const u8* bptr = B_col0 + k / 4;
                    const int8_t* aptr = A_row + k;
                    
                    for (int i = 0; i < 16; i++) {
                        u8 wb = bptr[i];
                        int8_t w0b = (wb & 1) - ((wb >> 1) & 1);
                        int8_t w1b = ((wb >> 2) & 1) - ((wb >> 3) & 1);
                        int8_t w2b = ((wb >> 4) & 1) - ((wb >> 5) & 1);
                        int8_t w3b = ((wb >> 6) & 1) - ((wb >> 7) & 1);
                        
                        local_sum += w0b * aptr[i*4 + 0];
                        local_sum += w1b * aptr[i*4 + 1];
                        local_sum += w2b * aptr[i*4 + 2];
                        local_sum += w3b * aptr[i*4 + 3];
                    }
                    sum0 += local_sum;
                }
                
                /* Columns 1, 2, 3 - same pattern */
                {
                    int32_t local_sum = 0;
                    const u8* bptr = B_col1 + k / 4;
                    const int8_t* aptr = A_row + k;
                    for (int i = 0; i < 16; i++) {
                        u8 wb = bptr[i];
                        int8_t w0b = (wb & 1) - ((wb >> 1) & 1);
                        int8_t w1b = ((wb >> 2) & 1) - ((wb >> 3) & 1);
                        int8_t w2b = ((wb >> 4) & 1) - ((wb >> 5) & 1);
                        int8_t w3b = ((wb >> 6) & 1) - ((wb >> 7) & 1);
                        local_sum += w0b * aptr[i*4 + 0] + w1b * aptr[i*4 + 1] + w2b * aptr[i*4 + 2] + w3b * aptr[i*4 + 3];
                    }
                    sum1 += local_sum;
                }
                {
                    int32_t local_sum = 0;
                    const u8* bptr = B_col2 + k / 4;
                    const int8_t* aptr = A_row + k;
                    for (int i = 0; i < 16; i++) {
                        u8 wb = bptr[i];
                        int8_t w0b = (wb & 1) - ((wb >> 1) & 1);
                        int8_t w1b = ((wb >> 2) & 1) - ((wb >> 3) & 1);
                        int8_t w2b = ((wb >> 4) & 1) - ((wb >> 5) & 1);
                        int8_t w3b = ((wb >> 6) & 1) - ((wb >> 7) & 1);
                        local_sum += w0b * aptr[i*4 + 0] + w1b * aptr[i*4 + 1] + w2b * aptr[i*4 + 2] + w3b * aptr[i*4 + 3];
                    }
                    sum2 += local_sum;
                }
                {
                    int32_t local_sum = 0;
                    const u8* bptr = B_col3 + k / 4;
                    const int8_t* aptr = A_row + k;
                    for (int i = 0; i < 16; i++) {
                        u8 wb = bptr[i];
                        int8_t w0b = (wb & 1) - ((wb >> 1) & 1);
                        int8_t w1b = ((wb >> 2) & 1) - ((wb >> 3) & 1);
                        int8_t w2b = ((wb >> 4) & 1) - ((wb >> 5) & 1);
                        int8_t w3b = ((wb >> 6) & 1) - ((wb >> 7) & 1);
                        local_sum += w0b * aptr[i*4 + 0] + w1b * aptr[i*4 + 1] + w2b * aptr[i*4 + 2] + w3b * aptr[i*4 + 3];
                    }
                    sum3 += local_sum;
                }
            }
            
            C_row[n + 0] = sum0;
            C_row[n + 1] = sum1;
            C_row[n + 2] = sum2;
            C_row[n + 3] = sum3;
            
            /* Remainder */
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

/*
 * v4: Actually vectorize the multiply approach
 * Convert weights to {-1,0,+1} then use vmull for the multiply
 */
void hs_ml_gemm_ternary_v4(int32_t* C,
                           const int8_t* A,
                           const u8* B_ternary,
                           u32 M, u32 N, u32 K) {
    
    memset(C, 0, M * N * sizeof(int32_t));
    
    const u32 K32 = K & ~31u;
    const u32 N4 = N & ~3u;
    
    const uint8x8_t mask_03 = vdup_n_u8(0x03);
    const uint8x8_t mask_01 = vdup_n_u8(0x01);
    
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
            
            for (u32 k = 0; k < K32; k += 32) {
                /* Load 8 weight bytes (32 weights) */
                uint8x8_t w0 = vld1_u8(B0 + k / 4);
                uint8x8_t w1 = vld1_u8(B1 + k / 4);
                uint8x8_t w2 = vld1_u8(B2 + k / 4);
                uint8x8_t w3 = vld1_u8(B3 + k / 4);
                
                /* Load 32 activations */
                int8x8_t a0 = vld1_s8(A_row + k + 0);
                int8x8_t a1 = vld1_s8(A_row + k + 8);
                int8x8_t a2 = vld1_s8(A_row + k + 16);
                int8x8_t a3 = vld1_s8(A_row + k + 24);
                
                /* 
                 * For each weight byte, extract 4 x 2-bit weights
                 * Convert to signed: low_bit - high_bit
                 */
                
                /* Process w0 → weights for B0 column */
                #define EXTRACT_AND_MUL(w, a_vec, result) do { \
                    /* Extract 4 groups of 2-bit weights */ \
                    uint8x8_t g0 = vand_u8(w, mask_03); \
                    uint8x8_t g1 = vand_u8(vshr_n_u8(w, 2), mask_03); \
                    uint8x8_t g2 = vand_u8(vshr_n_u8(w, 4), mask_03); \
                    uint8x8_t g3 = vshr_n_u8(w, 6); \
                    \
                    /* Convert to signed: (bits & 1) - (bits >> 1) */ \
                    int8x8_t s0 = vsub_s8(vreinterpret_s8_u8(vand_u8(g0, mask_01)), \
                                          vreinterpret_s8_u8(vshr_n_u8(g0, 1))); \
                    int8x8_t s1 = vsub_s8(vreinterpret_s8_u8(vand_u8(g1, mask_01)), \
                                          vreinterpret_s8_u8(vshr_n_u8(g1, 1))); \
                    int8x8_t s2 = vsub_s8(vreinterpret_s8_u8(vand_u8(g2, mask_01)), \
                                          vreinterpret_s8_u8(vshr_n_u8(g2, 1))); \
                    int8x8_t s3 = vsub_s8(vreinterpret_s8_u8(vand_u8(g3, mask_01)), \
                                          vreinterpret_s8_u8(vshr_n_u8(g3, 1))); \
                    \
                    /* s0..s3 now contain signed weights {-1, 0, +1} */ \
                    /* But they're interleaved: s0[i] is for a[i*4+0], s1[i] is for a[i*4+1] */ \
                    \
                    /* We need to multiply and accumulate */ \
                    /* Use vmlal_s8: multiply-accumulate long */ \
                    /* Problem: s0[0] needs a[0], s0[1] needs a[4], etc */ \
                    \
                    /* Deinterleave a_vec to match: */ \
                    /* From a[0,1,2,3,4,5,6,7], extract a[0,4] for s0[0,1] */ \
                    \
                    /* This is the same gather problem... */ \
                    /* Let's just do the multiply with wrong pairing and sum */ \
                    /* Actually for inner product, sum(s*a) = sum(any_pairing(s)*a) if we sum all */ \
                    /* No that's wrong too */ \
                    \
                    /* Correct approach: interleave the weights to match sequential a */ \
                    /* vzip s0,s1,s2,s3 to get [s0[0],s1[0],s2[0],s3[0], s0[1],s1[1],...] */ \
                    int8x8x2_t z01 = vzip_s8(s0, s1); \
                    int8x8x2_t z23 = vzip_s8(s2, s3); \
                    int8x8x2_t zlo = vzip_s8(z01.val[0], z23.val[0]); \
                    int8x8x2_t zhi = vzip_s8(z01.val[1], z23.val[1]); \
                    /* Now zlo.val[0] = [s0[0],s2[0],s1[0],s3[0], s0[1],s2[1],s1[1],s3[1]] */ \
                    /* Still not right order. Need [s0[0],s1[0],s2[0],s3[0],...] */ \
                    \
                    /* Use vtrn instead */ \
                    int8x8x2_t t01 = vtrn_s8(s0, s1); \
                    int8x8x2_t t23 = vtrn_s8(s2, s3); \
                    int8x8x2_t tlo = vtrn_s8(t01.val[0], t23.val[0]); \
                    /* Hmm still complex */ \
                    \
                    /* Simpler: just multiply s0*a at positions 0,4,8,... */ \
                    /* and use multiple vmlal with different strides */ \
                    /* But NEON doesn't support strided loads easily */ \
                    \
                    /* Fallback: scalar inner loop with compiler vectorization */ \
                    int32_t sum = 0; \
                    for (int i = 0; i < 8; i++) { \
                        sum += vget_lane_s8(s0, i) * (A_row[k + i*4 + 0]); \
                        sum += vget_lane_s8(s1, i) * (A_row[k + i*4 + 1]); \
                        sum += vget_lane_s8(s2, i) * (A_row[k + i*4 + 2]); \
                        sum += vget_lane_s8(s3, i) * (A_row[k + i*4 + 3]); \
                    } \
                    result = vsetq_lane_s32(vgetq_lane_s32(result, 0) + sum, result, 0); \
                } while(0)
                
                /* This approach is getting too messy. Let me try pure scalar. */
                /* The compiler should auto-vectorize better than this. */
                
                (void)a0; (void)a1; (void)a2; (void)a3;
                (void)w0; (void)w1; (void)w2; (void)w3;
                (void)acc0; (void)acc1; (void)acc2; (void)acc3;
                
                #undef EXTRACT_AND_MUL
            }
            
            /* Pure scalar fallback for now */
            int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (u32 k = 0; k < K; k++) {
                int8_t a = A_row[k];
                u8 b0 = B0[k/4], b1 = B1[k/4], b2 = B2[k/4], b3 = B3[k/4];
                int shift = (k % 4) * 2;
                int w0 = ((b0 >> shift) & 1) - ((b0 >> (shift+1)) & 1);
                int w1 = ((b1 >> shift) & 1) - ((b1 >> (shift+1)) & 1);
                int w2 = ((b2 >> shift) & 1) - ((b2 >> (shift+1)) & 1);
                int w3 = ((b3 >> shift) & 1) - ((b3 >> (shift+1)) & 1);
                s0 += w0 * a;
                s1 += w1 * a;
                s2 += w2 * a;
                s3 += w3 * a;
            }
            C_row[n+0] = s0;
            C_row[n+1] = s1;
            C_row[n+2] = s2;
            C_row[n+3] = s3;
        }
        
        for (u32 n = N4; n < N; n++) {
            const u8* B_col = B_ternary + n * (K / 4);
            int32_t sum = 0;
            for (u32 k = 0; k < K; k++) {
                u8 byte = B_col[k / 4];
                int shift = (k % 4) * 2;
                int w = ((byte >> shift) & 1) - ((byte >> (shift+1)) & 1);
                sum += w * A_row[k];
            }
            C_row[n] = sum;
        }
    }
}

