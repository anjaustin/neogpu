/*
 * NeoGPU ML - Ternary GEMM v13 - Cache-blocking for 6 GOPS
 * 
 * Analysis:
 * - V8 peaks at 4.8 GOPS for small matrices but drops to 4.1-4.6 for large
 * - This suggests cache effects - when K is large, we're thrashing L1
 * - Pi4 Cortex-A72 has 32KB L1D, 1MB L2
 * 
 * Strategy: Block K dimension to keep working set in L1
 * - For K block of 256: activations = 256 bytes, weights per col = 64 bytes
 * - With 4 columns: 256 + 256 = 512 bytes << 32KB L1
 */

#include "hs_ml.h"
#include <arm_neon.h>
#include <string.h>

/* Block size for K - chosen to fit comfortably in L1 */
#define KBLOCK 256

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

void hs_ml_gemm_ternary_v13(int32_t* C,
                            const int8_t* A,
                            const u8* B_ternary,
                            u32 M, u32 N, u32 K) {
    
    memset(C, 0, M * N * sizeof(int32_t));
    
    const u32 N4 = N & ~3u;
    const u32 Kstride = K / 4;
    
    for (u32 m = 0; m < M; m++) {
        const int8_t* A_row = A + m * K;
        int32_t* C_row = C + m * N;
        
        /* Process columns in groups of 4 */
        for (u32 n = 0; n < N4; n += 4) {
            const u8* B0 = B_ternary + (n + 0) * Kstride;
            const u8* B1 = B_ternary + (n + 1) * Kstride;
            const u8* B2 = B_ternary + (n + 2) * Kstride;
            const u8* B3 = B_ternary + (n + 3) * Kstride;
            
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);
            
            /* Block over K to stay in L1 cache */
            for (u32 kb = 0; kb < K; kb += KBLOCK) {
                u32 kend = kb + KBLOCK;
                if (kend > K) kend = K;
                u32 klen = kend - kb;
                u32 k64 = (klen / 64) * 64;  /* Full 64-element blocks */
                
                /* Process full 64-element blocks */
                for (u32 ki = 0; ki < k64; ki += 64) {
                    u32 k = kb + ki;
                    
                    /* Prefetch next block */
                    __builtin_prefetch(A_row + k + 64, 0, 3);
                    __builtin_prefetch(B0 + (k + 64) / 4, 0, 3);
                    
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
                    
                    /* Process all 4 columns */
                    #define PROC_COL(Bptr, acc) do { \
                        uint8x16_t wb = vld1q_u8(Bptr + ko); \
                        int8x16_t w0, w1, w2, w3; \
                        DECODE_WEIGHTS(wb, w0, w1, w2, w3); \
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
                    
                    #undef PROC_COL
                }
                
                /* Handle remaining elements in this K block */
                for (u32 k = kb + k64; k < kend; k++) {
                    int8_t a = A_row[k];
                    u32 ko4 = k / 4;
                    u32 shift = (k % 4) * 2;
                    
                    for (u32 col = 0; col < 4; col++) {
                        const u8* B_col = B_ternary + (n + col) * Kstride;
                        u8 bits = (B_col[ko4] >> shift) & 0x03;
                        if (bits == 1) C_row[n + col] += a;
                        else if (bits == 2) C_row[n + col] -= a;
                    }
                }
            }
            
            C_row[n + 0] += vaddvq_s32(acc0);
            C_row[n + 1] += vaddvq_s32(acc1);
            C_row[n + 2] += vaddvq_s32(acc2);
            C_row[n + 3] += vaddvq_s32(acc3);
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
 * v14: Process more columns (8) with cache blocking
 * Try to find sweet spot between activation reuse and register pressure
 */
void hs_ml_gemm_ternary_v14(int32_t* C,
                            const int8_t* A,
                            const u8* B_ternary,
                            u32 M, u32 N, u32 K) {
    
    memset(C, 0, M * N * sizeof(int32_t));
    
    const u32 N8 = N & ~7u;
    const u32 N4 = N & ~3u;
    const u32 Kstride = K / 4;
    
    for (u32 m = 0; m < M; m++) {
        const int8_t* A_row = A + m * K;
        int32_t* C_row = C + m * N;
        
        /* Process columns in groups of 4 (8 columns caused register spilling) */
        for (u32 n = 0; n < N4; n += 4) {
            const u8* B0 = B_ternary + (n + 0) * Kstride;
            const u8* B1 = B_ternary + (n + 1) * Kstride;
            const u8* B2 = B_ternary + (n + 2) * Kstride;
            const u8* B3 = B_ternary + (n + 3) * Kstride;
            
            /* Use two sets of accumulators to break dependency chains */
            int32x4_t acc0a = vdupq_n_s32(0);
            int32x4_t acc1a = vdupq_n_s32(0);
            int32x4_t acc2a = vdupq_n_s32(0);
            int32x4_t acc3a = vdupq_n_s32(0);
            int32x4_t acc0b = vdupq_n_s32(0);
            int32x4_t acc1b = vdupq_n_s32(0);
            int32x4_t acc2b = vdupq_n_s32(0);
            int32x4_t acc3b = vdupq_n_s32(0);
            
            /* Process 128 elements at a time (2x64) */
            u32 K128 = K & ~127u;
            for (u32 k = 0; k < K128; k += 128) {
                /* First batch of 64 */
                {
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
                    
                    #define PROC_COL_A(Bptr, acc) do { \
                        uint8x16_t wb = vld1q_u8(Bptr + ko); \
                        int8x16_t w0, w1, w2, w3; \
                        DECODE_WEIGHTS(wb, w0, w1, w2, w3); \
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
                    
                    PROC_COL_A(B0, acc0a);
                    PROC_COL_A(B1, acc1a);
                    PROC_COL_A(B2, acc2a);
                    PROC_COL_A(B3, acc3a);
                    #undef PROC_COL_A
                }
                
                /* Second batch of 64 into separate accumulators */
                {
                    int8x16_t a0 = vld1q_s8(A_row + k + 64);
                    int8x16_t a1 = vld1q_s8(A_row + k + 80);
                    int8x16_t a2 = vld1q_s8(A_row + k + 96);
                    int8x16_t a3 = vld1q_s8(A_row + k + 112);
                    
                    int8x16x2_t d0 = vuzpq_s8(a0, a1);
                    int8x16x2_t d1 = vuzpq_s8(a2, a3);
                    int8x16x2_t g01 = vuzpq_s8(d0.val[0], d1.val[0]);
                    int8x16x2_t g23 = vuzpq_s8(d0.val[1], d1.val[1]);
                    
                    int8x16_t ag0 = g01.val[0];
                    int8x16_t ag2 = g01.val[1];
                    int8x16_t ag1 = g23.val[0];
                    int8x16_t ag3 = g23.val[1];
                    
                    u32 ko = (k + 64) / 4;
                    
                    #define PROC_COL_B(Bptr, acc) do { \
                        uint8x16_t wb = vld1q_u8(Bptr + ko); \
                        int8x16_t w0, w1, w2, w3; \
                        DECODE_WEIGHTS(wb, w0, w1, w2, w3); \
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
                    
                    PROC_COL_B(B0, acc0b);
                    PROC_COL_B(B1, acc1b);
                    PROC_COL_B(B2, acc2b);
                    PROC_COL_B(B3, acc3b);
                    #undef PROC_COL_B
                }
            }
            
            /* Merge accumulators */
            int32x4_t acc0 = vaddq_s32(acc0a, acc0b);
            int32x4_t acc1 = vaddq_s32(acc1a, acc1b);
            int32x4_t acc2 = vaddq_s32(acc2a, acc2b);
            int32x4_t acc3 = vaddq_s32(acc3a, acc3b);
            
            /* Remaining 64-element blocks */
            for (u32 k = K128; k + 64 <= K; k += 64) {
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
                    int8x16_t w0, w1, w2, w3; \
                    DECODE_WEIGHTS(wb, w0, w1, w2, w3); \
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
                #undef PROC_COL
            }
            
            C_row[n + 0] = vaddvq_s32(acc0);
            C_row[n + 1] = vaddvq_s32(acc1);
            C_row[n + 2] = vaddvq_s32(acc2);
            C_row[n + 3] = vaddvq_s32(acc3);
            
            /* Remainder K */
            u32 K64 = K & ~63u;
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

