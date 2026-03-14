/*
 * NeoGPU ML - Ternary GEMM v5 - Targeting 6 GOPS
 * 
 * Best approach: Process weights in natural order, use vmlal for multiply
 * Avoid table lookups entirely by processing 4 activations per weight byte
 * sequentially with proper interleaving.
 * 
 * Key: Load 8 bytes of weights, expand each byte to 4 signed weights,
 * then multiply with corresponding 32 activations.
 */

#include "hs_ml.h"
#include <arm_neon.h>
#include <string.h>

void hs_ml_gemm_ternary_v5(int32_t* C,
                           const int8_t* A,
                           const u8* B_ternary,
                           u32 M, u32 N, u32 K) {
    
    memset(C, 0, M * N * sizeof(int32_t));
    
    const u32 K32 = K & ~31u;
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
            
            for (u32 k = 0; k < K32; k += 32) {
                __builtin_prefetch(A_row + k + 64, 0, 1);
                __builtin_prefetch(B0 + (k + 64) / 4, 0, 1);
                
                /* Load 8 weight bytes = 32 weights */
                uint8x8_t wb0 = vld1_u8(B0 + k / 4);
                uint8x8_t wb1 = vld1_u8(B1 + k / 4);
                uint8x8_t wb2 = vld1_u8(B2 + k / 4);
                uint8x8_t wb3 = vld1_u8(B3 + k / 4);
                
                /* 
                 * Each byte has 4 weights for consecutive activations.
                 * wb0[i] has weights for A[k + i*4 + 0..3]
                 * 
                 * Extract and convert to signed:
                 * For bits b1:b0: signed_weight = (b0) - (b1)
                 * 01 → 1-0 = +1
                 * 10 → 0-1 = -1
                 * 00 → 0-0 = 0
                 * 11 → 1-1 = 0 (undefined but safe)
                 */
                
                /* Process 4 activations per weight byte, 8 bytes at a time */
                /* Use 4-way unrolled scalar loop - compiler will vectorize */
                
                #define PROCESS_8_BYTES(WB, ACC) do { \
                    int32_t local = 0; \
                    for (int i = 0; i < 8; i++) { \
                        u8 b = vget_lane_u8(WB, i); \
                        int8_t w0 = (b & 1) - ((b >> 1) & 1); \
                        int8_t w1 = ((b >> 2) & 1) - ((b >> 3) & 1); \
                        int8_t w2 = ((b >> 4) & 1) - ((b >> 5) & 1); \
                        int8_t w3 = ((b >> 6) & 1) - ((b >> 7) & 1); \
                        const int8_t* ap = A_row + k + i * 4; \
                        local += w0 * ap[0] + w1 * ap[1] + w2 * ap[2] + w3 * ap[3]; \
                    } \
                    ACC = vsetq_lane_s32(vgetq_lane_s32(ACC, 0) + local, ACC, 0); \
                } while(0)
                
                /* Actually vget_lane in a loop is terrible. Use pure scalar. */
                #undef PROCESS_8_BYTES
                
                /* Column 0 */
                {
                    int32_t local = 0;
                    const u8* bp = B0 + k / 4;
                    for (int i = 0; i < 8; i++) {
                        u8 b = bp[i];
                        int8_t w0 = (b & 1) - ((b >> 1) & 1);
                        int8_t w1 = ((b >> 2) & 1) - ((b >> 3) & 1);
                        int8_t w2 = ((b >> 4) & 1) - ((b >> 5) & 1);
                        int8_t w3 = ((b >> 6) & 1) - ((b >> 7) & 1);
                        const int8_t* ap = A_row + k + i * 4;
                        local += w0 * ap[0] + w1 * ap[1] + w2 * ap[2] + w3 * ap[3];
                    }
                    acc0 = vsetq_lane_s32(vgetq_lane_s32(acc0, 0) + local, acc0, 0);
                }
                
                /* Column 1 */
                {
                    int32_t local = 0;
                    const u8* bp = B1 + k / 4;
                    for (int i = 0; i < 8; i++) {
                        u8 b = bp[i];
                        int8_t w0 = (b & 1) - ((b >> 1) & 1);
                        int8_t w1 = ((b >> 2) & 1) - ((b >> 3) & 1);
                        int8_t w2 = ((b >> 4) & 1) - ((b >> 5) & 1);
                        int8_t w3 = ((b >> 6) & 1) - ((b >> 7) & 1);
                        const int8_t* ap = A_row + k + i * 4;
                        local += w0 * ap[0] + w1 * ap[1] + w2 * ap[2] + w3 * ap[3];
                    }
                    acc1 = vsetq_lane_s32(vgetq_lane_s32(acc1, 0) + local, acc1, 0);
                }
                
                /* Column 2 */
                {
                    int32_t local = 0;
                    const u8* bp = B2 + k / 4;
                    for (int i = 0; i < 8; i++) {
                        u8 b = bp[i];
                        int8_t w0 = (b & 1) - ((b >> 1) & 1);
                        int8_t w1 = ((b >> 2) & 1) - ((b >> 3) & 1);
                        int8_t w2 = ((b >> 4) & 1) - ((b >> 5) & 1);
                        int8_t w3 = ((b >> 6) & 1) - ((b >> 7) & 1);
                        const int8_t* ap = A_row + k + i * 4;
                        local += w0 * ap[0] + w1 * ap[1] + w2 * ap[2] + w3 * ap[3];
                    }
                    acc2 = vsetq_lane_s32(vgetq_lane_s32(acc2, 0) + local, acc2, 0);
                }
                
                /* Column 3 */
                {
                    int32_t local = 0;
                    const u8* bp = B3 + k / 4;
                    for (int i = 0; i < 8; i++) {
                        u8 b = bp[i];
                        int8_t w0 = (b & 1) - ((b >> 1) & 1);
                        int8_t w1 = ((b >> 2) & 1) - ((b >> 3) & 1);
                        int8_t w2 = ((b >> 4) & 1) - ((b >> 5) & 1);
                        int8_t w3 = ((b >> 6) & 1) - ((b >> 7) & 1);
                        const int8_t* ap = A_row + k + i * 4;
                        local += w0 * ap[0] + w1 * ap[1] + w2 * ap[2] + w3 * ap[3];
                    }
                    acc3 = vsetq_lane_s32(vgetq_lane_s32(acc3, 0) + local, acc3, 0);
                }
            }
            
            C_row[n + 0] = vgetq_lane_s32(acc0, 0);
            C_row[n + 1] = vgetq_lane_s32(acc1, 0);
            C_row[n + 2] = vgetq_lane_s32(acc2, 0);
            C_row[n + 3] = vgetq_lane_s32(acc3, 0);
            
            /* Remainder */
            for (u32 col = 0; col < 4; col++) {
                const u8* B_col = B_ternary + (n + col) * (K / 4);
                for (u32 k = K32; k < K; k++) {
                    u8 byte = B_col[k / 4];
                    u8 bits = (byte >> ((k % 4) * 2)) & 0x03;
                    int8_t a = A_row[k];
                    if (bits == 1) C_row[n + col] += a;
                    else if (bits == 2) C_row[n + col] -= a;
                }
            }
        }
        
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
 * v6: Pure optimized scalar with compiler auto-vectorization hints
 * Let -O3 -ffast-math do its magic
 */
void hs_ml_gemm_ternary_v6(int32_t* restrict C,
                           const int8_t* restrict A,
                           const u8* restrict B_ternary,
                           u32 M, u32 N, u32 K) {
    
    memset(C, 0, M * N * sizeof(int32_t));
    
    for (u32 m = 0; m < M; m++) {
        const int8_t* restrict A_row = A + m * K;
        int32_t* restrict C_row = C + m * N;
        
        for (u32 n = 0; n < N; n++) {
            const u8* restrict B_col = B_ternary + n * (K / 4);
            int32_t sum = 0;
            
            /* Process 4 activations per byte */
            u32 num_bytes = K / 4;
            
            #pragma GCC unroll 8
            for (u32 i = 0; i < num_bytes; i++) {
                u8 b = B_col[i];
                const int8_t* ap = A_row + i * 4;
                
                /* Convert 2-bit weights to signed */
                int8_t w0 = (b & 1) - ((b >> 1) & 1);
                int8_t w1 = ((b >> 2) & 1) - ((b >> 3) & 1);
                int8_t w2 = ((b >> 4) & 1) - ((b >> 5) & 1);
                int8_t w3 = (b >> 6) - ((b >> 7) & 1);
                
                sum += w0 * ap[0] + w1 * ap[1] + w2 * ap[2] + w3 * ap[3];
            }
            
            C_row[n] = sum;
        }
    }
}

