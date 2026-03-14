/*
 * NeoGPU ML - True Ternary GEMM Kernel
 * 
 * Optimized for BitNet 1.58-bit weights on ARMv8.0 (Cortex-A72, Pi4)
 * No unpacking to INT8, no multiplication - just conditional add/sub
 * 
 * Weight encoding: 00 = 0, 01 = +1, 10 = -1, 11 = unused
 */

#include "hs_ml.h"
#include <arm_neon.h>
#include <string.h>

/*
 * hs_ml_gemm_ternary_neon - True ternary GEMM without multiply
 * 
 * C[M,N] = A[M,K] × B[K,N]  where B is ternary {-1, 0, +1}
 * 
 * A:         INT8 activations, row-major [M × K]
 * B_ternary: 2-bit packed weights, layout [N × K/4] (each output channel contiguous)
 * C:         INT32 accumulators, row-major [M × N]
 */
void hs_ml_gemm_ternary_neon(int32_t* C,
                              const int8_t* A,
                              const u8* B_ternary,
                              u32 M, u32 N, u32 K) {
    
    memset(C, 0, M * N * sizeof(int32_t));
    
    /* Process 32 weights per iteration (8 packed bytes) */
    const u32 K32 = K & ~31u;
    
    /* Constants */
    const uint8x8_t mask_03 = vdup_n_u8(0x03);
    const uint8x8_t val_01  = vdup_n_u8(0x01);
    const uint8x8_t val_02  = vdup_n_u8(0x02);
    
    /* Preload index arrays - cast to int8 for vtbl4_s8 */
    static const int8_t idx0_arr[8] = {0, 4, 8, 12, 16, 20, 24, 28}; 
    static const int8_t idx1_arr[8] = {1, 5, 9, 13, 17, 21, 25, 29};
    static const int8_t idx2_arr[8] = {2, 6, 10, 14, 18, 22, 26, 30};
    static const int8_t idx3_arr[8] = {3, 7, 11, 15, 19, 23, 27, 31};
    
    int8x8_t idx0 = vld1_s8(idx0_arr);
    int8x8_t idx1 = vld1_s8(idx1_arr);
    int8x8_t idx2 = vld1_s8(idx2_arr);
    int8x8_t idx3 = vld1_s8(idx3_arr);
    
    for (u32 m = 0; m < M; m++) {
        const int8_t* A_row = A + m * K;
        int32_t* C_row = C + m * N;
        
        for (u32 n = 0; n < N; n++) {
            const u8* B_col = B_ternary + n * (K / 4);
            
            int32x4_t acc = vdupq_n_s32(0);
            
            for (u32 k = 0; k < K32; k += 32) {
                /* Load 8 bytes of packed weights (32 ternary values) */
                uint8x8_t packed = vld1_u8(B_col + k / 4);
                
                /* Load 32 activations as 4 x int8x8 */
                int8x8_t a0 = vld1_s8(A_row + k + 0);
                int8x8_t a1 = vld1_s8(A_row + k + 8);
                int8x8_t a2 = vld1_s8(A_row + k + 16);
                int8x8_t a3 = vld1_s8(A_row + k + 24);
                
                /* Extract 2-bit fields */
                uint8x8_t bits0 = vand_u8(packed, mask_03);
                uint8x8_t bits1 = vand_u8(vshr_n_u8(packed, 2), mask_03);
                uint8x8_t bits2 = vand_u8(vshr_n_u8(packed, 4), mask_03);
                uint8x8_t bits3 = vshr_n_u8(packed, 6);
                
                /* Create +1/-1 masks */
                uint8x8_t plus0  = vceq_u8(bits0, val_01);
                uint8x8_t minus0 = vceq_u8(bits0, val_02);
                uint8x8_t plus1  = vceq_u8(bits1, val_01);
                uint8x8_t minus1 = vceq_u8(bits1, val_02);
                uint8x8_t plus2  = vceq_u8(bits2, val_01);
                uint8x8_t minus2 = vceq_u8(bits2, val_02);
                uint8x8_t plus3  = vceq_u8(bits3, val_01);
                uint8x8_t minus3 = vceq_u8(bits3, val_02);
                
                /* Combine activations into table and gather */
                int8x8x4_t a_tbl = {{a0, a1, a2, a3}};
                
                int8x8_t ag0 = vtbl4_s8(a_tbl, idx0);
                int8x8_t ag1 = vtbl4_s8(a_tbl, idx1);
                int8x8_t ag2 = vtbl4_s8(a_tbl, idx2);
                int8x8_t ag3 = vtbl4_s8(a_tbl, idx3);
                
                /* Apply masks: select activations for +1 and -1 weights */
                int8x8_t p0 = vand_s8(ag0, vreinterpret_s8_u8(plus0));
                int8x8_t m0 = vand_s8(ag0, vreinterpret_s8_u8(minus0));
                int8x8_t p1 = vand_s8(ag1, vreinterpret_s8_u8(plus1));
                int8x8_t m1 = vand_s8(ag1, vreinterpret_s8_u8(minus1));
                int8x8_t p2 = vand_s8(ag2, vreinterpret_s8_u8(plus2));
                int8x8_t m2 = vand_s8(ag2, vreinterpret_s8_u8(minus2));
                int8x8_t p3 = vand_s8(ag3, vreinterpret_s8_u8(plus3));
                int8x8_t m3 = vand_s8(ag3, vreinterpret_s8_u8(minus3));
                
                /* Sum across groups using pairwise add */
                int16x4_t sum_plus = vadd_s16(vadd_s16(vpaddl_s8(p0), vpaddl_s8(p1)),
                                              vadd_s16(vpaddl_s8(p2), vpaddl_s8(p3)));
                int16x4_t sum_minus = vadd_s16(vadd_s16(vpaddl_s8(m0), vpaddl_s8(m1)),
                                               vadd_s16(vpaddl_s8(m2), vpaddl_s8(m3)));
                
                /* Widen to int32 and accumulate */
                acc = vaddw_s16(acc, sum_plus);
                acc = vsubw_s16(acc, sum_minus);
            }
            
            /* Horizontal sum */
            int32_t result = vaddvq_s32(acc);
            
            /* Handle remainder */
            for (u32 k = K32; k < K; k++) {
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
 * Optimized version processing 4 output columns in parallel
 * Reuses activation loads across columns for better cache utilization
 */
void hs_ml_gemm_ternary_neon_4col(int32_t* C,
                                   const int8_t* A,
                                   const u8* B_ternary,
                                   u32 M, u32 N, u32 K) {
    
    memset(C, 0, M * N * sizeof(int32_t));
    
    const u32 K32 = K & ~31u;
    const u32 N4 = N & ~3u;
    
    const uint8x8_t mask_03 = vdup_n_u8(0x03);
    const uint8x8_t val_01  = vdup_n_u8(0x01);
    const uint8x8_t val_02  = vdup_n_u8(0x02);
    
    static const int8_t idx0_arr[8] = {0, 4, 8, 12, 16, 20, 24, 28}; 
    static const int8_t idx1_arr[8] = {1, 5, 9, 13, 17, 21, 25, 29};
    static const int8_t idx2_arr[8] = {2, 6, 10, 14, 18, 22, 26, 30};
    static const int8_t idx3_arr[8] = {3, 7, 11, 15, 19, 23, 27, 31};
    
    int8x8_t idx0 = vld1_s8(idx0_arr);
    int8x8_t idx1 = vld1_s8(idx1_arr);
    int8x8_t idx2 = vld1_s8(idx2_arr);
    int8x8_t idx3 = vld1_s8(idx3_arr);
    
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
            
            for (u32 k = 0; k < K32; k += 32) {
                /* Load activations once, reuse for 4 columns */
                int8x8_t a0 = vld1_s8(A_row + k + 0);
                int8x8_t a1 = vld1_s8(A_row + k + 8);
                int8x8_t a2 = vld1_s8(A_row + k + 16);
                int8x8_t a3 = vld1_s8(A_row + k + 24);
                
                int8x8x4_t a_tbl = {{a0, a1, a2, a3}};
                
                /* Gather activations */
                int8x8_t ag0 = vtbl4_s8(a_tbl, idx0);
                int8x8_t ag1 = vtbl4_s8(a_tbl, idx1);
                int8x8_t ag2 = vtbl4_s8(a_tbl, idx2);
                int8x8_t ag3 = vtbl4_s8(a_tbl, idx3);
                
                /* Macro for processing each column */
                #define PROCESS_COL(colptr, accvar) do {                     uint8x8_t packed = vld1_u8(colptr + k / 4);                     uint8x8_t bits0 = vand_u8(packed, mask_03);                     uint8x8_t bits1 = vand_u8(vshr_n_u8(packed, 2), mask_03);                     uint8x8_t bits2 = vand_u8(vshr_n_u8(packed, 4), mask_03);                     uint8x8_t bits3 = vshr_n_u8(packed, 6);                                         uint8x8_t plus0  = vceq_u8(bits0, val_01);                     uint8x8_t minus0 = vceq_u8(bits0, val_02);                     uint8x8_t plus1  = vceq_u8(bits1, val_01);                     uint8x8_t minus1 = vceq_u8(bits1, val_02);                     uint8x8_t plus2  = vceq_u8(bits2, val_01);                     uint8x8_t minus2 = vceq_u8(bits2, val_02);                     uint8x8_t plus3  = vceq_u8(bits3, val_01);                     uint8x8_t minus3 = vceq_u8(bits3, val_02);                                         int8x8_t p0 = vand_s8(ag0, vreinterpret_s8_u8(plus0));                     int8x8_t _m0 = vand_s8(ag0, vreinterpret_s8_u8(minus0));                     int8x8_t p1 = vand_s8(ag1, vreinterpret_s8_u8(plus1));                     int8x8_t _m1 = vand_s8(ag1, vreinterpret_s8_u8(minus1));                     int8x8_t p2 = vand_s8(ag2, vreinterpret_s8_u8(plus2));                     int8x8_t _m2 = vand_s8(ag2, vreinterpret_s8_u8(minus2));                     int8x8_t p3 = vand_s8(ag3, vreinterpret_s8_u8(plus3));                     int8x8_t _m3 = vand_s8(ag3, vreinterpret_s8_u8(minus3));                                         int16x4_t sum_plus = vadd_s16(vadd_s16(vpaddl_s8(p0), vpaddl_s8(p1)),                                                   vadd_s16(vpaddl_s8(p2), vpaddl_s8(p3)));                     int16x4_t sum_minus = vadd_s16(vadd_s16(vpaddl_s8(_m0), vpaddl_s8(_m1)),                                                    vadd_s16(vpaddl_s8(_m2), vpaddl_s8(_m3)));                                         accvar = vaddw_s16(accvar, sum_plus);                     accvar = vsubw_s16(accvar, sum_minus);                 } while(0)
                
                PROCESS_COL(B_col0, acc0);
                PROCESS_COL(B_col1, acc1);
                PROCESS_COL(B_col2, acc2);
                PROCESS_COL(B_col3, acc3);
                
                #undef PROCESS_COL
            }
            
            /* Store results */
            C_row[n + 0] = vaddvq_s32(acc0);
            C_row[n + 1] = vaddvq_s32(acc1);
            C_row[n + 2] = vaddvq_s32(acc2);
            C_row[n + 3] = vaddvq_s32(acc3);
            
            /* Handle remainder K */
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
        
        /* Handle remainder N (less than 4 columns) - use single-column version */
        for (u32 n = N4; n < N; n++) {
            const u8* B_col = B_ternary + n * (K / 4);
            int32x4_t acc = vdupq_n_s32(0);
            
            for (u32 k = 0; k < K32; k += 32) {
                uint8x8_t packed = vld1_u8(B_col + k / 4);
                
                int8x8_t a0 = vld1_s8(A_row + k + 0);
                int8x8_t a1 = vld1_s8(A_row + k + 8);
                int8x8_t a2 = vld1_s8(A_row + k + 16);
                int8x8_t a3 = vld1_s8(A_row + k + 24);
                
                uint8x8_t bits0 = vand_u8(packed, mask_03);
                uint8x8_t bits1 = vand_u8(vshr_n_u8(packed, 2), mask_03);
                uint8x8_t bits2 = vand_u8(vshr_n_u8(packed, 4), mask_03);
                uint8x8_t bits3 = vshr_n_u8(packed, 6);
                
                uint8x8_t plus0  = vceq_u8(bits0, val_01);
                uint8x8_t minus0 = vceq_u8(bits0, val_02);
                uint8x8_t plus1  = vceq_u8(bits1, val_01);
                uint8x8_t minus1 = vceq_u8(bits1, val_02);
                uint8x8_t plus2  = vceq_u8(bits2, val_01);
                uint8x8_t minus2 = vceq_u8(bits2, val_02);
                uint8x8_t plus3  = vceq_u8(bits3, val_01);
                uint8x8_t minus3 = vceq_u8(bits3, val_02);
                
                int8x8x4_t a_tbl = {{a0, a1, a2, a3}};
                int8x8_t ag0 = vtbl4_s8(a_tbl, idx0);
                int8x8_t ag1 = vtbl4_s8(a_tbl, idx1);
                int8x8_t ag2 = vtbl4_s8(a_tbl, idx2);
                int8x8_t ag3 = vtbl4_s8(a_tbl, idx3);
                
                int8x8_t p0 = vand_s8(ag0, vreinterpret_s8_u8(plus0));
                int8x8_t m0 = vand_s8(ag0, vreinterpret_s8_u8(minus0));
                int8x8_t p1 = vand_s8(ag1, vreinterpret_s8_u8(plus1));
                int8x8_t m1 = vand_s8(ag1, vreinterpret_s8_u8(minus1));
                int8x8_t p2 = vand_s8(ag2, vreinterpret_s8_u8(plus2));
                int8x8_t m2 = vand_s8(ag2, vreinterpret_s8_u8(minus2));
                int8x8_t p3 = vand_s8(ag3, vreinterpret_s8_u8(plus3));
                int8x8_t m3 = vand_s8(ag3, vreinterpret_s8_u8(minus3));
                
                int16x4_t sum_plus = vadd_s16(vadd_s16(vpaddl_s8(p0), vpaddl_s8(p1)),
                                              vadd_s16(vpaddl_s8(p2), vpaddl_s8(p3)));
                int16x4_t sum_minus = vadd_s16(vadd_s16(vpaddl_s8(m0), vpaddl_s8(m1)),
                                               vadd_s16(vpaddl_s8(m2), vpaddl_s8(m3)));
                
                acc = vaddw_s16(acc, sum_plus);
                acc = vsubw_s16(acc, sum_minus);
            }
            
            int32_t result = vaddvq_s32(acc);
            
            for (u32 k = K32; k < K; k++) {
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

