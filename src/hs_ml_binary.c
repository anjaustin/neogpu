/*
 * NeoGPU ML - Binary Neural Network Operations
 * 
 * Pure binary: weights and activations are {-1, +1}
 * Operations become bit logic: XNOR + popcount
 * 
 * Encoding: bit 0 = -1, bit 1 = +1
 * 
 * dot(a, w) = count(matches) - count(mismatches)
 *           = 2 * popcount(a XNOR w) - K
 *           
 * This achieves ~30 GOPS on Pi4, limited by memory bandwidth.
 * 
 * See: docs/theory/GEOMETRIC_INFERENCE.md
 * See: include/hs_ml_routing.h
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/*
 * Binary dot product: 128 bits × 128 bits → int32
 * 
 * Inputs are packed bits: 16 bytes = 128 binary values
 * Uses {-1, +1} encoding where bit 0 = -1, bit 1 = +1
 * 
 * Result = count(matches) - count(mismatches)
 *        = 2 * count(matches) - 128
 */
#ifdef __ARM_NEON
static inline int32_t binary_dot_128(uint8x16_t a, uint8x16_t w) {
    /* XNOR: bits match where result is 1 */
    uint8x16_t xnor = vmvnq_u8(veorq_u8(a, w));
    
    /* Popcount per byte */
    uint8x16_t counts = vcntq_u8(xnor);
    
    /* Horizontal sum */
    uint16x8_t sum16 = vpaddlq_u8(counts);
    uint32x4_t sum32 = vpaddlq_u16(sum16);
    uint64x2_t sum64 = vpaddlq_u32(sum32);
    
    int32_t matches = (int32_t)(vgetq_lane_u64(sum64, 0) + vgetq_lane_u64(sum64, 1));
    
    /* Convert to signed result: matches - mismatches = 2*matches - 128 */
    return 2 * matches - 128;
}
#endif

/*
 * hs_ml_route_binary - Basic binary routing
 * 
 * C[M,N] = A[M,K] × B[K,N] using binary XNOR + popcount
 * 
 * A: binary activations, packed bits [M, K/8] bytes
 * B: binary weights, packed bits [N, K/8] bytes (transposed for efficiency)
 * C: int32 output [M, N]
 * K: must be multiple of 128
 */
void hs_ml_route_binary(int32_t* C,
                        const uint8_t* A,
                        const uint8_t* B,
                        uint32_t M, uint32_t N, uint32_t K) {
#ifdef __ARM_NEON
    const uint32_t K_bytes = K / 8;
    const uint32_t K128 = K / 128;
    
    for (uint32_t m = 0; m < M; m++) {
        const uint8_t* A_row = A + m * K_bytes;
        int32_t* C_row = C + m * N;
        
        for (uint32_t n = 0; n < N; n++) {
            const uint8_t* B_col = B + n * K_bytes;
            int32_t sum = 0;
            
            for (uint32_t k = 0; k < K128; k++) {
                uint8x16_t a = vld1q_u8(A_row + k * 16);
                uint8x16_t w = vld1q_u8(B_col + k * 16);
                sum += binary_dot_128(a, w);
            }
            
            C_row[n] = sum;
        }
    }
#else
    /* Scalar fallback */
    const uint32_t K_bytes = K / 8;
    
    for (uint32_t m = 0; m < M; m++) {
        const uint8_t* A_row = A + m * K_bytes;
        int32_t* C_row = C + m * N;
        
        for (uint32_t n = 0; n < N; n++) {
            const uint8_t* B_col = B + n * K_bytes;
            int32_t sum = 0;
            
            for (uint32_t kb = 0; kb < K_bytes; kb++) {
                uint8_t xnor = ~(A_row[kb] ^ B_col[kb]);
                /* Count set bits */
                int matches = __builtin_popcount(xnor);
                sum += 2 * matches - 8;
            }
            
            C_row[n] = sum;
        }
    }
#endif
}

/*
 * hs_ml_route_binary_opt - Optimized binary routing (4 columns at once)
 * 
 * Processes 4 output columns simultaneously to amortize activation load
 * and improve instruction-level parallelism.
 */
void hs_ml_route_binary_opt(int32_t* C,
                            const uint8_t* A,
                            const uint8_t* B,
                            uint32_t M, uint32_t N, uint32_t K) {
#ifdef __ARM_NEON
    const uint32_t K_bytes = K / 8;
    const uint32_t K128 = K / 128;
    const uint32_t N4 = N & ~3u;
    
    for (uint32_t m = 0; m < M; m++) {
        const uint8_t* A_row = A + m * K_bytes;
        int32_t* C_row = C + m * N;
        
        /* Process 4 columns at a time */
        for (uint32_t n = 0; n < N4; n += 4) {
            const uint8_t* B0 = B + (n + 0) * K_bytes;
            const uint8_t* B1 = B + (n + 1) * K_bytes;
            const uint8_t* B2 = B + (n + 2) * K_bytes;
            const uint8_t* B3 = B + (n + 3) * K_bytes;
            
            int32x4_t sum = vdupq_n_s32(0);
            
            for (uint32_t k = 0; k < K128; k++) {
                uint8x16_t a = vld1q_u8(A_row + k * 16);
                
                uint8x16_t w0 = vld1q_u8(B0 + k * 16);
                uint8x16_t w1 = vld1q_u8(B1 + k * 16);
                uint8x16_t w2 = vld1q_u8(B2 + k * 16);
                uint8x16_t w3 = vld1q_u8(B3 + k * 16);
                
                /* XNOR */
                uint8x16_t x0 = vmvnq_u8(veorq_u8(a, w0));
                uint8x16_t x1 = vmvnq_u8(veorq_u8(a, w1));
                uint8x16_t x2 = vmvnq_u8(veorq_u8(a, w2));
                uint8x16_t x3 = vmvnq_u8(veorq_u8(a, w3));
                
                /* Popcount */
                uint8x16_t c0 = vcntq_u8(x0);
                uint8x16_t c1 = vcntq_u8(x1);
                uint8x16_t c2 = vcntq_u8(x2);
                uint8x16_t c3 = vcntq_u8(x3);
                
                /* Horizontal sums */
                uint16x8_t s0 = vpaddlq_u8(c0);
                uint16x8_t s1 = vpaddlq_u8(c1);
                uint16x8_t s2 = vpaddlq_u8(c2);
                uint16x8_t s3 = vpaddlq_u8(c3);
                
                uint32x4_t ss0 = vpaddlq_u16(s0);
                uint32x4_t ss1 = vpaddlq_u16(s1);
                uint32x4_t ss2 = vpaddlq_u16(s2);
                uint32x4_t ss3 = vpaddlq_u16(s3);
                
                /* Sum all lanes */
                int32_t m0 = (int32_t)vaddvq_u32(ss0);
                int32_t m1 = (int32_t)vaddvq_u32(ss1);
                int32_t m2 = (int32_t)vaddvq_u32(ss2);
                int32_t m3 = (int32_t)vaddvq_u32(ss3);
                
                int32x4_t matches = {m0, m1, m2, m3};
                sum = vaddq_s32(sum, matches);
            }
            
            /* Convert: 2*matches - K */
            int32x4_t result = vsubq_s32(
                vshlq_n_s32(sum, 1),
                vdupq_n_s32((int32_t)K)
            );
            
            vst1q_s32(C_row + n, result);
        }
        
        /* Remainder columns */
        for (uint32_t n = N4; n < N; n++) {
            const uint8_t* B_col = B + n * K_bytes;
            int32_t sum = 0;
            
            for (uint32_t k = 0; k < K128; k++) {
                uint8x16_t a = vld1q_u8(A_row + k * 16);
                uint8x16_t w = vld1q_u8(B_col + k * 16);
                sum += binary_dot_128(a, w);
            }
            
            C_row[n] = sum;
        }
    }
#else
    /* Scalar fallback - use basic version */
    hs_ml_route_binary(C, A, B, M, N, K);
#endif
}

/*
 * hs_ml_route_ternary_x_ternary - Ternary activations × Ternary weights
 * 
 * Uses bitplane representation for maximum throughput:
 *   active_plane: 1 if value is non-zero
 *   sign_plane: 1 if value is negative (only meaningful where active)
 * 
 * Result computation:
 *   result_active = a_active AND w_active
 *   result_sign = a_sign XOR w_sign  
 *   output = count(result_active AND NOT result_sign)
 *          - count(result_active AND result_sign)
 *          = count(result_active) - 2*count(result_active AND result_sign)
 * 
 * Achieves ~15 GOPS on Pi4.
 */
void hs_ml_route_ternary_x_ternary(int32_t* C,
                                   const uint8_t* A_active,
                                   const uint8_t* A_sign,
                                   const uint8_t* B_active,
                                   const uint8_t* B_sign,
                                   uint32_t M, uint32_t N, uint32_t K) {
#ifdef __ARM_NEON
    const uint32_t K_bytes = K / 8;
    const uint32_t K128 = K / 128;
    
    for (uint32_t m = 0; m < M; m++) {
        const uint8_t* Aa_row = A_active + m * K_bytes;
        const uint8_t* As_row = A_sign + m * K_bytes;
        int32_t* C_row = C + m * N;
        
        for (uint32_t n = 0; n < N; n++) {
            const uint8_t* Ba_col = B_active + n * K_bytes;
            const uint8_t* Bs_col = B_sign + n * K_bytes;
            
            int32_t sum_active = 0;
            int32_t sum_negative = 0;
            
            for (uint32_t k = 0; k < K128; k++) {
                uint8x16_t a_act = vld1q_u8(Aa_row + k * 16);
                uint8x16_t a_sgn = vld1q_u8(As_row + k * 16);
                uint8x16_t b_act = vld1q_u8(Ba_col + k * 16);
                uint8x16_t b_sgn = vld1q_u8(Bs_col + k * 16);
                
                /* Result is active where both inputs are active */
                uint8x16_t r_act = vandq_u8(a_act, b_act);
                
                /* Result is negative where signs differ */
                uint8x16_t r_neg = vandq_u8(r_act, veorq_u8(a_sgn, b_sgn));
                
                /* Popcount */
                uint8x16_t cnt_act = vcntq_u8(r_act);
                uint8x16_t cnt_neg = vcntq_u8(r_neg);
                
                /* Horizontal sum */
                sum_active += vaddvq_u8(cnt_act);
                sum_negative += vaddvq_u8(cnt_neg);
            }
            
            /* result = active - 2*negative */
            C_row[n] = sum_active - 2 * sum_negative;
        }
    }
#else
    /* Scalar fallback */
    const uint32_t K_bytes = K / 8;
    
    for (uint32_t m = 0; m < M; m++) {
        const uint8_t* Aa_row = A_active + m * K_bytes;
        const uint8_t* As_row = A_sign + m * K_bytes;
        int32_t* C_row = C + m * N;
        
        for (uint32_t n = 0; n < N; n++) {
            const uint8_t* Ba_col = B_active + n * K_bytes;
            const uint8_t* Bs_col = B_sign + n * K_bytes;
            
            int32_t sum = 0;
            
            for (uint32_t kb = 0; kb < K_bytes; kb++) {
                uint8_t r_act = Aa_row[kb] & Ba_col[kb];
                uint8_t r_neg = r_act & (As_row[kb] ^ Bs_col[kb]);
                
                int active = __builtin_popcount(r_act);
                int negative = __builtin_popcount(r_neg);
                
                sum += active - 2 * negative;
            }
            
            C_row[n] = sum;
        }
    }
#endif
}
