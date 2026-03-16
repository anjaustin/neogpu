/*
 * NeoGPU ML — MTFP Ternary Projection APU (NEON, Cortex-A72)
 *
 * Ternary neural network projection without quantization or floating-point
 * multiply on the weight path. Ternary weights {-1, 0, +1} ROUTE float32
 * activations into positive/negative accumulators via masked addition.
 *
 * Core idea: for weight code c ∈ {0=-1, 1=0, 2=+1}:
 *   if c == 2: pos_acc += activation
 *   if c == 0: neg_acc += activation
 *   if c == 1 or 3: skip
 *   output = pos_acc - neg_acc
 *
 * No multiply. No quantize. No sign vector. Just conditional add.
 *
 * NEON strategy (16 bytes = 64 weights per iteration):
 *   1. Load 16 weight bytes as uint8x16
 *   2. Extract 4 groups of 16 2-bit codes via shift+mask
 *   3. Compare codes: build masks for +1 (code==2) and -1 (code==0)
 *   4. Use masks to selectively load and accumulate activations
 *   5. After all bytes: output = pos - neg
 *
 * On Cortex-A72 (ARMv8.0, no DOTPROD):
 *   - Each 16-byte block processes 64 weights
 *   - Uses vceqq/vandq for branchless masking
 *   - Float additions only (no FMA needed for ternary routing)
 *   - Target: 2-4 GOPS (10-20x current f32 kernel)
 *
 * Weight format: I2_S sequential 2-bit packing.
 *   Byte bits [1:0]=w0, [3:2]=w1, [5:4]=w2, [7:6]=w3
 *   Code: 0=-1, 1=0, 2=+1, 3=0
 */

#include <stdint.h>
#include <string.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/*============================================================================
 * NEON Ternary Routing Kernel
 *
 * Process 16 bytes (64 weights) per outer iteration.
 * Within each byte, 4 weights are packed sequentially.
 * We process 4 bytes at a time (16 weights) using NEON f32 lanes.
 *
 * For 4 bytes → 16 weights → 4 float32x4 activation loads:
 *   byte[0] → act[0..3],  byte[1] → act[4..7],
 *   byte[2] → act[8..11], byte[3] → act[12..15]
 *
 * Decode: for each byte, extract 4 codes. Build uint32 masks:
 *   mask_pos = (code == 2) → all-ones for +1 weights
 *   mask_neg = (code == 0) → all-ones for -1 weights
 * Apply: pos_acc += act & mask_pos,  neg_acc += act & mask_neg
 * This is branchless — no conditional, just bitwise AND + float add.
 *============================================================================*/

#ifdef __ARM_NEON

/* Decode 4 sequential codes from one byte into 4 uint32 mask values.
 * Returns mask_pos (1 where code==2) and mask_neg (1 where code==0)
 * as uint32x4 (all-ones or all-zeros per lane). */
static inline void decode_byte_masks(uint8_t b,
                                      uint32x4_t *mask_pos,
                                      uint32x4_t *mask_neg) {
    /* Extract 4 codes */
    uint32_t c0 = (b >> 0) & 3;
    uint32_t c1 = (b >> 2) & 3;
    uint32_t c2 = (b >> 4) & 3;
    uint32_t c3 = (b >> 6) & 3;

    /* Build code vector and compare */
    uint32x4_t codes = {c0, c1, c2, c3};
    *mask_pos = vceqq_u32(codes, vdupq_n_u32(2));  /* +1 weights */
    *mask_neg = vceqq_u32(codes, vdupq_n_u32(0));  /* -1 weights */
}

void hs_ml_ternary_neon_proj(float *out, const float *in,
                              const uint8_t *W, uint32_t N, uint32_t K) {
    const uint32_t row_bytes = K / 4;

    for (uint32_t n = 0; n < N; n++) {
        const uint8_t *wrow = W + n * row_bytes;
        float32x4_t pos_acc = vdupq_n_f32(0.0f);
        float32x4_t neg_acc = vdupq_n_f32(0.0f);

        /* Process 4 bytes (16 weights) per iteration */
        uint32_t b4 = row_bytes & ~3u;
        for (uint32_t bi = 0; bi < b4; bi += 4) {
            /* Prefetch next cache line of weights */
            __builtin_prefetch(wrow + bi + 64, 0, 3);

            for (uint32_t j = 0; j < 4; j++) {
                uint8_t b = wrow[bi + j];
                uint32_t k_base = (bi + j) * 4;

                uint32x4_t mp, mn;
                decode_byte_masks(b, &mp, &mn);

                /* Load 4 activations */
                float32x4_t act = vld1q_f32(in + k_base);

                /* Masked accumulate: branchless ternary routing
                 * pos_acc += act where weight == +1
                 * neg_acc += act where weight == -1 */
                pos_acc = vaddq_f32(pos_acc,
                    vreinterpretq_f32_u32(vandq_u32(
                        vreinterpretq_u32_f32(act), mp)));
                neg_acc = vaddq_f32(neg_acc,
                    vreinterpretq_f32_u32(vandq_u32(
                        vreinterpretq_u32_f32(act), mn)));
            }
        }

        /* Handle remaining bytes */
        for (uint32_t bi = b4; bi < row_bytes; bi++) {
            uint8_t b = wrow[bi];
            uint32_t k_base = bi * 4;
            uint32x4_t mp, mn;
            decode_byte_masks(b, &mp, &mn);
            float32x4_t act = vld1q_f32(in + k_base);
            pos_acc = vaddq_f32(pos_acc,
                vreinterpretq_f32_u32(vandq_u32(
                    vreinterpretq_u32_f32(act), mp)));
            neg_acc = vaddq_f32(neg_acc,
                vreinterpretq_f32_u32(vandq_u32(
                    vreinterpretq_u32_f32(act), mn)));
        }

        /* Final reduction: output = sum(pos) - sum(neg) */
        float32x4_t diff = vsubq_f32(pos_acc, neg_acc);
        out[n] = vaddvq_f32(diff);
    }
}

#else
/* Scalar fallback */
void hs_ml_ternary_neon_proj(float *out, const float *in,
                              const uint8_t *W, uint32_t N, uint32_t K) {
    uint32_t row_bytes = K / 4;
    for (uint32_t n = 0; n < N; n++) {
        const uint8_t *wrow = W + n * row_bytes;
        float pos = 0, neg = 0;
        for (uint32_t bi = 0; bi < row_bytes; bi++) {
            uint8_t b = wrow[bi];
            uint32_t k = bi * 4;
            for (int s = 0; s < 8; s += 2) {
                uint8_t c = (b >> s) & 3;
                if (c == 2) pos += in[k + s/2];
                else if (c == 0) neg += in[k + s/2];
            }
        }
        out[n] = pos - neg;
    }
}
#endif

/*============================================================================
 * Public API — drop-in replacement for hs_ml_ternary_f32_proj
 *============================================================================*/

void hs_ml_ternary_f32_proj(float *out, const float *in,
                             const uint8_t *W, uint32_t N, uint32_t K) {
    hs_ml_ternary_neon_proj(out, in, W, N, K);
}
