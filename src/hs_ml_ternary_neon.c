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

/*============================================================================
 * NEON batch decode: process 4 bytes (16 weights) per call.
 *
 * Input:  4 weight bytes from wrow
 * Output: 4 float32x4 mask vectors for pos, 4 for neg
 *
 * Decodes all 16 codes using integer NEON:
 *   1. Splat each byte to a uint32x4 (replicate across lanes)
 *   2. Shift by [0,2,4,6] to align each code to bits[1:0]
 *   3. Mask with 0x3 to isolate the 2-bit code
 *   4. Compare: code==2 for +1 mask, code==0 for -1 mask
 *============================================================================*/

/* Shift constants: extract codes at bit positions 0,2,4,6 */
static const int32_t SHIFTS[4] = {0, 2, 4, 6};

static inline void decode_4bytes_masks(
    const uint8_t *w,
    uint32x4_t mp[4],   /* 4 pos masks, one per byte */
    uint32x4_t mn[4])   /* 4 neg masks, one per byte */
{
    const uint32x4_t vshift = vld1q_u32((const uint32_t *)SHIFTS);
    const uint32x4_t vmask  = vdupq_n_u32(3);
    const uint32x4_t vtwo   = vdupq_n_u32(2);
    const uint32x4_t vzero  = vdupq_n_u32(0);

    for (int i = 0; i < 4; i++) {
        /* Splat byte to all 4 lanes, shift by [0,2,4,6], mask */
        uint32x4_t codes = vandq_u32(
            vshlq_u32(vdupq_n_u32(w[i]), vnegq_s32(vreinterpretq_s32_u32(vshift))),
            vmask);
        mp[i] = vceqq_u32(codes, vtwo);   /* +1 mask */
        mn[i] = vceqq_u32(codes, vzero);  /* -1 mask */
    }
}

void hs_ml_ternary_neon_proj(float *out, const float *in,
                              const uint8_t *W, uint32_t N, uint32_t K) {
    const uint32_t row_bytes = K / 4;

    for (uint32_t n = 0; n < N; n++) {
        const uint8_t *wrow = W + n * row_bytes;

        /* Use 4 pos/neg accumulator pairs to reduce dependency chains */
        float32x4_t pos0 = vdupq_n_f32(0.0f), neg0 = vdupq_n_f32(0.0f);
        float32x4_t pos1 = vdupq_n_f32(0.0f), neg1 = vdupq_n_f32(0.0f);
        float32x4_t pos2 = vdupq_n_f32(0.0f), neg2 = vdupq_n_f32(0.0f);
        float32x4_t pos3 = vdupq_n_f32(0.0f), neg3 = vdupq_n_f32(0.0f);

        /* Main loop: 16 bytes (64 weights) per iteration */
        uint32_t b16 = row_bytes & ~15u;
        for (uint32_t bi = 0; bi < b16; bi += 16) {
            __builtin_prefetch(wrow + bi + 64, 0, 3);
            __builtin_prefetch(in + (bi + 16) * 4, 0, 3);

            /* Decode 4 groups of 4 bytes each */
            for (uint32_t g = 0; g < 4; g++) {
                uint32x4_t mp[4], mn[4];
                decode_4bytes_masks(wrow + bi + g * 4, mp, mn);

                uint32_t k_base = (bi + g * 4) * 4;

                /* Load 4 × float32x4 activations and route */
                float32x4_t a0 = vld1q_f32(in + k_base);
                float32x4_t a1 = vld1q_f32(in + k_base + 4);
                float32x4_t a2 = vld1q_f32(in + k_base + 8);
                float32x4_t a3 = vld1q_f32(in + k_base + 12);

                /* Branchless masked accumulate */
                pos0 = vaddq_f32(pos0, vreinterpretq_f32_u32(
                    vandq_u32(vreinterpretq_u32_f32(a0), mp[0])));
                neg0 = vaddq_f32(neg0, vreinterpretq_f32_u32(
                    vandq_u32(vreinterpretq_u32_f32(a0), mn[0])));

                pos1 = vaddq_f32(pos1, vreinterpretq_f32_u32(
                    vandq_u32(vreinterpretq_u32_f32(a1), mp[1])));
                neg1 = vaddq_f32(neg1, vreinterpretq_f32_u32(
                    vandq_u32(vreinterpretq_u32_f32(a1), mn[1])));

                pos2 = vaddq_f32(pos2, vreinterpretq_f32_u32(
                    vandq_u32(vreinterpretq_u32_f32(a2), mp[2])));
                neg2 = vaddq_f32(neg2, vreinterpretq_f32_u32(
                    vandq_u32(vreinterpretq_u32_f32(a2), mn[2])));

                pos3 = vaddq_f32(pos3, vreinterpretq_f32_u32(
                    vandq_u32(vreinterpretq_u32_f32(a3), mp[3])));
                neg3 = vaddq_f32(neg3, vreinterpretq_f32_u32(
                    vandq_u32(vreinterpretq_u32_f32(a3), mn[3])));
            }
        }

        /* Remaining bytes (< 16) */
        for (uint32_t bi = b16; bi < row_bytes; bi++) {
            uint8_t b = wrow[bi];
            uint32_t k_base = bi * 4;
            uint32_t c0 = (b >> 0) & 3, c1 = (b >> 2) & 3;
            uint32_t c2 = (b >> 4) & 3, c3 = (b >> 6) & 3;
            uint32x4_t codes = {c0, c1, c2, c3};
            uint32x4_t mp = vceqq_u32(codes, vdupq_n_u32(2));
            uint32x4_t mn = vceqq_u32(codes, vdupq_n_u32(0));
            float32x4_t act = vld1q_f32(in + k_base);
            pos0 = vaddq_f32(pos0, vreinterpretq_f32_u32(
                vandq_u32(vreinterpretq_u32_f32(act), mp)));
            neg0 = vaddq_f32(neg0, vreinterpretq_f32_u32(
                vandq_u32(vreinterpretq_u32_f32(act), mn)));
        }

        /* Reduce: merge 4 accumulator pairs, then horizontal sum */
        float32x4_t pos_sum = vaddq_f32(vaddq_f32(pos0, pos1),
                                         vaddq_f32(pos2, pos3));
        float32x4_t neg_sum = vaddq_f32(vaddq_f32(neg0, neg1),
                                         vaddq_f32(neg2, neg3));
        out[n] = vaddvq_f32(vsubq_f32(pos_sum, neg_sum));
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
