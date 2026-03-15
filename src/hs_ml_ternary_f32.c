/*
 * NeoGPU ML - Pure Float32 Ternary Routing Kernel
 *
 * Ternary weights route float32 activations directly.
 * No int8 quantization. No dequantization. No scale factors.
 *
 * weight = {-1, 0, +1}
 * output[n] = sum(act[k] where w[n,k]=+1) - sum(act[k] where w[n,k]=-1)
 *
 * I2_S format: raw codes {0,1,2} mapped as {-1,0,+1}
 * Group layout: group_idx=j/16, group_pos=j%16, bits [7:6] first
 *
 * NEON strategy:
 *   - Load 4 float32 activations per vld1q_f32 (16 bytes)
 *   - Decode 4 weight codes from 1 byte
 *   - Use vbslq_f32 to select activations for +1 and -1
 *   - Accumulate positive and negative sums separately
 *   - Final: output = pos_sum - neg_sum
 */

#include <stdint.h>
#include <string.h>
#include <math.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#define I2S_QK 64u

/*============================================================================
 * Scalar reference: float32 activations × I2_S ternary weights
 *============================================================================*/

static void i2s_f32_proj_scalar(float *out, const float *in,
                                 const uint8_t *W, uint32_t N, uint32_t K) {
    uint32_t row_bytes = K / 4;
    uint32_t nblk = K / I2S_QK;

    for (uint32_t n = 0; n < N; n++) {
        const uint8_t *wrow = W + n * row_bytes;
        float acc = 0.0f;

        for (uint32_t bi = 0; bi < nblk; bi++) {
            const uint8_t *block = wrow + bi * (I2S_QK / 4);
            for (uint32_t j = 0; j < I2S_QK; j++) {
                uint32_t k = bi * I2S_QK + j;
                uint32_t gi = j / 16;
                uint32_t gp = j % 16;
                uint8_t raw = (block[gp] >> (6 - 2 * gi)) & 0x3;
                /* raw: 0=-1, 1=0, 2=+1 */
                if (raw == 2) acc += in[k];
                else if (raw == 0) acc -= in[k];
            }
        }
        out[n] = acc;
    }
}

/*============================================================================
 * NEON: float32 activations × I2_S ternary weights
 *
 * Per 16-byte weight block (16 bytes = 64 weights):
 *   - 4 groups of 16 weights
 *   - Each group: 16 bytes decode to 16 codes
 *   - Process 4 activations at a time (one float32x4)
 *
 * For each byte, bits [7:6] = group 0, [5:4] = group 1, [3:2] = group 2, [1:0] = group 3
 * Group g processes activations[bi*64 + g*16 + 0..15]
 *============================================================================*/

#ifdef __ARM_NEON
static void i2s_f32_proj_neon(float *out, const float *in,
                               const uint8_t *W, uint32_t N, uint32_t K) {
    const uint32_t row_bytes = K / 4;
    const uint32_t nblk = K / I2S_QK;

    for (uint32_t n = 0; n < N; n++) {
        const uint8_t *wrow = W + n * row_bytes;
        float32x4_t vacc = vdupq_n_f32(0.0f);

        for (uint32_t bi = 0; bi < nblk; bi++) {
            const uint8_t *block = wrow + bi * 16;

            /* Process 4 groups of 16 weights each */
            for (uint32_t g = 0; g < 4; g++) {
                uint32_t shift = 6 - 2 * g;
                uint32_t base_k = bi * 64 + g * 16;

                /* Process 16 activations in chunks of 4 */
                for (uint32_t p = 0; p < 16; p += 4) {
                    /* Decode 4 weight codes from 4 consecutive bytes */
                    uint8_t c0 = (block[p + 0] >> shift) & 3;
                    uint8_t c1 = (block[p + 1] >> shift) & 3;
                    uint8_t c2 = (block[p + 2] >> shift) & 3;
                    uint8_t c3 = (block[p + 3] >> shift) & 3;

                    float32x4_t act = vld1q_f32(in + base_k + p);

                    /* Build sign vector: +1 for code 2, -1 for code 0, 0 for code 1 */
                    float signs[4] = {
                        (c0 == 2) ? 1.0f : (c0 == 0) ? -1.0f : 0.0f,
                        (c1 == 2) ? 1.0f : (c1 == 0) ? -1.0f : 0.0f,
                        (c2 == 2) ? 1.0f : (c2 == 0) ? -1.0f : 0.0f,
                        (c3 == 2) ? 1.0f : (c3 == 0) ? -1.0f : 0.0f
                    };
                    float32x4_t vsign = vld1q_f32(signs);

                    vacc = vfmaq_f32(vacc, act, vsign);
                }
            }
        }
        out[n] = vaddvq_f32(vacc);
    }
}
#endif

/*============================================================================
 * Public API
 *============================================================================*/

void hs_ml_ternary_f32_proj(float *out, const float *in,
                             const uint8_t *W, uint32_t N, uint32_t K) {
#ifdef __ARM_NEON
    i2s_f32_proj_neon(out, in, W, N, K);
#else
    i2s_f32_proj_scalar(out, in, W, N, K);
#endif
}

/*============================================================================
 * Float32 RMSNorm + ternary projection fused
 *
 * Eliminates int8 quantization entirely:
 *   1. RMSNorm: out = (in / rms) * norm_weight
 *   2. Ternary projection: result = sum(normed[k] * ternary_sign[k])
 *
 * No quantize. No dequantize. No scale.
 *============================================================================*/

void hs_ml_ternary_f32_norm_proj(float *proj_out,
                                  const float *hidden, const float *norm_w,
                                  const uint8_t *W, uint32_t N, uint32_t K,
                                  float eps) {
    /* Allocate normed buffer on stack for typical sizes, heap for large */
    float stack_buf[8192];
    float *normed = (K <= 8192) ? stack_buf : (float *)__builtin_alloca(K * sizeof(float));

    /* RMSNorm */
    float ss = 0.0f;
#ifdef __ARM_NEON
    float32x4_t vss = vdupq_n_f32(0.0f);
    uint32_t k4 = K & ~3u;
    for (uint32_t i = 0; i < k4; i += 4) {
        float32x4_t v = vld1q_f32(hidden + i);
        vss = vfmaq_f32(vss, v, v);
    }
    ss = vaddvq_f32(vss);
    for (uint32_t i = k4; i < K; i++) ss += hidden[i] * hidden[i];
#else
    for (uint32_t i = 0; i < K; i++) ss += hidden[i] * hidden[i];
#endif

    float scale = 1.0f / sqrtf(ss / (float)K + eps);

#ifdef __ARM_NEON
    float32x4_t vscale = vdupq_n_f32(scale);
    for (uint32_t i = 0; i < k4; i += 4) {
        float32x4_t h = vld1q_f32(hidden + i);
        float32x4_t w = vld1q_f32(norm_w + i);
        vst1q_f32(normed + i, vmulq_f32(vmulq_f32(h, vscale), w));
    }
    for (uint32_t i = k4; i < K; i++) normed[i] = hidden[i] * scale * norm_w[i];
#else
    for (uint32_t i = 0; i < K; i++) normed[i] = hidden[i] * scale * norm_w[i];
#endif

    /* Ternary projection on normed activations */
    hs_ml_ternary_f32_proj(proj_out, normed, W, N, K);
}
