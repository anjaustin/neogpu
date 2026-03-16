/*
 * NeoGPU ML - Pure Float32 Ternary Routing Kernel
 *
 * Ternary weights route float32 activations directly.
 * No int8 quantization. No dequantization. No scale factors.
 *
 * weight = {-1, 0, +1}
 * output[n] = sum(act[k] where w[n,k]=+1) - sum(act[k] where w[n,k]=-1)
 *
 * BitNet I2_S format (GGUF type 36):
 *   Simple sequential 2-bit packing, 4 weights per byte, low bits first:
 *     byte bits [1:0] = weight 0
 *     byte bits [3:2] = weight 1
 *     byte bits [5:4] = weight 2
 *     byte bits [7:6] = weight 3
 *   Code mapping: 0 = -1, 1 = 0, 2 = +1, 3 = 0 (zero, for FFN tensors)
 *
 * This was confirmed by reading the Microsoft BitNet converter:
 *   data_torch = (data_torch.float() - 1)  =>  {0:-1, 1:0, 2:+1}
 *   shift = [0, 2, 4, 6]  =>  sequential low-bits-first
 *
 * NEON strategy:
 *   - Process 4 activations at a time per byte
 *   - Build sign vector from decoded codes
 *   - Accumulate via vfmaq_f32
 */

#include <stdint.h>
#include <string.h>
#include <math.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/*============================================================================
 * Scalar reference: float32 activations × I2_S ternary weights
 *
 * Sequential 2-bit packing, 4 weights per byte, low bits first.
 * Code mapping: 0=-1, 1=0, 2=+1, 3=0
 *============================================================================*/

static void i2s_f32_proj_scalar(float *out, const float *in,
                                 const uint8_t *W, uint32_t N, uint32_t K) {
    uint32_t row_bytes = K / 4;

    for (uint32_t n = 0; n < N; n++) {
        const uint8_t *wrow = W + n * row_bytes;
        float acc = 0.0f;

        for (uint32_t byte_idx = 0; byte_idx < row_bytes; byte_idx++) {
            uint8_t b = wrow[byte_idx];
            uint32_t k_base = byte_idx * 4;

            /* 4 weights per byte, sequential: bits[1:0], [3:2], [5:4], [7:6] */
            uint8_t c0 = (b >> 0) & 3;
            uint8_t c1 = (b >> 2) & 3;
            uint8_t c2 = (b >> 4) & 3;
            uint8_t c3 = (b >> 6) & 3;

            /* code: 0=-1, 1=0, 2=+1, 3=0 */
            if (c0 == 2)      acc += in[k_base + 0];
            else if (c0 == 0) acc -= in[k_base + 0];
            if (c1 == 2)      acc += in[k_base + 1];
            else if (c1 == 0) acc -= in[k_base + 1];
            if (c2 == 2)      acc += in[k_base + 2];
            else if (c2 == 0) acc -= in[k_base + 2];
            if (c3 == 2)      acc += in[k_base + 3];
            else if (c3 == 0) acc -= in[k_base + 3];
        }
        out[n] = acc;
    }
}

/*============================================================================
 * NEON: float32 activations × I2_S ternary weights (sequential packing)
 *
 * Each byte has 4 weights packed sequentially.
 * Process 4 bytes at a time → 16 weights → 4 float32x4 loads.
 * For each byte, decode 4 codes and build a sign vector.
 *============================================================================*/

#ifdef __ARM_NEON
static void i2s_f32_proj_neon(float *out, const float *in,
                               const uint8_t *W, uint32_t N, uint32_t K) {
    const uint32_t row_bytes = K / 4;

    for (uint32_t n = 0; n < N; n++) {
        const uint8_t *wrow = W + n * row_bytes;
        float32x4_t vacc = vdupq_n_f32(0.0f);

        for (uint32_t byte_idx = 0; byte_idx < row_bytes; byte_idx++) {
            uint8_t b = wrow[byte_idx];
            uint32_t k_base = byte_idx * 4;

            /* Decode 4 codes from this byte (sequential: low bits first) */
            uint8_t c0 = (b >> 0) & 3;
            uint8_t c1 = (b >> 2) & 3;
            uint8_t c2 = (b >> 4) & 3;
            uint8_t c3 = (b >> 6) & 3;

            /* Build sign vector: code 0=-1, 1=0, 2=+1, 3=0 */
            float signs[4] = {
                (c0 == 2) ? 1.0f : (c0 == 0) ? -1.0f : 0.0f,
                (c1 == 2) ? 1.0f : (c1 == 0) ? -1.0f : 0.0f,
                (c2 == 2) ? 1.0f : (c2 == 0) ? -1.0f : 0.0f,
                (c3 == 2) ? 1.0f : (c3 == 0) ? -1.0f : 0.0f
            };
            float32x4_t vsign = vld1q_f32(signs);
            float32x4_t act = vld1q_f32(in + k_base);
            vacc = vfmaq_f32(vacc, act, vsign);
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
