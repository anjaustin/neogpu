/*
 * NeoGPU ML - MT7 NEON-Optimized Kernel
 *
 * Cache-resident MT7 RMSNorm using ARM NEON intrinsics.
 *
 * Design:
 *   - 2187-entry decode LUT (8.5 KB) pinned in L1 data cache
 *   - Pass 1: NEON vfmaq sum-of-squares (4-wide)
 *   - Pass 2: fused unpack-decode-scale-multiply
 *     - Unpack 12 bytes → 8 MT7 codes (4 pairs of 3 bytes)
 *     - 8 LUT lookups → 8 float weights
 *     - 8 fused multiply-add: out[i] = in[i] * rms_scale * weight[i]
 *   - Prefetch MT7 bytes 2 cache lines ahead
 *
 * The LUT stays hot because:
 *   - 8.5 KB < 32 KB L1D (Cortex-A72)
 *   - Sequential access pattern on MT7 bytes
 *   - LUT indices are 0..2186 — bounded, no cache pollution
 *   - __builtin_prefetch on upcoming MT7 bytes keeps the pipeline fed
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/*============================================================================
 * Shared LUT (same as hs_ml_mt7.c but aligned for cache efficiency)
 *============================================================================*/

static float mt7_lut[2187] __attribute__((aligned(64)));
static int mt7_lut_ready = 0;

static const float POW3[9] = {
    1.0f/81, 1.0f/27, 1.0f/9, 1.0f/3, 1.0f,
    3.0f, 9.0f, 27.0f, 81.0f
};

static void mt7_ensure_lut(void) {
    if (mt7_lut_ready) return;
    for (int code = 0; code < 2187; code++) {
        int c = code;
        int t[7];
        for (int i = 0; i < 7; i++) { t[i] = c % 3; c /= 3; }
        int sign = t[0] - 1;
        int mant = t[1] + t[2]*3 + t[3]*9 + t[4]*27;
        int exp_idx = t[5] + t[6]*3;
        mt7_lut[code] = (float)sign * ((float)mant / 80.0f) * POW3[exp_idx];
    }
    mt7_lut_ready = 1;
}

/*============================================================================
 * Inline helpers
 *============================================================================*/

/* Unpack 3 bytes → two 12-bit MT7 codes */
static inline void unpack2(const uint8_t *p, uint16_t *c0, uint16_t *c1) {
    *c0 = (uint16_t)(p[0] | ((p[1] & 0x0F) << 8));
    *c1 = (uint16_t)((p[1] >> 4) | ((uint16_t)p[2] << 4));
}

/* Decode 8 MT7 codes from 12 bytes into a float32x4x2 pair */
static inline void decode8(const uint8_t *mt7, float *out) {
    uint16_t c0, c1;
    for (int i = 0; i < 4; i++) {
        unpack2(mt7 + i * 3, &c0, &c1);
        out[i * 2]     = mt7_lut[c0 < 2187 ? c0 : 0];
        out[i * 2 + 1] = mt7_lut[c1 < 2187 ? c1 : 0];
    }
}

/*============================================================================
 * mt7_rmsnorm_neon: Cache-optimized NEON RMSNorm with MT7 weights
 *
 * Two-pass design:
 *   Pass 1: NEON sum-of-squares (4-wide vfmaq, 128-bit)
 *   Pass 2: Fused decode-scale-multiply with prefetch
 *
 * The key optimization: decode MT7 in 8-element batches,
 * immediately consume the decoded weights in the same register set,
 * prefetch the next batch of MT7 bytes while processing current.
 *============================================================================*/

void mt7_rmsnorm_neon(float *out, const float *in, const uint8_t *mt7_weights,
                      float eps, uint32_t n) {
    mt7_ensure_lut();

#ifdef __ARM_NEON
    /* === Pass 1: sum of squares === */
    float32x4_t vss = vdupq_n_f32(0.0f);
    uint32_t n4 = n & ~3u;
    for (uint32_t i = 0; i < n4; i += 4) {
        float32x4_t v = vld1q_f32(in + i);
        vss = vfmaq_f32(vss, v, v);
    }
    float ss = vaddvq_f32(vss);
    for (uint32_t i = n4; i < n; i++) ss += in[i] * in[i];

    float rms_scale = 1.0f / sqrtf(ss / (float)n + eps);
    float32x4_t vscale = vdupq_n_f32(rms_scale);

    /* === Pass 2: fused decode-scale-multiply === */
    /* Process 8 elements at a time (4 pairs × 3 bytes = 12 bytes of MT7) */
    uint32_t n8 = n & ~7u;
    float wbuf[8];

    for (uint32_t i = 0; i < n8; i += 8) {
        const uint8_t *mp = mt7_weights + (i / 2) * 3;

        /* Prefetch next MT7 batch (2 cache lines ahead = 128 bytes = ~85 values) */
        __builtin_prefetch(mp + 64, 0, 3);

        /* Decode 8 MT7 weights via LUT */
        decode8(mp, wbuf);

        /* Load input, multiply by rms_scale and weight, store */
        float32x4_t in0 = vld1q_f32(in + i);
        float32x4_t in1 = vld1q_f32(in + i + 4);
        float32x4_t w0  = vld1q_f32(wbuf);
        float32x4_t w1  = vld1q_f32(wbuf + 4);

        vst1q_f32(out + i,     vmulq_f32(vmulq_f32(in0, vscale), w0));
        vst1q_f32(out + i + 4, vmulq_f32(vmulq_f32(in1, vscale), w1));
    }

    /* Remainder */
    for (uint32_t i = n8; i < n; i++) {
        uint32_t pair = i / 2;
        uint32_t slot = i % 2;
        uint16_t c0, c1;
        unpack2(mt7_weights + pair * 3, &c0, &c1);
        float w = mt7_lut[slot == 0 ? (c0 < 2187 ? c0 : 0) : (c1 < 2187 ? c1 : 0)];
        out[i] = in[i] * rms_scale * w;
    }

#else
    /* Scalar fallback */
    float ss = 0.0f;
    for (uint32_t i = 0; i < n; i++) ss += in[i] * in[i];
    float rms_scale = 1.0f / sqrtf(ss / (float)n + eps);
    uint32_t pairs = n / 2;
    for (uint32_t i = 0; i < pairs; i++) {
        uint16_t c0, c1;
        unpack2(mt7_weights + i * 3, &c0, &c1);
        out[i*2]   = in[i*2]   * rms_scale * mt7_lut[c0 < 2187 ? c0 : 0];
        out[i*2+1] = in[i*2+1] * rms_scale * mt7_lut[c1 < 2187 ? c1 : 0];
    }
    if (n & 1) {
        uint16_t c0, c1;
        unpack2(mt7_weights + pairs * 3, &c0, &c1);
        out[n-1] = in[n-1] * rms_scale * mt7_lut[c0 < 2187 ? c0 : 0];
    }
#endif
}

/*============================================================================
 * mt7_decode_neon: Cache-optimized bulk decode with prefetch
 *============================================================================*/

void mt7_decode_neon(float *out, const uint8_t *in, uint32_t n) {
    mt7_ensure_lut();

#ifdef __ARM_NEON
    uint32_t n8 = n & ~7u;
    float wbuf[8];

    for (uint32_t i = 0; i < n8; i += 8) {
        const uint8_t *mp = in + (i / 2) * 3;
        __builtin_prefetch(mp + 64, 0, 3);
        decode8(mp, wbuf);
        vst1q_f32(out + i,     vld1q_f32(wbuf));
        vst1q_f32(out + i + 4, vld1q_f32(wbuf + 4));
    }
    for (uint32_t i = n8; i < n; i++) {
        uint16_t c0, c1;
        unpack2(in + (i/2) * 3, &c0, &c1);
        out[i] = mt7_lut[(i%2 == 0 ? c0 : c1) < 2187 ? (i%2 == 0 ? c0 : c1) : 0];
    }
#else
    uint32_t pairs = n / 2;
    for (uint32_t i = 0; i < pairs; i++) {
        uint16_t c0, c1;
        unpack2(in + i * 3, &c0, &c1);
        out[i*2]   = mt7_lut[c0 < 2187 ? c0 : 0];
        out[i*2+1] = mt7_lut[c1 < 2187 ? c1 : 0];
    }
    if (n & 1) {
        uint16_t c0, c1;
        unpack2(in + pairs * 3, &c0, &c1);
        out[n-1] = mt7_lut[c0 < 2187 ? c0 : 0];
    }
#endif
}
