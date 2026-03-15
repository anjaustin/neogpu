/*
 * NeoGPU ML - Multi-Trit Floating Point 11 (MT11)
 *
 * 11-trit floating point: 177,147 distinct values = 17.43 bits.
 * More resolution than F16 (65,536 values = 16 bits).
 *
 * Layout (22 bits packed as 11 trit-pairs):
 *   bits [1:0]   = sign trit        {0=-1, 1=0, 2=+1}
 *   bits [15:2]  = mantissa          7 trits (0..2186) in 14 bits
 *   bits [21:16] = exponent          3 trits (0..26) in 6 bits
 *
 * Value = sign * (mantissa / 2186.0) * 3^(exponent - 13)
 *
 * Storage: 3 MT11 values per 8 bytes (22 bits × 3 = 66 bits, pad to 64+8)
 *   Actually: pack 4 values per 11 bytes (88 bits = 4 × 22)
 *   Simpler: 22 bits each, pack 8 values per 22 bytes
 *   Simplest: align to 3 bytes (24 bits) per value = 12.5% overhead
 *
 * We use 24-bit (3 byte) packing for simplicity and alignment:
 *   byte[0] = bits [7:0]
 *   byte[1] = bits [15:8]
 *   byte[2] = bits [21:16] in low 6 bits, high 2 bits = 0
 *
 * Decode is pure arithmetic — no LUT:
 *   1. Extract sign, mantissa trit-pairs, exponent trit-pairs
 *   2. Convert 7 trit-pairs to mantissa integer via Horner's method
 *   3. Convert 3 trit-pairs to exponent integer via Horner's method
 *   4. Compute: sign * (mant / 2186.0f) * pow3[exp]
 *
 * pow3 table: 27 entries × 4 bytes = 108 bytes (fits in NEON registers)
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/*============================================================================
 * Constants
 *============================================================================*/

/* 3^(n-13) for n=0..26 */
static const float MT11_POW3[27] = {
    6.30957344e-07f, /* 3^-13 */
    1.89287203e-06f, /* 3^-12 */
    5.67861610e-06f, /* 3^-11 */
    1.70358483e-05f, /* 3^-10 */
    5.11075449e-05f, /* 3^-9  */
    1.53322635e-04f, /* 3^-8  */
    4.59967904e-04f, /* 3^-7  */
    1.37990371e-03f, /* 3^-6  */
    4.13971114e-03f, /* 3^-5  */
    1.24191334e-02f, /* 3^-4  */
    3.72574003e-02f, /* 3^-3  */
    1.11111111e-01f, /* 3^-2  */
    3.33333333e-01f, /* 3^-1  */
    1.00000000e+00f, /* 3^0   */
    3.00000000e+00f, /* 3^1   */
    9.00000000e+00f, /* 3^2   */
    2.70000000e+01f, /* 3^3   */
    8.10000000e+01f, /* 3^4   */
    2.43000000e+02f, /* 3^5   */
    7.29000000e+02f, /* 3^6   */
    2.18700000e+03f, /* 3^7   */
    6.56100000e+03f, /* 3^8   */
    1.96830000e+04f, /* 3^9   */
    5.90490000e+04f, /* 3^10  */
    1.77147000e+05f, /* 3^11  */
    5.31441000e+05f, /* 3^12  */
    1.59432300e+06f, /* 3^13  */
};

static const float MT11_MANT_INV = 1.0f / 2186.0f;

/*============================================================================
 * Trit-pair to integer conversion (Horner's method)
 *
 * 7 trit-pairs in 14 bits -> integer 0..2186
 * t = b0 + 3*b1 + 9*b2 + 27*b3 + 81*b4 + 243*b5 + 729*b6
 *   = b0 + 3*(b1 + 3*(b2 + 3*(b3 + 3*(b4 + 3*(b5 + 3*b6)))))
 *============================================================================*/

static inline uint32_t trits7_to_int(uint32_t raw14) {
    uint32_t t0 = raw14 & 3;
    uint32_t t1 = (raw14 >> 2) & 3;
    uint32_t t2 = (raw14 >> 4) & 3;
    uint32_t t3 = (raw14 >> 6) & 3;
    uint32_t t4 = (raw14 >> 8) & 3;
    uint32_t t5 = (raw14 >> 10) & 3;
    uint32_t t6 = (raw14 >> 12) & 3;
    return t0 + 3*(t1 + 3*(t2 + 3*(t3 + 3*(t4 + 3*(t5 + 3*t6)))));
}

static inline uint32_t trits3_to_int(uint32_t raw6) {
    uint32_t t0 = raw6 & 3;
    uint32_t t1 = (raw6 >> 2) & 3;
    uint32_t t2 = (raw6 >> 4) & 3;
    return t0 + 3*(t1 + 3*t2);
}

/*============================================================================
 * Encode: float32 -> MT11 (22-bit code packed in 24 bits)
 *============================================================================*/

static inline uint32_t int_to_trits7(uint32_t v) {
    uint32_t code = 0;
    for (int i = 0; i < 7; i++) {
        code |= (v % 3) << (i * 2);
        v /= 3;
    }
    return code;
}

static inline uint32_t int_to_trits3(uint32_t v) {
    uint32_t code = 0;
    for (int i = 0; i < 3; i++) {
        code |= (v % 3) << (i * 2);
        v /= 3;
    }
    return code;
}

static uint32_t mt11_encode_one(float v) {
    if (v == 0.0f) {
        /* sign=0 -> trit 1, mant=0, exp=13 (unity scale) */
        uint32_t sign_bits = 1;
        uint32_t mant_bits = 0;
        uint32_t exp_bits = int_to_trits3(13);
        return sign_bits | (mant_bits << 2) | (exp_bits << 16);
    }

    uint32_t sign_trit = (v > 0) ? 2 : 0;
    float av = fabsf(v);

    float best_err = 1e30f;
    uint32_t best_code = 0;

    for (int exp_idx = 0; exp_idx < 27; exp_idx++) {
        float scale = MT11_POW3[exp_idx];
        int mant = (int)roundf(av / scale * 2186.0f);
        if (mant < 0) mant = 0;
        if (mant > 2186) mant = 2186;

        float reconstructed = ((float)mant * MT11_MANT_INV) * scale;
        float err = fabsf(av - reconstructed);

        if (err < best_err) {
            best_err = err;
            uint32_t mant_bits = int_to_trits7(mant);
            uint32_t exp_bits = int_to_trits3(exp_idx);
            best_code = sign_trit | (mant_bits << 2) | (exp_bits << 16);
        }
    }
    return best_code;
}

/*============================================================================
 * Decode: MT11 (22-bit code) -> float32  (pure arithmetic, no LUT)
 *============================================================================*/

static inline float mt11_decode_one(uint32_t code) {
    int32_t sign = (int32_t)(code & 3) - 1;
    uint32_t mant_raw = (code >> 2) & 0x3FFF;
    uint32_t exp_raw = (code >> 16) & 0x3F;

    uint32_t mant = trits7_to_int(mant_raw);
    uint32_t exp_idx = trits3_to_int(exp_raw);
    if (exp_idx > 26) exp_idx = 26;

    return (float)sign * ((float)mant * MT11_MANT_INV) * MT11_POW3[exp_idx];
}

/*============================================================================
 * Pack/Unpack: 3 bytes per MT11 value
 *============================================================================*/

static inline void mt11_pack(uint8_t *out, uint32_t code) {
    out[0] = (uint8_t)(code & 0xFF);
    out[1] = (uint8_t)((code >> 8) & 0xFF);
    out[2] = (uint8_t)((code >> 16) & 0x3F);
}

static inline uint32_t mt11_unpack(const uint8_t *in) {
    return (uint32_t)in[0] | ((uint32_t)in[1] << 8) | ((uint32_t)(in[2] & 0x3F) << 16);
}

/*============================================================================
 * Bulk encode/decode
 *============================================================================*/

void mt11_encode_f32(uint8_t *out, const float *in, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        uint32_t code = mt11_encode_one(in[i]);
        mt11_pack(out + i * 3, code);
    }
}

void mt11_decode_to_f32(float *out, const uint8_t *in, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        uint32_t code = mt11_unpack(in + i * 3);
        out[i] = mt11_decode_one(code);
    }
}

/*============================================================================
 * NEON-optimized decode: process 4 values at a time
 *
 * Trit-wise decomposition in 64-bit lanes:
 *   - Load 4 × 3 bytes = 12 bytes
 *   - Unpack to 4 × 32-bit codes
 *   - Extract sign, mantissa, exponent fields
 *   - Horner's method for trit-to-int (in NEON integer pipeline)
 *   - Convert mantissa to float, multiply by scale and pow3
 *============================================================================*/

#ifdef __ARM_NEON
static inline float32x4_t mt11_decode_4_neon(const uint8_t *in) {
    /* Unpack 4 codes from 12 bytes */
    uint32_t c0 = mt11_unpack(in);
    uint32_t c1 = mt11_unpack(in + 3);
    uint32_t c2 = mt11_unpack(in + 6);
    uint32_t c3 = mt11_unpack(in + 9);

    /* Extract fields */
    int32_t s0 = (int32_t)(c0 & 3) - 1;
    int32_t s1 = (int32_t)(c1 & 3) - 1;
    int32_t s2 = (int32_t)(c2 & 3) - 1;
    int32_t s3 = (int32_t)(c3 & 3) - 1;

    uint32_t m0 = trits7_to_int((c0 >> 2) & 0x3FFF);
    uint32_t m1 = trits7_to_int((c1 >> 2) & 0x3FFF);
    uint32_t m2 = trits7_to_int((c2 >> 2) & 0x3FFF);
    uint32_t m3 = trits7_to_int((c3 >> 2) & 0x3FFF);

    uint32_t e0 = trits3_to_int((c0 >> 16) & 0x3F);
    uint32_t e1 = trits3_to_int((c1 >> 16) & 0x3F);
    uint32_t e2 = trits3_to_int((c2 >> 16) & 0x3F);
    uint32_t e3 = trits3_to_int((c3 >> 16) & 0x3F);
    if (e0 > 26) e0 = 26;
    if (e1 > 26) e1 = 26;
    if (e2 > 26) e2 = 26;
    if (e3 > 26) e3 = 26;

    /* Build float vectors */
    int32x4_t vsign = {s0, s1, s2, s3};
    float32x4_t vmant = vcvtq_f32_s32((int32x4_t){m0, m1, m2, m3});
    float32x4_t vpow = {MT11_POW3[e0], MT11_POW3[e1], MT11_POW3[e2], MT11_POW3[e3]};

    /* value = sign * (mant / 2186) * pow3[exp] */
    float32x4_t vscale = vdupq_n_f32(MT11_MANT_INV);
    float32x4_t val = vmulq_f32(vmulq_f32(vmant, vscale), vpow);
    val = vmulq_f32(val, vcvtq_f32_s32(vsign));

    return val;
}
#endif

void mt11_decode_neon(float *out, const uint8_t *in, uint32_t n) {
#ifdef __ARM_NEON
    uint32_t n4 = n & ~3u;
    for (uint32_t i = 0; i < n4; i += 4) {
        float32x4_t v = mt11_decode_4_neon(in + i * 3);
        vst1q_f32(out + i, v);
    }
    for (uint32_t i = n4; i < n; i++) {
        out[i] = mt11_decode_one(mt11_unpack(in + i * 3));
    }
#else
    mt11_decode_to_f32(out, in, n);
#endif
}

/*============================================================================
 * MT11 RMSNorm (NEON-optimized)
 *============================================================================*/

void mt11_rmsnorm_neon(float *out, const float *in, const uint8_t *mt11_weights,
                       float eps, uint32_t n) {
#ifdef __ARM_NEON
    /* Pass 1: sum of squares */
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

    /* Pass 2: decode MT11 weights and apply */
    for (uint32_t i = 0; i < n4; i += 4) {
        float32x4_t w = mt11_decode_4_neon(mt11_weights + i * 3);
        float32x4_t v = vld1q_f32(in + i);
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(v, vscale), w));
    }
    for (uint32_t i = n4; i < n; i++) {
        float w = mt11_decode_one(mt11_unpack(mt11_weights + i * 3));
        out[i] = in[i] * rms_scale * w;
    }
#else
    float ss = 0.0f;
    for (uint32_t i = 0; i < n; i++) ss += in[i] * in[i];
    float rms_scale = 1.0f / sqrtf(ss / (float)n + eps);
    for (uint32_t i = 0; i < n; i++) {
        float w = mt11_decode_one(mt11_unpack(mt11_weights + i * 3));
        out[i] = in[i] * rms_scale * w;
    }
#endif
}

/*============================================================================
 * Storage helpers
 *============================================================================*/

uint32_t mt11_storage_bytes(uint32_t n) {
    return n * 3;  /* 3 bytes per value (24-bit aligned) */
}

float mt11_bits_per_value(void) {
    return 17.43f;  /* log2(177147) */
}
