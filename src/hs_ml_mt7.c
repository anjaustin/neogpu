/*
 * NeoGPU ML - Multi-Trit Floating Point 7 (MT7)
 *
 * 7-trit floating point representation for norm weight quantization.
 *
 * Layout: 1 sign trit + 4 mantissa trits + 2 exponent trits
 *   sign:     {-1, 0, +1}
 *   mantissa: 0..80 (81 values from 4 trits)
 *   exponent: -4..+4 (9 values from 2 balanced trits)
 *   value = sign * (mantissa / 80) * 3^exponent
 *
 * Properties:
 *   - 3^7 = 2187 distinct values
 *   - 11.09 bits of information
 *   - Range: ±81 with zero as explicit state
 *   - Mean quantization error: 0.46% on norm weights
 *   - Packed as 12-bit codes (2187 < 4096 = 2^12)
 *   - Two MT7 values pack into 3 bytes (24 bits)
 *
 * Storage format:
 *   Two MT7 values per 3 bytes:
 *     byte[0] = low 8 bits of code_0
 *     byte[1] = high 4 bits of code_0 | low 4 bits of code_1
 *     byte[2] = high 8 bits of code_1
 *
 * See: docs/theory/GEOMETRIC_INFERENCE.md
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

/* 3^exp lookup table for exponents -4..+4 */
static const float MT7_POW3[9] = {
    0.01234567901f,  /* 3^-4 = 1/81 */
    0.03703703704f,  /* 3^-3 = 1/27 */
    0.11111111111f,  /* 3^-2 = 1/9  */
    0.33333333333f,  /* 3^-1 = 1/3  */
    1.0f,            /* 3^0  = 1    */
    3.0f,            /* 3^1  = 3    */
    9.0f,            /* 3^2  = 9    */
    27.0f,           /* 3^3  = 27   */
    81.0f,           /* 3^4  = 81   */
};

/* Precomputed decode table: code (0..2186) -> float32 value */
static float mt7_decode_lut[2187];
static int mt7_lut_built = 0;

static void mt7_build_lut(void) {
    if (mt7_lut_built) return;
    for (int code = 0; code < 2187; code++) {
        /* Decode 7 trits from code: base-3 digits */
        int c = code;
        int trits[7];
        for (int i = 0; i < 7; i++) {
            trits[i] = c % 3;
            c /= 3;
        }
        /* trit[0] = sign: 0->-1, 1->0, 2->+1 */
        int sign_val = trits[0] - 1;
        /* trit[1..4] = mantissa: 4 trits = 0..80 */
        int mant = trits[1] + trits[2] * 3 + trits[3] * 9 + trits[4] * 27;
        /* trit[5..6] = exponent: 2 trits = 0..8, mapped to -4..+4 */
        int exp_idx = trits[5] + trits[6] * 3;  /* 0..8 */

        float val = (float)sign_val * ((float)mant / 80.0f) * MT7_POW3[exp_idx];
        mt7_decode_lut[code] = val;
    }
    mt7_lut_built = 1;
}

/*============================================================================
 * Encode: float32 -> MT7 code (0..2186)
 *============================================================================*/

static uint16_t mt7_encode_scalar(float v) {
    if (v == 0.0f) {
        /* sign=0 -> trit 1, mant=0, exp=0 -> trits (1,0,0,0,0,4_idx)
         * sign trit=1 means 0. code = 1 + 0 + 0 + 0 + 0 + 4*243 + 0 = 1 */
        /* Actually: sign=0 is trit value 1 (since 0->-1, 1->0, 2->+1) */
        /* code = 1 (sign) + 0*3 + 0*9 + 0*27 + 0*81 + 4*243 + 0*729 */
        return 1 + 4 * 243;  /* = 973 */
    }

    int sign_trit = (v > 0) ? 2 : 0;  /* +1 -> trit 2, -1 -> trit 0 */
    float av = fabsf(v);

    float best_err = 1e30f;
    uint16_t best_code = 0;

    for (int exp_idx = 0; exp_idx < 9; exp_idx++) {
        float scale = MT7_POW3[exp_idx];
        int mant = (int)roundf(av / scale * 80.0f);
        if (mant < 0) mant = 0;
        if (mant > 80) mant = 80;

        float reconstructed = ((float)mant / 80.0f) * scale;
        float err = fabsf(av - reconstructed);

        if (err < best_err) {
            best_err = err;
            /* Encode as base-3 number */
            int m0 = mant % 3;
            int m1 = (mant / 3) % 3;
            int m2 = (mant / 9) % 3;
            int m3 = (mant / 27) % 3;
            int e0 = exp_idx % 3;
            int e1 = exp_idx / 3;

            best_code = (uint16_t)(sign_trit +
                                   m0 * 3 +
                                   m1 * 9 +
                                   m2 * 27 +
                                   m3 * 81 +
                                   e0 * 243 +
                                   e1 * 729);
        }
    }
    return best_code;
}

/*============================================================================
 * Decode: MT7 code -> float32
 *============================================================================*/

static inline float mt7_decode_scalar(uint16_t code) {
    return mt7_decode_lut[code];
}

/*============================================================================
 * Pack/Unpack: two 12-bit codes into 3 bytes
 *============================================================================*/

static inline void mt7_pack2(uint8_t *out, uint16_t code0, uint16_t code1) {
    out[0] = (uint8_t)(code0 & 0xFF);
    out[1] = (uint8_t)((code0 >> 8) | ((code1 & 0xF) << 4));
    out[2] = (uint8_t)(code1 >> 4);
}

static inline void mt7_unpack2(const uint8_t *in, uint16_t *code0, uint16_t *code1) {
    *code0 = (uint16_t)(in[0] | ((in[1] & 0x0F) << 8));
    *code1 = (uint16_t)((in[1] >> 4) | ((uint16_t)in[2] << 4));
}

/*============================================================================
 * Bulk encode: float32 array -> MT7 packed bytes
 *============================================================================*/

void mt7_encode_f32(uint8_t *out, const float *in, uint32_t n) {
    mt7_build_lut();
    uint32_t pairs = n / 2;
    for (uint32_t i = 0; i < pairs; i++) {
        uint16_t c0 = mt7_encode_scalar(in[i * 2]);
        uint16_t c1 = mt7_encode_scalar(in[i * 2 + 1]);
        mt7_pack2(out + i * 3, c0, c1);
    }
    if (n & 1) {
        uint16_t c0 = mt7_encode_scalar(in[n - 1]);
        mt7_pack2(out + pairs * 3, c0, 0);
    }
}

/*============================================================================
 * Bulk decode: MT7 packed bytes -> float32 array
 *============================================================================*/

void mt7_decode_to_f32(float *out, const uint8_t *in, uint32_t n) {
    mt7_build_lut();
    uint32_t pairs = n / 2;
    for (uint32_t i = 0; i < pairs; i++) {
        uint16_t c0, c1;
        mt7_unpack2(in + i * 3, &c0, &c1);
        out[i * 2]     = mt7_decode_lut[c0 < 2187 ? c0 : 0];
        out[i * 2 + 1] = mt7_decode_lut[c1 < 2187 ? c1 : 0];
    }
    if (n & 1) {
        uint16_t c0, c1;
        mt7_unpack2(in + pairs * 3, &c0, &c1);
        out[n - 1] = mt7_decode_lut[c0 < 2187 ? c0 : 0];
    }
}

/*============================================================================
 * MT7 RMSNorm: applies RMSNorm using MT7-encoded weights directly
 *
 * out[i] = (in[i] / rms) * mt7_decode(weight[i])
 *
 * The MT7 decode is done via LUT — one table lookup per weight.
 *============================================================================*/

void mt7_rmsnorm(float *out, const float *in, const uint8_t *mt7_weights,
                 float eps, uint32_t n) {
    mt7_build_lut();

    /* Compute RMS */
    float ss = 0.0f;
#ifdef __ARM_NEON
    float32x4_t vss = vdupq_n_f32(0.0f);
    uint32_t n4 = n & ~3u;
    for (uint32_t i = 0; i < n4; i += 4) {
        float32x4_t v = vld1q_f32(in + i);
        vss = vfmaq_f32(vss, v, v);
    }
    ss = vaddvq_f32(vss);
    for (uint32_t i = n4; i < n; i++) ss += in[i] * in[i];
#else
    for (uint32_t i = 0; i < n; i++) ss += in[i] * in[i];
#endif

    float scale = 1.0f / sqrtf(ss / (float)n + eps);

    /* Apply scale and MT7 weight */
    uint32_t pairs = n / 2;
    for (uint32_t i = 0; i < pairs; i++) {
        uint16_t c0, c1;
        mt7_unpack2(mt7_weights + i * 3, &c0, &c1);
        float w0 = mt7_decode_lut[c0 < 2187 ? c0 : 0];
        float w1 = mt7_decode_lut[c1 < 2187 ? c1 : 0];
        out[i * 2]     = in[i * 2]     * scale * w0;
        out[i * 2 + 1] = in[i * 2 + 1] * scale * w1;
    }
    if (n & 1) {
        uint16_t c0, c1;
        mt7_unpack2(mt7_weights + pairs * 3, &c0, &c1);
        float w0 = mt7_decode_lut[c0 < 2187 ? c0 : 0];
        out[n - 1] = in[n - 1] * scale * w0;
    }
}

/*============================================================================
 * Storage helpers
 *============================================================================*/

/* Bytes needed to store n MT7 values */
uint32_t mt7_storage_bytes(uint32_t n) {
    return ((n + 1) / 2) * 3;
}
