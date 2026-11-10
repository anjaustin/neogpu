/*
 * hs_neonCompat.h - ARM NEON intrinsics, or a portable fallback.
 *
 * On aarch64 this is <arm_neon.h>. Elsewhere (or on aarch64 with
 * -DHS_NEON_SCALAR, which is how the fallback is tested) it provides the
 * subset of NEON that NeoGPU uses, with the same semantics, on top of GCC/Clang
 * vector extensions. Only the intrinsics below are available in fallback mode.
 */

#ifndef HS_NEON_COMPAT_H
#define HS_NEON_COMPAT_H

#if defined(__aarch64__) && !defined(HS_NEON_SCALAR)
#include <arm_neon.h>
#else
#include <stdint.h>

typedef float float32x4_t __attribute__((vector_size(16)));
typedef float float32x2_t __attribute__((vector_size(8)));
typedef struct { float32x4_t val[2]; } float32x4x2_t;
typedef struct { float32x4_t val[4]; } float32x4x4_t;

static inline float32x4_t vdupq_n_f32(float s) { return (float32x4_t){s, s, s, s}; }
static inline float32x4_t vaddq_f32(float32x4_t a, float32x4_t b) { return a + b; }
static inline float32x4_t vsubq_f32(float32x4_t a, float32x4_t b) { return a - b; }
static inline float32x4_t vmulq_f32(float32x4_t a, float32x4_t b) { return a * b; }
/* Vector times scalar (the scalar is a float, as in NEON). */
static inline float32x4_t vmulq_n_f32(float32x4_t a, float s) { return a * vdupq_n_f32(s); }

static inline float32x4_t vminq_f32(float32x4_t a, float32x4_t b) {
    float32x4_t r;
    for (int i = 0; i < 4; i++) r[i] = a[i] < b[i] ? a[i] : b[i];
    return r;
}

static inline float32x4_t vmaxq_f32(float32x4_t a, float32x4_t b) {
    float32x4_t r;
    for (int i = 0; i < 4; i++) r[i] = a[i] > b[i] ? a[i] : b[i];
    return r;
}

/* Lanes shift..3 of a followed by lanes 0..shift-1 of b. */
static inline float32x4_t vextq_f32(float32x4_t a, float32x4_t b, int shift) {
    float32x4_t r;
    for (int i = 0; i < 4; i++) {
        int src = i + shift;
        r[i] = src < 4 ? a[src] : b[src - 4];
    }
    return r;
}

/* Pairwise add: {a0+a1, b0+b1}. */
static inline float32x2_t vpadd_f32(float32x2_t a, float32x2_t b) {
    return (float32x2_t){a[0] + a[1], b[0] + b[1]};
}

static inline float vget_lane_f32(float32x2_t v, int lane) { return v[lane]; }
static inline float vgetq_lane_f32(float32x4_t v, int lane) { return v[lane]; }
static inline float32x4_t vsetq_lane_f32(float value, float32x4_t v, int lane) {
    v[lane] = value;
    return v;
}
static inline void vst1q_f32(float* ptr, float32x4_t v) {
    for (int i = 0; i < 4; i++) ptr[i] = v[i];
}
static inline float32x2_t vget_low_f32(float32x4_t v) { return (float32x2_t){v[0], v[1]}; }
static inline float32x2_t vget_high_f32(float32x4_t v) { return (float32x2_t){v[2], v[3]}; }
static inline float32x4_t vcombine_f32(float32x2_t low, float32x2_t high) {
    return (float32x4_t){low[0], low[1], high[0], high[1]};
}

/* Transpose pairs: val[0] = {a0,b0,a2,b2}, val[1] = {a1,b1,a3,b3}. */
static inline float32x4x2_t vtrnq_f32(float32x4_t a, float32x4_t b) {
    float32x4x2_t r;
    r.val[0] = (float32x4_t){a[0], b[0], a[2], b[2]};
    r.val[1] = (float32x4_t){a[1], b[1], a[3], b[3]};
    return r;
}

#endif /* aarch64 */
#endif /* HS_NEON_COMPAT_H */
