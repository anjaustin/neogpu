/*
 * hs_neonCompat.h - Compatibility layer for ARM NEON on x86
 * 
 * This file provides scalar fallbacks when compiling on non-ARM targets.
 * On real ARM builds, this just includes <arm_neon.h>.
 */

#ifndef HS_NEON_COMPAT_H
#define HS_NEON_COMPAT_H

#ifdef __aarch64__
#include <arm_neon.h>
#else
/* Scalar fallbacks for x86_64 testing */
#include <stdint.h>
#include <math.h>

typedef float float32x4_t __attribute__((vector_size(16)));
typedef float32x4_t float32x4x4_t __attribute__((vector_size(64)));

/* Basic vector operations as inline functions */
static inline float32x4_t vaddq_f32(float32x4_t a, float32x4_t b) {
    return a + b;
}

static inline float32x4_t vsubq_f32(float32x4_t a, float32x4_t b) {
    return a - b;
}

static inline float32x4_t vmulq_f32(float32x4_t a, float32x4_t b) {
    return a * b;
}

static inline float32x4_t vmulq_n_f32(float32x4_t a, float32x4_t b) {
    return a * b;
}

static inline float32x4_t vminq_f32(float32x4_t a, float32x4_t b) {
    float32x4_t r;
    for (int i = 0; i < 4; i++) r[i] = (a[i] < b[i]) ? a[i] : b[i];
    return r;
}

static inline float32x4_t vmaxq_f32(float32x4_t a, float32x4_t b) {
    float32x4_t r;
    for (int i = 0; i < 4; i++) r[i] = (a[i] > b[i]) ? a[i] : b[i];
    return r;
}

static inline float32x4_t vextq_f32(float32x4_t a, float32x4_t b, int shift) {
    float32x4_t r;
    for (int i = 0; i < 4; i++) {
        int src = i + shift;
        r[i] = (src < 4) ? a[src] : b[src - 4];
    }
    return r;
}

static inline float32x2_t vpadd_f32(float32x2_t a, float32x2_t b) {
    return (float32x2_t){a[0] + a[1], b[0] + b[1]};
}

static inline float vget_lane_f32(float32x2_t v, int lane) {
    return v[lane];
}

static inline float vgetq_lane_f32(float32x4_t v, int lane) {
    return v[lane];
}

static inline void vst1q_f32(float* ptr, float32x4_t v) {
    for (int i = 0; i < 4; i++) ptr[i] = v[i];
}

static inline float32x2_t vget_low_f32(float32x4_t v) {
    return (float32x2_t){v[0], v[1]};
}

static inline float32x2_t vget_high_f32(float32x4_t v) {
    return (float32x2_t){v[2], v[3]};
}

static inline float32x4_t vcombine_f32(float32x2_t low, float32x2_t high) {
    return (float32x4_t){low[0], low[1], high[0], high[1]};
}

typedef struct { float32x4_t val[2]; } float32x4x2_t;

static inline float32x4x2_t vtrnq_f32(float32x4_t a, float32x4_t b) {
    return (float32x4x2_t){{a, b}};
}

#endif /* __aarch64__ */
