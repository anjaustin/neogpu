#ifndef HS_MATH_NEON_H
#define HS_MATH_NEON_H

#include "hs_core.h"
#include <math.h>

#define PI 3.14159265358979323846f
#define V4_EPSILON 1e-6f
#define M4_EPSILON 1e-5f

typedef float32x4_t vec4;
typedef float32x4x4_t mat4;

static inline vec4 v4_make(f32 x, f32 y, f32 z, f32 w) {
    return (vec4){x, y, z, w};
}

static inline vec4 v4_zero(void) {
    return v4_make(0, 0, 0, 0);
}

static inline vec4 v4_one(void) {
    return v4_make(1, 1, 1, 1);
}

static inline vec4 v4_add(vec4 a, vec4 b) {
    return vaddq_f32(a, b);
}

static inline vec4 v4_sub(vec4 a, vec4 b) {
    return vsubq_f32(a, b);
}

static inline vec4 v4_mul(vec4 a, vec4 b) {
    return vmulq_f32(a, b);
}

static inline vec4 v4_scale(vec4 a, f32 s) {
    return vmulq_n_f32(a, s);
}

static inline vec4 v4_div(vec4 a, f32 s) {
    if (fabsf(s) < V4_EPSILON) {
        return v4_zero();
    }
    return vmulq_n_f32(a, 1.0f / s);
}

static inline f32 v4_dot(vec4 a, vec4 b) {
    float32x4_t prod = vmulq_f32(a, b);
    float32x2_t sum = vpadd_f32(vget_low_f32(prod), vget_high_f32(prod));
    return vget_lane_f32(vpadd_f32(sum, sum), 0);
}

static inline f32 v4_length(vec4 v) {
    return sqrtf(v4_dot(v, v));
}

static inline f32 v4_length_sq(vec4 v) {
    return v4_dot(v, v);
}

static inline vec4 v4_normalize(vec4 v) {
    f32 len = v4_length(v);
    if (len < V4_EPSILON) {
        return v4_zero();
    }
    return v4_div(v, len);
}

static inline vec4 v4_normalize_safe(vec4 v) {
    f32 len_sq = v4_length_sq(v);
    if (len_sq < V4_EPSILON * V4_EPSILON) {
        return v4_zero();
    }
    f32 len = sqrtf(len_sq);
    return vmulq_n_f32(v, 1.0f / len);
}

static inline vec4 v4_cross(vec4 a, vec4 b) {
    /*
     * Cross product: only uses xyz, result.w = 0.
     * result.x = a.y*b.z - a.z*b.y
     * result.y = a.z*b.x - a.x*b.z
     * result.z = a.x*b.y - a.y*b.x
     *
     * a_yzx = {a.y, a.z, a.x, a.w}   b_zxy = {b.z, b.x, b.y, b.w}
     * a_zxy = {a.z, a.x, a.y, a.w}   b_yzx = {b.y, b.z, b.x, b.w}
     */
    float32x4_t a_yzx = __builtin_shufflevector(a, a, 1, 2, 0, 3);
    float32x4_t b_yzx = __builtin_shufflevector(b, b, 1, 2, 0, 3);
    float32x4_t c = vsubq_f32(vmulq_f32(a, b_yzx), vmulq_f32(a_yzx, b));
    /* c = {a.x*b.y - a.y*b.x, a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, 0}
     * which is {z, x, y, 0} -- need to shuffle to {x, y, z, 0} */
    return __builtin_shufflevector(c, c, 1, 2, 0, 3);
}

static inline bool v4_equal(vec4 a, vec4 b, f32 eps) {
    vec4 diff = v4_sub(a, b);
    return v4_length_sq(diff) < eps * eps;
}

static inline vec4 v4_lerp(vec4 a, vec4 b, f32 t) {
    return v4_add(v4_scale(a, 1.0f - t), v4_scale(b, t));
}

static inline vec4 v4_bezier2(vec4 p0, vec4 p1, vec4 p2, f32 t) {
    f32 omt = 1.0f - t;
    f32 omt2 = omt * omt;
    f32 t2 = t * t;
    return v4_add(v4_add(v4_scale(p0, omt2), v4_scale(p1, 2.0f * omt * t)), v4_scale(p2, t2));
}

static inline vec4 v4_bezier3(vec4 p0, vec4 p1, vec4 p2, vec4 p3, f32 t) {
    f32 omt = 1.0f - t;
    f32 omt2 = omt * omt;
    f32 omt3 = omt2 * omt;
    f32 t2 = t * t;
    f32 t3 = t2 * t;
    return v4_add(v4_add(v4_scale(p0, omt3), v4_scale(p1, 3.0f * omt2 * t)), 
                  v4_add(v4_scale(p2, 3.0f * omt * t2), v4_scale(p3, t3)));
}

static inline vec4 v4_catmull_rom(vec4 p0, vec4 p1, vec4 p2, vec4 p3, f32 t) {
    f32 t2 = t * t;
    f32 t3 = t2 * t;
    f32 a = -0.5f * t3 + t2 - 0.5f * t;
    f32 b = 1.5f * t3 - 2.5f * t2 + 1.0f;
    f32 c = -1.5f * t3 + 2.0f * t2 + 0.5f * t;
    f32 d = 0.5f * t3 - 0.5f * t2;
    return v4_add(v4_add(v4_scale(p0, a), v4_scale(p1, b)), v4_add(v4_scale(p2, c), v4_scale(p3, d)));
}

static inline vec4 v4_hermite(vec4 p0, vec4 m0, vec4 p1, vec4 m1, f32 t) {
    f32 t2 = t * t;
    f32 t3 = t2 * t;
    f32 h00 = 2.0f * t3 - 3.0f * t2 + 1.0f;
    f32 h10 = t3 - 2.0f * t2 + t;
    f32 h01 = -2.0f * t3 + 3.0f * t2;
    f32 h11 = t3 - t2;
    return v4_add(v4_add(v4_scale(p0, h00), v4_scale(m0, h10)), 
                  v4_add(v4_scale(p1, h01), v4_scale(m1, h11)));
}

static inline vec4 v4_min(vec4 a, vec4 b) {
    return vminq_f32(a, b);
}

static inline vec4 v4_max(vec4 a, vec4 b) {
    return vmaxq_f32(a, b);
}

static inline vec4 v4_clamp(vec4 v, vec4 min_val, vec4 max_val) {
    return v4_min(v4_max(v, min_val), max_val);
}

static inline mat4 m4_identity(void) {
    mat4 m;
    m.val[0] = v4_make(1, 0, 0, 0);
    m.val[1] = v4_make(0, 1, 0, 0);
    m.val[2] = v4_make(0, 0, 1, 0);
    m.val[3] = v4_make(0, 0, 0, 1);
    return m;
}

static inline mat4 m4_zero(void) {
    mat4 m;
    m.val[0] = v4_zero();
    m.val[1] = v4_zero();
    m.val[2] = v4_zero();
    m.val[3] = v4_zero();
    return m;
}

static inline mat4 m4_translation(f32 x, f32 y, f32 z) {
    mat4 m = m4_identity();
    m.val[3] = v4_make(x, y, z, 1);
    return m;
}

static inline mat4 m4_scale(f32 x, f32 y, f32 z) {
    mat4 m = m4_identity();
    m.val[0] = v4_make(x, 0, 0, 0);
    m.val[1] = v4_make(0, y, 0, 0);
    m.val[2] = v4_make(0, 0, z, 0);
    return m;
}

static inline mat4 m4_rotation_x(f32 angle) {
    f32 c = cosf(angle);
    f32 s = sinf(angle);
    mat4 m = m4_identity();
    m.val[1] = v4_make(0, c, s, 0);
    m.val[2] = v4_make(0, -s, c, 0);
    return m;
}

static inline mat4 m4_rotation_y(f32 angle) {
    f32 c = cosf(angle);
    f32 s = sinf(angle);
    mat4 m = m4_identity();
    m.val[0] = v4_make(c, 0, -s, 0);
    m.val[2] = v4_make(s, 0, c, 0);
    return m;
}

static inline mat4 m4_rotation_z(f32 angle) {
    f32 c = cosf(angle);
    f32 s = sinf(angle);
    mat4 m = m4_identity();
    m.val[0] = v4_make(c, s, 0, 0);
    m.val[1] = v4_make(-s, c, 0, 0);
    return m;
}

static inline mat4 m4_rotation_axis(vec4 axis, f32 angle) {
    axis = v4_normalize_safe(axis);
    f32 c = cosf(angle);
    f32 s = sinf(angle);
    f32 t = 1.0f - c;
    
    f32 x = vgetq_lane_f32(axis, 0);
    f32 y = vgetq_lane_f32(axis, 1);
    f32 z = vgetq_lane_f32(axis, 2);
    
    mat4 m = m4_identity();
    m.val[0] = v4_make(t*x*x + c,   t*x*y + s*z, t*x*z - s*y, 0);
    m.val[1] = v4_make(t*x*y - s*z, t*y*y + c,   t*y*z + s*x, 0);
    m.val[2] = v4_make(t*x*z + s*y, t*y*z - s*x, t*z*z + c,   0);
    return m;
}

static inline mat4 m4_multiply(mat4 a, mat4 b) {
    /* For column-major storage: M.val[i] = column i of M.
     * Standard matrix multiply: (A*B)[i][j] = sum_k A[i][k] * B[k][j]
     * In column-major: A.col_j[i] = A.val[j][i]
     * We need to compute columns of result from columns of B:
     * result.col_j = A * B.col_j
     * = sum_k B.col_j[k] * A.col_k
     */
    mat4 result;
    for (int i = 0; i < 4; i++) {
        float32x4_t col_i = b.val[i];
        /* A.col0 * B[i][0] + A.col1 * B[i][1] + A.col2 * B[i][2] + A.col3 * B[i][3] */
        float bx = vgetq_lane_f32(col_i, 0);
        float by = vgetq_lane_f32(col_i, 1);
        float bz = vgetq_lane_f32(col_i, 2);
        float bw = vgetq_lane_f32(col_i, 3);
        result.val[i] = v4_add(
            v4_add(v4_scale(a.val[0], bx), v4_scale(a.val[1], by)),
            v4_add(v4_scale(a.val[2], bz), v4_scale(a.val[3], bw))
        );
    }
    return result;
}

static inline mat4 m4_transpose(mat4 m) {
    mat4 result;
    float32x4x2_t t1 = vtrnq_f32(m.val[0], m.val[1]);
    float32x4x2_t t2 = vtrnq_f32(m.val[2], m.val[3]);
    result.val[0] = vcombine_f32(vget_low_f32(t1.val[0]), vget_low_f32(t2.val[0]));
    result.val[1] = vcombine_f32(vget_high_f32(t1.val[0]), vget_high_f32(t2.val[0]));
    result.val[2] = vcombine_f32(vget_low_f32(t1.val[1]), vget_low_f32(t2.val[1]));
    result.val[3] = vcombine_f32(vget_high_f32(t1.val[1]), vget_high_f32(t2.val[1]));
    return result;
}

static inline mat4 m4_perspective(f32 fov_y, f32 aspect, f32 near, f32 far) {
    if (fabsf(aspect) < M4_EPSILON || fabsf(far - near) < M4_EPSILON) {
        return m4_zero();
    }
    f32 tan_half_fov = tanf(fov_y * 0.5f);
    mat4 m = m4_zero();
    m.val[0] = v4_make(1.0f / (aspect * tan_half_fov), 0, 0, 0);
    m.val[1] = v4_make(0, 1.0f / tan_half_fov, 0, 0);
    m.val[2] = v4_make(0, 0, -(far + near) / (far - near), -1);
    m.val[3] = v4_make(0, 0, -(2 * far * near) / (far - near), 0);
    return m;
}

static inline mat4 m4_ortho(f32 left, f32 right, f32 bottom, f32 top, f32 near, f32 far) {
    f32 lr = 1.0f / (right - left);
    f32 bt = 1.0f / (top - bottom);
    f32 nf = 1.0f / (far - near);
    mat4 m = m4_identity();
    m.val[0] = v4_make(2.0f * lr, 0, 0, 0);
    m.val[1] = v4_make(0, 2.0f * bt, 0, 0);
    m.val[2] = v4_make(0, 0, -2.0f * nf, 0);
    m.val[3] = v4_make(-(right + left) * lr, -(top + bottom) * bt, -(far + near) * nf, 1);
    return m;
}

/* Forward declarations for vec4 component accessors */
static inline f32 v4_x(vec4 v);
static inline f32 v4_y(vec4 v);
static inline f32 v4_z(vec4 v);

static inline mat4 m4_look_at(vec4 eye, vec4 center, vec4 up) {
    vec4 f = v4_normalize_safe(v4_sub(center, eye));
    vec4 s = v4_normalize_safe(v4_cross(f, up));
    vec4 u = v4_cross(s, f);
    
    /* Column-major: val[i] = column i.
     * Row 0 = s components, Row 1 = u components, Row 2 = -f components.
     * So column 0 = {s.x, u.x, -f.x, 0}, etc. */
    f32 sx = v4_x(s), sy = v4_y(s), sz = v4_z(s);
    f32 ux = v4_x(u), uy = v4_y(u), uz = v4_z(u);
    f32 fx = v4_x(f), fy = v4_y(f), fz = v4_z(f);
    
    mat4 m;
    m.val[0] = v4_make( sx,  ux, -fx, 0);
    m.val[1] = v4_make( sy,  uy, -fy, 0);
    m.val[2] = v4_make( sz,  uz, -fz, 0);
    m.val[3] = v4_make(
        -v4_dot(s, eye),
        -v4_dot(u, eye),
         v4_dot(f, eye),
        1
    );
    return m;
}

static inline mat4 m4_invert(mat4 m) {
    /* Extract all 16 scalar elements from columns */
    f32 m00 = vgetq_lane_f32(m.val[0], 0), m01 = vgetq_lane_f32(m.val[0], 1);
    f32 m02 = vgetq_lane_f32(m.val[0], 2), m03 = vgetq_lane_f32(m.val[0], 3);
    f32 m10 = vgetq_lane_f32(m.val[1], 0), m11 = vgetq_lane_f32(m.val[1], 1);
    f32 m12 = vgetq_lane_f32(m.val[1], 2), m13 = vgetq_lane_f32(m.val[1], 3);
    f32 m20 = vgetq_lane_f32(m.val[2], 0), m21 = vgetq_lane_f32(m.val[2], 1);
    f32 m22 = vgetq_lane_f32(m.val[2], 2), m23 = vgetq_lane_f32(m.val[2], 3);
    f32 m30 = vgetq_lane_f32(m.val[3], 0), m31 = vgetq_lane_f32(m.val[3], 1);
    f32 m32 = vgetq_lane_f32(m.val[3], 2), m33 = vgetq_lane_f32(m.val[3], 3);

    /* 2x2 sub-determinants */
    f32 s0 = m00 * m11 - m10 * m01;
    f32 s1 = m00 * m12 - m10 * m02;
    f32 s2 = m00 * m13 - m10 * m03;
    f32 s3 = m01 * m12 - m11 * m02;
    f32 s4 = m01 * m13 - m11 * m03;
    f32 s5 = m02 * m13 - m12 * m03;

    f32 c5 = m22 * m33 - m32 * m23;
    f32 c4 = m21 * m33 - m31 * m23;
    f32 c3 = m21 * m32 - m31 * m22;
    f32 c2 = m20 * m33 - m30 * m23;
    f32 c1 = m20 * m32 - m30 * m22;
    f32 c0 = m20 * m31 - m30 * m21;

    f32 det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
    if (fabsf(det) < M4_EPSILON) {
        return m4_zero();
    }
    f32 inv_det = 1.0f / det;

    mat4 result;
    result.val[0] = v4_make(
        ( m11 * c5 - m12 * c4 + m13 * c3) * inv_det,
        (-m01 * c5 + m02 * c4 - m03 * c3) * inv_det,
        ( m31 * s5 - m32 * s4 + m33 * s3) * inv_det,
        (-m21 * s5 + m22 * s4 - m23 * s3) * inv_det
    );
    result.val[1] = v4_make(
        (-m10 * c5 + m12 * c2 - m13 * c1) * inv_det,
        ( m00 * c5 - m02 * c2 + m03 * c1) * inv_det,
        (-m30 * s5 + m32 * s2 - m33 * s1) * inv_det,
        ( m20 * s5 - m22 * s2 + m23 * s1) * inv_det
    );
    result.val[2] = v4_make(
        ( m10 * c4 - m11 * c2 + m13 * c0) * inv_det,
        (-m00 * c4 + m01 * c2 - m03 * c0) * inv_det,
        ( m30 * s4 - m31 * s2 + m33 * s0) * inv_det,
        (-m20 * s4 + m21 * s2 - m23 * s0) * inv_det
    );
    result.val[3] = v4_make(
        (-m10 * c3 + m11 * c1 - m12 * c0) * inv_det,
        ( m00 * c3 - m01 * c1 + m02 * c0) * inv_det,
        (-m30 * s3 + m31 * s1 - m32 * s0) * inv_det,
        ( m20 * s3 - m21 * s1 + m22 * s0) * inv_det
    );
    return result;
}

typedef struct {
    float x, y, z, w;
} Vec4;

typedef struct {
    float m[16];
} Mat4;

static inline void m4_to_array_inline(mat4 m, float* out) {
    for (int i = 0; i < 4; i++) {
        out[i*4+0] = vgetq_lane_f32(m.val[i], 0);
        out[i*4+1] = vgetq_lane_f32(m.val[i], 1);
        out[i*4+2] = vgetq_lane_f32(m.val[i], 2);
        out[i*4+3] = vgetq_lane_f32(m.val[i], 3);
    }
}
static inline mat4 m4_from_array_inline(float* m) {
    mat4 res;
    for (int i = 0; i < 4; i++) {
        float32x4_t col = vdupq_n_f32(0);
        col = vsetq_lane_f32(m[i*4+0], col, 0);
        col = vsetq_lane_f32(m[i*4+1], col, 1);
        col = vsetq_lane_f32(m[i*4+2], col, 2);
        col = vsetq_lane_f32(m[i*4+3], col, 3);
        res.val[i] = col;
    }
    return res;
}

#define m4_to_array(m, out) m4_to_array_inline(m, out)
#define m4_from_array(m) m4_from_array_inline(m)

/* ============================================================
 * Scalar math helpers (ported from PicoApi.hx)
 * ============================================================ */

#include <stdlib.h>

static inline f32 hs_rnd(f32 max) {
    return ((f32)rand() / (f32)RAND_MAX) * max;
}

static inline s32 hs_random(s32 max) {
    if (max <= 0) return 0;
    return rand() % max;
}

static inline f32 hs_abs(f32 v)          { return fabsf(v); }
static inline f32 hs_cos(f32 v)          { return cosf(v); }
static inline f32 hs_sin(f32 v)          { return sinf(v); }
static inline f32 hs_tan(f32 v)          { return tanf(v); }
static inline f32 hs_acos(f32 v)         { return acosf(v); }
static inline f32 hs_asin(f32 v)         { return asinf(v); }
static inline f32 hs_atan(f32 v)         { return atanf(v); }
static inline f32 hs_atan2(f32 y, f32 x) { return atan2f(y, x); }
static inline f32 hs_ceil(f32 v)         { return ceilf(v); }
static inline f32 hs_floor(f32 v)        { return floorf(v); }
static inline f32 hs_round(f32 v)        { return roundf(v); }
static inline f32 hs_exp(f32 v)          { return expf(v); }
static inline f32 hs_log(f32 v)          { return logf(v); }
static inline f32 hs_pow(f32 b, f32 e)   { return powf(b, e); }
static inline f32 hs_sqrt(f32 v)         { return sqrtf(v); }

static inline f32 hs_fmin(f32 a, f32 b) { return a < b ? a : b; }
static inline f32 hs_fmax(f32 a, f32 b) { return a > b ? a : b; }
static inline s32 hs_imin(s32 a, s32 b) { return a < b ? a : b; }
static inline s32 hs_imax(s32 a, s32 b) { return a > b ? a : b; }

static inline f32 hs_clamp(f32 v, f32 lo, f32 hi) {
    return hs_fmin(hs_fmax(v, lo), hi);
}

static inline f32 hs_lerp(f32 a, f32 b, f32 t) {
    return a + (b - a) * t;
}

static inline f32 hs_deg_to_rad(f32 deg) { return deg * (PI / 180.0f); }
static inline f32 hs_rad_to_deg(f32 rad) { return rad * (180.0f / PI); }

/* vec4 component accessors */
static inline f32 v4_x(vec4 v) { return vgetq_lane_f32(v, 0); }
static inline f32 v4_y(vec4 v) { return vgetq_lane_f32(v, 1); }
static inline f32 v4_z(vec4 v) { return vgetq_lane_f32(v, 2); }
static inline f32 v4_w(vec4 v) { return vgetq_lane_f32(v, 3); }

/* Convenience: vec3 constructor (w=0) */
static inline vec4 v4_make3(f32 x, f32 y, f32 z) {
    return v4_make(x, y, z, 0.0f);
}

/* Quaternion from axis-angle */
static inline vec4 v4_quat_axis_angle(vec4 axis, f32 angle) {
    f32 half = angle * 0.5f;
    f32 s = sinf(half);
    vec4 n = v4_normalize_safe(axis);
    return v4_make(
        vgetq_lane_f32(n, 0) * s,
        vgetq_lane_f32(n, 1) * s,
        vgetq_lane_f32(n, 2) * s,
        cosf(half)
    );
}

/* Quaternion multiplication */
static inline vec4 v4_quat_mul(vec4 a, vec4 b) {
    f32 ax = v4_x(a), ay = v4_y(a), az = v4_z(a), aw = v4_w(a);
    f32 bx = v4_x(b), by = v4_y(b), bz = v4_z(b), bw = v4_w(b);
    return v4_make(
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz
    );
}

/* Convert quaternion to rotation matrix */
static inline mat4 m4_from_quat(vec4 q) {
    f32 x = v4_x(q), y = v4_y(q), z = v4_z(q), w = v4_w(q);
    f32 x2 = x+x, y2 = y+y, z2 = z+z;
    f32 xx = x*x2, xy = x*y2, xz = x*z2;
    f32 yy = y*y2, yz = y*z2, zz = z*z2;
    f32 wx = w*x2, wy = w*y2, wz = w*z2;

    mat4 m;
    m.val[0] = v4_make(1.0f-(yy+zz), xy+wz,        xz-wy,        0.0f);
    m.val[1] = v4_make(xy-wz,         1.0f-(xx+zz), yz+wx,        0.0f);
    m.val[2] = v4_make(xz+wy,         yz-wx,         1.0f-(xx+yy), 0.0f);
    m.val[3] = v4_make(0.0f,          0.0f,          0.0f,          1.0f);
    return m;
}

#endif
