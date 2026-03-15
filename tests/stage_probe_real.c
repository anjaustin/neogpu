#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arm_neon.h>
#include "hs_ml_infer.h"

#define I2S_QK 64u

static void statsf(const char *name, const float *x, u32 n) {
    float min=x[0], max=x[0], sum=0, absmax=0;
    for (u32 i=0;i<n;i++) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        sum += x[i];
        float a = fabsf(x[i]);
        if (a > absmax) absmax = a;
    }
    printf("%s: min=%g max=%g mean=%g absmax=%g\n", name, min, max, sum/n, absmax);
}

static void statsi(const char *name, const int32_t *x, u32 n) {
    int32_t min=x[0], max=x[0], absmax=0;
    long long sum=0;
    for (u32 i=0;i<n;i++) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        sum += x[i];
        int32_t a = x[i] < 0 ? -x[i] : x[i];
        if (a > absmax) absmax = a;
    }
    printf("%s: min=%d max=%d mean=%.3f absmax=%d\n", name, min, max, (double)sum/n, absmax);
}

static void rmsnorm_local(float *out, const float *in, const float *w, float eps, u32 n) {
    float ss = 0.0f;
    for (u32 i = 0; i < n; i++) ss += in[i] * in[i];
    float scale = 1.0f / sqrtf(ss / n + eps);
    for (u32 i = 0; i < n; i++) out[i] = in[i] * scale * w[i];
}

static float quantize_local(int8_t *out, const float *in, u32 n) {
    float absmax = 0.0f;
    for (u32 i = 0; i < n; i++) {
        float a = fabsf(in[i]);
        if (a > absmax) absmax = a;
    }
    if (absmax == 0.0f) {
        memset(out, 0, n);
        return 1.0f;
    }
    float scale = 127.0f / absmax;
    for (u32 i = 0; i < n; i++) {
        float v = in[i] * scale;
        out[i] = v > 127 ? 127 : v < -128 ? -128 : (int8_t)v;
    }
    return scale;
}

static void dequant_local(float *out, const int32_t *in, u32 n, float act_scale) {
    for (u32 i = 0; i < n; i++) out[i] = (float)in[i] / act_scale;
}

static void i2s_proj_neon(int32_t* out, const int8_t* in, const u8* W, u32 N, u32 K) {
    const uint8x16_t mask = vdupq_n_u8(3);
    const u32 nblk = K / I2S_QK;
    const u32 row_bytes = K / 4;
    int32_t sum_x = 0;
    for (u32 i = 0; i < K; i++) sum_x += (int32_t)in[i];
    for (u32 n = 0; n < N; n++) {
        const uint8_t* wrow = W + n * row_bytes;
        int32x4_t acc = vdupq_n_s32(0);
        for (u32 b = 0; b < nblk; b++) {
            uint8x16_t x3 = vld1q_u8(wrow + b * 16);
            uint8x16_t x2 = vshrq_n_u8(x3, 2);
            uint8x16_t x1 = vshrq_n_u8(x3, 4);
            uint8x16_t x0 = vshrq_n_u8(x3, 6);
            int8x16_t q0 = vreinterpretq_s8_u8(vandq_u8(x0, mask));
            int8x16_t q1 = vreinterpretq_s8_u8(vandq_u8(x1, mask));
            int8x16_t q2 = vreinterpretq_s8_u8(vandq_u8(x2, mask));
            int8x16_t q3 = vreinterpretq_s8_u8(vandq_u8(x3, mask));
            const int8x16_t y0 = vld1q_s8(in + b * 64 + 0);
            const int8x16_t y1 = vld1q_s8(in + b * 64 + 16);
            const int8x16_t y2 = vld1q_s8(in + b * 64 + 32);
            const int8x16_t y3 = vld1q_s8(in + b * 64 + 48);
            int16x8_t acc16 = vdupq_n_s16(0);
            acc16 = vmlal_s8(acc16, vget_low_s8(q0), vget_low_s8(y0));
            acc16 = vmlal_s8(acc16, vget_high_s8(q0), vget_high_s8(y0));
            acc16 = vmlal_s8(acc16, vget_low_s8(q1), vget_low_s8(y1));
            acc16 = vmlal_s8(acc16, vget_high_s8(q1), vget_high_s8(y1));
            acc16 = vmlal_s8(acc16, vget_low_s8(q2), vget_low_s8(y2));
            acc16 = vmlal_s8(acc16, vget_high_s8(q2), vget_high_s8(y2));
            acc16 = vmlal_s8(acc16, vget_low_s8(q3), vget_low_s8(y3));
            acc16 = vmlal_s8(acc16, vget_high_s8(q3), vget_high_s8(y3));
            acc = vaddq_s32(acc, vmovl_s16(vget_low_s16(acc16)));
            acc = vaddq_s32(acc, vmovl_high_s16(acc16));
        }
        out[n] = vaddvq_s32(acc) - sum_x;
    }
}

int main(void) {
    HSMLTernary m;
    hs_mlt_init(&m);
    if (hs_mlt_load_gguf(&m, "/home/ztflynn/001/neogpu/models/ggml-model-i2_s.gguf")) return 1;
    for (u32 l = 0; l < m.num_layers; l++) {
        for (u32 i = 0; i < m.hidden_size; i++) {
            m.layers[l].attn_norm[i] = 1.0f;
            m.layers[l].attn_sub_norm[i] = 1.0f;
            m.layers[l].ffn_norm[i] = 1.0f;
        }
        for (u32 i = 0; i < m.ffn_hidden_size; i++) {
            m.layers[l].ffn_sub_norm[i] = 1.0f;
        }
    }
    HSTernaryLayer *lay = &m.layers[0];
    u32 H = m.hidden_size, F = m.ffn_hidden_size, nh = m.num_heads, nkv = m.num_kv_heads, hd = m.head_dim, kv = nkv * hd;

    float *hidden = malloc(H * sizeof(float));
    memcpy(hidden, m.embedding + (size_t)128000 * H, H * sizeof(float));
    statsf("embed", hidden, H);

    float *tmp = malloc((F > H ? F : H) * sizeof(float));
    int8_t *qbuf = malloc((F > H ? F : H));
    int32_t *gbuf = malloc((F > H ? F : H) * sizeof(int32_t));
    float *q = malloc(H * sizeof(float));
    float *k = malloc(kv * sizeof(float));
    float *v = malloc(kv * sizeof(float));
    float *attn = calloc(H, sizeof(float));
    float *gate = malloc(F * sizeof(float));
    float *up = malloc(F * sizeof(float));
    float *ff = malloc(F * sizeof(float));

    rmsnorm_local(tmp, hidden, lay->attn_norm, 1e-5f, H);
    statsf("attn_norm", tmp, H);
    float as = quantize_local(qbuf, tmp, H);
    printf("act_scale=%g\n", as);

    i2s_proj_neon(gbuf, qbuf, lay->q_proj, H, H);
    statsi("q_gemm", gbuf, H);
    dequant_local(q, gbuf, H, as);
    statsf("q", q, H);

    i2s_proj_neon(gbuf, qbuf, lay->k_proj, kv, H);
    statsi("k_gemm", gbuf, kv);
    dequant_local(k, gbuf, kv, as);
    statsf("k", k, kv);

    i2s_proj_neon(gbuf, qbuf, lay->v_proj, kv, H);
    statsi("v_gemm", gbuf, kv);
    dequant_local(v, gbuf, kv, as);
    statsf("v", v, kv);

    memset(attn, 0, H * sizeof(float));
    for (u32 h = 0; h < nh; h++) {
        u32 kv_h = h * nkv / nh;
        for (u32 d = 0; d < hd; d++) attn[h * hd + d] = v[kv_h * hd + d];
    }
    statsf("attn_pre_subnorm", attn, H);
    rmsnorm_local(attn, attn, lay->attn_sub_norm, 1e-5f, H);
    statsf("attn_sub_norm", attn, H);

    as = quantize_local(qbuf, attn, H);
    i2s_proj_neon(gbuf, qbuf, lay->o_proj, H, H);
    statsi("o_gemm", gbuf, H);
    dequant_local(tmp, gbuf, H, as);
    statsf("o_proj", tmp, H);

    for (u32 i = 0; i < H; i++) hidden[i] += tmp[i];
    statsf("after_attn_resid", hidden, H);

    rmsnorm_local(tmp, hidden, lay->ffn_norm, 1e-5f, H);
    statsf("ffn_norm", tmp, H);
    as = quantize_local(qbuf, tmp, H);
    i2s_proj_neon(gbuf, qbuf, lay->gate_proj, F, H);
    statsi("gate_gemm", gbuf, F);
    dequant_local(gate, gbuf, F, as);
    statsf("gate", gate, F);
    i2s_proj_neon(gbuf, qbuf, lay->up_proj, F, H);
    statsi("up_gemm", gbuf, F);
    dequant_local(up, gbuf, F, as);
    statsf("up", up, F);
    for (u32 i = 0; i < F; i++) {
        float rg = gate[i] > 0.0f ? gate[i] : 0.0f;
        ff[i] = (rg * rg) * up[i];
    }
    statsf("relu2*up", ff, F);
    rmsnorm_local(ff, ff, lay->ffn_sub_norm, 1e-5f, F);
    statsf("ffn_sub_norm", ff, F);
    as = quantize_local(qbuf, ff, F);
    i2s_proj_neon(gbuf, qbuf, lay->down_proj, H, F);
    statsi("down_gemm", gbuf, H);
    dequant_local(tmp, gbuf, H, as);
    statsf("down", tmp, H);
    for (u32 i = 0; i < H; i++) hidden[i] += tmp[i];
    statsf("after_ffn_resid", hidden, H);
    return 0;
}
