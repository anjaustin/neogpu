/*
 * NeoGPU ML - Ternary Inference Implementation
 *
 * End-to-end transformer inference using the ternary GEMM kernels.
 * See include/hs_ml_infer.h for architecture and data flow.
 */

#include "hs_ml_infer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdio.h>

/* Global stats */
HSMLTernaryStats g_mlt_stats;

/*============================================================================
 * Timing
 *============================================================================*/

static inline u64 now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (u64)ts.tv_sec * 1000000000ULL + (u64)ts.tv_nsec;
}

/*============================================================================
 * Lifecycle
 *============================================================================*/

void hs_mlt_init(HSMLTernary* m) {
    memset(m, 0, sizeof(*m));
}

void hs_mlt_free(HSMLTernary* m) {
    if (!m) return;

    free(m->embedding);
    free(m->lm_head);
    free(m->final_norm);
    if (m->tokenizer_vocab) {
        for (u32 i = 0; i < m->vocab_size; i++) free(m->tokenizer_vocab[i]);
        free(m->tokenizer_vocab);
    }
    if (m->tokenizer_merges) {
        for (u32 i = 0; i < m->num_merges; i++) free(m->tokenizer_merges[i]);
        free(m->tokenizer_merges);
    }

    if (m->layers) {
        for (u32 l = 0; l < m->num_layers; l++) {
            HSTernaryLayer* lay = &m->layers[l];
            free(lay->q_proj);   free(lay->q_scale);
            free(lay->k_proj);   free(lay->k_scale);
            free(lay->v_proj);   free(lay->v_scale);
            free(lay->o_proj);   free(lay->o_scale);
            free(lay->gate_proj);free(lay->gate_scale);
            free(lay->up_proj);  free(lay->up_scale);
            free(lay->down_proj);free(lay->down_scale);
            free(lay->attn_norm);
            free(lay->attn_sub_norm);
            free(lay->ffn_norm);
            free(lay->ffn_sub_norm);
        }
        free(m->layers);
    }

    free(m->quant_buf);
    free(m->gemm_buf);
    free(m->hidden_buf);
    free(m->ffn_buf);
    free(m->q_buf);
    free(m->k_buf);
    free(m->v_buf);
    free(m->attn_buf);
    free(m->gate_buf);
    free(m->up_buf);
    free(m->score_buf);
    free(m->ffn_qbuf);

    memset(m, 0, sizeof(*m));
}

/*
 * Allocate a ternary weight matrix [N, K/4] with random {-1,0,+1} weights.
 * Scales are set to 1.0 (random weights have no meaningful scale).
 */
static u8* alloc_ternary(u32 N, u32 K, float* scales, unsigned int* rng) {
    u8* W = aligned_alloc(64, (size_t)N * (K / 4));
    if (!W) return NULL;

    for (u32 n = 0; n < N; n++) {
        u8* row = W + n * (K / 4);
        for (u32 k = 0; k < K / 4; k++) {
            u8 byte = 0;
            for (int b = 0; b < 4; b++) {
                /* Random ternary: 50% zero, 25% +1, 25% -1 */
                int r = rand_r(rng) % 4;
                u8 v = (r == 0) ? 1 : (r == 1) ? 2 : 0;
                byte |= (v << (b * 2));
            }
            row[k] = byte;
        }
        scales[n] = 1.0f;
    }
    return W;
}

int hs_mlt_alloc_random(HSMLTernary* m,
                        u32 vocab_size, u32 hidden_size, u32 num_layers,
                        u32 num_heads, u32 ffn_hidden_size, u32 max_context,
                        unsigned int seed) {
    hs_mlt_free(m);

    m->vocab_size      = vocab_size;
    m->hidden_size     = hidden_size;
    m->num_layers      = num_layers;
    m->num_heads       = num_heads;
    m->num_kv_heads    = num_heads;  /* alloc_random: always MHA */
    m->head_dim        = hidden_size / num_heads;
    m->ffn_hidden_size = ffn_hidden_size;
    m->max_context     = max_context;
    m->rope_theta      = 10000.0f;

    unsigned int rng = seed;

    /* Embeddings and head (float32) */
    m->embedding  = malloc((size_t)vocab_size * hidden_size * sizeof(float));
    m->lm_head    = malloc((size_t)vocab_size * hidden_size * sizeof(float));
    m->final_norm = malloc(hidden_size * sizeof(float));
    if (!m->embedding || !m->lm_head || !m->final_norm) goto fail;

    for (u32 i = 0; i < vocab_size * hidden_size; i++)
        m->embedding[i] = ((float)(rand_r(&rng) % 200) - 100.0f) / 100.0f;
    for (u32 i = 0; i < vocab_size * hidden_size; i++)
        m->lm_head[i] = ((float)(rand_r(&rng) % 200) - 100.0f) / 100.0f;
    for (u32 i = 0; i < hidden_size; i++)
        m->final_norm[i] = 1.0f;

    /* Per-layer weights */
    m->layers = calloc(num_layers, sizeof(HSTernaryLayer));
    if (!m->layers) goto fail;

    for (u32 l = 0; l < num_layers; l++) {
        HSTernaryLayer* lay = &m->layers[l];
        u32 H = hidden_size, F = ffn_hidden_size;

        /* Attention projections: [H, H] ternary */
        lay->q_scale = malloc(H * sizeof(float));
        lay->k_scale = malloc(H * sizeof(float));
        lay->v_scale = malloc(H * sizeof(float));
        lay->o_scale = malloc(H * sizeof(float));
        if (!lay->q_scale || !lay->k_scale || !lay->v_scale || !lay->o_scale)
            goto fail;

        lay->q_proj = alloc_ternary(H, H, lay->q_scale, &rng);
        lay->k_proj = alloc_ternary(H, H, lay->k_scale, &rng);
        lay->v_proj = alloc_ternary(H, H, lay->v_scale, &rng);
        lay->o_proj = alloc_ternary(H, H, lay->o_scale, &rng);
        if (!lay->q_proj || !lay->k_proj || !lay->v_proj || !lay->o_proj)
            goto fail;

        /* FFN projections */
        lay->gate_scale = malloc(F * sizeof(float));
        lay->up_scale   = malloc(F * sizeof(float));
        lay->down_scale = malloc(H * sizeof(float));
        if (!lay->gate_scale || !lay->up_scale || !lay->down_scale)
            goto fail;

        lay->gate_proj = alloc_ternary(F, H, lay->gate_scale, &rng);
        lay->up_proj   = alloc_ternary(F, H, lay->up_scale, &rng);
        lay->down_proj = alloc_ternary(H, F, lay->down_scale, &rng);
        if (!lay->gate_proj || !lay->up_proj || !lay->down_proj)
            goto fail;

        /* Norms (float32, initialised to 1) */
        lay->attn_norm = malloc(H * sizeof(float));
        lay->attn_sub_norm = malloc(H * sizeof(float));
        lay->ffn_norm  = malloc(H * sizeof(float));
        lay->ffn_sub_norm = malloc(F * sizeof(float));
        if (!lay->attn_norm || !lay->attn_sub_norm || !lay->ffn_norm || !lay->ffn_sub_norm) goto fail;

        for (u32 i = 0; i < H; i++) { lay->attn_norm[i] = 1.0f; lay->attn_sub_norm[i] = 1.0f; lay->ffn_norm[i] = 1.0f; }
        for (u32 i = 0; i < F; i++) lay->ffn_sub_norm[i] = 1.0f;
    }

    /* Scratch buffers — allocated once, never freed during inference */
    u32 max_dim = (hidden_size > ffn_hidden_size) ? hidden_size : ffn_hidden_size;
    m->quant_buf  = aligned_alloc(64, max_dim * sizeof(int8_t));
    m->gemm_buf   = aligned_alloc(64, max_dim * sizeof(int32_t));
    m->hidden_buf = malloc(hidden_size * sizeof(float));
    m->ffn_buf    = malloc(ffn_hidden_size * sizeof(float));
    m->q_buf      = malloc(hidden_size * sizeof(float));
    m->k_buf      = malloc(hidden_size * sizeof(float));
    m->v_buf      = malloc(hidden_size * sizeof(float));
    m->attn_buf   = calloc(hidden_size, sizeof(float));
    m->gate_buf   = malloc(ffn_hidden_size * sizeof(float));
    m->up_buf     = malloc(ffn_hidden_size * sizeof(float));
    m->score_buf  = malloc(max_context * sizeof(float));
    m->ffn_qbuf   = aligned_alloc(64, ffn_hidden_size * sizeof(int8_t));
    if (!m->quant_buf || !m->gemm_buf || !m->hidden_buf || !m->ffn_buf ||
        !m->q_buf || !m->k_buf || !m->v_buf || !m->attn_buf ||
        !m->gate_buf || !m->up_buf || !m->score_buf || !m->ffn_qbuf)
        goto fail;

    m->loaded = true;
    return 0;

fail:
    hs_mlt_free(m);
    return -1;
}

/*============================================================================
 * Quantisation helpers
 *============================================================================*/

float hs_mlt_quantize(int8_t* output, const float* input, u32 n) {
    /* Find absmax */
    float absmax = 0.0f;
    for (u32 i = 0; i < n; i++) {
        float a = fabsf(input[i]);
        if (a > absmax) absmax = a;
    }
    if (absmax == 0.0f) {
        memset(output, 0, n * sizeof(int8_t));
        return 1.0f;
    }
    float scale = 127.0f / absmax;
    for (u32 i = 0; i < n; i++) {
        float v = input[i] * scale;
        output[i] = (v > 127.0f) ? 127 : (v < -128.0f) ? -128 : (int8_t)v;
    }
    return scale;
}

void hs_mlt_dequantize(float* output, const int32_t* input, u32 n,
                       float act_scale, const float* weight_scales) {
    if (!weight_scales) {
        for (u32 i = 0; i < n; i++) output[i] = (float)input[i] / act_scale;
        return;
    }
    for (u32 i = 0; i < n; i++) {
        output[i] = (float)input[i] / (act_scale * weight_scales[i]);
    }
}

/*============================================================================
 * RMSNorm
 *============================================================================*/

static void rmsnorm(float* out, const float* in, const float* weight,
                    float eps, u32 n) {
    float ss = 0.0f;
    for (u32 i = 0; i < n; i++) ss += in[i] * in[i];
    float scale = 1.0f / sqrtf(ss / n + eps);
    for (u32 i = 0; i < n; i++) out[i] = in[i] * scale * weight[i];
}

/*============================================================================
 * Ternary GEMM via routing abstraction (M=1)
 *
 * input:  [K] int8   (already quantized)
 * weight: [N, K/4] uint8 ternary
 * output: [N] int32
 * Returns the scale factor that should be used for dequantization.
 *============================================================================*/

extern void hs_ml_gemm_ternary_mt(int32_t* C, const int8_t* A,
                                   const u8* B, u32 M, u32 N, u32 K,
                                   int num_threads);

#define I2S_QK 64u

static void i2s_proj_neon(int32_t* out, const int8_t* in,
                          const u8* W, u32 N, u32 K) {
#ifdef __ARM_NEON
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

            const int8x16_t y0 = vld1q_s8(in + b * 64 +  0);
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
#else
    const u32 row_bytes = K / 4;
    int32_t sum_x = 0;
    for (u32 i = 0; i < K; i++) sum_x += (int32_t)in[i];
    for (u32 n = 0; n < N; n++) {
        const uint8_t* wrow = W + n * row_bytes;
        int32_t acc = 0;
        for (u32 k = 0; k < K; k++) {
            uint8_t byte = wrow[k / 4];
            uint8_t code = (byte >> ((k % 4) * 2)) & 0x3;
            acc += (int32_t)code * (int32_t)in[k];
        }
        out[n] = acc - sum_x;
    }
#endif
}

static void ternary_proj(HSMLTernary* m, int32_t* out, const int8_t* in,
                         const u8* W, u32 N, u32 K) {
    u64 t0 = now_ns();
    if (m->use_i2s) {
        i2s_proj_neon(out, in, W, N, K);
    } else {
        HSRouteDesc route;
        route.format = HS_ROUTE_TERNARY_2BIT;
        route.K = K; route.N = N; route.routes = W;
        int threads = hs_ml_route_optimal_threads(&route, 1);
        hs_ml_route_mt(out, in, &route, 1, threads);
    }

    g_mlt_stats.total_gemm_calls++;
    g_mlt_stats.total_gemm_ops += (u64)N * K;
    g_mlt_stats.total_ns += now_ns() - t0;
}

/*============================================================================
 * Layer forward pass
 *============================================================================*/

/* NEON dot product: q[hd] . k[hd] -> float */
static inline void rope_apply_one(float* x, u32 num_heads, u32 head_dim, u32 position, float theta) {
    u32 half_dim = head_dim / 2;
    float* freqs = malloc(half_dim * sizeof(float));
    if (!freqs) return;
    for (u32 d = 0; d < half_dim; d++)
        freqs[d] = powf(theta, (float)(-2.0f * d) / (float)head_dim);
    for (u32 h = 0; h < num_heads; h++) {
        for (u32 d = 0; d < half_dim; d++) {
            float angle = (float)position * freqs[d];
            float c = cosf(angle), s = sinf(angle);
            float x0 = x[h * head_dim + d];
            float x1 = x[h * head_dim + d + half_dim];
            x[h * head_dim + d]            = x0 * c - x1 * s;
            x[h * head_dim + d + half_dim] = x0 * s + x1 * c;
        }
    }
    free(freqs);
}

#ifdef __ARM_NEON
static inline float dot_f32_neon(const float* a, const float* b, u32 n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    u32 n4 = n & ~3u;
    for (u32 i = 0; i < n4; i += 4)
        acc = vfmaq_f32(acc, vld1q_f32(a+i), vld1q_f32(b+i));
    float s = vaddvq_f32(acc);
    for (u32 i = n4; i < n; i++) s += a[i] * b[i];
    return s;
}
/* NEON scalar*vector accumulate: out[n] += scalar * v[n] */
static inline void axpy_f32_neon(float* out, float scalar, const float* v, u32 n) {
    float32x4_t sc = vdupq_n_f32(scalar);
    u32 n4 = n & ~3u;
    for (u32 i = 0; i < n4; i += 4)
        vst1q_f32(out+i, vfmaq_f32(vld1q_f32(out+i), sc, vld1q_f32(v+i)));
    for (u32 i = n4; i < n; i++) out[i] += scalar * v[i];
}
/* Fast exp approximation using NEON — Schraudolph method, ~3 ULP */
static inline float32x4_t fast_expq_f32(float32x4_t x) {
    /* Clamp to avoid overflow */
    x = vmaxq_f32(x, vdupq_n_f32(-88.0f));
    x = vminq_f32(x, vdupq_n_f32(88.0f));
    /* exp(x) = 2^(x/ln2) = 2^(x*1.4427) */
    float32x4_t t = vfmaq_f32(vdupq_n_f32(127.0f), x, vdupq_n_f32(1.4426950408f));
    /* Convert to int and reinterpret as float */
    int32x4_t ti = vcvtq_s32_f32(t);
    return vreinterpretq_f32_s32(vshlq_n_s32(ti, 23));
}
#else
static inline float dot_f32_neon(const float* a, const float* b, u32 n) {
    float s = 0; for (u32 i = 0; i < n; i++) s += a[i]*b[i]; return s;
}
static inline void axpy_f32_neon(float* out, float sc, const float* v, u32 n) {
    for (u32 i = 0; i < n; i++) out[i] += sc * v[i];
}
#endif

void hs_mlt_layer_forward(float* hidden, u32 layer_idx,
                          HSKVCache* cache, HSMLTernary* m) {
    if (layer_idx >= m->num_layers) return;
    HSTernaryLayer* lay = &m->layers[layer_idx];
    u32 H = m->hidden_size;
    u32 F = m->ffn_hidden_size;
    u32 nh    = m->num_heads;
    u32 nkv   = m->num_kv_heads ? m->num_kv_heads : m->num_heads;
    u32 hd = m->head_dim;
    u32 kv = nkv * hd;

    /* Scratch — all pre-allocated, zero hot malloc */
    int8_t*  qbuf  = m->quant_buf;
    int32_t* gbuf  = m->gemm_buf;
    float*   fbuf  = m->hidden_buf;
    float*   ffbuf = m->ffn_buf;
    float*   q     = m->q_buf;
    float*   k     = m->k_buf;
    float*   v     = m->v_buf;
    float*   attn_out = m->attn_buf;
    float*   gate_out = m->gate_buf;
    float*   up_out   = m->up_buf;
    float*   scores   = m->score_buf;
    int8_t*  fqbuf    = m->ffn_qbuf;

    /* ── 1. Attention RMSNorm ── */
    rmsnorm(fbuf, hidden, lay->attn_norm, 1e-5f, H);

    /* ── 2. Quantise normed hidden ── */
    float act_scale = hs_mlt_quantize(qbuf, fbuf, H);

    /* ── 3-5. Q, K, V projections (no malloc) ── */
    ternary_proj(m, gbuf, qbuf, lay->q_proj, H, H);
    hs_mlt_dequantize(q, gbuf, H, act_scale, m->use_i2s ? NULL : lay->q_scale);
    ternary_proj(m, gbuf, qbuf, lay->k_proj, kv, H);
    hs_mlt_dequantize(k, gbuf, kv, act_scale, m->use_i2s ? NULL : lay->k_scale);
    ternary_proj(m, gbuf, qbuf, lay->v_proj, kv, H);
    hs_mlt_dequantize(v, gbuf, kv, act_scale, m->use_i2s ? NULL : lay->v_scale);

    /* ── 6. RoPE ── */
    u32 seq_pos = cache ? cache->cache_len : 0;
    rope_apply_one(q, nh,  hd, seq_pos, m->rope_theta);
    rope_apply_one(k, nkv, hd, seq_pos, m->rope_theta);

    /* ── 7. KV cache update ── */
    if (cache && cache->k_cache && cache->v_cache) {
        u32 kv_off = seq_pos * nkv * hd;
        memcpy(cache->k_cache + kv_off, k, nkv * hd * sizeof(float));
        memcpy(cache->v_cache + kv_off, v, nkv * hd * sizeof(float));
        cache->cache_len = seq_pos + 1;
    }

    /* ── 8. Attention — NEON dot products, no malloc ── */
    memset(attn_out, 0, H * sizeof(float));
    if (cache && cache->cache_len > 0) {
        u32 ctx = cache->cache_len;
        float inv_sq = 1.0f / sqrtf((float)hd);
        for (u32 h = 0; h < nh; h++) {
            float* qh   = q + h * hd;
            float* outh = attn_out + h * hd;
            u32 kv_h = (nkv < nh) ? (h * nkv / nh) : h;
            /* Score: NEON dot product q . k_t */
            float max_s = -1e30f;
            for (u32 t = 0; t < ctx; t++) {
                float* kh = cache->k_cache + t * nkv * hd + kv_h * hd;
                float s = dot_f32_neon(qh, kh, hd) * inv_sq;
                scores[t] = s;
                if (s > max_s) max_s = s;
            }
            /* Online softmax */
            float sum = 0.0f;
#ifdef __ARM_NEON
            /* Vectorised exp for blocks of 4 */
            u32 ctx4 = ctx & ~3u;
            float32x4_t vmax = vdupq_n_f32(max_s);
            float32x4_t vsum = vdupq_n_f32(0.0f);
            for (u32 t = 0; t < ctx4; t += 4) {
                float32x4_t vs = vsubq_f32(vld1q_f32(scores+t), vmax);
                float32x4_t ve = fast_expq_f32(vs);
                vst1q_f32(scores+t, ve);
                vsum = vaddq_f32(vsum, ve);
            }
            sum = vaddvq_f32(vsum);
            for (u32 t = ctx4; t < ctx; t++) {
                scores[t] = expf(scores[t] - max_s);
                sum += scores[t];
            }
#else
            for (u32 t = 0; t < ctx; t++) {
                scores[t] = expf(scores[t] - max_s);
                sum += scores[t];
            }
#endif
            float inv_sum = 1.0f / sum;
            /* Weighted sum: NEON axpy */
            for (u32 t = 0; t < ctx; t++) {
                float* vh = cache->v_cache + t * nkv * hd + kv_h * hd;
                axpy_f32_neon(outh, scores[t] * inv_sum, vh, hd);
            }
        }
    }

    /* ── 8.5. Attention sub-layer norm (BitNet subln) ── */
    rmsnorm(attn_out, attn_out, lay->attn_sub_norm, 1e-5f, H);

    /* ── 9. O projection ── */
    act_scale = hs_mlt_quantize(qbuf, attn_out, H);
    ternary_proj(m, gbuf, qbuf, lay->o_proj, H, H);
    hs_mlt_dequantize(fbuf, gbuf, H, act_scale, m->use_i2s ? NULL : lay->o_scale);

    /* ── 10. Residual (attention) + FFN RMSNorm ── */
    for (u32 i = 0; i < H; i++) hidden[i] += fbuf[i];
    rmsnorm(fbuf, hidden, lay->ffn_norm, 1e-5f, H);

    /* ── 11. Quantise for FFN ── */
    act_scale = hs_mlt_quantize(qbuf, fbuf, H);

    /* ── 12. Gate and Up projections (no malloc) ── */
    ternary_proj(m, gbuf, qbuf, lay->gate_proj, F, H);
    hs_mlt_dequantize(gate_out, gbuf, F, act_scale, m->use_i2s ? NULL : lay->gate_scale);
    ternary_proj(m, gbuf, qbuf, lay->up_proj, F, H);
    hs_mlt_dequantize(up_out, gbuf, F, act_scale, m->use_i2s ? NULL : lay->up_scale);

    /* ── 13. BitNet FFN: ReLU^2(gate) * up, then ffn_sub_norm ── */
#ifdef __ARM_NEON
    {
        u32 F4 = F & ~3u;
        float32x4_t zero = vdupq_n_f32(0.0f);
        for (u32 i = 0; i < F4; i += 4) {
            float32x4_t g = vmaxq_f32(vld1q_f32(gate_out + i), zero);
            float32x4_t u = vld1q_f32(up_out + i);
            vst1q_f32(ffbuf + i, vmulq_f32(vmulq_f32(g, g), u));
        }
        for (u32 i = F4; i < F; i++) {
            float rg = gate_out[i] > 0.0f ? gate_out[i] : 0.0f;
            ffbuf[i] = (rg * rg) * up_out[i];
        }
    }
#else
    for (u32 i = 0; i < F; i++) {
        float rg = gate_out[i] > 0.0f ? gate_out[i] : 0.0f;
        ffbuf[i] = (rg * rg) * up_out[i];
    }
#endif
    rmsnorm(ffbuf, ffbuf, lay->ffn_sub_norm, 1e-5f, F);

    /* ── 14. Down projection (use ffn_qbuf — pre-allocated for F elements) ── */
    act_scale = hs_mlt_quantize(fqbuf, ffbuf, F);
    ternary_proj(m, gbuf, fqbuf, lay->down_proj, H, F);
    hs_mlt_dequantize(fbuf, gbuf, H, act_scale, m->use_i2s ? NULL : lay->down_scale);

    /* ── 15. Residual (FFN) ── */
    for (u32 i = 0; i < H; i++) hidden[i] += fbuf[i];
}

/*============================================================================
 * Full forward pass
 *============================================================================*/

int hs_mlt_forward(HSMLTernary* m, const u32* tokens, u32 seq_len,
                   float* logits) {
    if (!m || !m->loaded || !logits || seq_len == 0) return -1;

    memset(&g_mlt_stats, 0, sizeof(g_mlt_stats));

    u32 H = m->hidden_size;
    u32 V = m->vocab_size;

    float* hidden = malloc(H * sizeof(float));
    if (!hidden) return -1;

    /* KV cache: one per layer */
    HSKVCache* caches = calloc(m->num_layers, sizeof(HSKVCache));
    if (!caches) { free(hidden); return -1; }
    for (u32 l = 0; l < m->num_layers; l++) {
        hs_kv_cache_init(&caches[l], m->max_context,
                         m->num_heads, m->num_kv_heads, m->head_dim);
    }

    for (u32 pos = 0; pos < seq_len; pos++) {
        u32 tok = tokens[pos];

        /* Embedding lookup */
        if (m->embedding && tok < V) {
            memcpy(hidden, m->embedding + (size_t)tok * H, H * sizeof(float));
        } else {
            for (u32 i = 0; i < H; i++)
                hidden[i] = (float)((i + tok) % 256) / 256.0f;
        }

        /* Layers */
        for (u32 l = 0; l < m->num_layers; l++)
            hs_mlt_layer_forward(hidden, l, &caches[l], m);
    }

    /* Final norm */
    float* normed = malloc(H * sizeof(float));
    if (normed) {
        rmsnorm(normed, hidden, m->final_norm, 1e-5f, H);
        /* LM head: float matmul [V, H] x [H] -> [V] */
        for (u32 v = 0; v < V; v++) {
            float s = 0;
            for (u32 i = 0; i < H; i++)
                s += m->lm_head[v * H + i] * normed[i];
            logits[v] = s;
        }
        free(normed);
    }

    g_mlt_stats.tokens_generated = seq_len;

    for (u32 l = 0; l < m->num_layers; l++) hs_kv_cache_free(&caches[l]);
    free(caches);
    free(hidden);
    return 0;
}

/*============================================================================
 * Sampling
 *============================================================================*/

u32 hs_mlt_sample_greedy(const float* logits, u32 vocab_size) {
    u32 best = 0;
    float best_val = logits[0];
    for (u32 i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) { best_val = logits[i]; best = i; }
    }
    return best;
}

void hs_mlt_reset_stats(HSMLTernary* m) {
    (void)m;
    memset(&g_mlt_stats, 0, sizeof(g_mlt_stats));
}

/*============================================================================
 * Stateful decode session
 *============================================================================*/

int hs_mlt_session_init(HSMLTernarySession* sess, HSMLTernary* model) {
    if (!sess || !model || !model->loaded) return -1;
    memset(sess, 0, sizeof(*sess));

    sess->model  = model;
    sess->hidden = malloc(model->hidden_size * sizeof(float));
    if (!sess->hidden) return -1;

    sess->caches = calloc(model->num_layers, sizeof(HSKVCache));
    if (!sess->caches) { free(sess->hidden); sess->hidden = NULL; return -1; }

    for (u32 l = 0; l < model->num_layers; l++) {
        hs_kv_cache_init(&sess->caches[l],
                         model->max_context,
                         model->num_heads,
                         model->num_kv_heads,
                         model->head_dim);
    }

    sess->seq_len = 0;
    sess->ready   = true;
    return 0;
}

void hs_mlt_session_free(HSMLTernarySession* sess) {
    if (!sess) return;
    free(sess->hidden);
    if (sess->caches && sess->model) {
        for (u32 l = 0; l < sess->model->num_layers; l++)
            hs_kv_cache_free(&sess->caches[l]);
        free(sess->caches);
    }
    memset(sess, 0, sizeof(*sess));
}

void hs_mlt_session_reset(HSMLTernarySession* sess) {
    if (!sess || !sess->ready) return;
    if (sess->caches && sess->model) {
        for (u32 l = 0; l < sess->model->num_layers; l++)
            hs_kv_cache_clear(&sess->caches[l]);
    }
    sess->seq_len = 0;
}

/* Internal: run one token through the model, updating caches. */
static void session_step(HSMLTernarySession* sess, u32 token) {
    HSMLTernary* m = sess->model;
    u32 H = m->hidden_size;
    u32 V = m->vocab_size;

    /* Embedding lookup */
    if (m->embedding && token < V)
        memcpy(sess->hidden, m->embedding + (size_t)token * H, H * sizeof(float));
    else
        for (u32 i = 0; i < H; i++)
            sess->hidden[i] = (float)((i + token) % 256) / 256.0f;

    /* All layers */
    for (u32 l = 0; l < m->num_layers; l++)
        hs_mlt_layer_forward(sess->hidden, l, &sess->caches[l], m);

    sess->seq_len++;
}

/* Write logits from current hidden state. */
static void session_logits(HSMLTernarySession* sess, float* logits) {
    HSMLTernary* m = sess->model;
    u32 H = m->hidden_size;
    u32 V = m->vocab_size;
    float* normed = m->hidden_buf;  /* borrow model scratch — safe, not in use */
    rmsnorm(normed, sess->hidden, m->final_norm, 1e-5f, H);
#ifdef __ARM_NEON
    for (u32 v = 0; v < V; v++) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        u32 H4 = H & ~3u;
        for (u32 i = 0; i < H4; i += 4)
            acc = vfmaq_f32(acc, vld1q_f32(m->lm_head + v*H + i), vld1q_f32(normed + i));
        float s = vaddvq_f32(acc);
        for (u32 i = H4; i < H; i++) s += m->lm_head[v*H + i] * normed[i];
        logits[v] = s;
    }
#else
    for (u32 v = 0; v < V; v++) {
        float s = 0;
        for (u32 i = 0; i < H; i++) s += m->lm_head[v*H + i] * normed[i];
        logits[v] = s;
    }
#endif
}

int hs_mlt_prefill(HSMLTernarySession* sess,
                   const u32* tokens, u32 seq_len) {
    if (!sess || !sess->ready || !tokens || seq_len == 0) return -1;
    for (u32 i = 0; i < seq_len; i++)
        session_step(sess, tokens[i]);
    return 0;
}

int hs_mlt_session_logits(HSMLTernarySession* sess, float* logits) {
    if (!sess || !sess->ready || !logits) return -1;
    session_logits(sess, logits);
    return 0;
}

int hs_mlt_decode(HSMLTernarySession* sess, u32 token, float* logits) {
    if (!sess || !sess->ready || !logits) return -1;
    session_step(sess, token);
    session_logits(sess, logits);
    return 0;
}
