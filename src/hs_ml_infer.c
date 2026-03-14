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
            free(lay->ffn_norm);
        }
        free(m->layers);
    }

    free(m->quant_buf);
    free(m->gemm_buf);
    free(m->hidden_buf);
    free(m->ffn_buf);

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
    m->head_dim        = hidden_size / num_heads;
    m->ffn_hidden_size = ffn_hidden_size;
    m->max_context     = max_context;

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
        lay->ffn_norm  = malloc(H * sizeof(float));
        if (!lay->attn_norm || !lay->ffn_norm) goto fail;

        for (u32 i = 0; i < H; i++) { lay->attn_norm[i] = 1.0f; lay->ffn_norm[i] = 1.0f; }
    }

    /* Scratch buffers */
    u32 max_dim = (hidden_size > ffn_hidden_size) ? hidden_size : ffn_hidden_size;
    m->quant_buf  = aligned_alloc(64, max_dim * sizeof(int8_t));
    m->gemm_buf   = aligned_alloc(64, max_dim * sizeof(int32_t));
    m->hidden_buf = malloc(hidden_size * sizeof(float));
    m->ffn_buf    = malloc(ffn_hidden_size * sizeof(float));
    if (!m->quant_buf || !m->gemm_buf || !m->hidden_buf || !m->ffn_buf)
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

static void ternary_proj(int32_t* out, const int8_t* in,
                         const u8* W, u32 N, u32 K) {
    u64 t0 = now_ns();
    /* Use routing abstraction — thread count auto-selected */
    HSRouteDesc route;
    route.format = HS_ROUTE_TERNARY_2BIT;
    route.K = K; route.N = N; route.routes = W;
    int threads = hs_ml_route_optimal_threads(&route, 1);
    hs_ml_route_mt(out, in, &route, 1, threads);

    g_mlt_stats.total_gemm_calls++;
    g_mlt_stats.total_gemm_ops += (u64)N * K;
    g_mlt_stats.total_ns += now_ns() - t0;
}

/*============================================================================
 * Layer forward pass
 *============================================================================*/

void hs_mlt_layer_forward(float* hidden, u32 layer_idx,
                          HSKVCache* cache, HSMLTernary* m) {
    if (layer_idx >= m->num_layers) return;
    HSTernaryLayer* lay = &m->layers[layer_idx];
    u32 H = m->hidden_size;
    u32 F = m->ffn_hidden_size;
    u32 nh = m->num_heads;
    u32 hd = m->head_dim;

    /* Scratch */
    int8_t*  qbuf  = m->quant_buf;
    int32_t* gbuf  = m->gemm_buf;
    float*   fbuf  = m->hidden_buf;  /* reused for Q, K, V, O, FFN outputs */
    float*   ffbuf = m->ffn_buf;

    /* ── 1. Attention RMSNorm ── */
    rmsnorm(fbuf, hidden, lay->attn_norm, 1e-5f, H);

    /* ── 2. Quantise normed hidden ── */
    float act_scale = hs_mlt_quantize(qbuf, fbuf, H);

    /* ── 3. Q projection ── */
    float* q = malloc(H * sizeof(float));
    if (!q) return;
    ternary_proj(gbuf, qbuf, lay->q_proj, H, H);
    hs_mlt_dequantize(q, gbuf, H, act_scale, lay->q_scale);

    /* ── 4. K projection ── */
    float* k = malloc(H * sizeof(float));
    if (!k) { free(q); return; }
    ternary_proj(gbuf, qbuf, lay->k_proj, H, H);
    hs_mlt_dequantize(k, gbuf, H, act_scale, lay->k_scale);

    /* ── 5. V projection ── */
    float* v = malloc(H * sizeof(float));
    if (!v) { free(q); free(k); return; }
    ternary_proj(gbuf, qbuf, lay->v_proj, H, H);
    hs_mlt_dequantize(v, gbuf, H, act_scale, lay->v_scale);

    /* ── 6. RoPE ── */
    u32 seq_pos = cache ? cache->cache_len : 0;
    hs_rope_apply(q, k, nh, hd, seq_pos, 10000.0f);

    /* ── 7. KV cache update ── */
    if (cache && cache->k_cache && cache->v_cache) {
        u32 kv_off = seq_pos * nh * hd;
        memcpy(cache->k_cache + kv_off, k, H * sizeof(float));
        memcpy(cache->v_cache + kv_off, v, H * sizeof(float));
        cache->cache_len = seq_pos + 1;
    }

    /* ── 8. Attention (float32, per-head) ── */
    float* attn_out = calloc(H, sizeof(float));
    if (!attn_out) { free(q); free(k); free(v); return; }

    if (cache && cache->cache_len > 0) {
        u32 ctx = cache->cache_len;
        float scale = 1.0f / sqrtf((float)hd);
        float* scores = malloc(ctx * sizeof(float));
        if (scores) {
            for (u32 h = 0; h < nh; h++) {
                float* qh = q + h * hd;
                float* out_h = attn_out + h * hd;
                /* Scores */
                float max_s = -1e30f;
                for (u32 t = 0; t < ctx; t++) {
                    float* kh = cache->k_cache + t * nh * hd + h * hd;
                    float s = 0;
                    for (u32 d = 0; d < hd; d++) s += qh[d] * kh[d];
                    scores[t] = s * scale;
                    if (scores[t] > max_s) max_s = scores[t];
                }
                /* Softmax */
                float sum = 0;
                for (u32 t = 0; t < ctx; t++) {
                    scores[t] = expf(scores[t] - max_s);
                    sum += scores[t];
                }
                for (u32 t = 0; t < ctx; t++) scores[t] /= sum;
                /* Weighted sum */
                for (u32 d = 0; d < hd; d++) {
                    float val = 0;
                    for (u32 t = 0; t < ctx; t++) {
                        float* vh = cache->v_cache + t * nh * hd + h * hd;
                        val += scores[t] * vh[d];
                    }
                    out_h[d] = val;
                }
            }
            free(scores);
        }
    }
    free(q); free(k); free(v);

    /* ── 9. O projection ── */
    act_scale = hs_mlt_quantize(qbuf, attn_out, H);
    ternary_proj(gbuf, qbuf, lay->o_proj, H, H);
    hs_mlt_dequantize(fbuf, gbuf, H, act_scale, lay->o_scale);
    free(attn_out);

    /* ── 10. Residual (attention) + FFN RMSNorm ── */
    for (u32 i = 0; i < H; i++) hidden[i] += fbuf[i];
    rmsnorm(fbuf, hidden, lay->ffn_norm, 1e-5f, H);

    /* ── 11. Quantise for FFN ── */
    act_scale = hs_mlt_quantize(qbuf, fbuf, H);

    /* ── 12. Gate and Up projections ── */
    float* gate_out = malloc(F * sizeof(float));
    float* up_out   = malloc(F * sizeof(float));
    if (!gate_out || !up_out) {
        free(gate_out); free(up_out); return;
    }

    ternary_proj(gbuf, qbuf, lay->gate_proj, F, H);
    hs_mlt_dequantize(gate_out, gbuf, F, act_scale, lay->gate_scale);

    ternary_proj(gbuf, qbuf, lay->up_proj, F, H);
    hs_mlt_dequantize(up_out, gbuf, F, act_scale, lay->up_scale);

    /* ── 13. SwiGLU ── */
    for (u32 i = 0; i < F; i++) {
        float silu = gate_out[i] / (1.0f + expf(-gate_out[i]));
        ffbuf[i] = silu * up_out[i];
    }
    free(gate_out); free(up_out);

    /* ── 14. Down projection ── */
    act_scale = hs_mlt_quantize(qbuf, ffbuf, F);
    ternary_proj(gbuf, qbuf, lay->down_proj, H, F);
    hs_mlt_dequantize(fbuf, gbuf, H, act_scale, lay->down_scale);

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
                         m->num_heads, m->num_heads, m->head_dim);
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
