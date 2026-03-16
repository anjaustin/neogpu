/*
 * profile_step.c — Per-section decode-step profiler.
 *
 * Loads the real model, prefills a short prompt to populate the KV cache,
 * then runs ONE decode step and reports wall-time for every major section
 * across all 30 layers, plus the lm_head pass.
 *
 * Sections timed:
 *   attn_norm      — RMSNorm before attention (per layer)
 *   q_proj         — Q projection [H×H]
 *   k_proj         — K projection [kv×H]
 *   v_proj         — V projection [kv×H]
 *   rope           — Rotary embedding apply
 *   kv_cache       — KV cache write (memcpy)
 *   attn_score     — Q·K^T dot products + softmax
 *   attn_agg       — weighted V sum (axpy)
 *   attn_sub_norm  — BitNet subln after attention
 *   o_proj         — O projection [H×H]
 *   residual_attn  — hidden += attn_out
 *   ffn_norm       — RMSNorm before FFN
 *   gate_proj      — Gate projection [F×H]
 *   up_proj        — Up projection [F×H]
 *   relu2_mul      — ReLU^2(gate) * up
 *   ffn_sub_norm   — BitNet subln after FFN activation
 *   down_proj      — Down projection [H×F]
 *   residual_ffn   — hidden += ffn_out
 *   lm_head        — Final norm + F16 logit matmul [V×H]
 *
 * Usage:
 *   ./tests/profile_step models/bitnet-2b4t-i2s.gguf [norms_v2.bin]
 *
 * Build (on Pi):
 *   gcc -O3 -march=armv8-a+simd+fp16 -mtune=cortex-a72 -ffast-math \
 *       -funroll-loops -DNDEBUG -Iinclude \
 *       tests/profile_step.c \
 *       src/hs_ml_ternary_neon.c src/hs_ml_loader_ternary.c \
 *       src/hs_ml_infer.c src/hs_ml_routing.c \
 *       src/hs_ml_ternary_mt.c src/hs_ml_binary.c src/hs_ml.c \
 *       -lm -lpthread -o tests/profile_step
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#include "hs_ml_infer.h"

/* ------------------------------------------------------------------ */
/* Timing                                                              */
/* ------------------------------------------------------------------ */

static inline uint64_t ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* ------------------------------------------------------------------ */
/* Section accumulators                                                */
/* ------------------------------------------------------------------ */

typedef struct {
    uint64_t attn_norm;
    uint64_t q_proj;
    uint64_t k_proj;
    uint64_t v_proj;
    uint64_t rope;
    uint64_t kv_cache;
    uint64_t attn_score;
    uint64_t attn_agg;
    uint64_t attn_sub_norm;
    uint64_t o_proj;
    uint64_t residual_attn;
    uint64_t ffn_norm;
    uint64_t gate_proj;
    uint64_t up_proj;
    uint64_t relu2_mul;
    uint64_t ffn_sub_norm;
    uint64_t down_proj;
    uint64_t residual_ffn;
    uint64_t lm_head;
    uint64_t total_step;
} ProfAccum;

/* ------------------------------------------------------------------ */
/* Primitives (copied from hs_ml_infer.c — forward declarations not   */
/* exported, so we re-implement the same logic inline here)           */
/* ------------------------------------------------------------------ */

static inline float fp16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t f;
    if (exp == 0)       f = sign;
    else if (exp == 31) f = sign | 0x7F800000u | (mant << 13);
    else                f = sign | ((exp + 112u) << 23) | (mant << 13);
    float out; memcpy(&out, &f, 4); return out;
}

static void rmsnorm_local(float *out, const float *in, const float *w,
                          float eps, uint32_t n) {
    float ss = 0.0f;
    for (uint32_t i = 0; i < n; i++) ss += in[i] * in[i];
    float scale = 1.0f / sqrtf(ss / (float)n + eps);
    for (uint32_t i = 0; i < n; i++) out[i] = in[i] * scale * w[i];
}

static float dot_f32(const float *a, const float *b, uint32_t n) {
#ifdef __ARM_NEON
    float32x4_t acc = vdupq_n_f32(0.0f);
    uint32_t n4 = n & ~3u;
    for (uint32_t i = 0; i < n4; i += 4)
        acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    float s = vaddvq_f32(acc);
    for (uint32_t i = n4; i < n; i++) s += a[i] * b[i];
    return s;
#else
    float s = 0.0f;
    for (uint32_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
#endif
}

static void axpy_f32(float *y, float a, const float *x, uint32_t n) {
#ifdef __ARM_NEON
    float32x4_t va = vdupq_n_f32(a);
    uint32_t n4 = n & ~3u;
    for (uint32_t i = 0; i < n4; i += 4)
        vst1q_f32(y + i, vfmaq_f32(vld1q_f32(y + i), va, vld1q_f32(x + i)));
    for (uint32_t i = n4; i < n; i++) y[i] += a * x[i];
#else
    for (uint32_t i = 0; i < n; i++) y[i] += a * x[i];
#endif
}

static void rope_apply(float *x, uint32_t num_heads, uint32_t head_dim,
                       uint32_t pos, float theta) {
    for (uint32_t h = 0; h < num_heads; h++) {
        float *xh = x + h * head_dim;
        for (uint32_t i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
            float angle = (float)pos * freq;
            float cs = cosf(angle), sn = sinf(angle);
            float x0 = xh[i], x1 = xh[i + head_dim / 2];
            xh[i]              = x0 * cs - x1 * sn;
            xh[i + head_dim/2] = x0 * sn + x1 * cs;
        }
    }
}

/* Wrapper: ternary f32 proj (calls the kernel) */
extern void hs_ml_ternary_f32_proj(float *out, const float *in,
                                    const uint8_t *W, uint32_t N, uint32_t K);

/* weight_scale is the per-tensor scale array; [0] holds the scalar value */
static void f32_proj_scaled(float *out, const float *in, const uint8_t *W,
                             const float *weight_scale, uint32_t N, uint32_t K) {
    hs_ml_ternary_f32_proj(out, in, W, N, K);
    if (weight_scale && weight_scale[0] != 1.0f) {
        float s = weight_scale[0];
        for (uint32_t i = 0; i < N; i++) out[i] *= s;
    }
}

/* ------------------------------------------------------------------ */
/* Instrumented single decode step                                      */
/* ------------------------------------------------------------------ */

static void profile_decode_step(HSMLTernary *m, HSKVCache *caches,
                                 float *hidden, float *logits,
                                 uint32_t seq_pos, ProfAccum *acc) {
    uint64_t t0, t1;
    uint64_t step_start = ns_now();

    uint32_t H  = m->hidden_size;
    uint32_t F  = m->ffn_hidden_size;
    uint32_t nh  = m->num_heads;
    uint32_t nkv = m->num_kv_heads ? m->num_kv_heads : m->num_heads;
    uint32_t hd  = m->head_dim;
    uint32_t kv  = nkv * hd;

    /* Scratch buffers — allocated once, reused across layers */
    float *fbuf     = malloc(H * sizeof(float));
    float *ffbuf    = malloc(F * sizeof(float));
    float *q        = malloc(H * sizeof(float));
    float *k        = malloc(kv * sizeof(float));
    float *v        = malloc(kv * sizeof(float));
    float *attn_out = calloc(H, sizeof(float));
    float *gate_out = malloc(F * sizeof(float));
    float *up_out   = malloc(F * sizeof(float));
    float *scores   = malloc(m->max_context * sizeof(float));
    if (!fbuf||!ffbuf||!q||!k||!v||!attn_out||!gate_out||!up_out||!scores) {
        fprintf(stderr, "OOM in profile_decode_step\n"); return;
    }

    for (uint32_t l = 0; l < m->num_layers; l++) {
        HSTernaryLayer *lay = &m->layers[l];
        HSKVCache *cache = &caches[l];

        /* ── 1. Attention RMSNorm ── */
        t0 = ns_now();
        rmsnorm_local(fbuf, hidden, lay->attn_norm, 1e-5f, H);
        acc->attn_norm += ns_now() - t0;

        /* ── 2. Q projection ── */
        t0 = ns_now();
        f32_proj_scaled(q, fbuf, lay->q_proj, lay->q_scale, H, H);
        acc->q_proj += ns_now() - t0;

        /* ── 3. K projection ── */
        t0 = ns_now();
        f32_proj_scaled(k, fbuf, lay->k_proj, lay->k_scale, kv, H);
        acc->k_proj += ns_now() - t0;

        /* ── 4. V projection ── */
        t0 = ns_now();
        f32_proj_scaled(v, fbuf, lay->v_proj, lay->v_scale, kv, H);
        acc->v_proj += ns_now() - t0;

        /* ── 5. RoPE ── */
        t0 = ns_now();
        rope_apply(q, nh,  hd, seq_pos, m->rope_theta);
        rope_apply(k, nkv, hd, seq_pos, m->rope_theta);
        acc->rope += ns_now() - t0;

        /* ── 6. KV cache write ── */
        t0 = ns_now();
        if (cache->k_cache && cache->v_cache) {
            uint32_t kv_off = seq_pos * nkv * hd;
            memcpy(cache->k_cache + kv_off, k, nkv * hd * sizeof(float));
            memcpy(cache->v_cache + kv_off, v, nkv * hd * sizeof(float));
            /* Note: don't advance cache_len here; caller already set it */
        }
        acc->kv_cache += ns_now() - t0;

        /* ── 7. Attention scores (Q·K^T + softmax) ── */
        t0 = ns_now();
        memset(attn_out, 0, H * sizeof(float));
        {
            uint32_t ctx = cache->cache_len;  /* includes current position */
            float inv_sq = 1.0f / sqrtf((float)hd);
            for (uint32_t h = 0; h < nh; h++) {
                float *qh = q + h * hd;
                uint32_t kv_h = (nkv < nh) ? (h * nkv / nh) : h;
                float max_s = -1e30f;
                for (uint32_t t2 = 0; t2 < ctx; t2++) {
                    float *kh = cache->k_cache + t2 * nkv * hd + kv_h * hd;
                    float s = dot_f32(qh, kh, hd) * inv_sq;
                    scores[t2] = s;
                    if (s > max_s) max_s = s;
                }
                float sum = 0.0f;
                for (uint32_t t2 = 0; t2 < ctx; t2++) {
                    scores[t2] = expf(scores[t2] - max_s);
                    sum += scores[t2];
                }
                float inv_sum = 1.0f / sum;
                for (uint32_t t2 = 0; t2 < ctx; t2++) scores[t2] *= inv_sum;

                /* store normalised scores back — aggregation timed separately */
                /* (no-op here: scores[] already normalised in place) */
                (void)kv_h; /* used above */

                /* aggregation — timed below, so we need the scores intact */
                /* We'll split: scores pass above, agg pass below.          */
                /* Re-use scores[] which now holds softmax weights.         */
                t1 = ns_now();
                acc->attn_score += t1 - t0;
                t0 = t1;

                /* ── 8. Attention aggregation (weighted V sum) ── */
                float *outh = attn_out + h * hd;
                for (uint32_t t2 = 0; t2 < ctx; t2++) {
                    float *vh = cache->v_cache + t2 * nkv * hd + kv_h * hd;
                    axpy_f32(outh, scores[t2], vh, hd);
                }
                t1 = ns_now();
                acc->attn_agg += t1 - t0;
                t0 = t1;
            }
        }

        /* ── 8.5. Attention sub-layer norm ── */
        t0 = ns_now();
        rmsnorm_local(attn_out, attn_out, lay->attn_sub_norm, 1e-5f, H);
        acc->attn_sub_norm += ns_now() - t0;

        /* ── 9. O projection ── */
        t0 = ns_now();
        f32_proj_scaled(fbuf, attn_out, lay->o_proj, lay->o_scale, H, H);
        acc->o_proj += ns_now() - t0;

        /* ── 10. Residual (attention) ── */
        t0 = ns_now();
        for (uint32_t i = 0; i < H; i++) hidden[i] += fbuf[i];
        acc->residual_attn += ns_now() - t0;

        /* ── 11. FFN RMSNorm ── */
        t0 = ns_now();
        rmsnorm_local(fbuf, hidden, lay->ffn_norm, 1e-5f, H);
        acc->ffn_norm += ns_now() - t0;

        /* ── 12. Gate projection ── */
        t0 = ns_now();
        f32_proj_scaled(gate_out, fbuf, lay->gate_proj, lay->gate_scale, F, H);
        acc->gate_proj += ns_now() - t0;

        /* ── 13. Up projection ── */
        t0 = ns_now();
        f32_proj_scaled(up_out, fbuf, lay->up_proj, lay->up_scale, F, H);
        acc->up_proj += ns_now() - t0;

        /* ── 14. ReLU^2(gate) * up ── */
        t0 = ns_now();
#ifdef __ARM_NEON
        {
            uint32_t F4 = F & ~3u;
            float32x4_t zero = vdupq_n_f32(0.0f);
            for (uint32_t i = 0; i < F4; i += 4) {
                float32x4_t g = vmaxq_f32(vld1q_f32(gate_out + i), zero);
                float32x4_t u = vld1q_f32(up_out + i);
                vst1q_f32(ffbuf + i, vmulq_f32(vmulq_f32(g, g), u));
            }
            for (uint32_t i = F4; i < F; i++) {
                float rg = gate_out[i] > 0.0f ? gate_out[i] : 0.0f;
                ffbuf[i] = rg * rg * up_out[i];
            }
        }
#else
        for (uint32_t i = 0; i < F; i++) {
            float rg = gate_out[i] > 0.0f ? gate_out[i] : 0.0f;
            ffbuf[i] = rg * rg * up_out[i];
        }
#endif
        acc->relu2_mul += ns_now() - t0;

        /* ── 15. FFN sub-layer norm ── */
        t0 = ns_now();
        rmsnorm_local(ffbuf, ffbuf, lay->ffn_sub_norm, 1e-5f, F);
        acc->ffn_sub_norm += ns_now() - t0;

        /* ── 16. Down projection ── */
        t0 = ns_now();
        f32_proj_scaled(fbuf, ffbuf, lay->down_proj, lay->down_scale, H, F);
        acc->down_proj += ns_now() - t0;

        /* ── 17. Residual (FFN) ── */
        t0 = ns_now();
        for (uint32_t i = 0; i < H; i++) hidden[i] += fbuf[i];
        acc->residual_ffn += ns_now() - t0;
    }

    /* ── LM head: final norm + F16 matmul ── */
    t0 = ns_now();
    {
        uint32_t V = m->vocab_size;
        float *normed = malloc(H * sizeof(float));
        if (normed) {
            rmsnorm_local(normed, hidden, m->final_norm, 1e-5f, H);
            for (uint32_t vv = 0; vv < V; vv++) {
                float s = 0.0f;
                const uint16_t *row = m->lm_head_f16 + (size_t)vv * H;
                for (uint32_t i = 0; i < H; i++)
                    s += fp16_to_f32(row[i]) * normed[i];
                if (logits) logits[vv] = s;
            }
            free(normed);
        }
    }
    acc->lm_head += ns_now() - t0;

    acc->total_step += ns_now() - step_start;

    free(scores); free(up_out); free(gate_out);
    free(attn_out); free(v); free(k); free(q);
    free(ffbuf); free(fbuf);
}

/* ------------------------------------------------------------------ */
/* Reporting                                                           */
/* ------------------------------------------------------------------ */

static void print_row(const char *name, uint64_t ns, uint64_t total_ns,
                      double ops, const char *unit) {
    double ms   = (double)ns / 1e6;
    double pct  = (double)ns / (double)total_ns * 100.0;
    if (ops > 0)
        printf("  %-18s  %7.1f ms  %5.1f%%  %6.2f %s\n",
               name, ms, pct, ops, unit);
    else
        printf("  %-18s  %7.1f ms  %5.1f%%\n", name, ms, pct);
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [norms.bin]\n", argv[0]);
        return 1;
    }
    const char *model_path = argv[1];
    const char *norms_path = argc > 2 ? argv[2] : NULL;

    HSMLTernary m;
    hs_mlt_init(&m);
    if (hs_mlt_load_gguf(&m, model_path) != 0) {
        fprintf(stderr, "Failed to load model\n"); return 1;
    }
    if (norms_path) {
        if (hs_mlt_load_norms_sidecar(&m, norms_path) != 0)
            fprintf(stderr, "warning: norms sidecar not loaded\n");
    }

    uint32_t H  = m.hidden_size;
    uint32_t nh  = m.num_heads;
    uint32_t nkv = m.num_kv_heads ? m.num_kv_heads : m.num_heads;
    uint32_t hd  = m.head_dim;
    uint32_t F   = m.ffn_hidden_size;
    uint32_t V   = m.vocab_size;
    uint32_t L   = m.num_layers;

    printf("\nModel: H=%u F=%u nh=%u nkv=%u hd=%u V=%u L=%u\n\n",
           H, F, nh, nkv, hd, V, L);

    /* Allocate KV caches — prefill 8 tokens so attention has real work */
    uint32_t prefill_len = 8;
    HSKVCache *caches = calloc(L, sizeof(HSKVCache));
    if (!caches) { fprintf(stderr, "OOM\n"); return 1; }
    for (uint32_t l = 0; l < L; l++)
        hs_kv_cache_init(&caches[l], m.max_context, nh, nkv, hd);

    /* Build a small hidden state from embedding[1] as a realistic vector */
    float *hidden = malloc(H * sizeof(float));
    float *logits = malloc(V * sizeof(float));
    if (!hidden || !logits) { fprintf(stderr, "OOM\n"); return 1; }
    if (m.embedding)
        memcpy(hidden, m.embedding + (size_t)1 * H, H * sizeof(float));
    else
        for (uint32_t i = 0; i < H; i++) hidden[i] = 0.01f * (float)(i % 64);

    /* Fake-prefill: write synthetic K/V entries so decode-step attention
     * has a realistic ctx of prefill_len positions to score against.     */
    for (uint32_t l = 0; l < L; l++) {
        for (uint32_t pos = 0; pos < prefill_len; pos++) {
            uint32_t kv_off = pos * nkv * hd;
            for (uint32_t i = 0; i < nkv * hd; i++) {
                caches[l].k_cache[kv_off + i] = 0.01f * (float)((i + pos) % 64);
                caches[l].v_cache[kv_off + i] = 0.01f * (float)((i * 2 + pos) % 64);
            }
        }
        caches[l].cache_len = prefill_len;
    }

    /* ── Warm-up run (not counted) ── */
    printf("Warming up...\n");
    {
        ProfAccum dummy; memset(&dummy, 0, sizeof(dummy));
        float *h2 = malloc(H * sizeof(float));
        if (h2) {
            memcpy(h2, hidden, H * sizeof(float));
            profile_decode_step(&m, caches, h2, NULL, prefill_len, &dummy);
            free(h2);
        }
        /* Reset cache_len back to prefill_len after warm-up wrote pos */
        for (uint32_t l = 0; l < L; l++)
            caches[l].cache_len = prefill_len;
    }

    /* ── Measured run ── */
    printf("Profiling decode step (ctx=%u)...\n\n", prefill_len + 1);
    ProfAccum acc; memset(&acc, 0, sizeof(acc));
    profile_decode_step(&m, caches, hidden, logits, prefill_len, &acc);

    /* ── Report ── */
    uint64_t total = acc.total_step;

    /* GFLOP/GOPS estimates (per 30-layer decode step):
     *   proj [N×K]:  2*N*K ops (one MAC per weight, weight is ternary so
     *                technically +/-/0 but counts as 1 multiply-add)
     *   attn_score: 2*nh*ctx*hd  (dot products)
     *   attn_agg:   2*nh*ctx*hd  (axpy)
     *   lm_head:    2*V*H        (F16 MAC)
     * All scaled by L=30 layers where applicable.
     */
    double kv_dim = (double)(nkv * hd);
    double qop  = 2.0 * H   * H   * L / 1e9;  /* q_proj  GOPS */
    double kop  = 2.0 * kv_dim * H * L / 1e9;  /* k_proj  GOPS */
    double vop  = 2.0 * kv_dim * H * L / 1e9;  /* v_proj  GOPS */
    double oop  = 2.0 * H   * H   * L / 1e9;  /* o_proj  GOPS */
    double gop  = 2.0 * F   * H   * L / 1e9;  /* gate    GOPS */
    double uop  = 2.0 * F   * H   * L / 1e9;  /* up      GOPS */
    double dop  = 2.0 * H   * F   * L / 1e9;  /* down    GOPS */
    double sop  = 2.0 * nh  * (prefill_len+1) * hd * L / 1e9; /* score */
    double aop  = 2.0 * nh  * (prefill_len+1) * hd * L / 1e9; /* agg   */
    double lop  = 2.0 * V   * H   / 1e9;       /* lm_head GOPS */

    printf("=== Decode step breakdown (ctx=%u) ===\n\n", prefill_len + 1);
    printf("  %-18s  %8s  %6s  %s\n", "Section", "Time", "Share", "Throughput");
    printf("  %-18s  %8s  %6s  %s\n",
           "------------------", "--------", "------", "----------");

    print_row("attn_norm",     acc.attn_norm,     total, 0, "");
    print_row("q_proj",        acc.q_proj,        total,
              qop / ((double)acc.q_proj / 1e9),  "GOPS");
    print_row("k_proj",        acc.k_proj,        total,
              kop / ((double)acc.k_proj / 1e9),  "GOPS");
    print_row("v_proj",        acc.v_proj,        total,
              vop / ((double)acc.v_proj / 1e9),  "GOPS");
    print_row("rope",          acc.rope,          total, 0, "");
    print_row("kv_cache_write",acc.kv_cache,      total, 0, "");
    print_row("attn_score",    acc.attn_score,    total,
              sop / ((double)acc.attn_score / 1e9), "GOPS");
    print_row("attn_agg",      acc.attn_agg,      total,
              aop / ((double)acc.attn_agg / 1e9),   "GOPS");
    print_row("attn_sub_norm", acc.attn_sub_norm, total, 0, "");
    print_row("o_proj",        acc.o_proj,        total,
              oop / ((double)acc.o_proj / 1e9),  "GOPS");
    print_row("residual_attn", acc.residual_attn, total, 0, "");
    print_row("ffn_norm",      acc.ffn_norm,      total, 0, "");
    print_row("gate_proj",     acc.gate_proj,     total,
              gop / ((double)acc.gate_proj / 1e9), "GOPS");
    print_row("up_proj",       acc.up_proj,       total,
              uop / ((double)acc.up_proj / 1e9),   "GOPS");
    print_row("relu2_mul",     acc.relu2_mul,     total, 0, "");
    print_row("ffn_sub_norm",  acc.ffn_sub_norm,  total, 0, "");
    print_row("down_proj",     acc.down_proj,     total,
              dop / ((double)acc.down_proj / 1e9), "GOPS");
    print_row("residual_ffn",  acc.residual_ffn,  total, 0, "");
    print_row("lm_head",       acc.lm_head,       total,
              lop / ((double)acc.lm_head / 1e9),  "GFLOPS");

    printf("\n  %-18s  %7.1f ms  100.0%%\n", "TOTAL",
           (double)total / 1e6);

    /* Projection totals */
    uint64_t all_proj = acc.q_proj + acc.k_proj + acc.v_proj +
                        acc.o_proj + acc.gate_proj + acc.up_proj + acc.down_proj;
    double all_proj_gops = (qop+kop+vop+oop+gop+uop+dop) /
                           ((double)all_proj / 1e9);
    printf("\n  %-18s  %7.1f ms  %5.1f%%  %6.2f GOPS (combined)\n",
           "all_projections",
           (double)all_proj / 1e6,
           (double)all_proj / (double)total * 100.0,
           all_proj_gops);

    /* Attention totals */
    uint64_t all_attn = acc.attn_score + acc.attn_agg;
    printf("  %-18s  %7.1f ms  %5.1f%%\n",
           "all_attention",
           (double)all_attn / 1e6,
           (double)all_attn / (double)total * 100.0);

    free(logits); free(hidden);
    for (uint32_t l = 0; l < L; l++) hs_kv_cache_free(&caches[l]);
    free(caches);
    hs_mlt_free(&m);
    return 0;
}
