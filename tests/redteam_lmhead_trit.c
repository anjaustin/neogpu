/*
 * redteam_lmhead_trit.c — Correctness test for the two-stage ternary spline lm_head.
 *
 * Tests:
 *   1. Encoding: lm_head_f16 → 8 trit planes reconstructs rows accurately
 *   2. Recall:   top-LMH_CANDIDATES from Stage-1 coarse pass contains all
 *                of top-42 from the exact F16 logits (100% recall required)
 *   3. Top-1:    top-1 from trit-spline matches top-1 from F16 exactly
 *   4. Logit quality: mean/max relative error for top-C candidates
 *
 * Uses real hidden states extracted from a short prefill so the recall
 * measurement reflects actual model activations, not random vectors.
 *
 * Usage:
 *   ./tests/redteam_lmhead_trit models/bitnet-2b4t-i2s.gguf [norms_v2.bin]
 *
 * Build:
 *   gcc -O3 -march=armv8-a+simd+fp16 -mtune=cortex-a72 -ffast-math \
 *       -funroll-loops -DNDEBUG -Iinclude \
 *       tests/redteam_lmhead_trit.c \
 *       src/hs_ml_ternary_neon.c src/hs_ml_loader_ternary.c \
 *       src/hs_ml_infer.c src/hs_ml_routing.c \
 *       src/hs_ml_ternary_mt.c src/hs_ml_binary.c src/hs_ml.c \
 *       -lm -lpthread -o tests/redteam_lmhead_trit
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "hs_ml_infer.h"

static inline uint64_t ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline float fp16_to_f32_local(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t f;
    if (exp == 0)       f = sign;
    else if (exp == 31) f = sign | 0x7F800000u | (mant << 13);
    else                f = sign | ((exp + 112u) << 23) | (mant << 13);
    float out; memcpy(&out, &f, 4); return out;
}

/* Reference F16 lm_head logits from a saved copy of lm_head_f16 */
static void compute_f16_logits(const uint16_t *lm_head_f16,
                                const float *normed,
                                float *logits,
                                uint32_t V, uint32_t H) {
    for (uint32_t v = 0; v < V; v++) {
        const uint16_t *row = lm_head_f16 + (size_t)v * H;
        float s = 0.0f;
        for (uint32_t i = 0; i < H; i++)
            s += fp16_to_f32_local(row[i]) * normed[i];
        logits[v] = s;
    }
}

static void rmsnorm_local(float *out, const float *in, const float *w,
                           float eps, uint32_t n) {
    float ss = 0.0f;
    for (uint32_t i = 0; i < n; i++) ss += in[i] * in[i];
    float sc = 1.0f / sqrtf(ss / (float)n + eps);
    for (uint32_t i = 0; i < n; i++) out[i] = in[i] * sc * w[i];
}

/* Partial argsort: return indices of top-K largest in src */
static void top_k_indices(const float *src, uint32_t N, uint32_t *out, uint32_t K) {
    for (uint32_t k = 0; k < K; k++) {
        float best = -1e30f; uint32_t bi = 0;
        for (uint32_t i = 0; i < N; i++) {
            int already = 0;
            for (uint32_t j = 0; j < k; j++) if (out[j] == i) { already = 1; break; }
            if (!already && src[i] > best) { best = src[i]; bi = i; }
        }
        out[k] = bi;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [norms.bin]\n", argv[0]);
        return 1;
    }

    /* ── Load model, save a copy of lm_head_f16 BEFORE encoding ── */
    HSMLTernary m;
    hs_mlt_init(&m);
    if (hs_mlt_load_gguf(&m, argv[1]) != 0) return 1;
    if (argc > 2) {
        if (hs_mlt_load_norms_sidecar(&m, argv[2]) != 0)
            fprintf(stderr, "warning: norms sidecar not loaded\n");
    }

    uint32_t V = m.vocab_size;
    uint32_t H = m.hidden_size;

    /* Save F16 lm_head before encode destroys it */
    uint16_t *lm_head_saved = malloc((size_t)V * H * sizeof(uint16_t));
    if (!lm_head_saved) { fprintf(stderr, "OOM saving lm_head_f16\n"); return 1; }
    memcpy(lm_head_saved, m.lm_head_f16, (size_t)V * H * sizeof(uint16_t));
    printf("Saved F16 lm_head (%u x %u)\n", V, H);

    /* ── Test 1: Trit plane encoding ── */
    printf("\n=== Test 1: Encoding ===\n");
    uint64_t t0 = ns_now();
    if (hs_mlt_lmhead_encode(&m) != 0) {
        fprintf(stderr, "FAIL: hs_mlt_lmhead_encode returned error\n"); return 1;
    }
    double enc_ms = (double)(ns_now() - t0) / 1e6;
    printf("Encoding time: %.1f ms\n", enc_ms);

    /* Reconstruct a sample of rows and measure error */
    uint32_t SAMPLE = 4096;
    if (SAMPLE > V) SAMPLE = V;

    float max_row_rel = 0.0f, sum_row_rel = 0.0f;
    for (uint32_t v = 0; v < SAMPLE; v++) {
        /* Reconstruct row from 8 planes */
        float recon[2560];  /* H=2560 */
        memset(recon, 0, H * sizeof(float));
        float row_scale = fp16_to_f32_local(m.lm_head_row_scale[v]);
        float plane_w[8] = {1.0f, 1/3.0f, 1/9.0f, 1/27.0f, 1/81.0f, 1/243.0f, 1/729.0f, 1/2187.0f};
        uint32_t row_bytes = H / 4;
        for (int k = 0; k < 8; k++) {
            const uint8_t *pr = m.lm_head_planes[k] + (size_t)v * row_bytes;
            for (uint32_t bi = 0; bi < row_bytes; bi++) {
                uint8_t b = pr[bi];
                for (int s = 0; s < 4; s++) {
                    int t = (int)((b >> (s*2)) & 3) - 1;  /* {-1,0,+1} */
                    recon[bi*4+s] += (float)t * plane_w[k];
                }
            }
        }
        /* Scale back */
        for (uint32_t i = 0; i < H; i++) recon[i] *= row_scale;

        /* Compare to original F16 */
        float max_abs_orig = 0.0f;
        for (uint32_t i = 0; i < H; i++) {
            float orig = fp16_to_f32_local(lm_head_saved[v*H+i]);
            float av = orig < 0.0f ? -orig : orig;
            if (av > max_abs_orig) max_abs_orig = av;
        }
        float row_err = 0.0f;
        for (uint32_t i = 0; i < H; i++) {
            float orig = fp16_to_f32_local(lm_head_saved[v*H+i]);
            float d = recon[i] - orig;
            if (d < 0) d = -d;
            if (d > row_err) row_err = d;
        }
        float rel = (max_abs_orig > 1e-9f) ? row_err / max_abs_orig : 0.0f;
        if (rel > max_row_rel) max_row_rel = rel;
        sum_row_rel += rel;
    }
    printf("Row reconstruction (sample %u): max_rel=%.5f  mean_rel=%.5f\n",
           SAMPLE, max_row_rel, sum_row_rel / SAMPLE);

    /* ── Get real hidden states via prefill ── */
    printf("\n=== Getting real hidden states via prefill ===\n");

    /* Encode a short prompt through the model, capture hidden states */
    /* Use the control prompt tokens */
    uint32_t prompt_tokens[16];
    uint32_t n_prompt = 0;
    prompt_tokens[n_prompt++] = m.tokenizer_bos;
    const char *prompt = "Hypothetically, might reflective recursion be a function of cognition?";
    n_prompt += hs_mlt_bpe_encode(&m, prompt, (uint32_t)strlen(prompt),
                                   prompt_tokens + n_prompt, 16 - n_prompt);
    printf("Prompt: %u tokens\n", n_prompt);

    /* Run one session prefill, capture hidden state after final layer */
    HSMLTernarySession sess;
    if (hs_mlt_session_init(&sess, &m) != 0) {
        fprintf(stderr, "FAIL: session_init\n"); return 1;
    }
    if (hs_mlt_prefill(&sess, prompt_tokens, n_prompt) != 0) {
        fprintf(stderr, "FAIL: prefill\n"); return 1;
    }

    /* hidden state is in sess->hidden after prefill */
    float *hidden_copy = malloc(H * sizeof(float));
    memcpy(hidden_copy, sess.hidden, H * sizeof(float));

    /* ── Test 2 & 3: Recall and top-1 on real hidden state ── */
    printf("\n=== Test 2: Top-K recall on real hidden state ===\n");

    /* Compute exact F16 logits (ground truth) */
    float *normed     = malloc(H * sizeof(float));
    float *logits_f16 = malloc(V * sizeof(float));
    float *logits_trit = malloc(V * sizeof(float));
    if (!normed || !logits_f16 || !logits_trit) { fprintf(stderr, "OOM\n"); return 1; }

    rmsnorm_local(normed, hidden_copy, m.final_norm, 1e-5f, H);
    t0 = ns_now();
    compute_f16_logits(lm_head_saved, normed, logits_f16, V, H);
    double f16_ms = (double)(ns_now() - t0) / 1e6;
    printf("F16 logits computed in %.1f ms\n", f16_ms);

    /* Compute trit-spline logits */
    t0 = ns_now();
    if (hs_mlt_session_logits(&sess, logits_trit) != 0) {
        fprintf(stderr, "FAIL: session_logits\n"); return 1;
    }
    double trit_ms = (double)(ns_now() - t0) / 1e6;
    printf("Trit-spline logits computed in %.1f ms  (%.2fx vs F16)\n",
           trit_ms, f16_ms / trit_ms);

    /* Top-1 match */
    uint32_t top1_f16 = 0;
    float    best_f16 = logits_f16[0];
    for (uint32_t v = 1; v < V; v++)
        if (logits_f16[v] > best_f16) { best_f16 = logits_f16[v]; top1_f16 = v; }

    uint32_t top1_trit = 0;
    float    best_trit = logits_trit[0];
    for (uint32_t v = 1; v < V; v++)
        if (logits_trit[v] > best_trit) { best_trit = logits_trit[v]; top1_trit = v; }

    int top1_match = (top1_f16 == top1_trit);
    printf("Top-1 F16:  token %u  logit=%.4f\n", top1_f16, best_f16);
    printf("Top-1 Trit: token %u  logit=%.4f\n", top1_trit, best_trit);
    printf("Top-1 match: %s\n", top1_match ? "PASS" : "FAIL");

    /* Top-K recall for various K */
    int all_recall_pass = 1;
    printf("\nTop-K recall (exact top-K in trit top-%d candidates):\n", LMH_CANDIDATES);
    for (int K = 1; K <= 42; K = (K == 1) ? 5 : (K == 5) ? 10 : (K == 10) ? 42 : K+1) {
        /* Find exact top-K from F16 */
        uint32_t *exact_topk = malloc(K * sizeof(uint32_t));
        top_k_indices(logits_f16, V, exact_topk, (uint32_t)K);

        /* Find trit top-LMH_CANDIDATES */
        uint32_t *trit_topC = malloc(LMH_CANDIDATES * sizeof(uint32_t));
        top_k_indices(logits_trit, V, trit_topC, LMH_CANDIDATES);

        /* Count how many exact top-K appear in trit top-C */
        int hits = 0;
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < LMH_CANDIDATES; j++) {
                if (exact_topk[i] == trit_topC[j]) { hits++; break; }
            }
        }
        float recall = (float)hits / K;
        int pass = (recall >= 1.0f);
        if (!pass) all_recall_pass = 0;
        printf("  top-%2d recall: %.4f  %s\n", K, recall, pass ? "PASS" : "FAIL");
        free(exact_topk);
        free(trit_topC);
    }

    /* ── Test 4: Logit quality for top-C candidates ── */
    printf("\n=== Test 4: Logit quality (top-C candidates) ===\n");
    float max_rel = 0.0f, sum_rel = 0.0f;
    int n_cands = 0;
    for (uint32_t v = 0; v < V; v++) {
        if (logits_trit[v] < -9e29f) continue;  /* non-candidate */
        float exact  = logits_f16[v];
        float approx = logits_trit[v];
        float denom  = exact < 0 ? -exact : exact;
        if (denom < 1e-9f) continue;
        float rel = (approx - exact) < 0 ? (exact - approx) / denom : (approx - exact) / denom;
        if (rel > max_rel) max_rel = rel;
        sum_rel += rel;
        n_cands++;
    }
    printf("Candidates: %d\n", n_cands);
    printf("Logit rel error (vs F16): max=%.5f  mean=%.5f\n",
           max_rel, n_cands > 0 ? sum_rel / n_cands : 0.0f);

    /* ── Summary ── */
    printf("\n=== Summary ===\n");
    printf("Top-1 match:    %s\n", top1_match ? "PASS" : "FAIL");
    printf("All recall:     %s\n", all_recall_pass ? "PASS" : "FAIL");
    printf("F16 time:       %.1f ms\n", f16_ms);
    printf("Trit time:      %.1f ms  (%.2fx speedup)\n", trit_ms, f16_ms/trit_ms);

    int overall = top1_match && all_recall_pass;
    printf("\n%s\n", overall ? "ALL PASS" : "FAILURES DETECTED");

    free(lm_head_saved);
    free(hidden_copy);
    free(normed);
    free(logits_f16);
    free(logits_trit);
    hs_mlt_session_free(&sess);
    hs_mlt_free(&m);
    return overall ? 0 : 1;
}
