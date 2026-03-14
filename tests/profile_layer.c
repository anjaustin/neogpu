/*
 * Profile: measure each stage of a layer forward pass.
 * Hidden=512, FFN=1024, 8 heads, ctx=64 tokens in KV cache.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "hs_ml_infer.h"

static double get_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

int main(void) {
    u32 H = 512, F = 1024, nh = 8, hd = 64;
    u32 ctx_depth = 64;

    HSMLTernary m;
    hs_mlt_init(&m);
    hs_mlt_alloc_random(&m, 512, H, 4, nh, F, 256, 42);

    /* Pre-fill a KV cache */
    HSKVCache cache;
    hs_kv_cache_init(&cache, 256, nh, nh, hd);
    float* hidden = malloc(H * sizeof(float));
    for (u32 i = 0; i < H; i++) hidden[i] = ((float)(i % 100) - 50.0f) / 50.0f;
    for (u32 t = 0; t < ctx_depth; t++)
        hs_mlt_layer_forward(hidden, 0, &cache, &m);
    printf("KV cache depth: %u\n\n", cache.cache_len);

    HSTernaryLayer* lay = &m.layers[0];
    int8_t*  qbuf = m.quant_buf;
    int32_t* gbuf = m.gemm_buf;
    float*   fbuf = m.hidden_buf;
    float*   ffbuf = m.ffn_buf;

    /* Allocate persistent scratch (no hot malloc) for fair comparison */
    float* q = malloc(H * sizeof(float));
    float* k = malloc(H * sizeof(float));
    float* v = malloc(H * sizeof(float));
    float* attn_out = malloc(H * sizeof(float));
    float* gate_out = malloc(F * sizeof(float));
    float* up_out   = malloc(F * sizeof(float));
    int8_t* fqbuf   = malloc(F * sizeof(int8_t));
    float*  scores  = malloc(256 * sizeof(float));

    int REPS = 200;
    double T_quant = 0, T_qkv = 0, T_attn = 0, T_oproj = 0;
    double T_ffn_norm = 0, T_gate_up = 0, T_swiglu = 0, T_down = 0;
    double T_total = 0;

    for (int r = 0; r < REPS; r++) {
        for (u32 i = 0; i < H; i++) hidden[i] = ((float)(i % 100) - 50.0f) / 50.0f;
        double t0 = get_ns();

        /* 1. Attn RMSNorm + quantize */
        double t1 = get_ns();
        float ss = 0;
        for (u32 i = 0; i < H; i++) ss += hidden[i] * hidden[i];
        float nsc = 1.0f / sqrtf(ss / H + 1e-5f);
        for (u32 i = 0; i < H; i++) fbuf[i] = hidden[i] * nsc * lay->attn_norm[i];
        float absmax = 0;
        for (u32 i = 0; i < H; i++) { float a = fabsf(fbuf[i]); if (a > absmax) absmax = a; }
        float act_sc = (absmax > 0) ? 127.0f / absmax : 1.0f;
        for (u32 i = 0; i < H; i++) { float vv = fbuf[i] * act_sc; qbuf[i] = vv > 127 ? 127 : vv < -128 ? -128 : (int8_t)vv; }
        T_quant += get_ns() - t1;

        /* 2. Q+K+V GEMMs */
        double t2 = get_ns();
        HSRouteDesc route; route.format = HS_ROUTE_TERNARY_2BIT; route.K = H; route.N = H;
        int th = hs_ml_route_optimal_threads(&route, 1);
        route.routes = lay->q_proj; hs_ml_route_mt(gbuf, qbuf, &route, 1, th);
        for (u32 i = 0; i < H; i++) q[i] = (float)gbuf[i] / (act_sc * lay->q_scale[i]);
        route.routes = lay->k_proj; hs_ml_route_mt(gbuf, qbuf, &route, 1, th);
        for (u32 i = 0; i < H; i++) k[i] = (float)gbuf[i] / (act_sc * lay->k_scale[i]);
        route.routes = lay->v_proj; hs_ml_route_mt(gbuf, qbuf, &route, 1, th);
        for (u32 i = 0; i < H; i++) v[i] = (float)gbuf[i] / (act_sc * lay->v_scale[i]);
        T_qkv += get_ns() - t2;

        /* 3. RoPE (in-place, fast — skip timing) */
        hs_rope_apply(q, k, nh, hd, cache.cache_len > 0 ? cache.cache_len - 1 : 0, 10000.0f);

        /* 4. Attention */
        double t3 = get_ns();
        u32 ctx_len = cache.cache_len;
        float inv_sq = 1.0f / sqrtf((float)hd);
        memset(attn_out, 0, H * sizeof(float));
        for (u32 h = 0; h < nh; h++) {
            float* qh = q + h * hd;
            float* outh = attn_out + h * hd;
            float max_s = -1e30f;
            for (u32 t2 = 0; t2 < ctx_len; t2++) {
                float* kh = cache.k_cache + t2 * nh * hd + h * hd;
                float s = 0;
                for (u32 d = 0; d < hd; d++) s += qh[d] * kh[d];
                scores[t2] = s * inv_sq;
                if (scores[t2] > max_s) max_s = scores[t2];
            }
            float sum = 0;
            for (u32 t2 = 0; t2 < ctx_len; t2++) { scores[t2] = expf(scores[t2] - max_s); sum += scores[t2]; }
            for (u32 t2 = 0; t2 < ctx_len; t2++) scores[t2] /= sum;
            for (u32 d = 0; d < hd; d++) {
                float val = 0;
                for (u32 t2 = 0; t2 < ctx_len; t2++) { float* vh = cache.v_cache + t2 * nh * hd + h * hd; val += scores[t2] * vh[d]; }
                outh[d] = val;
            }
        }
        T_attn += get_ns() - t3;

        /* 5. O projection */
        double t4 = get_ns();
        absmax = 0;
        for (u32 i = 0; i < H; i++) { float a = fabsf(attn_out[i]); if (a > absmax) absmax = a; }
        act_sc = (absmax > 0) ? 127.0f / absmax : 1.0f;
        for (u32 i = 0; i < H; i++) { float vv = attn_out[i] * act_sc; qbuf[i] = vv > 127 ? 127 : vv < -128 ? -128 : (int8_t)vv; }
        route.routes = lay->o_proj; hs_ml_route_mt(gbuf, qbuf, &route, 1, th);
        for (u32 i = 0; i < H; i++) { fbuf[i] = (float)gbuf[i] / (act_sc * lay->o_scale[i]); hidden[i] += fbuf[i]; }
        T_oproj += get_ns() - t4;

        /* 6. FFN RMSNorm + quantize */
        double t5 = get_ns();
        ss = 0; for (u32 i = 0; i < H; i++) ss += hidden[i] * hidden[i];
        nsc = 1.0f / sqrtf(ss / H + 1e-5f);
        for (u32 i = 0; i < H; i++) fbuf[i] = hidden[i] * nsc * lay->ffn_norm[i];
        absmax = 0; for (u32 i = 0; i < H; i++) { float a = fabsf(fbuf[i]); if (a > absmax) absmax = a; }
        act_sc = (absmax > 0) ? 127.0f / absmax : 1.0f;
        for (u32 i = 0; i < H; i++) { float vv = fbuf[i] * act_sc; qbuf[i] = vv > 127 ? 127 : vv < -128 ? -128 : (int8_t)vv; }
        T_ffn_norm += get_ns() - t5;

        /* 7. Gate + Up GEMMs */
        double t6 = get_ns();
        route.N = F;
        route.routes = lay->gate_proj; hs_ml_route_mt(gbuf, qbuf, &route, 1, th);
        for (u32 i = 0; i < F; i++) gate_out[i] = (float)gbuf[i] / (act_sc * lay->gate_scale[i]);
        route.routes = lay->up_proj; hs_ml_route_mt(gbuf, qbuf, &route, 1, th);
        for (u32 i = 0; i < F; i++) up_out[i] = (float)gbuf[i] / (act_sc * lay->up_scale[i]);
        T_gate_up += get_ns() - t6;

        /* 8. SwiGLU */
        double t7 = get_ns();
        for (u32 i = 0; i < F; i++) {
            float sg = gate_out[i] / (1.0f + expf(-gate_out[i]));
            ffbuf[i] = sg * up_out[i];
        }
        T_swiglu += get_ns() - t7;

        /* 9. Down projection */
        double t8 = get_ns();
        absmax = 0; for (u32 i = 0; i < F; i++) { float a = fabsf(ffbuf[i]); if (a > absmax) absmax = a; }
        act_sc = (absmax > 0) ? 127.0f / absmax : 1.0f;
        for (u32 i = 0; i < F; i++) { float vv = ffbuf[i] * act_sc; fqbuf[i] = vv > 127 ? 127 : vv < -128 ? -128 : (int8_t)vv; }
        route.K = F; route.N = H; route.routes = lay->down_proj;
        hs_ml_route_mt(gbuf, fqbuf, &route, 1, th);
        for (u32 i = 0; i < H; i++) { fbuf[i] = (float)gbuf[i] / (act_sc * lay->down_scale[i]); hidden[i] += fbuf[i]; }
        T_down += get_ns() - t8;

        T_total += get_ns() - t0;
    }

    double sc = 1.0 / (REPS * 1e6);
    double tot = T_total * sc;
    printf("Layer forward profile  H=%u F=%u nh=%u hd=%u ctx=%u  (%d reps)\n",
           H, F, nh, hd, ctx_depth, REPS);
    printf("--------------------------------------------------------------\n");
    printf("  1. Attn RMSNorm+quant:   %6.3f ms  %5.1f%%\n", T_quant*sc,   100.0*T_quant/T_total);
    printf("  2. Q+K+V GEMMs+dequant:  %6.3f ms  %5.1f%%\n", T_qkv*sc,     100.0*T_qkv/T_total);
    printf("  3. Attention scores:     %6.3f ms  %5.1f%%\n", T_attn*sc,     100.0*T_attn/T_total);
    printf("  4. O proj+dequant:       %6.3f ms  %5.1f%%\n", T_oproj*sc,    100.0*T_oproj/T_total);
    printf("  5. FFN RMSNorm+quant:    %6.3f ms  %5.1f%%\n", T_ffn_norm*sc, 100.0*T_ffn_norm/T_total);
    printf("  6. Gate+Up GEMMs:        %6.3f ms  %5.1f%%\n", T_gate_up*sc,  100.0*T_gate_up/T_total);
    printf("  7. SwiGLU:               %6.3f ms  %5.1f%%\n", T_swiglu*sc,   100.0*T_swiglu/T_total);
    printf("  8. Down proj+dequant:    %6.3f ms  %5.1f%%\n", T_down*sc,     100.0*T_down/T_total);
    printf("--------------------------------------------------------------\n");
    printf("  Total:                   %6.3f ms\n", tot);
    printf("\n  Bottlenecks (>10%% of total):\n");
    double stages[] = {T_quant,T_qkv,T_attn,T_oproj,T_ffn_norm,T_gate_up,T_swiglu,T_down};
    const char* names[] = {"quant","Q+K+V","attn","O-proj","ffn-norm","gate+up","SwiGLU","down"};
    for (int i = 0; i < 8; i++)
        if (stages[i]/T_total > 0.10)
            printf("  -> %-12s  %.1f%%\n", names[i], 100.0*stages[i]/T_total);

    free(q); free(k); free(v); free(attn_out);
    free(gate_out); free(up_out); free(fqbuf); free(scores);
    hs_kv_cache_free(&cache);
    hs_mlt_free(&m);
    free(hidden);
    return 0;
}
