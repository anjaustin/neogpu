/*
 * Simple CPU benchmark
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

extern void hs_ml_ternary_f32_proj(float *out, const float *in,
                                    const uint8_t *W, uint32_t N, uint32_t K);

static uint64_t ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main(void) {
    printf("=== CPU Benchmark ===\n\n");
    
    uint32_t H = 2560, F = 6912, kv = 640;
    
    float* hidden = malloc(H * sizeof(float));
    float* out_q = malloc(H * sizeof(float));
    float* out_k = malloc(kv * sizeof(float));
    float* out_v = malloc(kv * sizeof(float));
    float* out_o = malloc(H * sizeof(float));
    float* out_gate = malloc(F * sizeof(float));
    float* out_up = malloc(F * sizeof(float));
    float* out_down = malloc(H * sizeof(float));
    
    uint8_t* w_q = malloc((H/4) * H);
    uint8_t* w_k = malloc((kv/4) * H);
    uint8_t* w_v = malloc((kv/4) * H);
    uint8_t* w_o = malloc((H/4) * H);
    uint8_t* w_gate = malloc((F/4) * H);
    uint8_t* w_up = malloc((F/4) * H);
    uint8_t* w_down = malloc((H/4) * F);
    
    printf("Allocated\n");
    
    srand(42);
    for (uint32_t i = 0; i < H; i++) hidden[i] = (float)(rand() % 100) / 10.0f;
    for (uint32_t i = 0; i < (H/4)*H; i++) w_q[i] = rand() % 256;
    for (uint32_t i = 0; i < (kv/4)*H; i++) w_k[i] = rand() % 256;
    for (uint32_t i = 0; i < (kv/4)*H; i++) w_v[i] = rand() % 256;
    for (uint32_t i = 0; i < (H/4)*H; i++) w_o[i] = rand() % 256;
    for (uint32_t i = 0; i < (F/4)*H; i++) w_gate[i] = rand() % 256;
    for (uint32_t i = 0; i < (F/4)*H; i++) w_up[i] = rand() % 256;
    for (uint32_t i = 0; i < (H/4)*F; i++) w_down[i] = rand() % 256;
    
    printf("Initialized\n");
    
    printf("Running layer projections...\n");
    fflush(stdout);
    
    uint64_t t0 = ns_now();
    hs_ml_ternary_f32_proj(out_q, hidden, w_q, H, H);
    printf("q_proj done\n");
    hs_ml_ternary_f32_proj(out_k, hidden, w_k, kv, H);
    printf("k_proj done\n");
    hs_ml_ternary_f32_proj(out_v, hidden, w_v, kv, H);
    printf("v_proj done\n");
    hs_ml_ternary_f32_proj(out_o, out_q, w_o, H, H);
    printf("o_proj done\n");
    hs_ml_ternary_f32_proj(out_gate, out_q, w_gate, F, H);
    printf("gate_proj done\n");
    hs_ml_ternary_f32_proj(out_up, out_q, w_up, F, H);
    printf("up_proj done\n");
    hs_ml_ternary_f32_proj(out_down, out_gate, w_down, H, F);
    printf("down_proj done\n");
    uint64_t t1 = ns_now();
    
    double ms = (t1 - t0) / 1000000.0;
    printf("\nTotal layer time: %.1f ms\n", ms);
    
    free(hidden);
    free(out_q); free(out_k); free(out_v); free(out_o);
    free(out_gate); free(out_up); free(out_down);
    free(w_q); free(w_k); free(w_v); free(w_o);
    free(w_gate); free(w_up); free(w_down);
    
    return 0;
}
