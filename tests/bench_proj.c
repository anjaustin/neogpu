/*
 * Benchmark: Projection throughput measurement
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
    printf("=== Projection Throughput Benchmark ===\n\n");
    
    /* BitNet specs */
    uint32_t H = 2560, F = 6912, kv = 640;
    
    /* Allocate */
    float* hidden = malloc(H * sizeof(float));
    float* out = malloc(H * sizeof(float));
    uint8_t* w_q = malloc((H/4) * H);
    uint8_t* w_k = malloc((kv/4) * H);
    uint8_t* w_v = malloc((kv/4) * H);
    uint8_t* w_o = malloc((H/4) * H);
    uint8_t* w_gate = malloc((F/4) * H);
    uint8_t* w_up = malloc((F/4) * H);
    uint8_t* w_down = malloc((H/4) * F);
    
    srand(42);
    for (uint32_t i = 0; i < H; i++) hidden[i] = (float)(rand() % 100) / 10.0f;
    for (uint32_t i = 0; i < (H/4)*H; i++) w_q[i] = rand() % 256;
    for (uint32_t i = 0; i < (kv/4)*H; i++) w_k[i] = rand() % 256;
    for (uint32_t i = 0; i < (kv/4)*H; i++) w_v[i] = rand() % 256;
    for (uint32_t i = 0; i < (H/4)*H; i++) w_o[i] = rand() % 256;
    for (uint32_t i = 0; i < (F/4)*H; i++) w_gate[i] = rand() % 256;
    for (uint32_t i = 0; i < (F/4)*H; i++) w_up[i] = rand() % 256;
    for (uint32_t i = 0; i < (H/4)*F; i++) w_down[i] = rand() % 256;
    
    /* Single projection warmup */
    printf("Running warmup projection...\n");
    fflush(stdout);
    hs_ml_ternary_f32_proj(out, hidden, w_q, H, H);
    printf("Warmup done\n");
    fflush(stdout);
    
    /* Profile individual projections */
    #define RUN(proj, name, N, K, W) do { \
        uint64_t t0 = ns_now(); \
        for (int i = 0; i < 10; i++) proj(out, hidden, W, N, K); \
        uint64_t t1 = ns_now(); \
        double ms = (t1 - t0) / 1000000.0 / 10.0; \
        double gops = (double)N * K / ms / 1e6; \
        printf("%-12s %5u x %-5u: %7.2f ms  %6.2f GOPS\n", name, N, K, ms, gops); \
    } while(0)
    
    printf("Individual projections (10 iter avg):\n");
    fflush(stdout);
    printf("Running q_proj...\n"); fflush(stdout);
    RUN(hs_ml_ternary_f32_proj, "q_proj", H, H, w_q);
    printf("q_proj done\n"); fflush(stdout);
    printf("Running k_proj...\n"); fflush(stdout);
    RUN(hs_ml_ternary_f32_proj, "k_proj", kv, H, w_k);
    printf("k_proj done\n"); fflush(stdout);
    printf("Running v_proj...\n"); fflush(stdout);
    RUN(hs_ml_ternary_f32_proj, "v_proj", kv, H, w_v);
    printf("v_proj done\n"); fflush(stdout);
    printf("Running o_proj...\n"); fflush(stdout);
    RUN(hs_ml_ternary_f32_proj, "o_proj", H, H, w_o);
    printf("o_proj done\n"); fflush(stdout);
    printf("Running gate_proj...\n"); fflush(stdout);
    RUN(hs_ml_ternary_f32_proj, "gate_proj", F, H, w_gate);
    printf("gate_proj done\n"); fflush(stdout);
    printf("Running up_proj...\n"); fflush(stdout);
    RUN(hs_ml_ternary_f32_proj, "up_proj", F, H, w_up);
    printf("up_proj done\n"); fflush(stdout);
    printf("Running down_proj...\n"); fflush(stdout);
    RUN(hs_ml_ternary_f32_proj, "down_proj", H, F, w_down);
    printf("down_proj done\n"); fflush(stdout);
    
    /* Full layer (7 projections) */
    printf("\nFull layer (7 projections):\n");
    uint64_t t0 = ns_now();
    for (int iter = 0; iter < 10; iter++) {
        hs_ml_ternary_f32_proj(out, hidden, w_q, H, H);
        hs_ml_ternary_f32_proj(out, hidden, w_k, kv, H);
        hs_ml_ternary_f32_proj(out, hidden, w_v, kv, H);
        hs_ml_ternary_f32_proj(out, hidden, w_o, H, H);
        hs_ml_ternary_f32_proj(out, hidden, w_gate, F, H);
        hs_ml_ternary_f32_proj(out, hidden, w_up, F, H);
        hs_ml_ternary_f32_proj(out, hidden, w_down, H, F);
    }
    uint64_t t1 = ns_now();
    double ms_layer = (t1 - t0) / 1000000.0 / 10.0;
    printf("  Single layer: %.1f ms\n", ms_layer);
    printf("  30 layers:     %.1f ms\n", ms_layer * 30);
    
    printf("\n=== GPU Potential ===\n");
    printf("Current CPU (4-thread): %.1f ms/layer\n", ms_layer);
    printf("GPU (12 QPUs x 16 SIMD): ~%.0f-%.0f ms/layer (est. 3-5x)\n", 
           ms_layer/5.0, ms_layer/3.0);
    printf("GPU 30 layers: ~%.0f-%.0f ms (vs CPU %.0f ms)\n",
           ms_layer * 30 / 5, ms_layer * 30 / 3, ms_layer * 30);
    
    free(hidden); free(out);
    free(w_q); free(w_k); free(w_v); free(w_o);
    free(w_gate); free(w_up); free(w_down);
    return 0;
}
