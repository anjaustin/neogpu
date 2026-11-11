/*
 * Benchmark: Ternary Coprocessor vs CPU
 * 
 * Measures projection throughput with CPU (4-thread NEON)
 * and estimates GPU potential.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include "hs_ml_ternary_coproc.h"

/* CPU fallback projection */
extern void hs_ml_ternary_f32_proj(float *out, const float *in,
                                    const uint8_t *W, uint32_t N, uint32_t K);

static uint64_t ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

#define N_LAYERS 30

/* Simulate BitNet b1.58-2B-4T layer projections */
typedef struct {
    uint32_t H;   /* hidden = 2560 */
    uint32_t F;   /* ffn = 6912 */
    uint32_t kv;  /* kv heads * head_dim = 5 * 128 = 640 */
} LayerSpec;

static void run_cpu_projections(const LayerSpec* spec, float* hidden, 
                                uint8_t** weights, float** outputs) {
    /* q_proj: H x H */
    hs_ml_ternary_f32_proj(outputs[0], hidden, weights[0], spec->H, spec->H);
    /* k_proj: kv x H */
    hs_ml_ternary_f32_proj(outputs[1], hidden, weights[1], spec->kv, spec->H);
    /* v_proj: kv x H */
    hs_ml_ternary_f32_proj(outputs[2], hidden, weights[2], spec->kv, spec->H);
    /* o_proj: H x H */
    hs_ml_ternary_f32_proj(outputs[3], hidden, weights[3], spec->H, spec->H);
    /* gate_proj: F x H */
    hs_ml_ternary_f32_proj(outputs[4], hidden, weights[4], spec->F, spec->H);
    /* up_proj: F x H */
    hs_ml_ternary_f32_proj(outputs[5], hidden, weights[5], spec->F, spec->H);
    /* down_proj: H x F */
    hs_ml_ternary_f32_proj(outputs[6], hidden, weights[6], spec->H, spec->F);
}

int main(void) {
    printf("=== Ternary Coprocessor Benchmark ===\n\n");
    
    /* Initialize coprocessor */
    int avail = ternary_coproc_available();
    printf("Coprocessor available: %s\n", avail ? "yes" : "no");
    fflush(stdout);
    
    ternary_coproc_init();
    printf("After init: %s\n", ternary_coproc_available() ? "yes" : "no");
    fflush(stdout);
    
    /* BitNet b1.58-2B-4T specs */
    LayerSpec spec = { .H = 2560, .F = 6912, .kv = 640 };
    printf("Allocating test data...\n");
    fflush(stdout);
    
    /* Allocate test data */
    float* hidden = malloc(spec.H * sizeof(float));
    float* outputs[7];
    uint8_t* weights[7];
    
    for (int i = 0; i < 7; i++) outputs[i] = malloc(spec.H * sizeof(float));
    
    weights[0] = malloc((spec.H / 4) * spec.H);  /* q */
    weights[1] = malloc((spec.kv / 4) * spec.H); /* k */
    weights[2] = malloc((spec.kv / 4) * spec.H); /* v */
    weights[3] = malloc((spec.H / 4) * spec.H);  /* o */
    weights[4] = malloc((spec.F / 4) * spec.H);  /* gate */
    weights[5] = malloc((spec.F / 4) * spec.H);  /* up */
    weights[6] = malloc((spec.H / 4) * spec.F);  /* down */
    
    printf("Initializing data...\n");
    fflush(stdout);
    
    /* Initialize with random data */
    srand(42);
    for (uint32_t i = 0; i < spec.H; i++) hidden[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 7; i++) {
        uint32_t size = (i == 0 || i == 3) ? (spec.H/4)*spec.H :
                       (i == 1 || i == 2) ? (spec.kv/4)*spec.H :
                       (i == 6) ? (spec.H/4)*spec.F : (spec.F/4)*spec.H;
        for (uint32_t j = 0; j < size; j++) weights[i][j] = rand() % 256;
    }
    
    printf("Building projection descriptors...\n");
    fflush(stdout);
    
    /* Build projection descriptors */
    TernaryProj projs[7 * N_LAYERS];
    for (int l = 0; l < N_LAYERS; l++) {
        int base = l * 7;
        projs[base + 0] = (TernaryProj){ outputs[0], hidden, weights[0], spec.H, spec.H };
        projs[base + 1] = (TernaryProj){ outputs[1], hidden, weights[1], spec.kv, spec.H };
        projs[base + 2] = (TernaryProj){ outputs[2], hidden, weights[2], spec.kv, spec.H };
        projs[base + 3] = (TernaryProj){ outputs[3], outputs[0], weights[3], spec.H, spec.H };
        projs[base + 4] = (TernaryProj){ outputs[4], outputs[0], weights[4], spec.F, spec.H };
        projs[base + 5] = (TernaryProj){ outputs[5], outputs[0], weights[5], spec.F, spec.H };
        projs[base + 6] = (TernaryProj){ outputs[6], outputs[4], weights[6], spec.H, spec.F };
    }
    
    printf("Starting benchmark...\n");
    fflush(stdout);
    
    printf("Testing single-layer projections (CPU, 4-thread NEON)...\n");
    fflush(stdout);
    uint64_t t0 = ns_now();
    printf("  Running first projection...\n");
    fflush(stdout);
    run_cpu_projections(&spec, hidden, weights, outputs);
    printf("  First projection done, running rest...\n");
    fflush(stdout);
    for (int iter = 1; iter < 10; iter++) {
        run_cpu_projections(&spec, hidden, weights, outputs);
    }
    uint64_t t1 = ns_now();
    double ms_per_layer = (t1 - t0) / 1000000.0 / 10.0;
    printf("  Single layer: %.1f ms\n", ms_per_layer);
    printf("  All 30 layers: %.1f ms\n", ms_per_layer * N_LAYERS);
    
    printf("\nTesting batch API (CPU fallback)...\n");
    fflush(stdout);
    ternary_coproc_reset_stats();
    t0 = ns_now();
    printf("  Calling ternary_coproc_run_batch...\n");
    fflush(stdout);
    int ret = ternary_coproc_run_batch(projs, 7 * N_LAYERS);
    printf("  ternary_coproc_run_batch returned %d\n", ret);
    fflush(stdout);
    t1 = ns_now();
    
    TernaryCoprocStats stats;
    ternary_coproc_get_stats(&stats);
    double ms_batch = (t1 - t0) / 1000000.0;
    printf("  Batch of %d projections: %.1f ms\n", 7 * N_LAYERS, ms_batch);
    printf("  Avg per projection: %.2f ms\n", ms_batch / (7.0 * N_LAYERS));
    
    printf("\n=== Analysis ===\n");
    printf("Current (CPU 4-thread): ~%d ms for all projections\n", (int)(ms_per_layer * N_LAYERS));
    printf("GPU potential (12 QPUs × 16 SIMD):\n");
    printf("  Theoretical: 192x parallelism\n");
    printf("  Estimated: 50-100 ms (4-6x speedup)\n");
    
    printf("\nBottlenecks to address:\n");
    printf("  1. Weight loading: weights stay in RAM (need GPU memory prepin)\n");
    printf("  2. QPU program loading: need vc4asm to compile real shaders\n");
    printf("  3. VPM DMA: need optimized strided access patterns\n");
    
    /* Cleanup */
    free(hidden);
    for (int i = 0; i < 7; i++) {
        free(outputs[i]);
        free(weights[i]);
    }
    
    ternary_coproc_shutdown();
    return 0;
}
