/*
 * Benchmark: Zero-copy GPU GEMM vs old upload-per-call approach
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;

/* New zero-copy API */
extern int gpu_gemm_init(void);
extern void gpu_gemm_shutdown(void);
extern void* gpu_gemm_alloc_weights(u32 layer_idx, size_t bytes);
extern void gpu_gemm_set_dims(u32 H, u32 kv, u32 F);
extern int gpu_gemm_run(u32 layer_idx, u32 weight_offset,
                        const float* input, float* output, u32 N, u32 K);
extern int gpu_gemm_run_batch(u32 layer_idx,
                              const float* input, u32 input_K,
                              float* const* outputs,
                              const u32* weight_offsets,
                              const u32* Ns, const u32* Ks, u32 count);

/* CPU reference */
extern void hs_ml_ternary_f32_proj(float* out, const float* in,
                                   const u8* W, u32 N, u32 K);

static u64 now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (u64)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/* Generate random ternary weights in I2_S format */
static void gen_weights(u8* W, u32 N, u32 K) {
    u32 bytes = N * K / 4;
    for (u32 i = 0; i < bytes; i++) {
        /* Random 2-bit codes: 0,1,2 (avoid 3 which is also zero) */
        u8 c0 = rand() % 3;
        u8 c1 = rand() % 3;
        u8 c2 = rand() % 3;
        u8 c3 = rand() % 3;
        W[i] = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6);
    }
}

static void gen_input(float* in, u32 K) {
    for (u32 i = 0; i < K; i++)
        in[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

static float max_diff(const float* a, const float* b, u32 n) {
    float m = 0;
    for (u32 i = 0; i < n; i++) {
        float d = a[i] - b[i];
        if (d < 0) d = -d;
        if (d > m) m = d;
    }
    return m;
}

int main(void) {
    srand(42);

    /* BitNet 2B4T dimensions */
    const u32 H = 2560;
    const u32 kv = 640;
    const u32 F = 6912;

    /* Calculate weight sizes for one layer */
    size_t q_size  = H * H / 4;
    size_t k_size  = kv * H / 4;
    size_t v_size  = kv * H / 4;
    size_t o_size  = H * H / 4;
    size_t gate_size = F * H / 4;
    size_t up_size   = F * H / 4;
    size_t down_size = H * F / 4;
    size_t layer_size = q_size + k_size + v_size + o_size + 
                        gate_size + up_size + down_size;

    printf("Layer weight size: %.2f MB\n", layer_size / (1024.0 * 1024.0));
    fflush(stdout);

    /* Initialize GPU */
    fprintf(stderr, "DEBUG: About to call gpu_gemm_init\n");
    if (gpu_gemm_init() != 0) {
        printf("GPU init failed, running CPU-only\n");
        return 1;
    }
    fprintf(stderr, "DEBUG: gpu_gemm_init returned\n");
    fflush(stderr);
    
    printf("gpu_gemm_init done, calling gpu_gemm_set_dims...\n");
    fflush(stdout);
    gpu_gemm_set_dims(H, kv, F);
    printf("gpu_gemm_set_dims done\n");
    fflush(stdout);

    /* Allocate and fill weights in shared memory */
    printf("Allocating weights in shared memory...\n");
    fflush(stdout);
    fprintf(stderr, "DEBUG: About to call gpu_gemm_alloc_weights\n");
    fflush(stderr);
    u8* weights = (u8*)gpu_gemm_alloc_weights(0, layer_size);
    fprintf(stderr, "DEBUG: gpu_gemm_alloc_weights returned %p\n", weights);
    fflush(stderr);
    if (!weights) {
        printf("Failed to allocate shared weight buffer\n");
        gpu_gemm_shutdown();
        return 1;
    }
    gen_weights(weights, H, H);  /* Generate Q projection weights: N=H, K=H */
    fprintf(stderr, "DEBUG: gen_weights done\n");
    fflush(stderr);

    /* Weight offsets within the buffer */
    u32 off_q    = 0;
    u32 off_k    = off_q + q_size;
    u32 off_v    = off_k + k_size;
    u32 off_o    = off_v + v_size;
    u32 off_gate = off_o + o_size;
    u32 off_up   = off_gate + gate_size;
    u32 off_down = off_up + up_size;

    /* Allocate activation buffers */
    fprintf(stderr, "DEBUG: About to allocate buffers\n");
    fflush(stderr);
    float* input = malloc(H * sizeof(float));
    float* q_out = malloc(H * sizeof(float));
    float* k_out = malloc(kv * sizeof(float));
    float* v_out = malloc(kv * sizeof(float));
    float* o_out = malloc(H * sizeof(float));
    float* gate_out = malloc(F * sizeof(float));
    float* up_out = malloc(F * sizeof(float));
    float* down_out = malloc(H * sizeof(float));

    float* q_cpu = malloc(H * sizeof(float));
    float* k_cpu = malloc(kv * sizeof(float));
    float* v_cpu = malloc(kv * sizeof(float));
    fprintf(stderr, "DEBUG: Buffers allocated\n");
    fflush(stderr);

    gen_input(input, H);
    fprintf(stderr, "DEBUG: gen_input done\n");
    fflush(stderr);

    /* ========== Correctness test ========== */
    printf("\nCorrectness test (Q projection):\n");
    fflush(stdout);
    
    /* GPU */
    fprintf(stderr, "DEBUG: About to call gpu_gemm_run\n");
    fflush(stderr);
    gpu_gemm_run(0, off_q, input, q_out, H, H);
    fprintf(stderr, "DEBUG: gpu_gemm_run done\n");
    fflush(stderr);
    
    /* CPU reference */
    hs_ml_ternary_f32_proj(q_cpu, input, weights + off_q, H, H);
    
    float diff = max_diff(q_out, q_cpu, H);
    printf("  Max diff: %e %s\n", diff, diff < 1e-5 ? "PASS" : "FAIL");

    /* ========== Single projection benchmark ========== */
    printf("\nSingle projection benchmark (Q: %ux%u):\n", H, H);
    
    /* GPU - warm up */
    for (int i = 0; i < 3; i++)
        gpu_gemm_run(0, off_q, input, q_out, H, H);
    
    /* GPU - timed */
    int iters = 20;
    u64 t0 = now_ns();
    for (int i = 0; i < iters; i++)
        gpu_gemm_run(0, off_q, input, q_out, H, H);
    u64 t1 = now_ns();
    double gpu_ms = (t1 - t0) / 1e6 / iters;
    
    /* CPU - timed */
    t0 = now_ns();
    for (int i = 0; i < iters; i++)
        hs_ml_ternary_f32_proj(q_cpu, input, weights + off_q, H, H);
    t1 = now_ns();
    double cpu_ms = (t1 - t0) / 1e6 / iters;
    
    printf("  GPU: %.2f ms\n", gpu_ms);
    printf("  CPU: %.2f ms\n", cpu_ms);
    printf("  Speedup: %.2fx\n", cpu_ms / gpu_ms);

    /* ========== Layer batch benchmark (QKV) ========== */
    printf("\nBatched QKV projections:\n");
    
    float* outputs[3] = { q_out, k_out, v_out };
    u32 offsets[3] = { off_q, off_k, off_v };
    u32 Ns[3] = { H, kv, kv };
    u32 Ks[3] = { H, H, H };
    
    /* GPU batch - warm up */
    for (int i = 0; i < 3; i++)
        gpu_gemm_run_batch(0, input, H, outputs, offsets, Ns, Ks, 3);
    
    /* GPU batch - timed */
    t0 = now_ns();
    for (int i = 0; i < iters; i++)
        gpu_gemm_run_batch(0, input, H, outputs, offsets, Ns, Ks, 3);
    t1 = now_ns();
    double gpu_batch_ms = (t1 - t0) / 1e6 / iters;
    
    /* CPU - timed */
    t0 = now_ns();
    for (int i = 0; i < iters; i++) {
        hs_ml_ternary_f32_proj(q_cpu, input, weights + off_q, H, H);
        hs_ml_ternary_f32_proj(k_cpu, input, weights + off_k, kv, H);
        hs_ml_ternary_f32_proj(v_cpu, input, weights + off_v, kv, H);
    }
    t1 = now_ns();
    double cpu_batch_ms = (t1 - t0) / 1e6 / iters;
    
    printf("  GPU batch: %.2f ms\n", gpu_batch_ms);
    printf("  CPU sequential: %.2f ms\n", cpu_batch_ms);
    printf("  Speedup: %.2fx\n", cpu_batch_ms / gpu_batch_ms);

    /* ========== Full layer simulation (7 projections) ========== */
    printf("\nFull layer (7 projections):\n");
    
    float* all_outputs[7] = { q_out, k_out, v_out, o_out, gate_out, up_out, down_out };
    u32 all_offsets[7] = { off_q, off_k, off_v, off_o, off_gate, off_up, off_down };
    u32 all_Ns[7] = { H, kv, kv, H, F, F, H };
    u32 all_Ks[7] = { H, H, H, H, H, H, F };
    /* Note: In reality, projections use different inputs. This is a simplification. */
    
    /* GPU - all 7 projections */
    t0 = now_ns();
    for (int i = 0; i < iters; i++) {
        /* Attention: Q, K, V use same input */
        gpu_gemm_run_batch(0, input, H, all_outputs, all_offsets, all_Ns, all_Ks, 3);
        /* O uses attention output (but we use same input for simplicity) */
        gpu_gemm_run(0, off_o, input, o_out, H, H);
        /* FFN: gate, up use layer output */
        float* ffn_outputs[2] = { gate_out, up_out };
        u32 ffn_offsets[2] = { off_gate, off_up };
        u32 ffn_Ns[2] = { F, F };
        u32 ffn_Ks[2] = { H, H };
        gpu_gemm_run_batch(0, input, H, ffn_outputs, ffn_offsets, ffn_Ns, ffn_Ks, 2);
        /* Down uses gate*up output */
        float* down_in = gate_out;  /* Would be gate*up in reality */
        gpu_gemm_run(0, off_down, down_in, down_out, H, F);
    }
    t1 = now_ns();
    double gpu_layer_ms = (t1 - t0) / 1e6 / iters;
    
    /* CPU - all 7 projections */
    t0 = now_ns();
    for (int i = 0; i < iters; i++) {
        hs_ml_ternary_f32_proj(q_cpu, input, weights + off_q, H, H);
        hs_ml_ternary_f32_proj(k_cpu, input, weights + off_k, kv, H);
        hs_ml_ternary_f32_proj(v_cpu, input, weights + off_v, kv, H);
        hs_ml_ternary_f32_proj(o_out, input, weights + off_o, H, H);
        hs_ml_ternary_f32_proj(gate_out, input, weights + off_gate, F, H);
        hs_ml_ternary_f32_proj(up_out, input, weights + off_up, F, H);
        hs_ml_ternary_f32_proj(down_out, gate_out, weights + off_down, H, F);
    }
    t1 = now_ns();
    double cpu_layer_ms = (t1 - t0) / 1e6 / iters;
    
    printf("  GPU: %.2f ms/layer\n", gpu_layer_ms);
    printf("  CPU: %.2f ms/layer\n", cpu_layer_ms);
    printf("  Speedup: %.2fx\n", cpu_layer_ms / gpu_layer_ms);
    
    /* Projection of 30 layers */
    printf("\nProjected 30-layer decode step:\n");
    printf("  GPU: %.0f ms\n", gpu_layer_ms * 30);
    printf("  CPU: %.0f ms\n", cpu_layer_ms * 30);

    /* Cleanup */
    free(input); free(q_out); free(k_out); free(v_out);
    free(o_out); free(gate_out); free(up_out); free(down_out);
    free(q_cpu); free(k_cpu); free(v_cpu);
    gpu_gemm_shutdown();

    return 0;
}
