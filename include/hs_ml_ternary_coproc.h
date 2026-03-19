/*
 * NeoGPU - V3D Ternary GEMM Coprocessor Interface
 *
 * Provides GPU-accelerated ternary GEMM for BitNet inference.
 * Falls back to CPU (NEON) when GPU is unavailable.
 */

#ifndef HS_ML_TERNARY_COPROC_H
#define HS_ML_TERNARY_COPROC_H

#include <stdint.h>

/*============================================================================
 * Projection Descriptor
 * 
 * Describes a single ternary matrix-vector multiplication:
 *   output[N] = W[N,K] @ input[K]
 * 
 * Where W is packed 2-bit ternary: {-1, 0, +1} encoded as {0, 1, 2}.
 *============================================================================*/

typedef struct {
    float*         output;   /* Output vector [N] */
    const float*   input;    /* Input vector [K] */
    const uint8_t* weights;  /* Packed ternary weights [N * K / 4] */
    uint32_t       N;        /* Output dimension */
    uint32_t       K;        /* Input dimension (must be multiple of 4) */
} TernaryProj;

/*============================================================================
 * Statistics
 *============================================================================*/

typedef struct {
    uint64_t total_time_ns;    /* Total GPU time in nanoseconds */
    uint32_t num_projections;  /* Number of projections completed */
    uint32_t num_qpus_used;    /* Number of QPU threads used */
} TernaryCoprocStats;

/*============================================================================
 * Lifecycle
 *============================================================================*/

/* Initialize V3D coprocessor. Returns 0 on success, -1 on failure. */
int ternary_coproc_init(void);

/* Shutdown V3D coprocessor. */
void ternary_coproc_shutdown(void);

/* Check if coprocessor is available. Returns 1 if active, 0 otherwise. */
int ternary_coproc_available(void);

/*============================================================================
 * Weight Management
 *============================================================================*/

/* 
 * Preload a transformer layer's weights to GPU memory.
 * This avoids per-projection DMA overhead.
 * 
 * Each projection weight matrix is packed ternary: N*K/4 bytes.
 */
int ternary_coproc_load_layer(uint32_t layer_idx,
                               const uint8_t* q_proj,
                               const uint8_t* k_proj,
                               const uint8_t* v_proj,
                               const uint8_t* o_proj,
                               const uint8_t* gate_proj,
                               const uint8_t* up_proj,
                               const uint8_t* down_proj);

/* Unload all preloaded weights from GPU memory. */
void ternary_coproc_unload(void);

/*============================================================================
 * Execution
 *============================================================================*/

/*
 * Submit a batch of projections to the GPU.
 * 
 * Returns:
 *   0  = GPU completed successfully
 *  -1  = GPU not available (use CPU fallback)
 *  -2  = GPU submission failed (use CPU fallback)
 * 
 * When GPU is unavailable or fails, caller should fall back to CPU:
 *   hs_ml_ternary_f32_proj(out, in, W, N, K);
 */
int ternary_coproc_run_batch(const TernaryProj* projs, uint32_t count);

/*
 * Synchronous batch - runs on CPU if GPU unavailable.
 * Always completes, never returns error.
 */
int ternary_coproc_batch(const TernaryProj* projs, uint32_t count);

/*============================================================================
 * Layer-Batched Execution (Optimized)
 * 
 * These functions minimize GPU<->CPU transfer by:
 *  1. Uploading all layer weights once
 *  2. Running all projections with weights resident
 *  3. Keeping activations on GPU between projections where possible
 *============================================================================*/

/* Layer weight descriptor */
typedef struct {
    const uint8_t* q_proj;      /* [H, H] */
    const uint8_t* k_proj;      /* [kv, H] */
    const uint8_t* v_proj;      /* [kv, H] */
    const uint8_t* o_proj;      /* [H, H] */
    const uint8_t* gate_proj;   /* [F, H] */
    const uint8_t* up_proj;     /* [F, H] */
    const uint8_t* down_proj;   /* [H, F] */
    const float* q_scale;       /* Per-tensor scale for q */
    const float* k_scale;
    const float* v_scale;
    const float* o_scale;
    const float* gate_scale;
    const float* up_scale;
    const float* down_scale;
    uint32_t H;                 /* Hidden dimension */
    uint32_t kv;                /* KV dimension (num_kv_heads * head_dim) */
    uint32_t F;                 /* FFN intermediate dimension */
} TernaryLayerWeights;

/* Layer activation buffers (caller-allocated) */
typedef struct {
    const float* input;         /* Layer input [H] */
    float* q;                   /* Q projection output [H] */
    float* k;                   /* K projection output [kv] */
    float* v;                   /* V projection output [kv] */
    float* attn_out;            /* Input to o_proj [H] (after attention) */
    float* o_out;               /* O projection output [H] */
    float* gate_out;            /* Gate projection output [F] */
    float* up_out;              /* Up projection output [F] */
    float* ffn_in;              /* Input to down_proj [F] (after SiLU*gate) */
    float* down_out;            /* Down projection output [H] */
} TernaryLayerActivations;

/*
 * Run attention projections (q, k, v) with weights uploaded once.
 * Returns 0 on GPU success, -1 to fall back to CPU.
 */
int ternary_coproc_attn_qkv(const TernaryLayerWeights* w,
                            const float* input,
                            float* q, float* k, float* v);

/*
 * Run o_proj with weights uploaded once.
 */
int ternary_coproc_attn_o(const TernaryLayerWeights* w,
                          const float* attn_out, float* o_out);

/*
 * Run FFN projections (gate, up, down) with weights uploaded once.
 * Note: SiLU activation happens on CPU between gate/up and down.
 */
int ternary_coproc_ffn(const TernaryLayerWeights* w,
                       const float* input, float* gate_out, float* up_out,
                       const float* ffn_in, float* down_out);

/*============================================================================
 * Statistics
 *============================================================================*/

/* Get current statistics. */
void ternary_coproc_get_stats(TernaryCoprocStats* stats);

/* Reset statistics counters. */
void ternary_coproc_reset_stats(void);

#endif /* HS_ML_TERNARY_COPROC_H */
