/*
 * NeoGPU - Zero-Copy GPU Ternary GEMM
 *
 * High-performance GPU acceleration for ternary matrix-vector multiplication.
 * Uses persistent-mapped buffers for zero-copy CPU/GPU memory sharing.
 *
 * Usage:
 *   1. Call gpu_gemm_init() once at startup
 *   2. Call gpu_gemm_alloc_weights() for each layer, load weights into returned pointer
 *   3. Call gpu_gemm_set_dims() with model dimensions
 *   4. Call gpu_gemm_run() or gpu_gemm_run_batch() during inference
 *   5. Call gpu_gemm_shutdown() at cleanup
 */

#ifndef HS_ML_GPU_GEMM_H
#define HS_ML_GPU_GEMM_H

#include <stdint.h>
#include <stddef.h>

/*
 * Initialize GPU GEMM engine.
 * Returns 0 on success, -1 if GPU unavailable.
 */
int gpu_gemm_init(void);

/*
 * Shutdown GPU GEMM engine.
 */
void gpu_gemm_shutdown(void);

/*
 * Check if GPU GEMM is available.
 */
int gpu_gemm_available(void);

/*
 * Allocate weight buffer in shared CPU/GPU memory.
 * 
 * Returns CPU pointer where caller should load ternary weights.
 * GPU will read from same memory with zero copy.
 * 
 * layer_idx: Layer index (0 to MAX_LAYERS-1)
 * bytes: Size in bytes (N * K / 4 for ternary)
 * 
 * Returns NULL on failure.
 */
void* gpu_gemm_alloc_weights(uint32_t layer_idx, size_t bytes);

/*
 * Set model dimensions (needed for buffer sizing).
 */
void gpu_gemm_set_dims(uint32_t H, uint32_t kv, uint32_t F);

/*
 * Run a single ternary projection.
 * 
 * layer_idx: Which layer's weights to use
 * weight_offset: Byte offset within layer's weight buffer
 * input: Input vector [K floats]
 * output: Output vector [N floats]
 * N: Output dimension
 * K: Input dimension (must be multiple of 4)
 * 
 * Returns 0 on success, -1 on failure.
 */
int gpu_gemm_run(uint32_t layer_idx, uint32_t weight_offset,
                 const float* input, float* output,
                 uint32_t N, uint32_t K);

/*
 * Like gpu_gemm_run but doesn't sync. Caller must call gpu_gemm_sync() 
 * after all projections are dispatched.
 */
int gpu_gemm_run_nosync(uint32_t layer_idx, uint32_t weight_offset,
                        const float* input, float* output,
                        uint32_t N, uint32_t K);

/*
 * Sync after batched dispatches.
 */
void gpu_gemm_sync(void);

/*
 * Run multiple projections with shared input, single sync.
 * 
 * layer_idx: Which layer's weights to use
 * input: Input vector [input_K floats]
 * input_K: Input dimension
 * outputs: Array of output pointers
 * weight_offsets: Byte offset for each projection
 * Ns: Output dimension for each projection
 * Ks: Input dimension for each projection (usually all same)
 * count: Number of projections (max 16)
 * 
 * Returns 0 on success, -1 on failure.
 */
int gpu_gemm_run_batch(uint32_t layer_idx,
                       const float* input, uint32_t input_K,
                       float* const* outputs,
                       const uint32_t* weight_offsets,
                       const uint32_t* Ns, const uint32_t* Ks,
                       uint32_t count);

/*
 * Allocate unified weight buffer for batched all-layers processing.
 * All layers' weights go in one buffer.
 */
int gpu_gemm_alloc_unified_weights(size_t total_bytes);

/*
 * Get pointer to unified weight buffer for loading weights.
 */
void* gpu_gemm_get_unified_weight_ptr(void);

/*
 * Get direct pointer to input/output buffers for benchmarking.
 */
void gpu_gemm_get_buffer_ptrs(void** input, void** output);

/*
 * Run Q projection for ALL layers in ONE dispatch.
 * Each workgroup handles one layer.
 * 
 * input: [K] input activations (same for all layers)
 * outputs: [num_layers][N] output for each layer
 * layer_offsets: byte offset for each layer's Q weights in unified buffer
 * N: output dimension per layer
 * K: input dimension  
 * num_layers: number of layers
 */
int gpu_gemm_run_all_layers_q(const float* input, 
                               float** outputs,
                               const uint32_t* layer_offsets,
                               uint32_t N, uint32_t K, 
                               uint32_t num_layers);

/*
 * Run full layer's QKV + O projections (attention block) in one GPU batch.
 * Single sync point for 4 projections.
 * 
 * layer_idx: Layer index for weight lookup
 * input: Layer input [H floats]
 * q_out: Q projection output [H floats]
 * k_out: K projection output [kv floats]
 * v_out: V projection output [kv floats]
 * attn_out: Attention output (after attention) [H floats]
 * o_out: O projection output [H floats]
 * 
 * Returns 0 on GPU success, -1 on fallback.
 */
int gpu_gemm_run_attn(uint32_t layer_idx,
                      const float* input,
                      float* q_out, float* k_out, float* v_out,
                      const float* attn_out, float* o_out);

/*
 * Run full layer's FFN projections (gate + up + down) in one GPU batch.
 * Single sync point for 3 projections.
 * 
 * layer_idx: Layer index for weight lookup
 * input: FFN input [H floats]
 * gate_out: Gate projection output [F floats]
 * up_out: Up projection output [F floats]
 * ffn_in: SiLU(gate)*up output [F floats] (input to down_proj)
 * down_out: Down projection output [H floats]
 * 
 * Returns 0 on GPU success, -1 on fallback.
 */
int gpu_gemm_run_ffn(uint32_t layer_idx,
                     const float* input,
                     float* gate_out, float* up_out,
                     const float* ffn_in, float* down_out);

/*
 * Allocate lm_head weights on GPU (for vocab projection).
 * Returns pointer to copy weights into.
 */
void* gpu_gemm_alloc_lmhead(size_t bytes);

/*
 * Run lm_head projection on GPU.
 * input: [H] input vector (normalized hidden)
 * output: [V] output logits
 * Returns 0 on success, -1 on failure.
 */
int gpu_gemm_run_lmhead(const float* input, float* output, uint32_t V, uint32_t H);

/*
 * Async lm_head: starts GPU compute, returns immediately.
 * Returns 0 on launch success, -1 on failure.
 * Call gpu_gemm_poll_lmhead() or gpu_gemm_wait_lmhead() to complete.
 */
int gpu_gemm_run_lmhead_async(const float* input, float* output, uint32_t V, uint32_t H);

/*
 * Poll for async lm_head completion.
 * Returns: 0 = still running, 1 = complete, -1 = error.
 */
int gpu_gemm_poll_lmhead(void);

/*
 * Wait for async lm_head to complete.
 * Returns 0 on success, -1 on error.
 */
int gpu_gemm_wait_lmhead(void);

#endif /* HS_ML_GPU_GEMM_H */
