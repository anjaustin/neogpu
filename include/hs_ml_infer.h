/*
 * NeoGPU ML - Ternary Inference Layer
 *
 * End-to-end inference using the ternary GEMM kernels.
 *
 * Two weight paths:
 *   - use_i2s=false: 2-bit packed ternary with int8 quantized activations
 *   - use_i2s=true:  raw I2_S (GGUF type 36) with pure float32 activations
 *
 * BitNet b1.58 layer order (per experimental/BitNet/gpu/model.py):
 *   hidden -> attn_norm -> Q/K/V proj -> RoPE -> attention
 *          -> attn_sub_norm -> o_proj -> residual
 *          -> ffn_norm -> gate/up proj -> ReLU^2(gate)*up
 *          -> ffn_sub_norm -> down_proj -> residual
 */

#ifndef HS_ML_INFER_H
#define HS_ML_INFER_H

#include "hs_core.h"
#include "hs_ml.h"
#include "hs_ml_routing.h"

/*
 * Per-layer ternary weights.
 * All projection matrices are 2-bit packed ternary with per-row float scales.
 * I2_S format: raw GGUF I2_S bytes, NOT re-packed. Decoded on-the-fly.
 */
typedef struct {
    /* Attention projections */
    u8*    q_proj;      /* [hidden, hidden/4] packed ternary */
    float* q_scale;     /* [hidden] per-row scales */

    u8*    k_proj;      /* [kv_size, hidden/4] */
    float* k_scale;

    u8*    v_proj;      /* [kv_size, hidden/4] */
    float* v_scale;

    u8*    o_proj;      /* [hidden, hidden/4] */
    float* o_scale;

    /* FFN projections */
    u8*    gate_proj;   /* [ffn_hidden, hidden/4] */
    float* gate_scale;

    u8*    up_proj;     /* [ffn_hidden, hidden/4] */
    float* up_scale;

    u8*    down_proj;   /* [hidden, ffn_hidden/4] */
    float* down_scale;

    /* Learned norms (float32 — not quantized) */
    float* attn_norm;       /* [hidden]     pre-attention RMSNorm */
    float* attn_sub_norm;   /* [hidden]     BitNet subln after attention, before o_proj */
    float* ffn_norm;        /* [hidden]     pre-FFN RMSNorm */
    float* ffn_sub_norm;    /* [ffn_hidden] BitNet subln after gate*up, before down_proj */
} HSTernaryLayer;

/*
 * Full ternary model.
 */
typedef struct {
    /* Architecture config */
    u32   vocab_size;
    u32   hidden_size;
    u32   num_layers;
    u32   num_heads;
    u32   num_kv_heads;     /* GQA: may be < num_heads */
    u32   head_dim;         /* hidden_size / num_heads */
    u32   ffn_hidden_size;
    u32   max_context;
    float rope_theta;       /* RoPE base frequency (500000 for BitNet 2B-4T) */

    /* Weight format flag */
    bool  use_i2s;          /* true = raw I2_S bytes + f32 activation path */

    /* Non-quantized weights */
    float* embedding;      /* [vocab, hidden] float32 (for embedding lookup) */
    u16*   lm_head_f16;    /* [vocab, hidden] float16 — freed after trit encode if use_trit_lmhead */
    float* final_norm;     /* [hidden] float32 */

    /* Ternary spline lm_head (two-stage, replaces scalar F16 loop when use_trit_lmhead=true)
     *
     * Encoding: for each vocab row v, row_scale[v] = max(|emb[v,:]|).
     * 8 I2_S planes encode the row as a balanced ternary residual series:
     *   plane[0] = round(clip(row/scale,      -1,+1))        weight 1
     *   plane[1] = round(clip(residual*3,     -1,+1))        weight 1/3
     *   ...
     *   plane[k] = round(clip(residual*3^k,   -1,+1))        weight (1/3)^k
     *
     * Inference:
     *   Stage 1: accumulate planes 0..3 over all V rows   -> coarse logits
     *   Stage 2: accumulate planes 4..7 over top-C rows   -> residual refinement
     *   Final:   logit[v] = row_scale[v] * (coarse[v] + residual[v])
     */
    bool   use_trit_lmhead;           /* true after hs_mlt_lmhead_encode() */
    u8*    lm_head_planes[8];         /* [vocab * hidden/4] each, I2_S packed */
    u16*   lm_head_row_scale;         /* [vocab] F16 per-row scale */

    /* Per-layer ternary weights */
    HSTernaryLayer* layers; /* [num_layers] */

    /* Runtime scratch (allocated once — no hot malloc in forward pass) */
    int8_t*  quant_buf;  /* [max(hidden, ffn_hidden)] int8 activations */
    int32_t* gemm_buf;   /* [max(hidden, ffn_hidden)] GEMM int32 output */
    float*   hidden_buf; /* [hidden] general float scratch */
    float*   ffn_buf;    /* [ffn_hidden] FFN float scratch */
    float*   q_buf;      /* [hidden] Q projection output */
    float*   k_buf;      /* [kv_size] K projection output */
    float*   v_buf;      /* [kv_size] V projection output */
    float*   attn_buf;   /* [hidden] attention output */
    float*   gate_buf;   /* [ffn_hidden] gate projection output */
    float*   up_buf;     /* [ffn_hidden] up projection output */
    float*   score_buf;  /* [max_context] attention scores */
    int8_t*  ffn_qbuf;   /* [ffn_hidden] quantized FFN activations (int8 path) */

    /* Tokenizer (optional — loaded from GGUF tokenizer.ggml.* metadata) */
    char**  tokenizer_vocab;    /* [vocab_size] token strings */
    char**  tokenizer_merges;   /* BPE merge rules */
    u32     num_merges;
    u32     tokenizer_bos;      /* BOS token ID */
    u32     tokenizer_eos;      /* EOS token ID */

    /* BPE merge rank lookup (built from tokenizer_merges on first use) */
    u32*    merge_ranks;        /* hash table: merge pair → rank */
    u32     merge_hash_size;

    bool loaded;
} HSMLTernary;

/*
 * BPE tokenizer encode (uses tokenizer_vocab + tokenizer_merges).
 * Returns number of tokens written to output.
 */
u32 hs_mlt_bpe_encode(HSMLTernary* m, const char* text, u32 text_len,
                      u32* output, u32 max_tokens);

/*============================================================================
 * Lifecycle
 *============================================================================*/

/* Initialize (zero the struct) */
void hs_mlt_init(HSMLTernary* m);

/* Free all allocations */
void hs_mlt_free(HSMLTernary* m);

/*
 * Allocate synthetic random weights for testing.
 * Ternary weights are random {-1,0,+1}, norms are 1.0, embeddings are random.
 */
int hs_mlt_alloc_random(HSMLTernary* m,
                        u32 vocab_size, u32 hidden_size, u32 num_layers,
                        u32 num_heads, u32 ffn_hidden_size, u32 max_context,
                        unsigned int seed);

/*
 * Load model from GGUF file.
 * Automatically detects I2_S format and sets use_i2s=true.
 * Loads tokenizer vocab from GGUF metadata if present.
 */
int hs_mlt_load_gguf(HSMLTernary* m, const char* path);

/*
 * Fused 4-plane lm_head Stage 1 kernel (NEON + pthreads), float input.
 * out[n] = row_scale[n] * sum_{k=0..3} (1/3)^k * dot(in, Pk[n])
 */
void hs_ml_lmhead_stage1(float *out, const float *in,
                          const u8 *P0, const u8 *P1,
                          const u8 *P2, const u8 *P3,
                          const u16 *row_scale,
                          u32 N, u32 K);

/*
 * Fused 4-plane lm_head Stage 1 kernel — int8 activation path (fastest).
 * Pre-quantize hidden to int8 (act_scale = 127/max(|h|)), then call this.
 * Uses vmull_s8 integer multiply — 3.4× faster than F16 baseline.
 * out[n] = (1/act_scale) * row_scale[n] * sum_{k=0..3} (1/3)^k * dot_i8(in_i8, Pk[n])
 */
void hs_ml_lmhead_stage1_i8(float *out, const int8_t *in_i8, float act_scale,
                              const u8 *P0, const u8 *P1,
                              const u8 *P2, const u8 *P3,
                              const u16 *row_scale,
                              u32 N, u32 K);

/*
 * Encode lm_head F16 weights into 8 ternary spline planes.
 * Called once after hs_mlt_load_gguf(). Frees lm_head_f16 on success.
 * Sets m->use_trit_lmhead = true.
 */
int hs_mlt_lmhead_encode(HSMLTernary* m);

/*
 * Load BF16 norm weights from sidecar binary.
 * Overwrites layer norms and final_norm loaded from GGUF.
 * Format: magic(u32) + n_tensors(u32) + [name_len(u32) + name + n_values(u32) + bf16_data]*
 */
int hs_mlt_load_norms_sidecar(HSMLTernary* m, const char* path);

/*============================================================================
 * Inference
 *============================================================================*/

/*
 * Quantize float32 vector to int8 using per-tensor absmax scaling.
 * Returns the scale factor used (127 / max_abs).
 */
float hs_mlt_quantize(int8_t* output, const float* input, u32 n);

/*
 * Dequantize int32 GEMM output to float32.
 */
void hs_mlt_dequantize(float* output, const int32_t* input, u32 n,
                       float act_scale, const float* weight_scales);

/*
 * Run one transformer layer forward pass (decode mode, M=1).
 * hidden: [hidden_size] float32, modified in place.
 */
void hs_mlt_layer_forward(float* hidden, u32 layer_idx,
                          HSKVCache* cache, HSMLTernary* m);

/*
 * Full forward pass: token sequence -> logits.
 */
int hs_mlt_forward(HSMLTernary* m, const u32* tokens, u32 seq_len,
                   float* logits);

/*
 * Greedy token sampling.
 */
u32 hs_mlt_sample_greedy(const float* logits, u32 vocab_size);

/*============================================================================
 * Stateful decode session
 *============================================================================*/

/* Candidate set size for two-stage lm_head.
 * Stage 1 nominates this many rows; Stage 2 refines them.
 * 200 gives 100% top-42 recall on measured data with a safety margin. */
#define LMH_CANDIDATES 200

typedef struct {
    HSMLTernary*  model;
    float*        hidden;    /* [hidden_size] current hidden state */
    HSKVCache*    caches;    /* [num_layers] KV caches */
    u32           seq_len;
    bool          ready;

    /* Scratch for two-stage ternary spline lm_head (allocated in session_init) */
    float*   lmh_coarse;      /* [vocab_size] coarse logit accumulator (Stage 1) */
    float*   lmh_tmp;         /* [vocab_size] single-plane projection scratch     */
    u32*     lmh_candidates;  /* [LMH_CANDIDATES] top-C vocab indices             */
    u8*      lmh_plane_buf;   /* [LMH_CANDIDATES * hidden/4] Stage 2 row copy buf */
    float*   lmh_tmp_c;       /* [LMH_CANDIDATES] Stage 2 projection scratch      */
    int8_t*  lmh_in_i8;       /* [hidden_size] int8 quantized normed hidden       */
} HSMLTernarySession;

int  hs_mlt_session_init(HSMLTernarySession* sess, HSMLTernary* model);
void hs_mlt_session_free(HSMLTernarySession* sess);
void hs_mlt_session_reset(HSMLTernarySession* sess);

/* Prefill: process prompt tokens, update KV cache. */
int hs_mlt_prefill(HSMLTernarySession* sess, const u32* tokens, u32 seq_len);

/* Get logits from current hidden state (after prefill). */
int hs_mlt_session_logits(HSMLTernarySession* sess, float* logits);

/* Decode: step one new token, update KV cache, write logits. */
int hs_mlt_decode(HSMLTernarySession* sess, u32 token, float* logits);

/*============================================================================
 * Stats
 *============================================================================*/

typedef struct {
    u64 total_gemm_calls;
    u64 total_gemm_ops;    /* M*N*K accumulated */
    u64 total_ns;
    u32 tokens_generated;
} HSMLTernaryStats;

void hs_mlt_reset_stats(HSMLTernary* m);
extern HSMLTernaryStats g_mlt_stats;

#endif /* HS_ML_INFER_H */
