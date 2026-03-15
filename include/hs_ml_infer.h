/*
 * NeoGPU ML - Ternary Inference Layer
 *
 * End-to-end inference using the ternary GEMM kernels.
 * Parallel to HSMLSystem (which uses float32 weights) — this struct
 * holds weights in the formats the actual kernels consume:
 *
 *   - Projection weights:  uint8_t* 2-bit packed ternary [N, K/4]
 *   - Projection scales:   float* per-row scale [N]
 *   - Norms:               float* [hidden]
 *   - Embeddings:          float* [vocab, hidden]
 *   - LM head:             float* [vocab, hidden]
 *
 * Data flow for one transformer layer (decode, M=1):
 *
 *   hidden[hidden]  float32
 *       |
 *   RMSNorm (attn)
 *       |
 *   Quantize -> int8[hidden]  (per-tensor, scale = 127/max_abs)
 *       |
 *   Q/K/V proj  -> ternary GEMM -> int32[hidden] -> dequant -> float32
 *       |
 *   RoPE
 *       |
 *   Attention (float32 scores, KV cache)
 *       |
 *   O proj  -> ternary GEMM -> int32 -> dequant -> float32
 *       |
 *   Residual add
 *       |
 *   RMSNorm (ffn)
 *       |
 *   Quantize -> int8
 *       |
 *   Gate proj  -> ternary GEMM -> int32 -> dequant
 *   Up proj    -> ternary GEMM -> int32 -> dequant
 *       |
 *   SwiGLU (SiLU(gate) * up)
 *       |
 *   Quantize -> int8
 *       |
 *   Down proj  -> ternary GEMM -> int32 -> dequant
 *       |
 *   Residual add
 *       |
 *   hidden[hidden]  float32
 */

#ifndef HS_ML_INFER_H
#define HS_ML_INFER_H

#include "hs_core.h"
#include "hs_ml.h"
#include "hs_ml_routing.h"

/*
 * Per-layer ternary weights.
 * All projection matrices are 2-bit packed ternary with per-row float scales.
 */
typedef struct {
    /* Attention projections: [hidden, hidden] ternary */
    u8*    q_proj;      /* [hidden, hidden/4] packed 2-bit */
    float* q_scale;     /* [hidden] per-row scales */

    u8*    k_proj;
    float* k_scale;

    u8*    v_proj;
    float* v_scale;

    u8*    o_proj;
    float* o_scale;

    /* FFN projections */
    u8*    gate_proj;   /* [ffn_hidden, hidden/4] */
    float* gate_scale;

    u8*    up_proj;
    float* up_scale;

    u8*    down_proj;   /* [hidden, ffn_hidden/4] */
    float* down_scale;

    /* Learned norms (float -- not quantized) */
    float* attn_norm;      /* [hidden] */
    float* attn_sub_norm;  /* [hidden] post-attention subln */
    float* ffn_norm;       /* [hidden] */
    float* ffn_sub_norm;   /* [ffn_hidden] post-activation subln */
} HSTernaryLayer;

/*
 * Full ternary model.
 */
typedef struct {
    /* Config */
    u32 vocab_size;
    u32 hidden_size;
    u32 num_layers;
    u32 num_heads;
    u32 num_kv_heads;  /* KV heads for GQA */
    u32 head_dim;
    u32 ffn_hidden_size;
    u32 max_context;
    float rope_theta;

    /* Non-quantized weights */
    float* embedding;   /* [vocab, hidden] float32 */
    float* lm_head;     /* [vocab, hidden] float32 */
    float* final_norm;  /* [hidden] float32 */

    /* Tokenizer metadata from GGUF */
    char** tokenizer_vocab;
    u32    tokenizer_bos;
    u32    tokenizer_eos;
    u32    tokenizer_pad;
    char** tokenizer_merges;
    u32    num_merges;

    /* Per-layer ternary weights */
    HSTernaryLayer* layers; /* [num_layers] */

    /* Runtime scratch (allocated once — no hot malloc in forward pass) */
    int8_t*  quant_buf;  /* [max(hidden, ffn_hidden)] quantized activations */
    int32_t* gemm_buf;   /* [max(hidden, ffn_hidden)] GEMM output */
    float*   hidden_buf; /* [hidden] */
    float*   ffn_buf;    /* [ffn_hidden] */
    float*   q_buf;      /* [hidden] Q projection output */
    float*   k_buf;      /* [hidden] K projection output */
    float*   v_buf;      /* [hidden] V projection output */
    float*   attn_buf;   /* [hidden] attention output */
    float*   gate_buf;   /* [ffn_hidden] gate projection output */
    float*   up_buf;     /* [ffn_hidden] up projection output */
    float*   score_buf;  /* [max_context] attention scores */
    int8_t*  ffn_qbuf;   /* [ffn_hidden] quantized FFN activations */

    bool use_i2s;
    bool loaded;
} HSMLTernary;

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
 * seed: random seed for reproducibility
 */
int hs_mlt_alloc_random(HSMLTernary* m,
                        u32 vocab_size, u32 hidden_size, u32 num_layers,
                        u32 num_heads, u32 ffn_hidden_size, u32 max_context,
                        unsigned int seed);

/*============================================================================
 * Inference
 *============================================================================*/

/*
 * Quantize float32 vector to int8 using per-tensor absmax scaling.
 * Returns the scale factor used (127 / max_abs).
 * output: [n] int8
 * input:  [n] float32
 */
float hs_mlt_quantize(int8_t* output, const float* input, u32 n);

/*
 * Dequantize int32 GEMM output to float32.
 * output: [n] float32
 * input:  [n] int32
 * act_scale:    scale from activation quantization
 * weight_scales: [n] per-row weight scales
 */
void hs_mlt_dequantize(float* output, const int32_t* input, u32 n,
                       float act_scale, const float* weight_scales);

/*
 * Run one transformer layer forward pass (decode mode, M=1).
 *
 * hidden:    [hidden_size] float32, modified in place
 * layer_idx: which layer
 * cache:     KV cache for this layer (updated in place)
 * m:         model
 */
void hs_mlt_layer_forward(float* hidden, u32 layer_idx,
                          HSKVCache* cache, HSMLTernary* m);

/*
 * Full forward pass: token -> logits.
 * For each token in sequence, runs all layers, returns logits for last token.
 *
 * logits: [vocab_size] float32 output
 * tokens: [seq_len] token IDs
 */
int hs_mlt_forward(HSMLTernary* m, const u32* tokens, u32 seq_len,
                   float* logits);

/*
 * Greedy token sampling.
 */
u32 hs_mlt_sample_greedy(const float* logits, u32 vocab_size);

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
/* Stats are stored inline in the struct for simplicity */
extern HSMLTernaryStats g_mlt_stats;  /* global, reset on each forward pass */

#endif /* HS_ML_INFER_H */


/* Stateful decode session */
typedef struct {
    HSMLTernary* model;
    float* hidden;
    HSKVCache* caches;
    u32 seq_len;
    bool ready;
} HSMLTernarySession;

int  hs_mlt_load_gguf(HSMLTernary* m, const char* path);
int  hs_mlt_session_init(HSMLTernarySession* sess, HSMLTernary* model);
void hs_mlt_session_free(HSMLTernarySession* sess);
void hs_mlt_session_reset(HSMLTernarySession* sess);
int  hs_mlt_prefill(HSMLTernarySession* sess, const u32* tokens, u32 seq_len);
int  hs_mlt_decode(HSMLTernarySession* sess, u32 token, float* logits);
int  hs_mlt_session_logits(HSMLTernarySession* sess, float* logits);
