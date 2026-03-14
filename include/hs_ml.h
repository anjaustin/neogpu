/*
 * NeoGPU ML - INT8/Ternary GEMM for BitNet
 * 
 * ARM NEON optimized matrix operations for 1.58-bit LLM inference.
 */

#ifndef HS_ML_H
#define HS_ML_H

#include "hs_core.h"

#define HS_ML_QK_I2_S     64          /* 64 weights per ternary block */
#define HS_ML_MAX_LAYERS   32          /* Max transformer layers */
#define HS_ML_MAX_VOCAB   128256      /* Max vocabulary size */

/*
 * Tensor: multi-dimensional array
 */
typedef struct {
    u32 shape[4];          /* N, C, H, W */
    u32 stride[4];
    void* data;
} HSTensor;

/*
 * ML System for BitNet inference
 */
typedef struct {
    /* Model weights - per-layer */
    float* embedding;               /* Token embeddings [vocab, hidden] */
    float* lm_head;                /* LM head [vocab, hidden] */
    float* final_norm;             /* Output layer norm [hidden] */
    
    /* Per-layer weights (num_layers of each) */
    float* attn_q_proj;           /* [layers][hidden, hidden] */
    float* attn_k_proj;           /* [layers][hidden, hidden] */
    float* attn_v_proj;           /* [layers][hidden, hidden] */
    float* attn_o_proj;           /* [layers][hidden, hidden] */
    float* ffn_gate_proj;         /* [layers][hidden, ffn_hidden] */
    float* ffn_up_proj;           /* [layers][hidden, ffn_hidden] */
    float* ffn_down_proj;         /* [layers][ffn_hidden, hidden] */
    float* attn_norm;             /* [layers][hidden] */
    float* ffn_norm;              /* [layers][hidden] */
    
    /* Config */
    u32 vocab_size;
    u32 hidden_size;
    u32 num_layers;
    u32 num_heads;
    u32 head_dim;
    u32 ffn_hidden_size;
    u32 max_context;
    
    /* Tokenizer (minimal BPE) */
    u32* tokenizer_table;
    char** tokenizer_vocab;
    
    /* Workspace */
    void* work_buffer;
    size_t work_size;
    
    /* State */
    bool loaded;
} HSMLSystem;

/*
 * Ternary weight block
 * 64 weights packed into 32 bytes (2 bits each)
 * Plus scale factor
 */
typedef struct {
    u8 weight[32];                /* Packed ternary weights */
    float scale;                   /* Dequantization scale */
} __attribute__((aligned(32))) HSMaterialBlock;

/*
 * Initialize ML system
 */
void hs_ml_init(HSMLSystem* ml);

/*
 * Free ML system resources
 */
void hs_ml_free(HSMLSystem* ml);

/*
 * INT8 GEMM: C = A × B
 * 
 * A: INT8 activations [M × K]
 * B: Ternary weights [K × N] (packed 2-bit)
 * C: INT32 accumulator [M × N]
 * 
 * This is the core BitNet operation:
 * Instead of FP32 multiply, we use INT8 dot product
 * because B weights are ternary (-1, 0, +1)
 */
void hs_ml_gemm_int8(int32_t* C, 
                     const int8_t* A, 
                     const u8* B_ternary,
                     const float* B_scale,
                     u32 M, u32 N, u32 K);

/*
 * NEON DOTPROD version (ARMv8.2+ only, e.g., Cortex-A55, A75)
 * Uses vdotq_s32 for maximum throughput.
 * Falls back to non-DOTPROD path on ARMv8.0 (Cortex-A72, Pi4).
 */
void hs_ml_gemm_int8_neon_dotprod(int32_t* C,
                                  const int8_t* A,
                                  const u8* B_ternary,
                                  const float* B_scale,
                                  u32 M, u32 N, u32 K);

/*
 * Non-DOTPROD fallback - optimized for ternary weights
 * 
 * Key insight: ternary weights are -1, 0, +1
 * Multiplication is wasteful. Instead:
 *   +1 → add activation
 *   -1 → subtract activation  
 *   0  → skip
 * 
 * This is 3x faster than multiplication!
 */
void hs_ml_gemm_int8_neon_fallback(int32_t* C,
                                   const int8_t* A,
                                   const u8* B_ternary,
                                   const float* B_scale,
                                   u32 M, u32 N, u32 K);

/*
 * Apply scale factors and convert to FP32
 */
void hs_ml_dequantize(const int32_t* C,
                      const float* scales,
                      float* output,
                      u32 M, u32 N);

/*
 * Quantize FP32 weights to ternary (training-time, not needed for inference)
 */
void hs_ml_quantize_ternary(const float* input,
                            u8* output,
                            float* scales,
                            u32 N);

/*
 * BitNet FFN activation: squared_relu on first half, multiply by second half
 * 
 * Input:  x [2*N] (first N: to be activated, second N: gate)
 * Output: y [N]   (activated * gated)
 * 
 * squared_relu(x) = (max(0, x))^2
 */
void hs_ml_ffn_activate_gate(const float* input,
                             float* output,
                             u32 N);

/*
 * Root Mean Square Layer Normalization
 * 
 * y = x * (1 / sqrt(mean(x^2) + ε))
 * 
 * Input:  x [N]
 * Output: y [N]
 */
void hs_ml_rmsnorm(const float* input,
                   float* output,
                   float epsilon,
                   u32 N);

/*
 * Minimal BPE Tokenizer
 */
typedef struct {
    u32 vocab_size;
    char** vocab;           /* id → string */
    u32* vocab_sizes;       /* length of each token string */
    u32* token_to_id;      /* string → id hash table (open addressing) */
    u32 hash_size;
    
    /* Special tokens */
    u32 unk_token;         /* unknown token id */
    u32 bos_token;         /* beginning of sequence */
    u32 eos_token;         /* end of sequence */
    u32 eod_token;         /* end of document */
    
    /* For encoding */
    u8* encode_buffer;
    u32 encode_buffer_size;
} HSTokenizer;

/*
 * Initialize tokenizer (loads vocab from memory)
 * 
 * vocab: array of vocab_size strings
 * unk/bos/eos/eod: special token ids (0 if not used)
 */
void hs_tokenizer_init(HSTokenizer* tok, 
                       char** vocab, 
                       u32 vocab_size,
                       u32 unk, u32 bos, u32 eos, u32 eod);

/*
 * Free tokenizer
 */
void hs_tokenizer_free(HSTokenizer* tok);

/*
 * Encode string to token ids
 * 
 * Returns: number of tokens
 * 
 * Handles: UTF-8 input, unknown tokens, special tokens
 */
u32 hs_tokenizer_encode(HSTokenizer* tok, 
                        const char* text, 
                        u32 text_len,
                        u32* output,
                        u32 output_size);

/*
 * Decode token ids to string
 * 
 * Returns: length of decoded string
 * 
 * Note: output buffer should be large enough
 */
u32 hs_tokenizer_decode(HSTokenizer* tok,
                        const u32* tokens,
                        u32 num_tokens,
                        char* output,
                        u32 output_size);

/*
 * KV Cache for attention
 * 
 * Stores key and value vectors for each layer
 */
typedef struct {
    u32 max_seq;         /* maximum sequence length */
    u32 num_heads;        /* number of attention heads */
    u32 num_kv_heads;    /* number of KV heads */
    u32 head_dim;        /* dimension of each head */
    
    float* k_cache;     /* [max_seq, num_kv_heads, head_dim] */
    float* v_cache;     /* [max_seq, num_kv_heads, head_dim] */
    
    u32 cache_len;       /* current cached length */
} HSKVCache;

/*
 * Initialize KV cache
 */
void hs_kv_cache_init(HSKVCache* cache,
                      u32 max_seq,
                      u32 num_heads,
                      u32 num_kv_heads,
                      u32 head_dim);

/*
 * Free KV cache
 */
void hs_kv_cache_free(HSKVCache* cache);

/*
 * Clear KV cache
 */
void hs_kv_cache_clear(HSKVCache* cache);

/*
 * RoPE (Rotary Position Embedding)
 * 
 * Applies rotary position embedding to query and key vectors
 * 
 * Input: q/k [num_heads, head_dim]
 * Output: q/k with RoPE applied in-place
 * 
 * theta: RoPE base frequency (typically 10000 or 500000)
 * position: position index
 */
void hs_rope_apply(float* q, float* k,
                   u32 num_heads, u32 head_dim,
                   u32 position, float theta);

/*
 * Attention score computation
 * 
 * Simplified attention: softmax(Q @ K^T / sqrt(d)) @ V
 * 
 * Input:  q [num_heads, head_dim], k [seq_len, head_dim], v [seq_len, head_dim]
 * Output: out [num_heads, head_dim]
 * 
 * Uses scaled dot-product attention
 * Note: K and V are flattened - assumes num_kv_heads == num_heads for simplicity
 */
void hs_attention_score(const float* q,
                       const float* k,
                       const float* v,
                       float* out,
                       u32 num_heads,
                       u32 head_dim,
                       u32 seq_len);

/*
 * GGUF Model Loader
 * 
 * Load BitNet 1.58b models in GGUF format
 * Returns 0 on success
 */
int hs_ml_load_gguf(HSMLSystem* ml, const char* path);

/*
 * Save model to GGUF (not implemented)
 */
void hs_ml_save_gguf(HSMLSystem* ml, const char* path);

/*
 * Forward pass through a single transformer layer
 */
void hs_ml_layer_forward(float* hidden,
                         u32 layer_idx,
                         HSKVCache* cache,
                         HSMLSystem* ml);

/*
 * Run forward pass on input tokens
 * Returns 0 on success
 */
int hs_ml_forward(HSMLSystem* ml,
                  const u32* tokens,
                  u32 seq_len,
                  float* logits);

/*
 * Greedy sampling - choose highest probability token
 */
u32 hs_ml_sample_greedy(const float* logits, u32 vocab_size);

/*
 * Top-k sampling
 */
u32 hs_ml_sample_topk(const float* logits, u32 vocab_size, u32 k);

/*
 * Autoregressive generation
 * Returns: number of tokens generated
 */
u32 hs_ml_generate(HSMLSystem* ml,
                   const char* prompt,
                   u32 max_new,
                   float temperature,
                   u32 top_k,
                   u32* output_tokens);

#endif /* HS_ML_H */
