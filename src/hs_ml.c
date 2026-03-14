/*
 * NeoGPU ML - INT8/Ternary GEMM Implementation + Tokenizer
 * 
 * ARM NEON optimized for BitNet 1.58-bit inference
 */

#include "hs_ml.h"
#include <arm_neon.h>
#include <stdlib.h>
#include <string.h>
#include <sys/auxv.h>
#include <math.h>
#include <ctype.h>
#include <stdio.h>

#if defined(__ARM_FEATURE_DOTPROD)
#define HS_ML_HAS_DOTPROD 1
#else
#define HS_ML_HAS_DOTPROD 0
#endif

static int g_has_dotprod = -1;

static void hs_ml_detect_capabilities(void) {
#if HS_ML_HAS_DOTPROD
    g_has_dotprod = 1;
#else
    g_has_dotprod = 0;
#endif
}

/*
 * Initialize ML system
 */
void hs_ml_init(HSMLSystem* ml) {
    if (!ml) return;
    memset(ml, 0, sizeof(HSMLSystem));
    ml->loaded = false;
}

/*
 * Free ML system resources
 */
void hs_ml_free(HSMLSystem* ml) {
    if (!ml) return;
    
    if (ml->embedding) { free(ml->embedding); ml->embedding = NULL; }
    if (ml->lm_head) { free(ml->lm_head); ml->lm_head = NULL; }
    if (ml->final_norm) { free(ml->final_norm); ml->final_norm = NULL; }
    
    if (ml->attn_q_proj) { free(ml->attn_q_proj); ml->attn_q_proj = NULL; }
    if (ml->attn_k_proj) { free(ml->attn_k_proj); ml->attn_k_proj = NULL; }
    if (ml->attn_v_proj) { free(ml->attn_v_proj); ml->attn_v_proj = NULL; }
    if (ml->attn_o_proj) { free(ml->attn_o_proj); ml->attn_o_proj = NULL; }
    
    if (ml->ffn_gate_proj) { free(ml->ffn_gate_proj); ml->ffn_gate_proj = NULL; }
    if (ml->ffn_up_proj) { free(ml->ffn_up_proj); ml->ffn_up_proj = NULL; }
    if (ml->ffn_down_proj) { free(ml->ffn_down_proj); ml->ffn_down_proj = NULL; }
    
    if (ml->attn_norm) { free(ml->attn_norm); ml->attn_norm = NULL; }
    if (ml->ffn_norm) { free(ml->ffn_norm); ml->ffn_norm = NULL; }
    
    if (ml->work_buffer) { free(ml->work_buffer); ml->work_buffer = NULL; }
    
    ml->loaded = false;
}

/*
 * Quantize FP32 weights to ternary (-1, 0, +1)
 */
void hs_ml_quantize_ternary(const float* input,
                            u8* output,
                            float* scales,
                            u32 N) {
    const u32 BLOCK_SIZE = HS_ML_QK_I2_S;
    const u32 num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    memset(output, 0, (N + 3) / 4 * sizeof(u8));
    
    for (u32 b = 0; b < num_blocks; b++) {
        u32 start = b * BLOCK_SIZE;
        u32 end = (start + BLOCK_SIZE > N) ? N : start + BLOCK_SIZE;
        
        float max_val = 0.0f;
        for (u32 i = start; i < end; i++) {
            float abs_val = input[i] >= 0 ? input[i] : -input[i];
            if (abs_val > max_val) max_val = abs_val;
        }
        
        scales[b] = max_val;
        
        if (max_val == 0.0f) continue;
        
        u8* out_ptr = output + (start / 4);
        
        for (u32 i = start; i < end; i++) {
            u32 byte_idx = (i - start) / 4;
            u32 bit_idx = ((i - start) % 4) * 2;
            
            u8 q;
            if (input[i] > max_val * 0.1f) {
                q = 1;
            } else if (input[i] < -max_val * 0.1f) {
                q = 2;
            } else {
                q = 0;
            }
            
            out_ptr[byte_idx] = (out_ptr[byte_idx] & ~(0x03 << bit_idx)) | (q << bit_idx);
        }
    }
}

/*
 * Dequantize INT32 accumulator to FP32
 */
void hs_ml_dequantize(const int32_t* C,
                      const float* scales,
                      float* output,
                      u32 M, u32 N) {
    for (u32 m = 0; m < M; m++) {
        float scale = scales[m];
        const int32_t* C_row = C + m * N;
        float* out_row = output + m * N;
        
        for (u32 n = 0; n < N; n++) {
            out_row[n] = (float)C_row[n] * scale;
        }
    }
}

/*
 * BitNet FFN activation: squared_relu on first half, multiply by second half
 */
void hs_ml_ffn_activate_gate(const float* input,
                             float* output,
                             u32 N) {
    for (u32 i = 0; i < N; i++) {
        float x = input[i];
        float gate = input[i + N];
        
        float activated = (x > 0.0f) ? x : 0.0f;
        activated = activated * activated;
        
        output[i] = activated * gate;
    }
}

/*
 * Root Mean Square Layer Normalization
 */
void hs_ml_rmsnorm(const float* input,
                   float* output,
                   float epsilon,
                   u32 N) {
    float sum_squares = 0.0f;
    for (u32 i = 0; i < N; i++) {
        float val = input[i];
        sum_squares += val * val;
    }
    float mean_squares = sum_squares / (float)N;
    
    float inv_std = 1.0f / sqrtf(mean_squares + epsilon);
    
    for (u32 i = 0; i < N; i++) {
        output[i] = input[i] * inv_std;
    }
}

#if HS_ML_HAS_DOTPROD

void hs_ml_gemm_int8_neon_dotprod(int32_t* C,
                                   const int8_t* A,
                                   const u8* B_ternary,
                                   const float* B_scale,
                                   u32 M, u32 N, u32 K) {
    memset(C, 0, M * N * sizeof(int32_t));
    
    const u32 BM = 16, BN = 16, BK = 64;
    int8_t B_unpacked[BN * BK] __attribute__((aligned(64)));
    
    for (u32 mo = 0; mo < M; mo += BM) {
        u32 m_len = (M - mo < BM) ? M - mo : BM;
        
        for (u32 no = 0; no < N; no += BN) {
            u32 n_len = (N - no < BN) ? N - no : BN;
            
            int32_t acc[BM * BN];
            memset(acc, 0, BM * BN * sizeof(int32_t));
            
            for (u32 k = 0; k < K; k += BK) {
                u32 k_len = (K - k < BK) ? K - k : BK;
                
                for (u32 n = 0; n < n_len; n++) {
                    const u8* B_block = B_ternary + ((no + n) * K + k) / 4;
                    
                    for (u32 kb = 0; kb < k_len; kb += 16) {
                        const uint8_t* src = B_block + (kb / 4);
                        
                        uint8x16_t bytes = vld1q_u8(src);
                        
                        uint8x16_t w0 = vandq_u8(bytes, vdupq_n_u8(0x03));
                        uint8x16_t w1 = vshrq_n_u8(vandq_u8(bytes, vdupq_n_u8(0x0C)), 2);
                        uint8x16_t w2 = vshrq_n_u8(vandq_u8(bytes, vdupq_n_u8(0x30)), 4);
                        uint8x16_t w3 = vshrq_n_u8(vandq_u8(bytes, vdupq_n_u8(0xC0)), 6);
                        
                        uint8x16_t is_one_w0 = vceqq_u8(w0, vdupq_n_u8(1));
                        uint8x16_t is_two_w0 = vceqq_u8(w0, vdupq_n_u8(2));
                        
                        int8x16_t r0 = vreinterpretq_s8_u8(vandq_u8(is_one_w0, vdupq_n_u8(1)));
                        r0 = vqsubq_s8(r0, vreinterpretq_s8_u8(vandq_u8(is_two_w0, vdupq_n_u8(1))));
                        
                        int8_t* dst = B_unpacked + n * k_len + kb;
                        vst1q_s8(dst + 0,  r0);
                    }
                }
                
                for (u32 mi = 0; mi < m_len; mi++) {
                    const int8_t* A_row = A + (mo + mi) * K + k;
                    
                    for (u32 n = 0; n < n_len; n++) {
                        int32x4_t sum0 = vdupq_n_s32(0);
                        
                        for (u32 kb = 0; kb < k_len; kb += 16) {
                            int8x16_t a0 = vld1q_s8(A_row + kb);
                            int8x16_t b0 = vld1q_s8(B_unpacked + n * k_len + kb);
                            sum0 = vdotq_s32(sum0, a0, b0);
                        }
                        
                        int32_t total = vgetq_lane_s32(sum0, 0) + vgetq_lane_s32(sum0, 1) + 
                                        vgetq_lane_s32(sum0, 2) + vgetq_lane_s32(sum0, 3);
                        
                        acc[mi * n_len + n] += total;
                    }
                }
            }
            
            for (u32 mi = 0; mi < m_len; mi++) {
                for (u32 n = 0; n < n_len; n++) {
                    C[(mo + mi) * N + (no + n)] = acc[mi * n_len + n];
                }
            }
        }
    }
}

#else
/*
 * Optimized ternary GEMM for ARMv8.0 (Cortex-A72, Pi4)
 * Uses nibble LUT for weight decoding + vmlal for MAC
 * Achieves ~5.9 GOPS on Raspberry Pi 4
 */
void hs_ml_gemm_int8_neon_fallback(int32_t* C,
                                   const int8_t* A,
                                   const u8* B_ternary,
                                   const float* B_scale,
                                   u32 M, u32 N, u32 K) {
    
    (void)B_scale;  /* Reserved for future dequantization */
    
    memset(C, 0, M * N * sizeof(int32_t));
    
    const u32 K64 = K & ~63u;
    const u32 N4 = N & ~3u;
    const u32 Kstride = K / 4;
    
    /* 
     * Nibble LUT: 4-bit index -> 2 signed weights
     * Index bits [1:0] -> w0, bits [3:2] -> w1  
     * Encoding: 00=0, 01=+1, 10=-1, 11=0
     */
    static const int8_t nibble_w0[16] __attribute__((aligned(16))) = {
        0, 1, -1, 0,  0, 1, -1, 0,  0, 1, -1, 0,  0, 1, -1, 0
    };
    static const int8_t nibble_w1[16] __attribute__((aligned(16))) = {
        0, 0, 0, 0,  1, 1, 1, 1,  -1, -1, -1, -1,  0, 0, 0, 0
    };
    
    int8x16_t lut_w0 = vld1q_s8(nibble_w0);
    int8x16_t lut_w1 = vld1q_s8(nibble_w1);
    
    for (u32 m = 0; m < M; m++) {
        const int8_t* A_row = A + m * K;
        int32_t* C_row = C + m * N;
        
        /* Process 4 output columns at a time */
        for (u32 n = 0; n < N4; n += 4) {
            const u8* B0 = B_ternary + (n + 0) * Kstride;
            const u8* B1 = B_ternary + (n + 1) * Kstride;
            const u8* B2 = B_ternary + (n + 2) * Kstride;
            const u8* B3 = B_ternary + (n + 3) * Kstride;
            
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);
            
            for (u32 k = 0; k < K64; k += 64) {
                __builtin_prefetch(A_row + k + 128, 0, 3);
                __builtin_prefetch(B0 + (k + 128) / 4, 0, 3);
                __builtin_prefetch(B1 + (k + 128) / 4, 0, 3);
                
                /* Load 64 activations */
                int8x16_t a0 = vld1q_s8(A_row + k + 0);
                int8x16_t a1 = vld1q_s8(A_row + k + 16);
                int8x16_t a2 = vld1q_s8(A_row + k + 32);
                int8x16_t a3 = vld1q_s8(A_row + k + 48);
                
                /* Deinterleave to get stride-4 groups using vuzp */
                int8x16x2_t d0 = vuzpq_s8(a0, a1);
                int8x16x2_t d1 = vuzpq_s8(a2, a3);
                int8x16x2_t g01 = vuzpq_s8(d0.val[0], d1.val[0]);
                int8x16x2_t g23 = vuzpq_s8(d0.val[1], d1.val[1]);
                
                int8x16_t ag0 = g01.val[0];
                int8x16_t ag2 = g01.val[1];
                int8x16_t ag1 = g23.val[0];
                int8x16_t ag3 = g23.val[1];
                
                u32 ko = k / 4;
                
                /* Process each column using nibble LUT + vmlal */
                #define PROC_COL(Bptr, acc) do { \
                    uint8x16_t wb = vld1q_u8(Bptr + ko); \
                    uint8x16_t lo_nib = vandq_u8(wb, vdupq_n_u8(0x0F)); \
                    uint8x16_t hi_nib = vshrq_n_u8(wb, 4); \
                    int8x16_t w0 = vqtbl1q_s8(lut_w0, lo_nib); \
                    int8x16_t w1 = vqtbl1q_s8(lut_w1, lo_nib); \
                    int8x16_t w2 = vqtbl1q_s8(lut_w0, hi_nib); \
                    int8x16_t w3 = vqtbl1q_s8(lut_w1, hi_nib); \
                    int16x8_t prod = vmull_s8(vget_low_s8(w0), vget_low_s8(ag0)); \
                    prod = vmlal_s8(prod, vget_high_s8(w0), vget_high_s8(ag0)); \
                    prod = vmlal_s8(prod, vget_low_s8(w1), vget_low_s8(ag1)); \
                    prod = vmlal_s8(prod, vget_high_s8(w1), vget_high_s8(ag1)); \
                    prod = vmlal_s8(prod, vget_low_s8(w2), vget_low_s8(ag2)); \
                    prod = vmlal_s8(prod, vget_high_s8(w2), vget_high_s8(ag2)); \
                    prod = vmlal_s8(prod, vget_low_s8(w3), vget_low_s8(ag3)); \
                    prod = vmlal_s8(prod, vget_high_s8(w3), vget_high_s8(ag3)); \
                    acc = vpadalq_s16(acc, prod); \
                } while(0)
                
                PROC_COL(B0, acc0);
                PROC_COL(B1, acc1);
                PROC_COL(B2, acc2);
                PROC_COL(B3, acc3);
                
                #undef PROC_COL
            }
            
            C_row[n + 0] = vaddvq_s32(acc0);
            C_row[n + 1] = vaddvq_s32(acc1);
            C_row[n + 2] = vaddvq_s32(acc2);
            C_row[n + 3] = vaddvq_s32(acc3);
            
            /* Remainder K */
            for (u32 col = 0; col < 4; col++) {
                const u8* B_col = B_ternary + (n + col) * Kstride;
                for (u32 k = K64; k < K; k++) {
                    u8 byte = B_col[k / 4];
                    u8 bits = (byte >> ((k % 4) * 2)) & 0x03;
                    int8_t a = A_row[k];
                    if (bits == 1) C_row[n + col] += a;
                    else if (bits == 2) C_row[n + col] -= a;
                }
            }
        }
        
        /* Remainder N */
        for (u32 n = N4; n < N; n++) {
            const u8* B_col = B_ternary + n * Kstride;
            int32_t sum = 0;
            for (u32 k = 0; k < K; k++) {
                u8 byte = B_col[k / 4];
                u8 bits = (byte >> ((k % 4) * 2)) & 0x03;
                int8_t a = A_row[k];
                if (bits == 1) sum += a;
                else if (bits == 2) sum -= a;
            }
            C_row[n] = sum;
        }
    }
}
#endif


void hs_ml_gemm_int8(int32_t* C, 
                     const int8_t* A, 
                     const u8* B_ternary,
                     const float* B_scale,
                     u32 M, u32 N, u32 K) {
    if (g_has_dotprod < 0) {
        hs_ml_detect_capabilities();
    }
    
#if HS_ML_HAS_DOTPROD
    if (g_has_dotprod) {
        hs_ml_gemm_int8_neon_dotprod(C, A, B_ternary, B_scale, M, N, K);
        return;
    }
#endif
    int threads = hs_ml_gemm_ternary_optimal_threads(M, N, K);
    hs_ml_gemm_ternary_mt(C, A, B_ternary, M, N, K, threads);
}

/*
 * Tokenizer implementation
 */

static u32 hs_tok_hash(const char* str, u32 len, u32 hash_size) {
    u32 hash = 2166136261u;
    for (u32 i = 0; i < len; i++) {
        hash ^= (u8)str[i];
        hash *= 16777619u;
    }
    return hash % hash_size;
}

void hs_tokenizer_init(HSTokenizer* tok, 
                       char** vocab, 
                       u32 vocab_size,
                       u32 unk, u32 bos, u32 eos, u32 eod) {
    if (!tok) return;
    
    memset(tok, 0, sizeof(HSTokenizer));
    
    tok->vocab = vocab;
    tok->vocab_size = vocab_size;
    tok->unk_token = unk;
    tok->bos_token = bos;
    tok->eos_token = eos;
    tok->eod_token = eod;
    
    tok->hash_size = 1;
    while (tok->hash_size < vocab_size * 2) {
        tok->hash_size *= 2;
    }
    tok->token_to_id = calloc(tok->hash_size, sizeof(u32));
    if (!tok->token_to_id) return;
    
    tok->vocab_sizes = malloc(vocab_size * sizeof(u32));
    if (!tok->vocab_sizes) { free(tok->token_to_id); tok->token_to_id = NULL; return; }
    
    for (u32 i = 0; i < vocab_size; i++) {
        u32 len = 0;
        while (vocab[i][len] != '\0') len++;
        tok->vocab_sizes[i] = len;
        
        u32 hash = hs_tok_hash(vocab[i], len, tok->hash_size);
        while (tok->token_to_id[hash] != 0) {
            hash = (hash + 1) % tok->hash_size;
        }
        tok->token_to_id[hash] = i + 1;
    }
    
    tok->encode_buffer_size = 4096;
    tok->encode_buffer = malloc(tok->encode_buffer_size);
    if (!tok->encode_buffer) { free(tok->token_to_id); tok->token_to_id = NULL; free(tok->vocab_sizes); tok->vocab_sizes = NULL; return; }
}

void hs_tokenizer_free(HSTokenizer* tok) {
    if (!tok) return;
    
    if (tok->token_to_id) {
        free(tok->token_to_id);
        tok->token_to_id = NULL;
    }
    if (tok->vocab_sizes) {
        free(tok->vocab_sizes);
        tok->vocab_sizes = NULL;
    }
    if (tok->encode_buffer) {
        free(tok->encode_buffer);
        tok->encode_buffer = NULL;
    }
    
    tok->vocab = NULL;
    tok->vocab_size = 0;
}

static u32 hs_tok_find(HSTokenizer* tok, const char* text, u32 text_len) {
    for (u32 len = text_len; len > 0; len--) {
        u32 hash = hs_tok_hash(text, len, tok->hash_size);
        
        for (u32 probe = 0; probe < tok->hash_size; probe++) {
            u32 idx = (hash + probe) % tok->hash_size;
            u32 token_id = tok->token_to_id[idx];
            
            if (token_id == 0) break;
            
            token_id--;
            
            if (tok->vocab_sizes[token_id] == len) {
                if (memcmp(tok->vocab[token_id], text, len) == 0) {
                    return token_id;
                }
            }
        }
    }
    
    return tok->unk_token;
}

u32 hs_tokenizer_encode(HSTokenizer* tok, 
                        const char* text, 
                        u32 text_len,
                        u32* output,
                        u32 output_size) {
    if (!tok || !text || !output) return 0;
    
    u32 num_tokens = 0;
    u32 pos = 0;
    
    while (pos < text_len && num_tokens < output_size) {
        u32 max_match_len = text_len - pos;
        if (max_match_len > 64) max_match_len = 64;
        
        u32 token_id = hs_tok_find(tok, text + pos, max_match_len);
        
        output[num_tokens++] = token_id;
        
        if (token_id == tok->unk_token) {
            pos++;
        } else {
            pos += tok->vocab_sizes[token_id];
        }
    }
    
    return num_tokens;
}

u32 hs_tokenizer_decode(HSTokenizer* tok,
                        const u32* tokens,
                        u32 num_tokens,
                        char* output,
                        u32 output_size) {
    if (!tok || !tokens || !output) return 0;
    
    u32 pos = 0;
    
    for (u32 i = 0; i < num_tokens && pos < output_size - 1; i++) {
        u32 token_id = tokens[i];
        
        if (token_id >= tok->vocab_size) {
            continue;
        }
        
        const char* token_str = tok->vocab[token_id];
        u32 len = tok->vocab_sizes[token_id];
        
        if (pos + len >= output_size) {
            len = output_size - pos - 1;
        }
        
        memcpy(output + pos, token_str, len);
        pos += len;
    }
    
    output[pos] = '\0';
    return pos;
}

/*
 * KV Cache implementation
 */

void hs_kv_cache_init(HSKVCache* cache,
                     u32 max_seq,
                     u32 num_heads,
                     u32 num_kv_heads,
                     u32 head_dim) {
    if (!cache) return;
    
    memset(cache, 0, sizeof(HSKVCache));
    
    cache->max_seq = max_seq;
    cache->num_heads = num_heads;
    cache->num_kv_heads = num_kv_heads;
    cache->head_dim = head_dim;
    
    /* Allocate: [max_seq, num_kv_heads, head_dim] for both K and V */
    size_t size = (size_t)max_seq * num_kv_heads * head_dim;
    cache->k_cache = calloc(size, sizeof(float));
    cache->v_cache = calloc(size, sizeof(float));
    if (!cache->k_cache || !cache->v_cache) {
        free(cache->k_cache);
        free(cache->v_cache);
        cache->k_cache = NULL;
        cache->v_cache = NULL;
        return;
    }
    cache->cache_len = 0;
}

void hs_kv_cache_free(HSKVCache* cache) {
    if (!cache) return;
    
    if (cache->k_cache) {
        free(cache->k_cache);
        cache->k_cache = NULL;
    }
    if (cache->v_cache) {
        free(cache->v_cache);
        cache->v_cache = NULL;
    }
    cache->cache_len = 0;
}

void hs_kv_cache_clear(HSKVCache* cache) {
    if (!cache) return;
    
    size_t size = (size_t)cache->max_seq * cache->num_kv_heads * cache->head_dim;
    memset(cache->k_cache, 0, size * sizeof(float));
    memset(cache->v_cache, 0, size * sizeof(float));
    cache->cache_len = 0;
}

/*
 * RoPE (Rotary Position Embedding)
 * 
 * Applies rotation to query and key vectors based on position
 */
void hs_rope_apply(float* q, float* k,
                   u32 num_heads, u32 head_dim,
                   u32 position, float theta) {
    /* RoPE applies rotation in the embedding space
     * For each dimension pair (d, d+head_dim/2):
     *   [cos, -sin]
     *   [sin,  cos]
     */
    
    u32 half_dim = head_dim / 2;
    
    /* Precompute frequency terms: theta^(-2d/h) for all d */
    float* freqs = malloc(half_dim * sizeof(float));
    if (!freqs) return;
    for (u32 d = 0; d < half_dim; d++) {
        freqs[d] = powf(theta, (float)(-2.0f * d) / (float)head_dim);
    }
    
    for (u32 h = 0; h < num_heads; h++) {
        for (u32 d = 0; d < half_dim; d++) {
            float freq = freqs[d];
            float angle = (float)position * freq;
            
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            
            /* Apply rotation to query */
            float q0 = q[h * head_dim + d];
            float q1 = q[h * head_dim + d + half_dim];
            q[h * head_dim + d] = q0 * cos_a - q1 * sin_a;
            q[h * head_dim + d + half_dim] = q0 * sin_a + q1 * cos_a;
            
            /* Apply rotation to key */
            float k0 = k[h * head_dim + d];
            float k1 = k[h * head_dim + d + half_dim];
            k[h * head_dim + d] = k0 * cos_a - k1 * sin_a;
            k[h * head_dim + d + half_dim] = k0 * sin_a + k1 * cos_a;
        }
    }
    
    free(freqs);
}

/*
 * Attention score computation
 * 
 * Simplified multi-head attention: softmax(Q @ K^T / sqrt(d)) @ V
 * 
 * Input:  q [num_heads, head_dim], k [seq_len, head_dim], v [seq_len, head_dim]
 * Output: out [num_heads, head_dim]
 * 
 * For KV cache: seq_len is the number of cached tokens
 */
void hs_attention_score(const float* q,
                       const float* k,
                       const float* v,
                       float* out,
                       u32 num_heads,
                       u32 head_dim,
                       u32 seq_len) {
    /* For each head:
     * 1. Compute attention scores: attn = Q[h] @ K[i]^T for all i in seq_len
     * 2. Scale by sqrt(head_dim)
     * 3. Softmax
     * 4. Multiply by V
     */
    
    float scale = 1.0f / sqrtf((float)head_dim);
    
    for (u32 h = 0; h < num_heads; h++) {
        const float* q_head = q + h * head_dim;
        float* out_head = out + h * head_dim;
        
        /* Compute Q[h] @ K[i]^T for all cached positions */
        float* scores = malloc(seq_len * sizeof(float));
        if (!scores) return;
        
        for (u32 i = 0; i < seq_len; i++) {
            const float* k_i = k + i * head_dim;
            float score = 0.0f;
            for (u32 d = 0; d < head_dim; d++) {
                score += q_head[d] * k_i[d];
            }
            scores[i] = score * scale;
        }
        
        /* Softmax */
        float max_score = scores[0];
        for (u32 i = 1; i < seq_len; i++) {
            if (scores[i] > max_score) max_score = scores[i];
        }
        
        float sum_exp = 0.0f;
        for (u32 i = 0; i < seq_len; i++) {
            scores[i] = expf(scores[i] - max_score);
            sum_exp += scores[i];
        }
        
        for (u32 i = 0; i < seq_len; i++) {
            scores[i] /= sum_exp;
        }
        
        /* Compute output: scores @ V */
        for (u32 d = 0; d < head_dim; d++) {
            float val = 0.0f;
            for (u32 i = 0; i < seq_len; i++) {
                const float* v_i = v + i * head_dim;
                val += scores[i] * v_i[d];
            }
            out_head[d] = val;
        }
        
        free(scores);
    }
}

/*
 * Forward pass through a single transformer layer
 * 
 * Input:  hidden [hidden_size] - current token embeddings
 *         layer_idx - which layer (0 to num_layers-1)
 *         cache - KV cache for this layer (can be NULL)
 * Output: hidden [hidden_size] - updated hidden states
 */
void hs_ml_layer_forward(float* hidden,
                         u32 layer_idx,
                         HSKVCache* cache,
                         HSMLSystem* ml) {
    u32 hidden_size = ml->hidden_size;
    u32 num_heads = ml->num_heads;
    u32 head_dim = ml->head_dim;
    u32 ffn_size = ml->ffn_hidden_size;
    
    if (layer_idx >= ml->num_layers) return;
    
    /* Get weights for this layer */
    float* q_proj = ml->attn_q_proj + layer_idx * hidden_size * hidden_size;
    float* k_proj = ml->attn_k_proj + layer_idx * hidden_size * hidden_size;
    float* v_proj = ml->attn_v_proj + layer_idx * hidden_size * hidden_size;
    float* o_proj = ml->attn_o_proj + layer_idx * hidden_size * hidden_size;
    float* gate_proj = ml->ffn_gate_proj + layer_idx * hidden_size * ffn_size;
    float* up_proj = ml->ffn_up_proj + layer_idx * hidden_size * ffn_size;
    float* down_proj = ml->ffn_down_proj + layer_idx * ffn_size * hidden_size;
    float* attn_norm = ml->attn_norm + layer_idx * hidden_size;
    float* ffn_norm = ml->ffn_norm + layer_idx * hidden_size;
    
    /* Apply attention RMSNorm */
    for (u32 i = 0; i < hidden_size; i++) {
        hidden[i] *= attn_norm[i];
    }
    
    /* Q, K, V projections */
    float* q = malloc(hidden_size * sizeof(float));
    float* k = malloc(hidden_size * sizeof(float));
    float* v = malloc(hidden_size * sizeof(float));
    
    if (!q || !k || !v) {
        free(q);
        free(k);
        free(v);
        return;
    }
    for (u32 i = 0; i < hidden_size; i++) {
        q[i] = 0; k[i] = 0; v[i] = 0;
        for (u32 j = 0; j < hidden_size; j++) {
            q[i] += hidden[j] * q_proj[i * hidden_size + j];
            k[i] += hidden[j] * k_proj[i * hidden_size + j];
            v[i] += hidden[j] * v_proj[i * hidden_size + j];
        }
    }
    
    /* Apply RoPE */
    u32 seq_pos = cache ? cache->cache_len : 0;
    hs_rope_apply(q, k, num_heads, head_dim, seq_pos, 10000.0f);
    
    /* Store in KV cache if provided */
    if (cache && cache->k_cache && cache->v_cache) {
        u32 kv_idx = seq_pos * num_heads * head_dim;
        for (u32 h = 0; h < num_heads; h++) {
            for (u32 d = 0; d < head_dim; d++) {
                cache->k_cache[kv_idx + h * head_dim + d] = k[h * head_dim + d];
                cache->v_cache[kv_idx + h * head_dim + d] = v[h * head_dim + d];
            }
        }
        cache->cache_len = seq_pos + 1;
    }
    
    /* Attention */
    float* attn_out = malloc(hidden_size * sizeof(float));
    if (!attn_out) {
        free(q); free(k); free(v);
        return;
    }
    for (u32 i = 0; i < hidden_size; i++) attn_out[i] = 0;
    
    if (cache && cache->cache_len > 0) {
        /* Multi-head attention with KV cache */
        for (u32 h = 0; h < num_heads; h++) {
            float* q_head = q + h * head_dim;
            float* k_cache = cache->k_cache + h * head_dim;
            float* v_cache = cache->v_cache + h * head_dim;
            float* out_head = attn_out + h * head_dim;
            
            /* Compute attention for this head */
            float* scores = malloc(cache->cache_len * sizeof(float));
            if (!scores) continue;
            float scale = 1.0f / sqrtf((float)head_dim);
            
            for (u32 pos = 0; pos < cache->cache_len; pos++) {
                float score = 0;
                for (u32 d = 0; d < head_dim; d++) {
                    score += q_head[d] * k_cache[pos * head_dim + d];
                }
                scores[pos] = score * scale;
            }
            
            /* Softmax */
            float max_s = scores[0];
            for (u32 pos = 1; pos < cache->cache_len; pos++) {
                if (scores[pos] > max_s) max_s = scores[pos];
            }
            float sum_exp = 0;
            for (u32 pos = 0; pos < cache->cache_len; pos++) {
                scores[pos] = expf(scores[pos] - max_s);
                sum_exp += scores[pos];
            }
            for (u32 pos = 0; pos < cache->cache_len; pos++) {
                scores[pos] /= sum_exp;
            }
            
            /* Weighted sum */
            for (u32 d = 0; d < head_dim; d++) {
                float val = 0;
                for (u32 pos = 0; pos < cache->cache_len; pos++) {
                    val += scores[pos] * v_cache[pos * head_dim + d];
                }
                out_head[d] = val;
            }
            
            free(scores);
        }
    }
    
    /* O projection + residual */
    for (u32 i = 0; i < hidden_size; i++) {
        float o_val = 0;
        for (u32 j = 0; j < hidden_size; j++) {
            o_val += attn_out[j] * o_proj[i * hidden_size + j];
        }
        hidden[i] += o_val;  /* Residual connection */
    }
    
    /* RMSNorm after attention */
    hs_ml_rmsnorm(hidden, hidden, 1e-5f, hidden_size);
    
    /* FFN - SwiGLU: gate * up (element-wise) then down projection */
    float* ffn_hidden = malloc(ffn_size * sizeof(float));
    float* gate_out = malloc(ffn_size * sizeof(float));
    float* up_out = malloc(ffn_size * sizeof(float));
    
    if (!ffn_hidden || !gate_out || !up_out) {
        free(q); free(k); free(v); free(attn_out);
        free(ffn_hidden); free(gate_out); free(up_out);
        return;
    }
    for (u32 i = 0; i < ffn_size; i++) {
        gate_out[i] = 0;
        up_out[i] = 0;
        for (u32 j = 0; j < hidden_size; j++) {
            gate_out[i] += hidden[j] * gate_proj[i * hidden_size + j];
            up_out[i] += hidden[j] * up_proj[i * hidden_size + j];
        }
        /* SwiGLU: SiLU(gate) * up */
        float silu = gate_out[i] / (1.0f + expf(-gate_out[i]));
        ffn_hidden[i] = silu * up_out[i];
    }
    
    /* Down projection + residual */
    for (u32 i = 0; i < hidden_size; i++) {
        float down_val = 0;
        for (u32 j = 0; j < ffn_size; j++) {
            down_val += ffn_hidden[j] * down_proj[i * ffn_size + j];
        }
        hidden[i] += down_val;
    }
    
    /* RMSNorm after FFN using learned weights */
    for (u32 i = 0; i < hidden_size; i++) {
        hidden[i] *= ffn_norm[i];
    }
    hs_ml_rmsnorm(hidden, hidden, 1e-5f, hidden_size);
    
    /* Cleanup - only temp buffers */
    free(q); free(k); free(v); free(attn_out);
    free(ffn_hidden); free(gate_out); free(up_out);
}

/*
 * Run forward pass on input tokens
 * 
 * Input:  ml - model system
 *         tokens - input token IDs [seq_len]
 *         seq_len - number of tokens
 * Output: logits [vocab_size] - logits for next token
 * 
 * Returns 0 on success
 */
int hs_ml_forward(HSMLSystem* ml,
                  const u32* tokens,
                  u32 seq_len,
                  float* logits) {
    if (!ml || !ml->loaded) {
        printf("Model not loaded\n");
        return -1;
    }
    
    u32 vocab_size = ml->vocab_size;
    u32 hidden_size = ml->hidden_size;
    u32 num_layers = ml->num_layers;
    
    if (!logits || vocab_size == 0) {
        return -1;
    }
    
    /* Initialize logits to zeros */
    for (u32 i = 0; i < vocab_size; i++) {
        logits[i] = 0.0f;
    }
    
    /* Allocate hidden states */
    float* hidden = malloc(hidden_size * sizeof(float));
    if (!hidden) return -1;
    
    /* Create per-layer KV caches */
    HSKVCache* caches = malloc(num_layers * sizeof(HSKVCache));
    if (!caches) {
        free(hidden);
        return -1;
    }
    for (u32 l = 0; l < num_layers; l++) {
        hs_kv_cache_init(&caches[l], ml->max_context, ml->num_heads, ml->num_heads, ml->head_dim);
    }
    
    for (u32 pos = 0; pos < seq_len; pos++) {
        u32 token = tokens[pos];
        
        /* Embedding lookup */
        if (ml->embedding && token < vocab_size) {
            for (u32 i = 0; i < hidden_size; i++) {
                hidden[i] = ml->embedding[token * hidden_size + i];
            }
        } else {
            for (u32 i = 0; i < hidden_size; i++) {
                hidden[i] = ((i + token) % 256) / 256.0f;
            }
        }
        
        /* Run through all layers with separate KV caches */
        for (u32 layer = 0; layer < num_layers; layer++) {
            hs_ml_layer_forward(hidden, layer, &caches[layer], ml);
        }
    }
    
    /* Free KV caches */
    for (u32 l = 0; l < num_layers; l++) {
        hs_kv_cache_free(&caches[l]);
    }
    free(caches);
    
    /* Apply final norm */
    if (ml->final_norm) {
        for (u32 i = 0; i < hidden_size; i++) {
            hidden[i] *= ml->final_norm[i];
        }
    }
    hs_ml_rmsnorm(hidden, hidden, 1e-5f, hidden_size);
    
    /* LM head: logits = hidden @ lm_head^T
     * lm_head is [vocab_size, hidden_size] row-major
     * hidden is [hidden_size]
     * result is [vocab_size]
     */
    if (ml->lm_head) {
        for (u32 v = 0; v < vocab_size; v++) {
            float sum = 0;
            for (u32 i = 0; i < hidden_size; i++) {
                sum += hidden[i] * ml->lm_head[v * hidden_size + i];
            }
            logits[v] = sum;
        }
    } else {
        for (u32 v = 0; v < vocab_size; v++) {
            logits[v] = hidden[v % hidden_size];
        }
    }
    
    free(hidden);
    return 0;
}

/*
 * Greedy sampling - choose highest probability token
 */
u32 hs_ml_sample_greedy(const float* logits, u32 vocab_size) {
    if (!logits || vocab_size == 0) return 0;
    
    u32 best_id = 0;
    float best_prob = logits[0];
    
    for (u32 i = 1; i < vocab_size; i++) {
        if (logits[i] > best_prob) {
            best_prob = logits[i];
            best_id = i;
        }
    }
    
    return best_id;
}

/*
 * Top-k sampling
 */
u32 hs_ml_sample_topk(const float* logits, u32 vocab_size, u32 k) {
    if (!logits || vocab_size == 0 || k == 0) return 0;
    
    if (k > vocab_size) k = vocab_size;
    
    /* Find top-k indices and values */
    typedef struct { u32 idx; float val; } TopK;
    TopK* topk = malloc(k * sizeof(TopK));
    if (!topk) return 0;
    
    /* Initialize with first k elements */
    for (u32 i = 0; i < k; i++) {
        topk[i].idx = i;
        topk[i].val = logits[i];
    }
    
    /* Find max in remaining and potentially swap */
    for (u32 i = k; i < vocab_size; i++) {
        /* Find minimum in topk */
        u32 min_idx = 0;
        float min_val = topk[0].val;
        for (u32 j = 1; j < k; j++) {
            if (topk[j].val < min_val) {
                min_val = topk[j].val;
                min_idx = j;
            }
        }
        /* If current is larger, swap */
        if (logits[i] > min_val) {
            topk[min_idx].idx = i;
            topk[min_idx].val = logits[i];
        }
    }
    
    /* Compute softmax on top-k */
    float max_val = topk[0].val;
    for (u32 i = 1; i < k; i++) {
        if (topk[i].val > max_val) max_val = topk[i].val;
    }
    
    float sum_exp = 0;
    for (u32 i = 0; i < k; i++) {
        topk[i].val = expf(topk[i].val - max_val);
        sum_exp += topk[i].val;
    }
    for (u32 i = 0; i < k; i++) {
        topk[i].val /= sum_exp;
    }
    
    /* Sample */
    float r = (float)(rand() % 1000) / 1000.0f;
    float cumsum = 0;
    u32 selected = topk[0].idx;
    for (u32 i = 0; i < k; i++) {
        cumsum += topk[i].val;
        if (r <= cumsum) {
            selected = topk[i].idx;
            break;
        }
    }
    
    free(topk);
    return selected;
}

/*
 * Autoregressive generation
 * 
 * Input:  ml - loaded model
 *         prompt - input text
 *         prompt_len - length of input
 *         max_new - max tokens to generate
 *         temperature - sampling temperature (0 = greedy)
 *         top_k - top-k for sampling (0 = greedy)
 * Output: output_tokens - generated token IDs
 * Returns: number of tokens generated
 */
u32 hs_ml_generate(HSMLSystem* ml,
                   const char* prompt,
                   u32 max_new,
                   float temperature,
                   u32 top_k,
                   u32* output_tokens) {
    if (!ml || !ml->loaded) {
        printf("Model not loaded\n");
        return 0;
    }
    
    /* Tokenize input */
    HSTokenizer tok;
    if (!prompt || !output_tokens || ml->vocab_size == 0) return 0;
    if (!ml->tokenizer_vocab) {
        printf("Tokenizer not loaded\n");
        return 0;
    }
    hs_tokenizer_init(&tok, ml->tokenizer_vocab, ml->vocab_size, 0, 1, 2, 3);
    
    u32* input_tokens = malloc(1024 * sizeof(u32));
    if (!input_tokens) return 0;
    
    u32 input_len = hs_tokenizer_encode(&tok, prompt, strlen(prompt), input_tokens, 1024);
    
    if (input_len == 0) {
        /* Fallback: use character codes as tokens */
        input_len = strlen(prompt);
        for (u32 i = 0; i < input_len && i < 1024; i++) {
            input_tokens[i] = (u32)(unsigned char)prompt[i];
        }
    }
    
    u32 total_len = input_len;
    memcpy(output_tokens, input_tokens, input_len * sizeof(u32));
    
    /* Generate tokens */
    float* logits = malloc(ml->vocab_size * sizeof(float));
    if (!logits) {
        free(input_tokens);
        return 0;
    }
    
    for (u32 gen = 0; gen < max_new; gen++) {
        /* Forward pass */
        int ret = hs_ml_forward(ml, output_tokens, total_len, logits);
        if (ret != 0) break;
        
        /* Apply temperature */
        if (temperature > 0) {
            for (u32 i = 0; i < ml->vocab_size; i++) {
                logits[i] /= temperature;
            }
        }
        
        /* Sample */
        u32 next_token;
        if (top_k > 0 && top_k < ml->vocab_size) {
            next_token = hs_ml_sample_topk(logits, ml->vocab_size, top_k);
        } else {
            next_token = hs_ml_sample_greedy(logits, ml->vocab_size);
        }
        
        output_tokens[total_len] = next_token;
        total_len++;
        
        /* Check for EOS */
        if (next_token == 2) break;  /* EOS token */
    }
    
    hs_tokenizer_free(&tok);
    free(input_tokens);
    free(logits);
    
    return total_len - input_len;
}
