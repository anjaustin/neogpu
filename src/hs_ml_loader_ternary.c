/*
 * NeoGPU ML - GGUF Loader for HSMLTernary
 *
 * Loads a GGUF file (real BitNet 2B-4T or compatible) into HSMLTernary.
 *
 * GGUF binary layout (little-endian):
 *   [magic u32][version u32][tensor_count u64][kv_count u64]
 *   [kv_pairs...]
 *   [tensor_infos...]
 *   [alignment padding]
 *   [tensor data...]
 *
 * Tensor naming (GGUF standard BitNet):
 *   token_embd.weight          [vocab, hidden]  F16
 *   output.weight              [vocab, hidden]  F16  (absent = weight tying)
 *   output_norm.weight         [hidden]         F32
 *   blk.{l}.attn_norm.weight   [hidden]         F32
 *   blk.{l}.attn_sub_norm.weight [hidden]       F32
 *   blk.{l}.ffn_norm.weight    [hidden]         F32
 *   blk.{l}.ffn_sub_norm.weight [ffn_hidden]    F32
 *   blk.{l}.attn_q.weight      [hidden, hidden] I2_S
 *   blk.{l}.attn_k.weight      [kv_size, hidden] I2_S
 *   blk.{l}.attn_v.weight      [kv_size, hidden] I2_S
 *   blk.{l}.attn_output.weight [hidden, hidden] I2_S
 *   blk.{l}.ffn_gate.weight    [ffn_hidden, hidden] I2_S
 *   blk.{l}.ffn_up.weight      [ffn_hidden, hidden] I2_S
 *   blk.{l}.ffn_down.weight    [hidden, ffn_hidden] I2_S
 *
 * I2_S format (GGML type 36):
 *   Contiguous packed bytes: tensor_bytes = (rows * cols) / 4 [+ small header]
 *   64-weight blocks: 16 bytes per block, 4 groups of 16 weights.
 *   Bit layout: bits [7:6] = group 0, [5:4] = group 1, [3:2] = group 2, [1:0] = group 3
 *   Raw codes: 0 = -1, 1 = 0, 2 = +1
 *
 * When I2_S weights are detected, m->use_i2s is set to true.
 * The inference engine then uses hs_ml_ternary_f32_proj() (pure f32 path).
 * Norms from GGUF are loaded as-is; call hs_mlt_load_norms_sidecar() to
 * override them with the BF16 values from models/norms_v2.bin.
 */

#include "hs_ml_infer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define GGUF_MAGIC   0x46554747u
#define GGUF_VERSION 3u
#define GGUF_ALIGNMENT 32u

/* GGUF metadata value types */
#define GGUF_TYPE_UINT8   0u
#define GGUF_TYPE_INT8    1u
#define GGUF_TYPE_UINT16  2u
#define GGUF_TYPE_INT16   3u
#define GGUF_TYPE_UINT32  4u
#define GGUF_TYPE_INT32   5u
#define GGUF_TYPE_FLOAT32 6u
#define GGUF_TYPE_BOOL    7u
#define GGUF_TYPE_STRING  8u
#define GGUF_TYPE_ARRAY   9u
#define GGUF_TYPE_UINT64  10u
#define GGUF_TYPE_INT64   11u
#define GGUF_TYPE_FLOAT64 12u

/* GGML tensor types */
#define GGML_TYPE_F32  0u
#define GGML_TYPE_F16  1u
#define GGML_TYPE_I8   24u
#define GGML_TYPE_I2_S 36u

#define MAX_TENSORS 4096
#define MAX_NAME    256

/*============================================================================
 * Read helpers
 *============================================================================*/

static int rd_u8(FILE* f, uint8_t* v)  { return fread(v, 1, 1, f) == 1 ? 0 : -1; }
static int rd_u16(FILE* f, uint16_t* v){ uint8_t b[2]; if(fread(b,1,2,f)!=2) return -1; *v=(uint16_t)(b[0]|(b[1]<<8)); return 0; }
static int rd_u32(FILE* f, uint32_t* v){ uint8_t b[4]; if(fread(b,1,4,f)!=4) return -1; *v=(uint32_t)b[0]|((uint32_t)b[1]<<8)|((uint32_t)b[2]<<16)|((uint32_t)b[3]<<24); return 0; }
static int rd_i32(FILE* f, int32_t*  v){ return rd_u32(f, (uint32_t*)v); }
static int rd_u64(FILE* f, uint64_t* v){
    uint8_t b[8]; if(fread(b,1,8,f)!=8) return -1;
    *v=(uint64_t)b[0]|((uint64_t)b[1]<<8)|((uint64_t)b[2]<<16)|((uint64_t)b[3]<<24)|
       ((uint64_t)b[4]<<32)|((uint64_t)b[5]<<40)|((uint64_t)b[6]<<48)|((uint64_t)b[7]<<56);
    return 0;
}
static int rd_f32(FILE* f, float* v){ uint32_t u; if(rd_u32(f,&u)) return -1; memcpy(v,&u,4); return 0; }

/* Read GGUF string (u64 len + bytes), return malloc'd null-terminated string. */
static char* rd_string(FILE* f) {
    uint64_t len;
    if(rd_u64(f, &len)) return NULL;
    if(len > 65535) { fseek(f, (long)len, SEEK_CUR); return strdup(""); }
    char* s = malloc(len + 1);
    if(!s) return NULL;
    if(len > 0 && fread(s, 1, (size_t)len, f) != (size_t)len) { free(s); return NULL; }
    s[len] = '\0';
    return s;
}

/* Skip a metadata value of the given type. */
static int skip_value(FILE* f, uint32_t type) {
    uint64_t len;
    char* s;
    switch (type) {
        case GGUF_TYPE_UINT8:  case GGUF_TYPE_INT8:  case GGUF_TYPE_BOOL:
            return fseek(f, 1, SEEK_CUR);
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16:
            return fseek(f, 2, SEEK_CUR);
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32:
            return fseek(f, 4, SEEK_CUR);
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64:
            return fseek(f, 8, SEEK_CUR);
        case GGUF_TYPE_STRING:
            s = rd_string(f); free(s); return s ? 0 : -1;
        case GGUF_TYPE_ARRAY: {
            uint32_t elem_type;
            if(rd_u32(f, &elem_type)) return -1;
            if(rd_u64(f, &len)) return -1;
            for(uint64_t i = 0; i < len; i++)
                if(skip_value(f, elem_type)) return -1;
            return 0;
        }
        default: return -1;
    }
}

/*============================================================================
 * Tensor info
 *============================================================================*/

typedef struct {
    char     name[MAX_NAME];
    uint32_t n_dims;
    uint64_t dims[4];      /* GGUF col-major: dims[0]=cols(K), dims[1]=rows(N) */
    uint32_t type;
    uint64_t offset;       /* offset from tensor_data_start */
} TensorInfo;

static int find_tensor(TensorInfo* tis, int n, const char* name) {
    for(int i = 0; i < n; i++)
        if(strcmp(tis[i].name, name) == 0) return i;
    return -1;
}

/*============================================================================
 * Float conversion helpers
 *============================================================================*/

static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x0400) == 0) { mant <<= 1; exp--; }
            mant &= 0x03FF;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out; memcpy(&out, &f, sizeof(float)); return out;
}

static float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float out; memcpy(&out, &u, sizeof(float)); return out;
}

/* Load tensor data as float32, handling F32, F16 types. */
static float* load_as_f32(FILE* f, long data_start,
                           TensorInfo* ti, uint32_t n_elem) {
    float* out = malloc((size_t)n_elem * sizeof(float));
    if(!out) return NULL;

    long pos = data_start + (long)ti->offset;
    if(fseek(f, pos, SEEK_SET)) { free(out); return NULL; }

    if(ti->type == GGML_TYPE_F32) {
        if(fread(out, sizeof(float), n_elem, f) != n_elem) { free(out); return NULL; }
    } else if(ti->type == GGML_TYPE_F16) {
        uint16_t* tmp = malloc((size_t)n_elem * sizeof(uint16_t));
        if(!tmp) { free(out); return NULL; }
        if(fread(tmp, sizeof(uint16_t), n_elem, f) != n_elem) { free(tmp); free(out); return NULL; }
        for(uint32_t i = 0; i < n_elem; i++) out[i] = fp16_to_f32(tmp[i]);
        free(tmp);
    } else if(ti->type == GGML_TYPE_I8) {
        int8_t* tmp = malloc(n_elem);
        if(!tmp) { free(out); return NULL; }
        if(fread(tmp, 1, n_elem, f) != n_elem) { free(tmp); free(out); return NULL; }
        for(uint32_t i = 0; i < n_elem; i++) out[i] = (float)tmp[i];
        free(tmp);
    } else {
        fprintf(stderr, "loader: unsupported tensor type %u for load_as_f32\n", ti->type);
        free(out); return NULL;
    }
    return out;
}

/*============================================================================
 * I2_S direct load
 *
 * I2_S layout in GGUF (type 36):
 *   Contiguous packed bytes, no per-row metadata.
 *   tensor_bytes = ceil(N * K / 4)   (rounded to GGUF_ALIGNMENT)
 *   Row r occupies bytes [r * (K/4), r * (K/4) + K/4).
 *   Within each row, K weights packed into K/4 bytes.
 *   Block of 64 weights (16 bytes):
 *     byte[i] holds 4 codes: bits[7:6]=group0, [5:4]=group1, [3:2]=group2, [1:0]=group3
 *     group g covers activations at positions bi*64 + g*16 + [0..15]
 *     code: 0=-1, 1=0, 2=+1
 *
 * We load the raw packed bytes directly — the f32 kernel reads I2_S in-place.
 * No re-packing needed.
 *============================================================================*/

static uint8_t* load_i2s_raw(FILE* f, long data_start,
                               TensorInfo* ti, uint32_t rows, uint32_t cols) {
    size_t row_bytes = cols / 4;
    size_t total     = (size_t)rows * row_bytes;
    uint8_t* buf = aligned_alloc(64, total);
    if(!buf) return NULL;

    long pos = data_start + (long)ti->offset;
    if(fseek(f, pos, SEEK_SET)) { free(buf); return NULL; }
    if(fread(buf, 1, total, f) != total) {
        fprintf(stderr, "loader: I2_S read failed for '%s' (want %zu bytes)\n",
                ti->name, total);
        free(buf); return NULL;
    }
    return buf;
}

/*============================================================================
 * F32 -> 2-bit packed ternary (for synthetic / F32 test models only)
 *
 * Our internal 2-bit code: 00=0, 01=+1, 10=-1
 * This is NOT used for real I2_S model weights.
 *============================================================================*/

static void f32_to_ternary_packed(uint8_t* out, const float* in, uint32_t N) {
    uint32_t N4 = N / 4;
    for(uint32_t i = 0; i < N4; i++) {
        uint8_t byte = 0;
        for(int b = 0; b < 4; b++) {
            float v = in[i*4 + b];
            uint8_t code = (v > 0.5f) ? 1 : (v < -0.5f) ? 2 : 0;
            byte |= (code << (b * 2));
        }
        out[i] = byte;
    }
    if(N % 4 != 0) {
        uint8_t byte = 0;
        for(uint32_t b = 0; b < N % 4; b++) {
            float v = in[N4*4 + b];
            uint8_t code = (v > 0.5f) ? 1 : (v < -0.5f) ? 2 : 0;
            byte |= (code << (b * 2));
        }
        out[N4] = byte;
    }
}

/*============================================================================
 * load_ternary_proj
 *
 * Loads one projection layer weight matrix.
 * For I2_S tensors: loads raw bytes, sets *is_i2s = true.
 * For F32 tensors: packs into 2-bit ternary.
 *============================================================================*/

static int load_ternary_proj(FILE* f, long data_start,
                              TensorInfo* tis, int n_tensors,
                              const char* wname, const char* sname,
                              uint32_t rows, uint32_t cols,
                              uint8_t** w_out, float** s_out,
                              bool* is_i2s) {
    int wi = find_tensor(tis, n_tensors, wname);
    int si = find_tensor(tis, n_tensors, sname);
    if(wi < 0) {
        fprintf(stderr, "loader: tensor '%s' not found\n", wname);
        return -1;
    }

    float* scales = malloc(rows * sizeof(float));
    if(!scales) return -1;
    for(uint32_t r = 0; r < rows; r++) scales[r] = 1.0f;

    if(tis[wi].type == GGML_TYPE_I2_S) {
        /* Load raw I2_S bytes — used directly by hs_ml_ternary_f32_proj */
        uint8_t* raw = load_i2s_raw(f, data_start, &tis[wi], rows, cols);
        if(!raw) { free(scales); return -1; }

        /* Load per-tensor weight scale if present.
         * BitNet uses a single scalar scale per projection:
         *   output = ternary_proj(input, W) / weight_scale
         * We store it in scales[0]; the inference code reads it from there. */
        if(si >= 0) {
            uint64_t scale_elems = tis[si].dims[0];
            if(scale_elems == 1) {
                /* Single scalar — broadcast to scales[0] */
                float s;
                long pos = data_start + (long)tis[si].offset;
                if(fseek(f, pos, SEEK_SET) == 0 && fread(&s, sizeof(float), 1, f) == 1) {
                    scales[0] = s;
                    /* Broadcast to all rows for uniformity */
                    for(uint32_t r = 1; r < rows; r++) scales[r] = s;
                }
            } else if(scale_elems == rows) {
                long pos = data_start + (long)tis[si].offset;
                if(fseek(f, pos, SEEK_SET) != 0 ||
                   fread(scales, sizeof(float), rows, f) != rows)
                    for(uint32_t r = 0; r < rows; r++) scales[r] = 1.0f;
            }
        }

        *w_out  = raw;
        *s_out  = scales;
        if(is_i2s) *is_i2s = true;
        return 0;

    } else if(tis[wi].type == GGML_TYPE_F32) {
        uint32_t packed_row = (cols + 3) / 4;
        uint8_t* packed = aligned_alloc(64, (size_t)rows * packed_row);
        if(!packed) { free(scales); return -1; }

        float* raw = load_as_f32(f, data_start, &tis[wi], rows * cols);
        if(!raw) { free(packed); free(scales); return -1; }
        for(uint32_t r = 0; r < rows; r++)
            f32_to_ternary_packed(packed + r * packed_row, raw + r * cols, cols);
        free(raw);

        if(si >= 0) {
            long pos = data_start + (long)tis[si].offset;
            if(fseek(f, pos, SEEK_SET) || fread(scales, sizeof(float), rows, f) != rows)
                for(uint32_t r = 0; r < rows; r++) scales[r] = 1.0f;
        }

        *w_out  = packed;
        *s_out  = scales;
        if(is_i2s) *is_i2s = false;
        return 0;

    } else {
        fprintf(stderr, "loader: unsupported weight type %u for '%s'\n",
                tis[wi].type, wname);
        free(scales);
        return -1;
    }
}

/*============================================================================
 * hs_mlt_load_gguf — main entry point
 *============================================================================*/

int hs_mlt_load_gguf(HSMLTernary* m, const char* path) {
    if(!m || !path) return -1;
    hs_mlt_free(m);
    hs_mlt_init(m);

    FILE* f = fopen(path, "rb");
    if(!f) { fprintf(stderr, "loader: cannot open '%s'\n", path); return -1; }

    /* ── Header ── */
    uint32_t magic, version;
    uint64_t tensor_count, kv_count;
    if(rd_u32(f, &magic) || magic != GGUF_MAGIC) {
        fprintf(stderr, "loader: not a GGUF file\n"); goto fail;
    }
    if(rd_u32(f, &version)) goto fail;
    if(version < 2 || version > 3) {
        fprintf(stderr, "loader: unsupported GGUF version %u\n", version); goto fail;
    }
    if(rd_u64(f, &tensor_count) || rd_u64(f, &kv_count)) goto fail;
    printf("loader: GGUF v%u, %llu tensors, %llu metadata pairs\n",
           version, (unsigned long long)tensor_count, (unsigned long long)kv_count);

    /* ── Metadata KV pairs ── */
    uint32_t hidden_size = 0, n_layers = 0, n_heads = 0, n_kv_heads = 0, ffn_size = 0;
    uint32_t vocab_size  = 0, ctx_len  = 0;
    uint32_t tokenizer_bos = 1, tokenizer_eos = 2;
    float    rope_freq   = 10000.0f;

    /* Tokenizer vocab storage */
    char**   tok_vocab   = NULL;
    uint32_t tok_vocab_n = 0;
    char**   tok_merges  = NULL;
    uint32_t tok_merges_n = 0;

    for(uint64_t i = 0; i < kv_count; i++) {
        char* key = rd_string(f);
        if(!key) goto fail;
        uint32_t vtype;
        if(rd_u32(f, &vtype)) { free(key); goto fail; }

#define HAS_SUFFIX(k, s) (strstr((k), (s)) && strcmp(strstr((k), (s)), (s)) == 0)
        if(HAS_SUFFIX(key, ".embedding_length") && vtype == GGUF_TYPE_UINT32)
            rd_u32(f, &hidden_size);
        else if(HAS_SUFFIX(key, ".block_count") && vtype == GGUF_TYPE_UINT32)
            rd_u32(f, &n_layers);
        else if(HAS_SUFFIX(key, ".attention.head_count_kv") && vtype == GGUF_TYPE_UINT32)
            rd_u32(f, &n_kv_heads);
        else if(HAS_SUFFIX(key, ".attention.head_count") && vtype == GGUF_TYPE_UINT32)
            rd_u32(f, &n_heads);
        else if(HAS_SUFFIX(key, ".feed_forward_length") && vtype == GGUF_TYPE_UINT32)
            rd_u32(f, &ffn_size);
        else if(HAS_SUFFIX(key, ".context_length") && vtype == GGUF_TYPE_UINT32)
            rd_u32(f, &ctx_len);
        else if(HAS_SUFFIX(key, ".vocab_size") && vtype == GGUF_TYPE_UINT32)
            rd_u32(f, &vocab_size);
        else if(HAS_SUFFIX(key, ".rope.freq_base") && vtype == GGUF_TYPE_FLOAT32)
            rd_f32(f, &rope_freq);
        else if(strcmp(key, "tokenizer.ggml.bos_token_id") == 0 && vtype == GGUF_TYPE_UINT32)
            rd_u32(f, &tokenizer_bos);
        else if(strcmp(key, "tokenizer.ggml.eos_token_id") == 0 && vtype == GGUF_TYPE_UINT32)
            rd_u32(f, &tokenizer_eos);
        else if(strcmp(key, "tokenizer.ggml.tokens") == 0 && vtype == GGUF_TYPE_ARRAY) {
            /* Load tokenizer vocab */
            uint32_t elem_type;
            uint64_t arr_len;
            rd_u32(f, &elem_type);
            rd_u64(f, &arr_len);
            if(!vocab_size) vocab_size = (uint32_t)arr_len;
            tok_vocab_n = (uint32_t)arr_len;
            tok_vocab = calloc(tok_vocab_n, sizeof(char*));
            for(uint64_t j = 0; j < arr_len; j++) {
                char* tok = rd_string(f);
                if(tok_vocab && j < tok_vocab_n) tok_vocab[j] = tok;
                else free(tok);
            }
        }
        else if(strcmp(key, "tokenizer.ggml.merges") == 0 && vtype == GGUF_TYPE_ARRAY) {
            /* Load BPE merge rules */
            uint32_t elem_type;
            uint64_t arr_len;
            rd_u32(f, &elem_type);
            rd_u64(f, &arr_len);
            tok_merges_n = (uint32_t)arr_len;
            tok_merges = calloc(tok_merges_n, sizeof(char*));
            for(uint64_t j = 0; j < arr_len; j++) {
                char* mg = rd_string(f);
                if(tok_merges && j < tok_merges_n) tok_merges[j] = mg;
                else free(mg);
            }
        }
#undef HAS_SUFFIX
        else {
            if(skip_value(f, vtype)) { free(key); goto fail; }
        }
        free(key);
    }

    if(!hidden_size || !n_layers || !n_heads || !ffn_size || !vocab_size) {
        fprintf(stderr, "loader: missing required metadata "
                "(hidden=%u layers=%u heads=%u ffn=%u vocab=%u)\n",
                hidden_size, n_layers, n_heads, ffn_size, vocab_size);
        goto fail;
    }
    if(!ctx_len)    ctx_len  = 4096;
    if(!n_kv_heads) n_kv_heads = n_heads;
    printf("loader: hidden=%u layers=%u Q-heads=%u KV-heads=%u ffn=%u "
           "vocab=%u ctx=%u rope=%.0f bos=%u eos=%u\n",
           hidden_size, n_layers, n_heads, n_kv_heads, ffn_size,
           vocab_size, ctx_len, rope_freq, tokenizer_bos, tokenizer_eos);

    /* ── Tensor infos ── */
    if(tensor_count > MAX_TENSORS) {
        fprintf(stderr, "loader: too many tensors (%llu)\n",
                (unsigned long long)tensor_count);
        goto fail;
    }
    TensorInfo* tis = calloc((size_t)tensor_count, sizeof(TensorInfo));
    if(!tis) goto fail;

    for(uint64_t i = 0; i < tensor_count; i++) {
        char* name = rd_string(f);
        if(!name) { free(tis); goto fail; }
        strncpy(tis[i].name, name, MAX_NAME - 1);
        free(name);

        if(rd_u32(f, &tis[i].n_dims)) { free(tis); goto fail; }
        for(uint32_t d = 0; d < tis[i].n_dims && d < 4; d++)
            if(rd_u64(f, &tis[i].dims[d])) { free(tis); goto fail; }
        if(rd_u32(f, &tis[i].type))   { free(tis); goto fail; }
        if(rd_u64(f, &tis[i].offset)) { free(tis); goto fail; }
    }

    /* ── Data start: align to GGUF_ALIGNMENT ── */
    long header_end = ftell(f);
    long data_start = (long)(((uint64_t)header_end + GGUF_ALIGNMENT - 1) &
                              ~(uint64_t)(GGUF_ALIGNMENT - 1));

    /* ── Allocate model ── */
    m->vocab_size      = vocab_size;
    m->hidden_size     = hidden_size;
    m->num_layers      = n_layers;
    m->num_heads       = n_heads;
    m->num_kv_heads    = n_kv_heads;
    m->head_dim        = hidden_size / n_heads;
    m->ffn_hidden_size = ffn_size;
    m->max_context     = ctx_len;
    m->rope_theta      = rope_freq;
    m->use_i2s         = false;  /* set true below when I2_S tensors found */
    m->tokenizer_bos   = tokenizer_bos;
    m->tokenizer_eos   = tokenizer_eos;
    m->tokenizer_vocab  = tok_vocab;
    m->tokenizer_merges = tok_merges;
    m->num_merges       = tok_merges_n;
    tok_vocab           = NULL;  /* model owns it now */
    tok_merges          = NULL;

    /* Non-quantized weights */
    size_t emb_n = (size_t)vocab_size * hidden_size;
    m->embedding   = malloc(emb_n * sizeof(float));
    m->lm_head_f16 = malloc(emb_n * sizeof(uint16_t));
    m->final_norm  = malloc(hidden_size * sizeof(float));
    if(!m->embedding || !m->lm_head_f16 || !m->final_norm) { free(tis); goto fail; }

    /* Load embedding (F16 in real model → F32 for embedding lookup) */
    {
        int idx = find_tensor(tis, (int)tensor_count, "token_embd.weight");
        if(idx < 0) {
            fprintf(stderr, "loader: missing token_embd.weight\n");
            free(tis); goto fail;
        }
        float* raw = load_as_f32(f, data_start, &tis[idx], vocab_size * hidden_size);
        if(!raw) { free(tis); goto fail; }
        memcpy(m->embedding, raw, emb_n * sizeof(float));
        free(raw);
        printf("loader: embedding loaded (type=%u)\n", tis[idx].type);
    }

    /* Load lm_head as F16 (for logit computation — halves memory bandwidth).
     * With weight tying, convert the F32 embedding back to F16.
     * With separate output.weight, load it directly as F16. */
    {
        int idx = find_tensor(tis, (int)tensor_count, "output.weight");
        if(idx >= 0 && tis[idx].type == GGML_TYPE_F16) {
            /* Direct F16 load */
            long pos = data_start + (long)tis[idx].offset;
            if(fseek(f, pos, SEEK_SET) == 0 &&
               fread(m->lm_head_f16, sizeof(uint16_t), emb_n, f) == emb_n) {
                printf("loader: lm_head loaded as F16\n");
            } else {
                /* Fallback: convert embedding F32 → F16 */
                for(size_t i = 0; i < emb_n; i++) {
                    float v = m->embedding[i];
                    /* F32 → F16 via bit manipulation */
                    uint32_t u; memcpy(&u, &v, 4);
                    uint32_t sign = (u >> 16) & 0x8000;
                    int32_t exp = ((u >> 23) & 0xFF) - 127 + 15;
                    uint32_t mant = (u >> 13) & 0x03FF;
                    if(exp <= 0) m->lm_head_f16[i] = (uint16_t)sign;
                    else if(exp >= 31) m->lm_head_f16[i] = (uint16_t)(sign | 0x7C00);
                    else m->lm_head_f16[i] = (uint16_t)(sign | (exp << 10) | mant);
                }
                printf("loader: lm_head F16 from embedding (conversion)\n");
            }
        } else {
            /* Weight tying: convert embedding F32 → F16 for lm_head */
            for(size_t i = 0; i < emb_n; i++) {
                float v = m->embedding[i];
                uint32_t u; memcpy(&u, &v, 4);
                uint32_t sign = (u >> 16) & 0x8000;
                int32_t exp = ((u >> 23) & 0xFF) - 127 + 15;
                uint32_t mant = (u >> 13) & 0x03FF;
                if(exp <= 0) m->lm_head_f16[i] = (uint16_t)sign;
                else if(exp >= 31) m->lm_head_f16[i] = (uint16_t)(sign | 0x7C00);
                else m->lm_head_f16[i] = (uint16_t)(sign | (exp << 10) | mant);
            }
            printf("loader: no output.weight, lm_head F16 from embedding (weight tying)\n");
        }
    }

    /* Load final norm */
    {
        int idx = find_tensor(tis, (int)tensor_count, "output_norm.weight");
        if(idx >= 0) {
            float* raw = load_as_f32(f, data_start, &tis[idx], hidden_size);
            if(raw) { memcpy(m->final_norm, raw, hidden_size * sizeof(float)); free(raw); }
            else { for(uint32_t i = 0; i < hidden_size; i++) m->final_norm[i] = 1.0f; }
        } else {
            for(uint32_t i = 0; i < hidden_size; i++) m->final_norm[i] = 1.0f;
        }
    }

    /* Per-layer weights */
    m->layers = calloc(n_layers, sizeof(HSTernaryLayer));
    if(!m->layers) { free(tis); goto fail; }

    u32 kv_size = n_kv_heads * (hidden_size / n_heads);

    for(uint32_t l = 0; l < n_layers; l++) {
        HSTernaryLayer* lay = &m->layers[l];
        char wname[256], sname[256];

        /* Norms — allocate and default to 1 */
        lay->attn_norm     = malloc(hidden_size * sizeof(float));
        lay->attn_sub_norm = malloc(hidden_size * sizeof(float));
        lay->ffn_norm      = malloc(hidden_size * sizeof(float));
        lay->ffn_sub_norm  = malloc(ffn_size * sizeof(float));
        if(!lay->attn_norm || !lay->attn_sub_norm ||
           !lay->ffn_norm  || !lay->ffn_sub_norm) { free(tis); goto fail; }
        for(uint32_t i = 0; i < hidden_size; i++) {
            lay->attn_norm[i]     = 1.0f;
            lay->attn_sub_norm[i] = 1.0f;
            lay->ffn_norm[i]      = 1.0f;
        }
        for(uint32_t i = 0; i < ffn_size; i++) lay->ffn_sub_norm[i] = 1.0f;

        /* Load norms from GGUF (may be garbage for the real model — override
         * with hs_mlt_load_norms_sidecar() after loading) */
        snprintf(wname, sizeof(wname), "blk.%u.attn_norm.weight", l);
        { int ni = find_tensor(tis, (int)tensor_count, wname);
          if(ni >= 0) {
              float* r = load_as_f32(f, data_start, &tis[ni], hidden_size);
              if(r) { memcpy(lay->attn_norm, r, hidden_size*sizeof(float)); free(r); }
          } }

        snprintf(wname, sizeof(wname), "blk.%u.attn_sub_norm.weight", l);
        { int ni = find_tensor(tis, (int)tensor_count, wname);
          if(ni >= 0) {
              float* r = load_as_f32(f, data_start, &tis[ni], hidden_size);
              if(r) { memcpy(lay->attn_sub_norm, r, hidden_size*sizeof(float)); free(r); }
          } }

        snprintf(wname, sizeof(wname), "blk.%u.ffn_norm.weight", l);
        { int ni = find_tensor(tis, (int)tensor_count, wname);
          if(ni >= 0) {
              float* r = load_as_f32(f, data_start, &tis[ni], hidden_size);
              if(r) { memcpy(lay->ffn_norm, r, hidden_size*sizeof(float)); free(r); }
          } }

        snprintf(wname, sizeof(wname), "blk.%u.ffn_sub_norm.weight", l);
        { int ni = find_tensor(tis, (int)tensor_count, wname);
          if(ni >= 0) {
              float* r = load_as_f32(f, data_start, &tis[ni], ffn_size);
              if(r) { memcpy(lay->ffn_sub_norm, r, ffn_size*sizeof(float)); free(r); }
          } }

        /* Allocate scale arrays (used only in int8 path; set to 1 for I2_S) */
        lay->q_scale    = malloc(hidden_size * sizeof(float));
        lay->k_scale    = malloc(kv_size * sizeof(float));
        lay->v_scale    = malloc(kv_size * sizeof(float));
        lay->o_scale    = malloc(hidden_size * sizeof(float));
        lay->gate_scale = malloc(ffn_size * sizeof(float));
        lay->up_scale   = malloc(ffn_size * sizeof(float));
        lay->down_scale = malloc(hidden_size * sizeof(float));
        if(!lay->q_scale || !lay->k_scale || !lay->v_scale || !lay->o_scale ||
           !lay->gate_scale || !lay->up_scale || !lay->down_scale) {
            free(tis); goto fail;
        }

        /* Load projection weights.
         * LOAD_PROJ detects I2_S and accumulates use_i2s flag. */
        bool layer_i2s = false;

#define LOAD_PROJ(fw, fs, base, rows, cols) do { \
    snprintf(wname, sizeof(wname), "blk.%u." base ".weight", l); \
    snprintf(sname, sizeof(sname), "blk.%u." base ".weight_scale", l); \
    bool _i2s = false; \
    if(load_ternary_proj(f, data_start, tis, (int)tensor_count, \
                         wname, sname, rows, cols, \
                         &lay->fw, &lay->fs, &_i2s)) { \
        fprintf(stderr, "loader: failed '%s'\n", wname); \
        free(tis); goto fail; \
    } \
    if(_i2s) layer_i2s = true; \
} while(0)

        LOAD_PROJ(q_proj,    q_scale,    "attn_q",      hidden_size, hidden_size);
        LOAD_PROJ(k_proj,    k_scale,    "attn_k",      kv_size,     hidden_size);
        LOAD_PROJ(v_proj,    v_scale,    "attn_v",      kv_size,     hidden_size);
        LOAD_PROJ(o_proj,    o_scale,    "attn_output", hidden_size, hidden_size);
        LOAD_PROJ(gate_proj, gate_scale, "ffn_gate",    ffn_size,    hidden_size);
        LOAD_PROJ(up_proj,   up_scale,   "ffn_up",      ffn_size,    hidden_size);
        LOAD_PROJ(down_proj, down_scale, "ffn_down",    hidden_size, ffn_size);

#undef LOAD_PROJ

        if(layer_i2s) m->use_i2s = true;

        if((l % 5 == 0) || l == n_layers - 1)
            printf("loader: layer %u/%u loaded%s\n", l+1, n_layers,
                   layer_i2s ? " (I2_S)" : " (F32)");
    }

    /* Scratch buffers */
    uint32_t max_dim = (hidden_size > ffn_size) ? hidden_size : ffn_size;
    m->quant_buf  = aligned_alloc(64, max_dim);           /* int8 */
    m->gemm_buf   = aligned_alloc(64, max_dim * sizeof(int32_t));
    m->hidden_buf = malloc(hidden_size * sizeof(float));
    m->ffn_buf    = malloc(ffn_size * sizeof(float));
    m->q_buf      = malloc(hidden_size * sizeof(float));
    m->k_buf      = malloc(kv_size * sizeof(float));
    m->v_buf      = malloc(kv_size * sizeof(float));
    m->attn_buf   = calloc(hidden_size, sizeof(float));
    m->gate_buf   = malloc(ffn_size * sizeof(float));
    m->up_buf     = malloc(ffn_size * sizeof(float));
    m->score_buf  = malloc(ctx_len * sizeof(float));
    m->ffn_qbuf   = aligned_alloc(64, ffn_size);          /* int8 */

    if(!m->quant_buf || !m->gemm_buf || !m->hidden_buf || !m->ffn_buf ||
       !m->q_buf || !m->k_buf || !m->v_buf || !m->attn_buf ||
       !m->gate_buf || !m->up_buf || !m->score_buf || !m->ffn_qbuf) {
        free(tis); goto fail;
    }

    free(tis);
    fclose(f);
    m->loaded = true;
    printf("loader: model loaded — use_i2s=%d rope_theta=%.0f\n",
           (int)m->use_i2s, m->rope_theta);
    return 0;

fail:
    if(tok_vocab) {
        for(uint32_t i = 0; i < tok_vocab_n; i++) free(tok_vocab[i]);
        free(tok_vocab);
    }
    if(tok_merges) {
        for(uint32_t i = 0; i < tok_merges_n; i++) free(tok_merges[i]);
        free(tok_merges);
    }
    fclose(f);
    hs_mlt_free(m);
    return -1;
}

/*============================================================================
 * hs_mlt_load_norms_sidecar
 *
 * Loads BF16 norm weights from norms_v2.bin sidecar.
 * These were extracted directly from the HuggingFace safetensors checkpoint
 * via HTTP range request because the GGUF converter produces corrupt norms.
 *
 * File format:
 *   magic:     u32 = 0x4E524D32 ("NRM2")
 *   n_tensors: u32
 *   for each tensor:
 *     name_len: u32
 *     name:     char[name_len]  (NOT null-terminated in file)
 *     n_values: u32
 *     data:     bf16[n_values]
 *
 * Tensor names are HuggingFace convention:
 *   model.layers.{l}.input_layernorm.weight         -> attn_norm
 *   model.layers.{l}.self_attn.attn_sub_norm.weight -> attn_sub_norm
 *   model.layers.{l}.post_attention_layernorm.weight -> ffn_norm
 *   model.layers.{l}.mlp.ffn_sub_norm.weight        -> ffn_sub_norm
 *   model.norm.weight                                -> final_norm
 *============================================================================*/

#define NORMS_SIDECAR_MAGIC 0x4E524D32u  /* "NRM2" little-endian */

int hs_mlt_load_norms_sidecar(HSMLTernary* m, const char* path) {
    if(!m || !m->loaded || !path) return -1;

    FILE* f = fopen(path, "rb");
    if(!f) {
        fprintf(stderr, "norms_sidecar: cannot open '%s'\n", path);
        return -1;
    }

    uint32_t magic, n_tensors;
    if(rd_u32(f, &magic) || magic != NORMS_SIDECAR_MAGIC) {
        fprintf(stderr, "norms_sidecar: bad magic 0x%08X (expected 0x%08X) in '%s'\n",
                magic, NORMS_SIDECAR_MAGIC, path);
        fclose(f); return -1;
    }
    if(rd_u32(f, &n_tensors)) { fclose(f); return -1; }
    printf("norms_sidecar: loading %u norm tensors from '%s'\n", n_tensors, path);

    uint32_t loaded = 0, skipped = 0;
    for(uint32_t t = 0; t < n_tensors; t++) {
        uint32_t name_len, n_values;
        if(rd_u32(f, &name_len)) break;
        if(name_len >= MAX_NAME) {
            /* Skip oversized name + data */
            fseek(f, name_len, SEEK_CUR);
            if(rd_u32(f, &n_values)) break;
            fseek(f, n_values * 2, SEEK_CUR);
            skipped++;
            continue;
        }

        char name[MAX_NAME] = {0};
        if(fread(name, 1, name_len, f) != name_len) break;
        name[name_len] = '\0';

        if(rd_u32(f, &n_values)) break;

        /* Read BF16 values and convert to float32 */
        uint16_t* bf16 = malloc(n_values * sizeof(uint16_t));
        if(!bf16) { fseek(f, n_values * 2, SEEK_CUR); skipped++; continue; }
        if(fread(bf16, 2, n_values, f) != n_values) { free(bf16); break; }

        float* vals = malloc(n_values * sizeof(float));
        if(!vals) { free(bf16); skipped++; continue; }
        for(uint32_t i = 0; i < n_values; i++) vals[i] = bf16_to_f32(bf16[i]);
        free(bf16);

        /* Route tensor to correct model field using HF name convention */
        bool placed = false;

        /* Final norm: model.norm.weight */
        if(strcmp(name, "model.norm.weight") == 0 && n_values == m->hidden_size) {
            memcpy(m->final_norm, vals, n_values * sizeof(float));
            placed = true;
        } else {
            /* Per-layer norms: model.layers.{l}.<path>.weight */
            uint32_t layer_idx = 0;
            char rest[MAX_NAME] = {0};
            if(sscanf(name, "model.layers.%u.%255s", &layer_idx, rest) == 2
               && layer_idx < m->num_layers) {
                HSTernaryLayer* lay = &m->layers[layer_idx];

                if(strcmp(rest, "input_layernorm.weight") == 0
                   && n_values == m->hidden_size) {
                    /* input_layernorm = attn_norm (pre-attention RMSNorm) */
                    memcpy(lay->attn_norm, vals, n_values * sizeof(float));
                    placed = true;
                } else if(strcmp(rest, "self_attn.attn_sub_norm.weight") == 0
                          && n_values == m->hidden_size) {
                    /* attn_sub_norm (BitNet subln after attention) */
                    memcpy(lay->attn_sub_norm, vals, n_values * sizeof(float));
                    placed = true;
                } else if(strcmp(rest, "post_attention_layernorm.weight") == 0
                          && n_values == m->hidden_size) {
                    /* post_attention_layernorm = ffn_norm (pre-FFN RMSNorm) */
                    memcpy(lay->ffn_norm, vals, n_values * sizeof(float));
                    placed = true;
                } else if(strcmp(rest, "mlp.ffn_sub_norm.weight") == 0
                          && n_values == m->ffn_hidden_size) {
                    /* ffn_sub_norm (BitNet subln after gate*up) */
                    memcpy(lay->ffn_sub_norm, vals, n_values * sizeof(float));
                    placed = true;
                }
            }
        }

        free(vals);
        if(placed) loaded++;
        else {
            if(skipped < 5)
                fprintf(stderr, "norms_sidecar: unrecognised '%s' (n=%u)\n",
                        name, n_values);
            skipped++;
        }
    }

    fclose(f);
    printf("norms_sidecar: placed %u/%u norm tensors (%u skipped)\n",
           loaded, n_tensors, skipped);

    /* Sanity check: we expect 121 tensors (30*4 + 1) */
    if(loaded < m->num_layers * 4) {
        fprintf(stderr, "norms_sidecar: WARNING — only %u norms placed, "
                "expected %u\n", loaded, m->num_layers * 4 + 1);
    }
    return (loaded > 0) ? 0 : -1;
}

/*============================================================================
 * hs_mlt_lmhead_encode — Ternary Spline LM Head Encoder
 *
 * Converts m->lm_head_f16 [V x H] into 8 I2_S ternary planes plus a
 * per-row F16 scale vector.  After encoding, lm_head_f16 is freed and
 * m->use_trit_lmhead is set to true.
 *
 * Encoding (per row v):
 *   scale    = max(|row|)
 *   remainder = row / scale          (float, in [-1, +1])
 *   For k = 0..7:
 *     plane[k][v] = I2_S_encode( round(clip(remainder, -1, +1)) )
 *     remainder   = remainder - decode(plane[k][v])
 *     remainder  *= 3                (magnify residual for next plane)
 *
 * I2_S packing: 4 ternary values per byte.
 *   Code 0 → -1,  code 1 → 0,  code 2 → +1
 *   byte = (c0 & 3) | ((c1 & 3) << 2) | ((c2 & 3) << 4) | ((c3 & 3) << 6)
 *============================================================================*/

/* Inline F16 helpers (duplicated from hs_ml_infer.c to avoid header exposure) */
static inline float ldr_fp16_to_f32(u16 h) {
    u32 sign = ((u32)h & 0x8000u) << 16;
    u32 exp  = (h >> 10) & 0x1Fu;
    u32 mant = h & 0x03FFu;
    u32 f;
    if (exp == 0)        f = sign;
    else if (exp == 31)  f = sign | 0x7F800000u | (mant << 13);
    else                 f = sign | ((exp + 112u) << 23) | (mant << 13);
    float out; memcpy(&out, &f, 4); return out;
}

static inline u16 ldr_f32_to_fp16(float v) {
    u32 u; memcpy(&u, &v, 4);
    u32 sign = (u >> 16) & 0x8000u;
    int  exp  = (int)((u >> 23) & 0xFFu) - 127 + 15;
    u32 mant = (u >> 13) & 0x03FFu;
    if (exp <= 0)  return (u16)sign;
    if (exp >= 31) return (u16)(sign | 0x7C00u);
    return (u16)(sign | ((u32)exp << 10) | mant);
}

int hs_mlt_lmhead_encode(HSMLTernary* m) {
    if (!m || !m->loaded || !m->lm_head_f16) return -1;
    if (m->use_trit_lmhead) return 0;  /* already encoded */

    u32 V = m->vocab_size;
    u32 H = m->hidden_size;
    u32 row_bytes = H / 4;            /* I2_S bytes per row */
    size_t plane_size = (size_t)V * row_bytes;

    /* Allocate 8 planes and the row scale array */
    for (int k = 0; k < 8; k++) {
        m->lm_head_planes[k] = malloc(plane_size);
        if (!m->lm_head_planes[k]) {
            for (int j = 0; j < k; j++) { free(m->lm_head_planes[j]); m->lm_head_planes[j] = NULL; }
            fprintf(stderr, "lmhead_encode: OOM allocating plane %d\n", k);
            return -1;
        }
    }
    m->lm_head_row_scale = malloc((size_t)V * sizeof(u16));
    if (!m->lm_head_row_scale) {
        for (int k = 0; k < 8; k++) { free(m->lm_head_planes[k]); m->lm_head_planes[k] = NULL; }
        fprintf(stderr, "lmhead_encode: OOM allocating row_scale\n");
        return -1;
    }

    /* Working buffer: one row of floats */
    float* row_f = malloc(H * sizeof(float));
    if (!row_f) {
        for (int k = 0; k < 8; k++) { free(m->lm_head_planes[k]); m->lm_head_planes[k] = NULL; }
        free(m->lm_head_row_scale);
        return -1;
    }

    printf("lmhead_encode: encoding %u rows x %u dims into 8 trit planes...\n", V, H);

    for (u32 v = 0; v < V; v++) {
        const u16* src = m->lm_head_f16 + (size_t)v * H;

        /* Convert F16 row to F32 and find row scale */
        float scale = 0.0f;
        for (u32 h = 0; h < H; h++) {
            row_f[h] = ldr_fp16_to_f32(src[h]);
            float av = row_f[h] < 0.0f ? -row_f[h] : row_f[h];
            if (av > scale) scale = av;
        }
        if (scale == 0.0f) scale = 1.0f;
        m->lm_head_row_scale[v] = ldr_f32_to_fp16(scale);

        /* Normalise to [-1, +1] */
        float inv_scale = 1.0f / scale;
        for (u32 h = 0; h < H; h++) row_f[h] *= inv_scale;

        /* Encode 8 planes */
        for (int k = 0; k < 8; k++) {
            u8* plane_row = m->lm_head_planes[k] + (size_t)v * row_bytes;

            /* Pack 4 ternary codes per byte */
            for (u32 bi = 0; bi < row_bytes; bi++) {
                u8 byte = 0;
                for (int s = 0; s < 4; s++) {
                    u32 hi = bi * 4 + s;
                    float r = row_f[hi];
                    /* Clip and round to {-1, 0, +1} */
                    int8_t t;
                    if      (r >  0.5f) t =  1;
                    else if (r < -0.5f) t = -1;
                    else                t =  0;
                    /* I2_S code: -1→0, 0→1, +1→2 */
                    u8 code = (u8)(t + 1);
                    byte |= (code & 3u) << (s * 2);
                    /* Subtract trit, magnify residual for next plane */
                    row_f[hi] = (row_f[hi] - (float)t) * 3.0f;
                }
                plane_row[bi] = byte;
            }
        }

        if (v % 16384 == 0)
            printf("lmhead_encode:   %u/%u rows\n", v, V);
    }

    free(row_f);

    /* Free the F16 lm_head — planes replace it */
    free(m->lm_head_f16);
    m->lm_head_f16 = NULL;
    m->use_trit_lmhead = true;

    printf("lmhead_encode: done. 8 planes encoded, lm_head_f16 freed.\n");
    return 0;
}
