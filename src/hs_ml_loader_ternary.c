/*
 * NeoGPU ML - GGUF Loader for HSMLTernary
 *
 * Loads a GGUF file produced by tools/write_test_gguf.py (or a compatible
 * real BitNet GGUF) into an HSMLTernary struct for inference.
 *
 * GGUF binary layout (little-endian):
 *   [magic u32][version u32][tensor_count u64][kv_count u64]
 *   [kv_pairs...]
 *   [tensor_infos...]
 *   [alignment padding]
 *   [tensor data...]
 *
 * Weight convention in our test GGUF:
 *   - All projection weights stored as F32 (GGML_TYPE_F32 = 0)
 *   - Shape in GGUF is [K, N] (column-major / transposed from row-major)
 *   - Companion tensor "<name>_scale" holds per-row F32 scales
 *   - We convert F32 weights to 2-bit packed ternary at load time
 *
 * Tensor naming (GGUF standard):
 *   token_embd.weight          [hidden, vocab]
 *   output.weight              [hidden, vocab]
 *   output_norm.weight         [hidden]
 *   blk.{l}.attn_norm.weight   [hidden]
 *   blk.{l}.ffn_norm.weight    [hidden]
 *   blk.{l}.attn_q.weight      [hidden, hidden]   + _scale [hidden]
 *   blk.{l}.attn_k.weight      [hidden, hidden]
 *   blk.{l}.attn_v.weight      [hidden, hidden]
 *   blk.{l}.attn_output.weight [hidden, hidden]
 *   blk.{l}.ffn_gate.weight    [hidden, ffn]
 *   blk.{l}.ffn_up.weight      [hidden, ffn]
 *   blk.{l}.ffn_down.weight    [ffn, hidden]
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

/* GGML tensor types we care about */
#define GGML_TYPE_F32  0u
#define GGML_TYPE_F16  1u
#define GGML_TYPE_I8   24u
#define GGML_TYPE_I2_S 36u  /* BitNet I2_S ternary blocks */

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
    uint64_t dims[4];      /* GGUF col-major: dims[0]=cols, dims[1]=rows */
    uint32_t type;         /* GGML_TYPE_* */
    uint64_t offset;       /* offset from tensor_data_start */
} TensorInfo;

static int find_tensor(TensorInfo* tis, int n, const char* name) {
    for(int i = 0; i < n; i++)
        if(strcmp(tis[i].name, name) == 0) return i;
    return -1;
}

/*============================================================================
 * F32 -> 2-bit packed ternary conversion
 *
 * Our kernel packing: 4 weights per byte
 *   00 = 0, 01 = +1, 10 = -1
 * Input: F32 array [N] (already ±1 or 0)
 * Output: uint8 array [N/4]
 *============================================================================*/

static void f32_to_ternary_packed(uint8_t* out, const float* in, uint32_t N) {
    uint32_t N4 = N / 4;
    for(uint32_t i = 0; i < N4; i++) {
        uint8_t byte = 0;
        for(int b = 0; b < 4; b++) {
            float v = in[i*4 + b];
            uint8_t code;
            if(v > 0.5f)       code = 1;  /* +1 */
            else if(v < -0.5f) code = 2;  /* -1 */
            else               code = 0;  /* 0  */
            byte |= (code << (b * 2));
        }
        out[i] = byte;
    }
    /* Handle remainder */
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
 * Load a specific F32 tensor from the GGUF data section.
 * Returns malloc'd float array, or NULL on error.
 * rows*cols == total number of floats.
 *============================================================================*/

static float* load_f32_tensor(FILE* f, long data_start,
                               TensorInfo* ti, uint32_t rows, uint32_t cols) {
    if(ti->type != GGML_TYPE_F32) {
        fprintf(stderr, "loader: tensor '%s' is not F32 (type=%u)\n",
                ti->name, ti->type);
        return NULL;
    }
    uint64_t n_elem = (uint64_t)rows * cols;
    float* data = malloc(n_elem * sizeof(float));
    if(!data) return NULL;

    long pos = data_start + (long)ti->offset;
    if(fseek(f, pos, SEEK_SET)) { free(data); return NULL; }
    if(fread(data, sizeof(float), n_elem, f) != n_elem) { free(data); return NULL; }
    return data;
}

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
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FF;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, sizeof(float));
    return out;
}

static float* load_as_f32(FILE* f, long data_start,
                           TensorInfo* ti, uint32_t n_elem) {
    float* out = malloc((size_t)n_elem * sizeof(float));
    if(!out) return NULL;

    long pos = data_start + (long)ti->offset;
    if(fseek(f, pos, SEEK_SET)) { free(out); return NULL; }

    if(ti->type == GGML_TYPE_F32) {
        if(fread(out, sizeof(float), n_elem, f) != n_elem) {
            free(out);
            return NULL;
        }
    } else if(ti->type == GGML_TYPE_F16) {
        uint16_t* tmp = malloc((size_t)n_elem * sizeof(uint16_t));
        if(!tmp) { free(out); return NULL; }
        if(fread(tmp, sizeof(uint16_t), n_elem, f) != n_elem) {
            free(tmp);
            free(out);
            return NULL;
        }
        for(uint32_t i = 0; i < n_elem; i++) out[i] = fp16_to_f32(tmp[i]);
        free(tmp);
    } else if(ti->type == GGML_TYPE_I8) {
        int8_t* tmp = malloc(n_elem);
        if(!tmp) { free(out); return NULL; }
        if(fread(tmp, 1, n_elem, f) != n_elem) {
            free(tmp);
            free(out);
            return NULL;
        }
        for(uint32_t i = 0; i < n_elem; i++) out[i] = (float)tmp[i];
        free(tmp);
    } else {
        fprintf(stderr, "loader: unsupported tensor type %u\n", ti->type);
        free(out);
        return NULL;
    }
    return out;
}

#define I2S_QK 64u

/* Real BitNet GGUF I2_S tensors are laid out as contiguous packed bytes:
 *   tensor_bytes = (rows * cols) / 4 + 32
 * There are no per-row scale blocks inside the payload for this model file.
 * Each 64-weight block occupies 16 bytes, with 4 groups of 16 weights packed
 * into bits [7:6], [5:4], [3:2], [1:0] of each byte.
 */
static void i2s_row_to_packed(uint8_t* out, const uint8_t* row, uint32_t cols) {
    uint32_t packed_row = (cols + 3) / 4;
    memset(out, 0, packed_row);

    uint32_t nblk = (cols + I2S_QK - 1) / I2S_QK;
    for(uint32_t bi = 0; bi < nblk; bi++) {
        const uint8_t* block = row + bi * (I2S_QK / 4);
        for(uint32_t j = 0; j < I2S_QK && bi * I2S_QK + j < cols; j++) {
            uint32_t k = bi * I2S_QK + j;
            uint32_t group_idx = j / 16;
            uint32_t group_pos = j % 16;
            uint8_t raw = (block[group_pos] >> (6 - 2*group_idx)) & 0x3;
            uint8_t code = (raw == 0) ? 2 : (raw == 1) ? 0 : 1;
            out[k / 4] |= (code << ((k % 4) * 2));
        }
    }
}

/*============================================================================
 * Load a ternary projection layer:
 *   weight tensor [cols, rows] F32  -> packed [rows, cols/4] uint8
 *   scale tensor  [rows]       F32  -> float array [rows]
 *
 * GGUF stores weights in column-major: dims[0]=K(cols), dims[1]=N(rows)
 * So the raw F32 data is laid out as: [row0_col0, row0_col1, ..., row1_col0, ...]
 * when dims are [K, N] — actually GGUF is row-by-row in the file for F32.
 *
 * We read it as a flat [rows * cols] F32 array and pack row by row.
 *============================================================================*/

static int load_ternary_proj(FILE* f, long data_start,
                              TensorInfo* tis, int n_tensors,
                              const char* wname, const char* sname,
                              uint32_t rows, uint32_t cols,
                              uint8_t** w_out, float** s_out) {
    int wi = find_tensor(tis, n_tensors, wname);
    int si = find_tensor(tis, n_tensors, sname);
    if(wi < 0) {
        fprintf(stderr, "loader: tensor '%s' not found\n", wname);
        return -1;
    }

    uint32_t packed_row = (cols + 3) / 4;
    uint8_t* packed = aligned_alloc(64, (size_t)rows * packed_row);
    float* scales = malloc(rows * sizeof(float));
    if(!packed || !scales) {
        free(packed);
        free(scales);
        return -1;
    }

    if(tis[wi].type == GGML_TYPE_F32) {
        float* raw = load_f32_tensor(f, data_start, &tis[wi], rows, cols);
        if(!raw) { free(packed); free(scales); return -1; }
        for(uint32_t r = 0; r < rows; r++)
            f32_to_ternary_packed(packed + r * packed_row, raw + r * cols, cols);
        free(raw);

        if(si >= 0) {
            long pos = data_start + (long)tis[si].offset;
            if(fseek(f, pos, SEEK_SET) || fread(scales, sizeof(float), rows, f) != rows) {
                for(uint32_t r = 0; r < rows; r++) scales[r] = 1.0f;
            }
        } else {
            for(uint32_t r = 0; r < rows; r++) scales[r] = 1.0f;
        }
    } else if(tis[wi].type == GGML_TYPE_I2_S) {
        size_t row_bytes = cols / 4;
        uint8_t* row = malloc(row_bytes);
        if(!row) { free(packed); free(scales); return -1; }
        for(uint32_t r = 0; r < rows; r++) {
            long pos = data_start + (long)tis[wi].offset + (long)r * (long)row_bytes;
            if(fseek(f, pos, SEEK_SET) || fread(row, 1, row_bytes, f) != row_bytes) {
                free(row);
                free(packed);
                free(scales);
                return -1;
            }
            i2s_row_to_packed(packed + r * packed_row, row, cols);
            scales[r] = 1.0f;
        }
        free(row);
    } else {
        fprintf(stderr, "loader: unsupported weight type %u for %s\n", tis[wi].type, wname);
        free(packed);
        free(scales);
        return -1;
    }

    *w_out = packed;
    *s_out = scales;
    return 0;
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
    float    rope_freq   = 10000.0f;
    char**   tokenizer_vocab = NULL;
    char**   tokenizer_merges = NULL;
    uint32_t num_merges = 0;
    uint32_t tokenizer_bos = 1, tokenizer_eos = 2, tokenizer_pad = 0;

    for(uint64_t i = 0; i < kv_count; i++) {
        char* key = rd_string(f);
        if(!key) goto fail;
        uint32_t vtype;
        if(rd_u32(f, &vtype)) { free(key); goto fail; }

        /* Extract keys we care about -- support llama.* and bitnet-b1.58.* */
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
#undef HAS_SUFFIX
        /* Tokenizer tokens */
        else if(strcmp(key, "tokenizer.ggml.tokens") == 0 && vtype == GGUF_TYPE_ARRAY) {
            uint32_t elem_type;
            uint64_t arr_len;
            rd_u32(f, &elem_type);
            rd_u64(f, &arr_len);
            vocab_size = (uint32_t)arr_len;
            tokenizer_vocab = calloc(vocab_size, sizeof(char*));
            if(!tokenizer_vocab) { free(key); goto fail; }
            for(uint64_t j = 0; j < arr_len; j++) tokenizer_vocab[j] = rd_string(f);
        } else if(strcmp(key, "tokenizer.ggml.merges") == 0 && vtype == GGUF_TYPE_ARRAY) {
            uint32_t elem_type;
            uint64_t arr_len;
            rd_u32(f, &elem_type);
            rd_u64(f, &arr_len);
            num_merges = (uint32_t)arr_len;
            tokenizer_merges = calloc(num_merges, sizeof(char*));
            if(!tokenizer_merges) { free(key); goto fail; }
            for(uint64_t j = 0; j < arr_len; j++) tokenizer_merges[j] = rd_string(f);
        } else if(strcmp(key, "tokenizer.ggml.bos_token_id") == 0 && vtype == GGUF_TYPE_UINT32) {
            rd_u32(f, &tokenizer_bos);
        } else if(strcmp(key, "tokenizer.ggml.eos_token_id") == 0 && vtype == GGUF_TYPE_UINT32) {
            rd_u32(f, &tokenizer_eos);
        } else if(strcmp(key, "tokenizer.ggml.padding_token_id") == 0 && vtype == GGUF_TYPE_UINT32) {
            rd_u32(f, &tokenizer_pad);
        } else {
            if(skip_value(f, vtype)) { free(key); goto fail; }
        }
        free(key);
    }

    if(!hidden_size || !n_layers || !n_heads || !ffn_size || !vocab_size) {
        fprintf(stderr, "loader: missing required metadata (hidden=%u layers=%u heads=%u ffn=%u vocab=%u)\n",
                hidden_size, n_layers, n_heads, ffn_size, vocab_size);
        goto fail;
    }
    if(!ctx_len) ctx_len = 4096;
    if(!n_kv_heads) n_kv_heads = n_heads;
    printf("loader: hidden=%u layers=%u Q-heads=%u KV-heads=%u ffn=%u vocab=%u ctx=%u rope=%.0f\n",
           hidden_size, n_layers, n_heads, n_kv_heads, ffn_size, vocab_size, ctx_len, rope_freq);

    /* ── Tensor infos ── */
    if(tensor_count > MAX_TENSORS) {
        fprintf(stderr, "loader: too many tensors (%llu)\n", (unsigned long long)tensor_count);
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
    m->use_i2s         = true;
    m->use_i2s         = true;
    m->tokenizer_vocab  = tokenizer_vocab;
    m->tokenizer_bos    = tokenizer_bos;
    m->tokenizer_eos    = tokenizer_eos;
    m->tokenizer_pad    = tokenizer_pad;
    m->tokenizer_merges = tokenizer_merges;
    m->num_merges       = num_merges;

    /* Non-quantized weights */
    m->embedding  = malloc((size_t)vocab_size * hidden_size * sizeof(float));
    m->lm_head    = malloc((size_t)vocab_size * hidden_size * sizeof(float));
    m->final_norm = malloc(hidden_size * sizeof(float));
    if(!m->embedding || !m->lm_head || !m->final_norm) { free(tis); goto fail; }

    /* Load embedding */
    {
        int idx = find_tensor(tis, (int)tensor_count, "token_embd.weight");
        if(idx < 0) { fprintf(stderr, "loader: missing token_embd.weight\n"); free(tis); goto fail; }
        /* GGUF shape [hidden, vocab] — read as vocab*hidden floats */
        float* raw = load_as_f32(f, data_start, &tis[idx], vocab_size * hidden_size);
        if(!raw) { free(tis); goto fail; }
        memcpy(m->embedding, raw, (size_t)vocab_size * hidden_size * sizeof(float));
        free(raw);
    }

    /* Load lm_head */
    {
        int idx = find_tensor(tis, (int)tensor_count, "output.weight");
        if(idx >= 0) {
            float* raw = load_as_f32(f, data_start, &tis[idx], vocab_size * hidden_size);
            if(raw) { memcpy(m->lm_head, raw, (size_t)vocab_size * hidden_size * sizeof(float)); free(raw); }
        } else {
            /* Use embedding weights as lm_head (weight tying) */
            memcpy(m->lm_head, m->embedding, (size_t)vocab_size * hidden_size * sizeof(float));
        }
    }

    /* Load final norm */
    {
        int idx = find_tensor(tis, (int)tensor_count, "output_norm.weight");
        if(idx >= 0) {
            float* raw = load_as_f32(f, data_start, &tis[idx], hidden_size);
            if(raw) { memcpy(m->final_norm, raw, hidden_size * sizeof(float)); free(raw); }
        } else {
            for(uint32_t i = 0; i < hidden_size; i++) m->final_norm[i] = 1.0f;
        }
    }

    /* Per-layer weights */
    m->layers = calloc(n_layers, sizeof(HSTernaryLayer));
    if(!m->layers) { free(tis); goto fail; }

    for(uint32_t l = 0; l < n_layers; l++) {
        HSTernaryLayer* lay = &m->layers[l];
        char wname[256], sname[256];

        /* Norms */
        lay->attn_norm     = malloc(hidden_size * sizeof(float));
        lay->attn_sub_norm = malloc(hidden_size * sizeof(float));
        lay->ffn_norm      = malloc(hidden_size * sizeof(float));
        lay->ffn_sub_norm  = malloc(ffn_size * sizeof(float));
        if(!lay->attn_norm || !lay->attn_sub_norm || !lay->ffn_norm || !lay->ffn_sub_norm) { free(tis); goto fail; }
        for(uint32_t i = 0; i < hidden_size; i++) { lay->attn_norm[i] = 1.0f; lay->attn_sub_norm[i] = 1.0f; lay->ffn_norm[i] = 1.0f; }
        for(uint32_t i = 0; i < ffn_size; i++) lay->ffn_sub_norm[i] = 1.0f;

        /* Load attn_norm */
        snprintf(wname, sizeof(wname), "blk.%u.attn_norm.weight", l);
        int ni = find_tensor(tis, (int)tensor_count, wname);
        if(ni >= 0) {
            float* raw = load_as_f32(f, data_start, &tis[ni], hidden_size);
            if(raw) { memcpy(lay->attn_norm, raw, hidden_size * sizeof(float)); free(raw); }
        }

        /* Load attn_sub_norm */
        snprintf(wname, sizeof(wname), "blk.%u.attn_sub_norm.weight", l);
        ni = find_tensor(tis, (int)tensor_count, wname);
        if(ni >= 0) {
            float* raw = load_as_f32(f, data_start, &tis[ni], hidden_size);
            if(raw) { memcpy(lay->attn_sub_norm, raw, hidden_size * sizeof(float)); free(raw); }
        }

        /* Load ffn_norm */
        snprintf(wname, sizeof(wname), "blk.%u.ffn_norm.weight", l);
        ni = find_tensor(tis, (int)tensor_count, wname);
        if(ni >= 0) {
            float* raw = load_as_f32(f, data_start, &tis[ni], hidden_size);
            if(raw) { memcpy(lay->ffn_norm, raw, hidden_size * sizeof(float)); free(raw); }
        }

        /* Load ffn_sub_norm */
        snprintf(wname, sizeof(wname), "blk.%u.ffn_sub_norm.weight", l);
        ni = find_tensor(tis, (int)tensor_count, wname);
        if(ni >= 0) {
            float* raw = load_as_f32(f, data_start, &tis[ni], ffn_size);
            if(raw) { memcpy(lay->ffn_sub_norm, raw, ffn_size * sizeof(float)); free(raw); }
        }

        /* Allocate scales */
        u32 kv_size = n_kv_heads * (hidden_size / n_heads);
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

/* Helper macro to load one ternary projection */
#define LOAD_PROJ(field_w, field_s, basename, rows, cols) do { \
    snprintf(wname, sizeof(wname), "blk.%u." basename ".weight", l); \
    snprintf(sname, sizeof(sname), "blk.%u." basename ".weight_scale", l); \
    if(load_ternary_proj(f, data_start, tis, (int)tensor_count, \
                         wname, sname, rows, cols, \
                         &lay->field_w, &lay->field_s)) { \
        fprintf(stderr, "loader: failed to load %s\n", wname); \
        free(tis); goto fail; \
    } \
} while(0)

        LOAD_PROJ(q_proj,    q_scale,    "attn_q",       hidden_size, hidden_size);
        LOAD_PROJ(k_proj,    k_scale,    "attn_k",       kv_size, hidden_size);
        LOAD_PROJ(v_proj,    v_scale,    "attn_v",       kv_size, hidden_size);
        LOAD_PROJ(o_proj,    o_scale,    "attn_output",  hidden_size, hidden_size);
        LOAD_PROJ(gate_proj, gate_scale, "ffn_gate",     ffn_size,    hidden_size);
        LOAD_PROJ(up_proj,   up_scale,   "ffn_up",       ffn_size,    hidden_size);
        LOAD_PROJ(down_proj, down_scale, "ffn_down",     hidden_size, ffn_size);

#undef LOAD_PROJ
    }

    /* Scratch buffers */
    uint32_t max_dim = (hidden_size > ffn_size) ? hidden_size : ffn_size;
    m->quant_buf  = aligned_alloc(64, max_dim * sizeof(int8_t));
    m->gemm_buf   = aligned_alloc(64, max_dim * sizeof(int32_t));
    m->hidden_buf = malloc(hidden_size * sizeof(float));
    m->ffn_buf    = malloc(ffn_size * sizeof(float));
    m->q_buf      = malloc(hidden_size * sizeof(float));
    m->k_buf      = malloc(hidden_size * sizeof(float));
    m->v_buf      = malloc(hidden_size * sizeof(float));
    m->attn_buf   = calloc(hidden_size, sizeof(float));
    m->gate_buf   = malloc(ffn_size * sizeof(float));
    m->up_buf     = malloc(ffn_size * sizeof(float));
    m->score_buf  = malloc(ctx_len * sizeof(float));
    m->ffn_qbuf   = aligned_alloc(64, ffn_size * sizeof(int8_t));

    if(!m->quant_buf || !m->gemm_buf || !m->hidden_buf || !m->ffn_buf ||
       !m->q_buf || !m->k_buf || !m->v_buf || !m->attn_buf ||
       !m->gate_buf || !m->up_buf || !m->score_buf || !m->ffn_qbuf) {
        free(tis); goto fail;
    }
    /* BitNet 2B-4T GGUF: norm tensors stored as F16 in F32-typed slots.
     * attn_norm slot: F16 pair (attn_norm + attn_sub_norm)
     * ffn_sub_norm slot: F16 (first half)
     * ffn_norm, output_norm: also read as F16 */
    if (m->use_i2s) {
        for (u32 l = 0; l < n_layers; l++) {
            HSTernaryLayer *lay = &m->layers[l];
            char wname[256];
            int ni;

            /* attn_norm slot -> attn_norm + attn_sub_norm as F16 pair */
            snprintf(wname, sizeof(wname), "blk.%u.attn_norm.weight", l);
            ni = find_tensor(tis, (int)tensor_count, wname);
            if (ni >= 0) {
                uint16_t *buf = malloc(hidden_size * 2 * sizeof(uint16_t));
                if (buf) {
                    fseek(f, data_start + (long)tis[ni].offset, SEEK_SET);
                    if (fread(buf, 2, hidden_size * 2, f) == hidden_size * 2) {
                        for (u32 i = 0; i < hidden_size; i++)
                            lay->attn_norm[i] = fp16_to_f32(buf[i]);
                        for (u32 i = 0; i < hidden_size; i++)
                            lay->attn_sub_norm[i] = fp16_to_f32(buf[hidden_size + i]);
                    }
                    free(buf);
                }
            }

            /* ffn_sub_norm slot -> ffn_sub_norm as F16 (first half) */
            snprintf(wname, sizeof(wname), "blk.%u.ffn_sub_norm.weight", l);
            ni = find_tensor(tis, (int)tensor_count, wname);
            if (ni >= 0) {
                uint16_t *buf = malloc(ffn_size * sizeof(uint16_t));
                if (buf) {
                    fseek(f, data_start + (long)tis[ni].offset, SEEK_SET);
                    if (fread(buf, 2, ffn_size, f) == (size_t)ffn_size) {
                        for (u32 i = 0; i < ffn_size; i++)
                            lay->ffn_sub_norm[i] = fp16_to_f32(buf[i]);
                    }
                    free(buf);
                }
            }

            /* ffn_norm slot -> try as F16 */
            snprintf(wname, sizeof(wname), "blk.%u.ffn_norm.weight", l);
            ni = find_tensor(tis, (int)tensor_count, wname);
            if (ni >= 0) {
                uint16_t *buf = malloc(hidden_size * sizeof(uint16_t));
                if (buf) {
                    fseek(f, data_start + (long)tis[ni].offset, SEEK_SET);
                    if (fread(buf, 2, hidden_size, f) == hidden_size) {
                        int sane = 1;
                        for (u32 i = 0; i < hidden_size && sane; i++) {
                            float v = fp16_to_f32(buf[i]);
                            if (v != v || v > 100.0f || v < -100.0f) sane = 0;
                        }
                        if (sane) {
                            for (u32 i = 0; i < hidden_size; i++)
                                lay->ffn_norm[i] = fp16_to_f32(buf[i]);
                        }
                    }
                    free(buf);
                }
            }
        }

        /* output_norm -> try as F16 */
        {
            int ni = find_tensor(tis, (int)tensor_count, "output_norm.weight");
            if (ni >= 0) {
                uint16_t *buf = malloc(hidden_size * sizeof(uint16_t));
                if (buf) {
                    fseek(f, data_start + (long)tis[ni].offset, SEEK_SET);
                    if (fread(buf, 2, hidden_size, f) == hidden_size) {
                        int sane = 1;
                        for (u32 i = 0; i < hidden_size && sane; i++) {
                            float v = fp16_to_f32(buf[i]);
                            if (v != v || v > 100.0f || v < -100.0f) sane = 0;
                        }
                        if (sane) {
                            for (u32 i = 0; i < hidden_size; i++)
                                m->final_norm[i] = fp16_to_f32(buf[i]);
                        }
                    }
                    free(buf);
                }
            }
        }
    }

    free(tis);
    fclose(f);
    m->loaded = true;
    printf("loader: model loaded successfully\n");
    return 0;

fail:
    fclose(f);
    hs_mlt_free(m);
    return -1;
}
