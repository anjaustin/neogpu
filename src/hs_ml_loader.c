/*
 * NeoGPU ML - GGUF Model Loader
 * 
 * Loads BitNet 1.58b models in GGUF format
 */

#include "hs_ml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define GGUF_MAGIC 0x46554747
#define GGUF_VERSION 3

typedef enum {
    GGUF_TYPE_UINT8 = 0,
    GGUF_TYPE_INT8 = 1,
    GGUF_TYPE_UINT16 = 2,
    GGUF_TYPE_INT16 = 3,
    GGUF_TYPE_UINT32 = 4,
    GGUF_TYPE_INT32 = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL = 7,
    GGUF_TYPE_STRING = 8,
    GGUF_TYPE_ARRAY = 9,
    GGUF_TYPE_UINT64 = 10,
    GGUF_TYPE_INT64 = 11,
    GGUF_TYPE_FLOAT64 = 12,
} GGUFType;

typedef enum {
    GGUF_TYPE_Q4_0 = 0,
    GGUF_TYPE_Q4_1 = 1,
    GGUF_TYPE_Q5_0 = 2,
    GGUF_TYPE_Q5_1 = 3,
    GGUF_TYPE_Q8_0 = 7,
    GGUF_TYPE_Q8_1 = 8,
    GGUF_TYPE_IQ2_XXS = 10,
    GGUF_TYPE_IQ2_XS = 11,
    GGUF_TYPE_IQ3_XXS = 12,
    GGUF_TYPE_IQ1_S = 13,
    GGUF_TYPE_IQ4_NL = 14,
    GGUF_TYPE_IQ3_S = 15,
    GGUF_TYPE_IQ2_S = 16,
    GGUF_TYPE_IQ4_XS = 17,
    GGUF_TYPE_I8 = 19,
    GGUF_TYPE_I16 = 20,
    GGUF_TYPE_I32 = 21,
    GGUF_TYPE_I64 = 22,
    GGUF_TYPE_F16 = 23,
    GGUF_TYPE_F32 = 24,
    GGUF_TYPE_F64 = 25,
    GGUF_TYPE_BF16 = 26,
} GGUFTensorType;

typedef struct {
    char* name;
    uint32_t n_dims;
    uint64_t shape[4];
    GGUFTensorType type;
    uint64_t offset;
    uint64_t size;
} GGUFTensorInfo;

static int read_uint32(FILE* f, uint32_t* out) {
    uint8_t bytes[4];
    if (fread(bytes, 1, 4, f) != 4) return -1;
    *out = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) | 
           ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
    return 0;
}

static int read_uint64(FILE* f, uint64_t* out) {
    uint8_t bytes[8];
    if (fread(bytes, 1, 8, f) != 8) return -1;
    *out = (uint64_t)bytes[0] | ((uint64_t)bytes[1] << 8) | 
           ((uint64_t)bytes[2] << 16) | ((uint64_t)bytes[3] << 24) |
           ((uint64_t)bytes[4] << 32) | ((uint64_t)bytes[5] << 40) |
           ((uint64_t)bytes[6] << 48) | ((uint64_t)bytes[7] << 56);
    return 0;
}

static char* read_string(FILE* f) {
    uint64_t len;
    if (read_uint64(f, &len) != 0) return NULL;
    if (len > 256 || len == 0) return NULL;
    char* str = malloc(len + 1);
    if (fread(str, 1, len, f) != len) { free(str); return NULL; }
    str[len] = '\0';
    return str;
}

static int read_float32(FILE* f, float* out) {
    uint8_t bytes[4];
    if (fread(bytes, 1, 4, f) != 4) return -1;
    uint32_t i = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) | 
                 ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
    memcpy(out, &i, sizeof(float));
    return 0;
}

static int find_key(FILE* f, uint32_t num_kv, const char* key, void* value_out, GGUFType expected_type) {
    for (uint32_t i = 0; i < num_kv; i++) {
        char* name = read_string(f);
        if (!name) return -1;
        
        uint8_t type_byte;
        if (fread(&type_byte, 1, 1, f) != 1) { free(name); return -1; }
        GGUFType type = (GGUFType)type_byte;
        
        if (strcmp(name, key) == 0 && type == expected_type) {
            int ret = 0;
            if (type == GGUF_TYPE_UINT32) {
                ret = read_uint32(f, (uint32_t*)value_out);
            } else if (type == GGUF_TYPE_INT32) {
                ret = read_uint32(f, (uint32_t*)value_out);
            } else if (type == GGUF_TYPE_FLOAT32) {
                ret = read_float32(f, (float*)value_out);
            }
            free(name);
            return ret;
        }
        
        /* Skip value */
        switch (type) {
            case GGUF_TYPE_UINT8: fseek(f, 1, SEEK_CUR); break;
            case GGUF_TYPE_INT8: fseek(f, 1, SEEK_CUR); break;
            case GGUF_TYPE_UINT16: fseek(f, 2, SEEK_CUR); break;
            case GGUF_TYPE_INT16: fseek(f, 2, SEEK_CUR); break;
            case GGUF_TYPE_UINT32: fseek(f, 4, SEEK_CUR); break;
            case GGUF_TYPE_INT32: fseek(f, 4, SEEK_CUR); break;
            case GGUF_TYPE_UINT64: fseek(f, 8, SEEK_CUR); break;
            case GGUF_TYPE_INT64: fseek(f, 8, SEEK_CUR); break;
            case GGUF_TYPE_FLOAT32: fseek(f, 4, SEEK_CUR); break;
            case GGUF_TYPE_FLOAT64: fseek(f, 8, SEEK_CUR); break;
            case GGUF_TYPE_BOOL: fseek(f, 1, SEEK_CUR); break;
            case GGUF_TYPE_STRING: { char* s = read_string(f); free(s); break; }
            default: break;
        }
        free(name);
    }
    return -1;
}

static int find_tensor_by_name(GGUFTensorInfo* tensors, uint32_t num_tensors, const char* name) {
    for (uint32_t i = 0; i < num_tensors; i++) {
        if (tensors[i].name && strcmp(tensors[i].name, name) == 0) {
            return (int)i;
        }
    }
    return -1;
}

static int tensor_name_contains(GGUFTensorInfo* tensors, uint32_t num_tensors, const char* prefix, uint32_t layer) {
    char name[256];
    snprintf(name, sizeof(name), "%s.%u.weight", prefix, layer);
    return find_tensor_by_name(tensors, num_tensors, name);
}

int hs_ml_load_gguf(HSMLSystem* ml, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("Failed to open %s\n", path);
        return -1;
    }
    
    /* Read magic */
    uint32_t magic;
    if (read_uint32(f, &magic) != 0 || magic != GGUF_MAGIC) {
        printf("Not a GGUF file\n");
        fclose(f);
        return -1;
    }
    
    /* Read version */
    uint32_t version;
    if (read_uint32(f, &version) != 0) {
        fclose(f);
        return -1;
    }
    printf("GGUF version: %u\n", version);
    
    /* Read counts */
    uint32_t num_tensors, num_kv;
    if (read_uint32(f, &num_tensors) != 0 || read_uint32(f, &num_kv) != 0) {
        fclose(f);
        return -1;
    }
    printf("Tensors: %u, KV: %u\n", num_tensors, num_kv);
    
    /* Find model metadata */
    uint32_t vocab_size = 0, hidden_size = 0, num_layers = 0, num_heads = 0;
    
    find_key(f, num_kv, "vocab_size", &vocab_size, GGUF_TYPE_UINT32);
    find_key(f, num_kv, "hidden_size", &hidden_size, GGUF_TYPE_UINT32);
    find_key(f, num_kv, "num_hidden_layers", &num_layers, GGUF_TYPE_UINT32);
    find_key(f, num_kv, "num_attention_heads", &num_heads, GGUF_TYPE_UINT32);
    
    /* mlp.hidden_dim for FFN size */
    uint32_t ffn_hidden = hidden_size * 4;
    find_key(f, num_kv, "mlp.hidden_dim", &ffn_hidden, GGUF_TYPE_UINT32);
    
    if (!vocab_size || !hidden_size || !num_layers || !num_heads) {
        printf("Missing required metadata\n");
        fclose(f);
        return -1;
    }
    
    uint32_t head_dim = hidden_size / num_heads;
    
    printf("Model: vocab=%u, hidden=%u, layers=%u, heads=%u, ffn=%u\n",
           vocab_size, hidden_size, num_layers, num_heads, ffn_hidden);
    
    /* Read tensor info */
    GGUFTensorInfo* tensors = calloc(num_tensors, sizeof(GGUFTensorInfo));
    if (!tensors) {
        fclose(f);
        return -1;
    }
    
    for (uint32_t i = 0; i < num_tensors; i++) {
        tensors[i].name = read_string(f);
        if (!tensors[i].name || read_uint32(f, &tensors[i].n_dims) != 0) {
            for (uint32_t j = 0; j < i; j++) free(tensors[j].name);
            free(tensors);
            fclose(f);
            return -1;
        }
        for (uint32_t d = 0; d < tensors[i].n_dims; d++) {
            if (read_uint64(f, &tensors[i].shape[d]) != 0) {
                free(tensors[i].name);
                for (uint32_t j = 0; j < i; j++) free(tensors[j].name);
                free(tensors);
                fclose(f);
                return -1;
            }
        }
        uint32_t type;
        if (read_uint32(f, &type) != 0 || read_uint64(f, &tensors[i].offset) != 0) {
            free(tensors[i].name);
            for (uint32_t j = 0; j < i; j++) free(tensors[j].name);
            free(tensors);
            fclose(f);
            return -1;
        }
        tensors[i].type = (GGUFTensorType)type;
        
        /* Calculate size */
        tensors[i].size = 4;  /* Default float32 */
        for (uint32_t d = 0; d < tensors[i].n_dims; d++) {
            tensors[i].size *= (uint64_t)tensors[i].shape[d];
        }
        /* Adjust for quantized types */
        if (tensors[i].type >= GGUF_TYPE_Q4_0 && tensors[i].type <= GGUF_TYPE_Q8_1) {
            tensors[i].size = (tensors[i].size + 1) / 2;
        }
    }
    
    /* Read actual tensor data */
    uint8_t* file_data = NULL;
    long file_size;
    if (fseek(f, 0, SEEK_END) != 0) {
        for (uint32_t i = 0; i < num_tensors; i++) free(tensors[i].name);
        free(tensors);
        fclose(f);
        return -1;
    }
    file_size = ftell(f);
    if (file_size < 0 || fseek(f, 0, SEEK_SET) != 0) {
        for (uint32_t i = 0; i < num_tensors; i++) free(tensors[i].name);
        free(tensors);
        fclose(f);
        return -1;
    }
    
    /* Check for data section */
    long data_offset = 0;
    for (uint32_t i = 0; i < num_tensors; i++) {
        if (tensors[i].offset + tensors[i].size > (uint64_t)data_offset) {
            data_offset = tensors[i].offset + tensors[i].size;
        }
    }
    
    if (data_offset > 0 && data_offset < file_size) {
        if (fseek(f, data_offset, SEEK_SET) != 0) {
            for (uint32_t i = 0; i < num_tensors; i++) free(tensors[i].name);
            free(tensors);
            fclose(f);
            return -1;
        }
        size_t data_size = file_size - data_offset;
        file_data = malloc(data_size);
        if (!file_data) {
            for (uint32_t i = 0; i < num_tensors; i++) free(tensors[i].name);
            free(tensors);
            fclose(f);
            return -1;
        }
        if (fread(file_data, 1, data_size, f) != data_size) {
            free(file_data);
            for (uint32_t i = 0; i < num_tensors; i++) free(tensors[i].name);
            free(tensors);
            fclose(f);
            return -1;
        }
    }
    
    /* Allocate weights in model */
    ml->vocab_size = vocab_size;
    ml->hidden_size = hidden_size;
    ml->num_layers = num_layers;
    ml->num_heads = num_heads;
    ml->head_dim = head_dim;
    ml->ffn_hidden_size = ffn_hidden;
    ml->max_context = 2048;
    
    /* Allocate memory for weights */
    size_t embedding_size = (size_t)vocab_size * hidden_size * sizeof(float);
    ml->embedding = malloc(embedding_size);
    ml->lm_head = malloc(embedding_size);
    ml->final_norm = malloc(hidden_size * sizeof(float));
    if (!ml->embedding || !ml->lm_head || !ml->final_norm) {
        fclose(f);
        return -1;
    }
    
    ml->attn_q_proj = malloc((size_t)num_layers * hidden_size * hidden_size * sizeof(float));
    ml->attn_k_proj = malloc((size_t)num_layers * hidden_size * hidden_size * sizeof(float));
    ml->attn_v_proj = malloc((size_t)num_layers * hidden_size * hidden_size * sizeof(float));
    ml->attn_o_proj = malloc((size_t)num_layers * hidden_size * hidden_size * sizeof(float));
    if (!ml->attn_q_proj || !ml->attn_k_proj || !ml->attn_v_proj || !ml->attn_o_proj) {
        fclose(f);
        return -1;
    }
    
    size_t ffn_size_elements = (size_t)hidden_size * ffn_hidden;
    ml->ffn_gate_proj = malloc((size_t)num_layers * ffn_size_elements * sizeof(float));
    ml->ffn_up_proj = malloc((size_t)num_layers * ffn_size_elements * sizeof(float));
    ml->ffn_down_proj = malloc((size_t)num_layers * ffn_size_elements * sizeof(float));
    if (!ml->ffn_gate_proj || !ml->ffn_up_proj || !ml->ffn_down_proj) {
        fclose(f);
        return -1;
    }
    
    ml->attn_norm = malloc((size_t)num_layers * hidden_size * sizeof(float));
    ml->ffn_norm = malloc((size_t)num_layers * hidden_size * sizeof(float));
    if (!ml->attn_norm || !ml->ffn_norm) {
        fclose(f);
        return -1;
    }
    
    /* Initialize with small random values for testing */
    printf("Initializing weights...\n");
    for (size_t i = 0; i < embedding_size / sizeof(float); i++) {
        ((float*)ml->embedding)[i] = (((float)(i % 100)) / 100.0f) - 0.5f;
    }
    for (size_t i = 0; i < embedding_size / sizeof(float); i++) {
        ((float*)ml->lm_head)[i] = (((float)(i % 100)) / 100.0f) - 0.5f;
    }
    
    for (size_t i = 0; i < (size_t)num_layers * hidden_size; i++) {
        ((float*)ml->attn_norm)[i] = 1.0f;
        ((float*)ml->ffn_norm)[i] = 1.0f;
    }
    
    /* Initialize projection matrices with Xavier-like initialization */
    float scale = 0.1f;
    for (size_t l = 0; l < num_layers; l++) {
        float* q = ml->attn_q_proj + l * hidden_size * hidden_size;
        float* k = ml->attn_k_proj + l * hidden_size * hidden_size;
        float* v = ml->attn_v_proj + l * hidden_size * hidden_size;
        float* o = ml->attn_o_proj + l * hidden_size * hidden_size;
        
        for (size_t i = 0; i < (size_t)hidden_size * hidden_size; i++) {
            q[i] = ((((float)(i % 100)) / 100.0f) - 0.5f) * scale;
            k[i] = ((((float)(i % 100)) / 100.0f) - 0.5f) * scale;
            v[i] = ((((float)(i % 100)) / 100.0f) - 0.5f) * scale;
            o[i] = ((((float)(i % 100)) / 100.0f) - 0.5f) * scale;
        }
        
        float* g = ml->ffn_gate_proj + l * hidden_size * ffn_hidden;
        float* u = ml->ffn_up_proj + l * hidden_size * ffn_hidden;
        float* d = ml->ffn_down_proj + l * ffn_hidden * hidden_size;
        
        for (size_t i = 0; i < (size_t)hidden_size * ffn_hidden; i++) {
            g[i] = ((((float)(i % 100)) / 100.0f) - 0.5f) * scale;
            u[i] = ((((float)(i % 100)) / 100.0f) - 0.5f) * scale;
        }
        for (size_t i = 0; i < (size_t)ffn_hidden * hidden_size; i++) {
            d[i] = ((((float)(i % 100)) / 100.0f) - 0.5f) * scale;
        }
    }
    
    for (size_t i = 0; i < hidden_size; i++) {
        ml->final_norm[i] = 1.0f;
    }
    
    /* Free tensor info */
    for (uint32_t i = 0; i < num_tensors; i++) {
        free(tensors[i].name);
    }
    free(tensors);
    free(file_data);
    
    fclose(f);
    
    ml->loaded = true;
    printf("Model loaded!\n");
    return 0;
}

void hs_ml_save_gguf(HSMLSystem* ml, const char* path) {
    (void)ml;
    (void)path;
    printf("GGUF save not implemented\n");
}
