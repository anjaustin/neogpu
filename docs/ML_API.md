# NeoGPU ML API

## Overview

ML module for BitNet 1.58-bit LLM inference on ARM NEON.

## Compilation

```bash
# With DOTPROD (ARMv8.2+)
gcc -O3 -march=armv8-a+simd+dotprod -mtune=cortex-a72 -Iinclude -c src/hs_ml.c

# Without DOTPROD (ARMv8.0, Pi4)
gcc -O3 -march=armv8-a -mtune=cortex-a72 -Iinclude -c src/hs_ml.c
```

## Core Types

### HSMLSystem

```c
typedef struct {
    void* weights;           // Ternary weights (packed 2-bit)
    float* embedding;        // Token embeddings [vocab, hidden]
    float* final_norm;       // Output layer norm
    float* lm_head;          // LM head [vocab, hidden]
    float* kv_cache;        // [layers][2][heads][seq][head_dim]
    u32 kv_cache_seq;       // Current sequence length
    
    // Config
    u32 vocab_size;
    u32 hidden_size;
    u32 num_layers;
    u32 num_heads;
    u32 head_dim;
    u32 max_context;
    
    // Tokenizer
    u32* tokenizer_table;
    char** tokenizer_vocab;
    
    // Workspace
    void* work_buffer;
    size_t work_size;
    
    bool loaded;
} HSMLSystem;
```

### Constants

```c
#define HS_ML_QK_I2_S     64    // 64 weights per ternary block
#define HS_ML_MAX_LAYERS  32    // Max transformer layers
#define HS_ML_MAX_VOCAB   128256
```

## Functions

### hs_ml_init

```c
void hs_ml_init(HSMLSystem* ml);
```

Initialize ML system. Must be called before use.

### hs_ml_free

```c
void hs_ml_free(HSMLSystem* ml);
```

Free all ML resources.

### hs_ml_gemm_int8

```c
void hs_ml_gemm_int8(int32_t* C, 
                     const int8_t* A, 
                     const u8* B_ternary,
                     const float* B_scale,
                     u32 M, u32 N, u32 K);
```

Ternary GEMM: C = A × B

| Param | Description |
|-------|-------------|
| C | Output [M × N], INT32 accumulator |
| A | Activations [M × K], INT8 |
| B | Weights [K × N], packed 2-bit ternary |
| B_scale | Scale factors [N] |
| M | Output rows |
| N | Output cols |
| K | Hidden dimension |

**Runtime:** Automatically selects DOTPROD or fallback based on hardware.

### hs_ml_quantize_ternary

```c
void hs_ml_quantize_ternary(const float* input,
                            u8* output,
                            float* scales,
                            u32 N);
```

Quantize FP32 weights to ternary (-1, 0, +1).

| Param | Description |
|-------|-------------|
| input | FP32 weights [N] |
| output | Packed ternary [N/4 bytes] |
| scales | Scale factors [N/64] |
| N | Number of weights |

### hs_ml_dequantize

```c
void hs_ml_dequantize(const int32_t* C,
                      const float* scales,
                      float* output,
                      u32 M, u32 N);
```

Convert INT32 accumulator to FP32.

## Data Format

### Ternary Weight Packing

```
64 weights → 32 bytes (2 bits each)
4 weights per byte:
  bits [1:0] = weight 0
  bits [3:2] = weight 1
  bits [5:4] = weight 2
  bits [7:6] = weight 3

Encoding:
  00 = 0
  01 = +1
  10 = -1
  11 = reserved
```

### Scale Factors

Each 64-weight block has one scale factor (FP32). During inference:
```
output = sum(activation * weight) * scale
```

## Performance

| Hardware | DOTPROD | GEMM Speed | Tokens/sec |
|----------|---------|------------|------------|
| Pi4 (A72) | No | ~2-5 GFLOPS | ~1-3 |
| ARMv8.2+ | Yes | ~10-20 GFLOPS | ~5-10 |

## Testing

```bash
# Build test
gcc -O0 -march=armv8-a -Iinclude -c tests/test_ml_gemm.c
gcc -O0 -march=armv8-a -Iinclude -c src/hs_ml.c
gcc test_ml_gemm.o hs_ml.o -o test_ml_gemm
./test_ml_gemm
```

Tests verify:
- Basic GEMM (+1 weights)
- Negative weights (-1)
- Mixed weights
- Random data (100 iterations)
- Quantization
- Dequantization

## Integration

1. Initialize: `hs_ml_init(&ml)`
2. Load model weights (quantized to ternary)
3. For each token:
   - Compute embedding
   - Run transformer layers (GEMM for Q, K, V, FFN)
   - Apply attention with KV cache
   - Sample next token
4. Cleanup: `hs_ml_free(&ml)`
