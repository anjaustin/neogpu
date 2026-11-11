# LMM: BitNet 1.58b Inference on Pi4 (SYNTHESIZE)

## P0 Spec: BitNet Inference Engine

### 1. INT8 GEMM (hs_ml.c)
```c
// NEON-optimized INT8 GEMM
// A: int8 [M × K], B: int8 [K × N], C: int32 [M × N]
void hs_ml_gemm_int8(int32_t* C, const int8_t* A, const int8_t* B,
                    u32 M, u32 N, u32 K);

// Accumulate to float
void hs_ml_gemm_accum(float* C, const int8_t* A, const int8_t* B,
                      u32 M, u32 N, u32 K, float scale);
```

### 2. BitNet Layers (hs_bitnet.c)
```c
typedef struct {
    u32 in_features;
    u32 out_features;
    int8_t* weight;  // quantized
    float* scale;     // dequantization scale
} BitLinear;

typedef struct {
    BitLinear qkv;
    BitLinear o_proj;
    // KV cache managed separately
} BitAttention;

typedef struct {
    BitLinear gate_proj;
    BitLinear up_proj;
    BitLinear down_proj;
} BitMLP;
```

### 3. Model Structure
```c
typedef struct {
    u32 vocab_size;
    u32 hidden_size;
    u32 num_layers;
    u32 num_heads;
    u32 head_dim;
    BitEmbedding embedding;
    BitAttention* layers;
    BitLinear final_norm;
    BitLinear lm_head;
    int8_t* kv_cache;  // [layers × 2 × seq_len × hidden]
} BitNetModel;
```

### 4. Inference Loop
```c
// Token by token generation
u32 token = hs_bitnet_decode(model, prompt_tokens, num_tokens);
while (!eos && token_count < max_tokens) {
    token = hs_bitnet_step(model, token);
    output_tokens[num_tokens++] = token;
}
```

### 5. CLI Tool
```bash
# Start server
./neogpu_bitnet --model bitnet-1.58b.gguf --port 8765

# Send prompt via IPC
./neogpu_tool --host localhost --port 8765 generate "Hello world"

# Stream output
./neogpu_bitnet --model bitnet-1.58b.gguf --interactive
```

## Acceptance Criteria
- [ ] INT8 GEMM compiles and works on Pi4
- [ ] Can load GGUF model file
- [ ] First token generates
- [ ] Tokens stream at >1/sec
- [ ] Graceful OOM handling
- [ ] IPC control works

## What's NOT P0
- INT4 quantization (just INT8)
- Paged KV cache (fixed context)
- Tokenizer optimization
- Sampling strategies (greedy only)
- GGUF parser (use llama.cpp or minimal)

## This is a BIG Step-Charge
This transforms NeoGPU from a graphics engine to a local LLM inference engine. The message-passing architecture + NEON is a good fit, but this is fundamentally different from graphics.

**Is this still "NeoGPU"? Or something new?**
