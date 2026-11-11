# LMM: NeoGPU + BitNet Integration (SYNTHESIZE)

## P0 Spec: BitNet Inference on NeoGPU

### Phase 1: INT8 GEMM Kernel

#### 1.1 hs_ml.h - ML Types
```c
// Tensor: multi-dimensional array
typedef struct {
    u32 shape[4];      // N, C, H, W
    u32 stride[4];
    void* data;
} HSTensor;

// ML System
typedef struct {
    void* weights;         // Ternary weights
    void* kv_cache;        // [layers][2][heads][seq][head_dim]
    u32* tokenizer;       // BPE tokenizer
    u32 vocab_size;
    u32 hidden_size;
    u32 num_layers;
    u32 max_context;
    void* workspace;       // Scratch buffer
} HSMLSystem;
```

#### 1.2 hs_ml.c - GEMM Implementation
```c
// Ternary GEMM: C = A × B where B is ternary (-1, 0, +1)
// A: INT8 activations [M × K]
// B: Ternary weights [K × N]  
// C: INT32 accumulator [M × N]
void hs_ml_gemm_int8(int32_t* C, 
                     const int8_t* A, 
                     const u8* B_ternary,  // Packed 2-bit weights
                     const float* B_scale, // Per-block scales
                     u32 M, u32 N, u32 K);

// With NEON DOTPROD (Pi4)
void hs_ml_gemm_int8_neon_dotprod(int32_t* C, 
                                   const int8_t* A,
                                   const u8* B_ternary,
                                   const float* B_scale,
                                   u32 M, u32 N, u32 K);
```

### Phase 2: Model Loading

#### 2.1 GGUF Parser (minimal)
```c
// Load only what we need:
// - Model architecture
// - Token embeddings
// - Layer weights (quantized)
// - Tokenizer
typedef struct {
    u32 magic;           // "GGUF"
    u32 version;
    // ... minimal header
} HSGGUFHeader;

bool hs_ml_load_gguf(HSMLSystem* ml, const char* path);
```

### Phase 3: Inference

#### 3.1 Forward Pass
```c
// Single forward pass: input_ids → logits
// Input: [batch, seq_len]
// Output: [batch, seq_len, vocab_size]
bool hs_ml_forward(HSMLSystem* ml, 
                   const u32* input_ids,
                   u32 input_len,
                   float* logits_out);
```

#### 3.2 Autoregressive Generation
```c
// Generate tokens one by one
// Uses KV cache for efficiency
u32 hs_ml_generate(HSMLSystem* ml,
                   const u32* prompt,
                   u32 prompt_len,
                   u32 max_tokens,
                   float* output_logits);
```

### Phase 4: Integration with NeoGPU

#### 4.1 Message Types
```c
// Add to OpCode enum
OP_ML_LOAD,      // Load model
OP_ML_FORWARD,   // Forward pass
OP_ML_GENERATE,   // Generate tokens
OP_ML_UNLOAD,    // Free memory
```

#### 4.2 IPC Commands
```bash
# Load model
./neogpu_tool --sock /tmp/neogpu.sock ml-load model.gguf

# Generate
./neogpu_tool --sock /tmp/neogpu.sock ml-generate "Hello world" --max-tokens 100

# Query status
./neogpu_tool --sock /tmp/neogpu.sock ml-stats
```

## Acceptance Criteria

### Phase 1 (GEMM)
- [ ] INT8 GEMM compiles on ARM
- [ ] Correctness test (compare to reference)
- [ ] Performance: >1 GFLOPS

### Phase 2 (Model)
- [ ] Can load minimal GGUF
- [ ] Memory usage <2GB

### Phase 3 (Inference)
- [ ] First token generates
- [ ] Token speed >1 token/sec
- [ ] Correct output (valid text)

### Phase 4 (Integration)
- [ ] IPC control works
- [ ] No crash under load
- [ ] Graceful error handling

## Implementation Priority

1. **hs_ml.h/c** - Core ML types and GEMM
2. **Model loader** - Load GGUF or custom format
3. **Simple inference** - Forward pass only
4. **Tokenizer** - BPE decode
5. **Generation loop** - Autoregressive
6. **IPC integration** - Control interface

## Files to Create

```
include/hs_ml.h      - ML types and declarations
src/hs_ml.c        - GEMM implementations
src/hs_ml_loader.c - Model loading
tools/neogpu_ml.c  - ML tool commands
```
