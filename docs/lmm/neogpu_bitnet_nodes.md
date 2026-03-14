# LMM: NeoGPU + BitNet Integration (NODES)

## Architecture Analysis

### What We Have (NeoGPU)

| Component | Status | ML Use |
|-----------|--------|--------|
| NEON math | Working | Extend to INT8 |
| Message queue | 8.5M/sec | ML inference requests |
| QoS channels | RT/RENDER/TELEM | Priority for ML |
| IPC | TCP + Unix | Remote control |
| Toolbus | Result correlation | Async results |
| Payloads | 96 bytes | Tensor storage |

### What We Need (BitNet)

| Component | Description |
|-----------|-------------|
| INT8 GEMM | Ternary weight × INT8 activation |
| Model loader | Load GGUF files |
| Tokenizer | BPE tokenization |
| KV cache | Attention cache |
| Inference loop | Autoregressive generation |

## Design: Message-Driven ML

### Option A: Integrated
ML runs inside NeoGPU message processing
- Use existing message queue
- ML ops as new message types
- Pros: Unified system
- Cons: Memory model mismatch

### Option B: Separate Thread
ML runs in dedicated thread, communicates via queue
- NeoGPU queue → ML thread → results
- Pros: Clean separation
- Cons: More complex

**Recommendation: Option A** - leverage existing infrastructure

## Integration Points

### 1. New Message Types (hs_msg.h)
```c
// ML-specific operations
OP_ML_EMBED,      // Get embedding for token
OP_ML_ATTENTION,  // Attention computation  
OP_ML_FORWARD,    // Full forward pass
OP_ML_GENERATE,   // Autoregressive generation
```

### 2. ML System (hs_ml.h)
```c
typedef struct {
    // Model
    void* weights;           // ~800MB for 1.58b
    void* kv_cache;          // Attention cache
    
    // State
    u32* tokenizer;
    u32 vocab_size;
    u32 hidden_size;
    u32 num_layers;
    
    // Runtime
    void* work_buffer;      // Scratch space
} HSMLSystem;
```

### 3. QoS Integration
- ML inference on RT channel (priority)
- Large model loads on TELEM (background)
- Results via toolbus

## Step-Change Ranking

1. **INT8 GEMM** - Core compute kernel
2. **Model loading** - Get weights in memory
3. **Tokenizer** - Convert text ↔ tokens
4. **Inference loop** - Generate tokens
5. **IPC integration** - Remote control
