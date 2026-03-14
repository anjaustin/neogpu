# LMM: BitNet 1.58b Inference on Pi4 (NODES)

## What Exists

### NeoGPU Assets
- v4_dot: 4-element dot product (NEON)
- m4_multiply: 4×4 matrix multiply (NEON)
- Message queue with QoS
- IPC for remote control

### What's Missing for BitNet

#### 1. INT8/INT4 GEMM
```c
// Need: C[M][N] += A[M][K] * B[K][N]
// BitNet: weight is -1, 0, or +1
// Can use lookup tables or bit manipulation
```

#### 2. Layer Implementations
- BitLinear (quantized linear)
- BitAttention (with KV cache)
- RMSNorm
- SwiGLU (activation)

#### 3. KV Cache
- Key/value cache for attention
- Paged attention for long context
- Memory management

#### 4. Tokenizer
- BPE or SentencePiece
- Need vocab (~50K)

#### 5. Model Loader
- Load GGUF format (llama.cpp)
- Or custom binary format
- Memory mapping for large models

## Architecture Proposal

### Standalone vs Integrated
**Option A: Standalone binary**
- `neogpu_bitnet` - separate from graphics
- Loads model, runs inference
- IPC to control

**Option B: Integrated**
- ML node handles BitNet
- Message-driven inference
- Results via IPC

**Recommendation: Option A first** - simpler, focus on inference first.

### Key Files
```
include/hs_ml.h      - Tensor types
include/hs_bitnet.h  - BitNet types and declarations
src/hs_ml.c        - GEMM, activations
src/hs_bitnet.c     - Model, layers
```

### Memory Budget
- Model: ~1.6GB (INT8)
- KV cache: ~512MB (depending on context)
- activations: ~100MB
- Total: ~2.2GB (need 4GB Pi4)
