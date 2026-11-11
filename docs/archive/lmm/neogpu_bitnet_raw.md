# LMM: NeoGPU + BitNet Integration (RAW)

## Current State
- NeoGPU: Message-passing GPU layer with NEON math
- BitNet: Ternary (1.58-bit) LLM inference engine
- Pi4: 8GB RAM, 4 cores @ 1.5GHz, ARM NEON + DOTPROD

## Step-Change Opportunities

### 1. INT8/TERNARY GEMM
- Current: v4_dot (4 elements), m4_multiply (4×4)
- Need: Arbitrary-size INT8 GEMM with ternary weights
- Gap: No quantized matrix operations

### 2. Message-Driven Inference
- Current: Graphics messages (draw, buffer, texture)
- Need: LLM inference messages (token, embedding, attention)
- Gap: No ML-specific message types

### 3. Async Inference Pipeline
- Current: Synchronous frame processing
- Need: Async token generation with callbacks
- Gap: No async ML result handling

### 4. Model Memory Management
- Current: Fixed payloads (96 bytes)
- Need: Large model weights (~800MB)
- Gap: No large allocation system

### 5. KV Cache
- Current: No cache
- Need: Attention key/value cache
- Gap: No caching infrastructure

## Questions for NODES
- Which opportunity is the biggest step-change?
- Build on existing message architecture or separate?
- What's the MVP vs what's the vision?
