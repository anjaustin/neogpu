# LMM: NeoGPU + BitNet Integration (REFLECT)

## Edge Cases

### 1. Memory Limits
- Model: ~800MB (1.58-bit)
- KV cache: ~512MB (configurable)
- Activation: ~100MB
- Total: ~1.5GB
- Pi4 8GB: plenty of headroom
- Pi4 4GB: tight, may need to reduce KV cache
- **Mitigation**: Configurable KV cache size

### 2. Thermal Throttling
- LLM inference is 100% CPU load
- Pi4 throttles at 80°C
- Sustained load = thermal throttling
- **Mitigation**: Reduce threads, clock down

### 3. First Token Latency
- Full forward pass: ~5-10 seconds
- Must process entire model
- **Mitigation**: Show progress, streaming

### 4. Token/s Speed
- Target: 5-10 tokens/sec
- Real: Depends on optimization
- **Mitigation**: Benchmark, tune GEMM

### 5. Model Format
- GGUF is complex
- Need: parser or external conversion
- **Mitigation**: Use pre-converted models

### 6. Context Length
- KV cache grows with context
- 1K tokens = ~256MB
- **Mitigation**: Sliding window, RoPE

## Failure Modes

| Scenario | Impact | Mitigation |
|----------|--------|------------|
| OOM | Crash | Configurable memory |
| Throttle | Slow | Reduce threads |
| No model | Fail | Clear error msg |
| Bad input | Garbage | Validate tokens |
| Long context | OOM | Limit context |

## GEMM Kernel Requirements

### Must Have
- INT8 × INT8 → INT32 accumulation
- Ternary weight handling
- Scale factor application
- NEON optimization (DOTPROD path)

### Nice to Have
- INT4 support
- Block-wise quantization
- Memory mapping

## Pi4-Specific Considerations

- ARMv8.2 (Cortex-A72): Has DOTPROD ✓
- L1D: 32KB - fits ~8K INT8
- L2: 512KB - fits ~64K INT8
- RAM: 8GB - model + KV cache fit
