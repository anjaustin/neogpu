# LMM: BitNet 1.58b Inference on Pi4 (REFLECT)

## Edge Cases

### 1. Memory
- 1.58B params × 1 byte = 1.58GB
- Plus KV cache = easily 2GB+
- Pi4 with 4GB RAM: tight but possible
- Pi4 with 1GB: impossible
- **Solution**: Quantize to INT4 (~800MB), swap to disk if needed

### 2. Thermal Throttling
- Pi4 throttles at 80°C
- Heavy compute = heat
- **Solution**: Clock down, limit tokens/sec

### 3. Model Download
- Where to get the model?
- GGUF format from llama.cpp
- Need: wget/curl to download
- **Solution**: Document download process

### 4. First Token vs Token/Sec
- First token: ~5-10 seconds (full forward pass)
- After: ~50ms/token (just one layer)
- **Solution**: Stream output, show progress

### 5. Input/Output
- How to prompt? Serial? IPC?
- How to see output? Display? File?
- **Solution**: IPC for in/out, simple CLI

## Failure Modes

| Issue | Impact | Fix |
|-------|--------|-----|
| OOM | Crash | Quantize more, reduce context |
| Throttle | Slow | Heat sink, limit QPS |
| No model | Fail | Clear error message |
| Slow | Poor UX | Async generation |

## Why This is "NeoGPU"
- Message-driven generation
- Async inference via queue
- IPC for control
- Uses our NEON math

But also: this is a COMPLETELY different use case. Graphics → LLM.

Is this still "NeoGPU" or a new project?
