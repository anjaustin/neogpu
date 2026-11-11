# LMM: ML Inference (REFLECT)

## Edge Cases

### 1. Model Loading
- Models stored as binary blobs
- Memory constraints on Pi4 (1-4GB RAM)
- Need: streaming load, not all at once

### 2. Input Size
- Camera: 640x480 or higher
- Tensor sizes can be large
- Need: resize/crop preprocessing

### 3. Output Processing
- Classification: top-1 or top-5
- Detection: bounding boxes
- Need: post-processing

### 4. Precision
- FP32: standard
- FP16: faster but may lose accuracy
- Quantization: INT8 (harder)

### 5. Frame Rate
- ML can be slow
- Async inference while rendering continues
- Need: non-blocking with callback

## Failure Modes

| Scenario | Impact | Mitigation |
|----------|--------|-------------|
| OOM | Crash | Limit model size |
| Slow inference | Frame drop | Async, skip frames |
| No model | Fail gracefully | Disable ML |
| Bad input | Garbage out | Validate |

## Pi4 Constraints
- No GPU compute (OpenCL)
- Limited NEON bandwidth
- 1-4GB RAM
- Thermal throttling

## What Makes This "NeoGPU"
- Message-driven ML inference
- Async via message queue
- Sync with rendering via channels
- Not: separate ML library
