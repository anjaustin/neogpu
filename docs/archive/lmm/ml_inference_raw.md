# LMM: ML Inference (RAW)

## Current State
- Message-passing GPU layer
- GLES shaders for graphics
- NEON math (mat4, vec4)
- 8.5M msgs/sec throughput

## ML Possibilities on Pi4

### What Pi4 Has
- CPU: Cortex-A72 (4 cores @ 1.5GHz)
- GPU: VideoCore VI (V3D 4.2)
- NEON: ARM SIMD (64/128-bit)

### ML Options

#### A. CPU-only (slow)
- ONNX Runtime, TensorFlow Lite
- Too slow for real-time

#### B. GPU Compute (limited)
- OpenCL not available on Pi4
- Can't do CUDA
- Could write custom shaders for inference

#### C. NEON-optimized (feasible)
- Matrix multiplication via NEON
- Convolution via NEON
- 1-2 GFLOPS possible

#### D. Hybrid (best)
- Preprocess on CPU (NEON)
- Inference via custom shaders
- Post-process on CPU

## What's Needed
- Neural network layer definitions
- NEON matrix ops (already have some)
- Convolution shaders
- Inference engine (forward pass only)

## Questions
- What networks to support?
- Input from where? Camera? Framebuffer?
- Output to where? Display? Serial?
