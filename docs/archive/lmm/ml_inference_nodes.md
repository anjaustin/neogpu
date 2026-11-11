# LMM: ML Inference (NODES)

## Existing Assets

### NEON Math (hs_math_neon.h)
- vec4: ARM NEON 128-bit vectors
- mat4: 4x4 matrices (NEON)
- m4_multiply: matrix multiply
- m4_invert: matrix inversion
- m4_transpose

### What's Missing for ML

#### Matrix Multiply (GEMM)
- Need: A×B + C for arbitrary sizes
- Current: only 4×4
- Would need: tiled/blocked GEMM for performance
- NEON can do 4x4 in one instruction

#### Activation Functions
- ReLU: max(0, x)
- Sigmoid: 1/(1+e^-x)
- Tanh: (e^x - e^-x)/(e^x + e^-x)
- Softmax

#### Convolution
- 2D convolution via shaders
- Or via NEON (slow)
- Or via GPU (fast but complex)

## Architecture Proposal

### 1. Tensor Type
```c
typedef struct {
    u32 shape[4];  // N, C, H, W
    float* data;
} HSTensor;
```

### 2. ML Node (hs_ml.c)
- OP_ML_INFERENCE: run inference
- OP_ML_LAYER: configure layer
- Input: tensor from message
- Output: tensor result

### 3. Message Flow
```
Frame → Extract Features → ML Inference → Render Overlay
         (NEON)          (Shaders)        (GLES)
```

### 4. Pre-trained Models
- MobileNetV2 (image classification)
- YOLO (object detection - simplified)
- Custom: simple CNN

## Performance Estimate
- MobileNetV2: ~50ms/frame on Pi4 (CPU)
- GPU would help but complex
- For simple tasks: < 10ms possible
