# LMM: ML Inference (SYNTHESIZE)

## P0 Spec: ML Inference

### 1. Tensor Type (hs_ml.h)
```c
#define HS_TENSOR_MAX_DIMS 4

typedef struct {
    u32 shape[HS_TENSOR_MAX_DIMS];
    u32 ndim;
    float* data;
} HSTensor;
```

### 2. Basic Math (hs_ml.c)
```c
// GEMM: C = alpha * A * beta * C
void hs_ml_gemm(float* C, const float* A, const float* B, 
                u32 M, u32 N, u32 K, float alpha, float beta);

// Activation
void hs_ml_relu(float* x, u32 n);
float hs_ml_sigmoid(float x);
void hs_ml_softmax(float* x, u32 n);
```

### 3. Simple Layers
- Dense (fully-connected)
- Conv2D (via NEON, small kernels)
- MaxPool
- ReLU
- Sigmoid

### 4. Pre-built Models
- Simple CNN (2 conv + dense) for demo
- Feature extraction from frames

### 5. Integration with Rendering
- Extract frame → ML → overlay result
- Use TELEM channel for ML messages
- Async with callback

## Acceptance Criteria
- [ ] GEMM works (correctness test)
- [ ] ReLU activation works
- [ ] Simple CNN loads and runs
- [ ] No frame drops during inference
- [ ] Graceful fallback if no ML

## What's NOT P0
- Full MobileNet support
- Training
- GPU compute shaders
- Model quantization
