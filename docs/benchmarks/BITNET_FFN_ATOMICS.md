# BitNet FFN Atomics

## Overview

BitNet's Feed-Forward Network (FFN) uses a gated linear unit variant with squared ReLU activation, optimized for ternary weight matrices.

## FFN Structure (from model.py)

```
Input: x [B, S, D]  (batch, seq, dim)
Linear1: x13 = W13 @ x  → [B, S, 2*FFN_dim]
Split:   x1, x3 = chunk(x13, 2, dim=-1)  → each [B, S, FFN_dim]
Act:     act = squared_relu(x1)          → [B, S, FFN_dim]
Gate:    gated = act * x3                → [B, S, FFN_dim]  (Hadamard product)
Norm:    norm = RMSNorm(gated)           → [B, S, FFN_dim]
Linear2: output = W2 @ norm              → [B, S, D]
```

Where:
- `squared_relu(x) = max(0, x)^2`
- W13 ∈ ℝ^(2*FFN_dim × D) (ternary weights)
- W2 ∈ ℝ^(D × FFN_dim) (ternary weights)

## Operation Breakdown

### 1. First GEMM: x13 = W13 @ x
- Input: x [B*S, D] (FP32 activations)
- Weight: W13 [2*FFN_dim, D] (ternary, packed 2-bit)
- Output: x13 [B*S, 2*FFN_dim] (FP32)
- Uses: hs_ml_gemm_int8 with dequantization

### 2. Split & Activation
- Split: x1 = x13[:, :FFN_dim], x3 = x13[:, FFN_dim:]
- squared_relu(x1):
  ```c
  // NEON pseudocode
  float32x4_t v = vld1q_f32(x1_ptr);
  float32x4_t zero = vdupq_n_f32(0.0f);
  float32x4_t relu = vmaxq_f32(v, zero);    // ReLU
  float32x4_t sqr = vmulq_f32(relu, relu);  // Square
  vst1q_f32(act_ptr, sqr);
  ```
- Gate: gated = act * x3
  ```c
  float32x4_t a = vld1q_f32(act_ptr);
  float32x4_t b = vld1q_f32(x3_ptr);
  float32x4_t gated = vmulq_f32(a, b);
  vst1q_f32(gated_ptr, gated);
  ```

### 3. RMSNorm
Compute: y = gated * (1 / sqrt(mean(gated^2) + ε))
- Step 1: Square elements: sq = gated * gated
- Step 2: Reduce sum: sum = Σ(sq) over last dim
- Step 3: Mean: mean = sum / FFN_dim
- Step 4: Variance: var = mean + ε
- Step 5: InvStd: invstd = 1 / sqrt(var)
- Step 6: Scale: y = gated * invstd

### 4. Second GEMM: output = W2 @ norm
- Input: norm [B*S, FFN_dim] (FP32)
- Weight: W2 [D, FFN_dim] (ternary, packed 2-bit)
- Output: output [B*S, D] (FP32)
- Uses: hs_ml_gemm_int8 with dequantization

## Weight Storage

Ternary weights stored as:
- Packed: 64 weights → 32 bytes (2 bits each)
- Scale: 1 FP32 per 64 weights
- Total: 36 bytes per 64 weights = 0.5625 bytes/weight
- Compression: 32× vs FP32 (4 bytes/weight)

## Computational Complexity

For FFN with dim=D, hidden=FFN_dim:
- GEMM1: 2 * B * S * D * FFN_dim MACs
- GEMM2: B * S * D * FFN_dim MACs
- Activations: ~5 * B * S * FFN_dim ops (ReLU^2, multiply, norm)
- Total: ~3.5 * B * S * D * FFN_dim MAC-equivalent ops

## ARM NEON Optimization Opportunities

1. **GEMMs**: Use hs_ml_gemm_int8 (ternary-optimized)
2. **Activation**: Vectorized squared_relu with NEON
3. **Gate**: Vectorized multiply
4. **RMSNorm**: 
   - Reduce sum with pairwise accumulation
   - Inverse sqrt via Newton-Raphson
   - Scale with vmulq_f32

## Performance Characteristics

- Memory bound: Weight loading dominates
- Compute: GEMMs are ~70% of FFN cost
- Activation: ~20% ( ReLU^2, gate, norm)
- Ternary advantage: 3.2× weight compression reduces memory bandwidth
