# BitNet 1.58b GEMM - The Atomics

## Overview

BitNet uses **ternary quantization** (1.58 bits) where weights are restricted to {-1, 0, +1}. This eliminates expensive FP32 multiplication entirely.

## The Core Trick

```
Traditional GEMM:     y = Σ(w × x)     [FP32 multiply]
BitNet GEMM:         y = LUT[w+1][x]   [Lookup table]
```

Since weights are only -1, 0, or +1:
- w = +1 → add x
- w = 0 → skip
- w = -1 → subtract x

This can be done with **INT8** arithmetic instead of FP32.

## Data Format

### Quantization Block (QK_I2_S = 64 for ARM)

```
Each block: 64 weights packed into 32 bytes

Weights: [-1, 0, +1] → encoded as 2 bits each
         64 weights × 2 bits = 128 bits = 16 bytes

Plus scale factor: 4 bytes (FP32)

Total: 20 bytes per 64 weights
Compression: 64 × 4 bytes → 20 bytes = 3.2× smaller
```

### Packing

```
Original FP32 weights:  [w0, w1, w2, w3, ...] (4 bytes each)
Packed to INT2:       Each weight becomes 2 bits
                       0b00 = 0
                       0b01 = +1  
                       0b10 = -1 (encoded differently)
                       0b11 = reserved
```

## ARM NEON Implementation

### Key Constants (gemm-config.h for ARM with DOTPROD)

```c
#define QK_I2_S         64        // 64 weights per block
#define ROW_BLOCK_SIZE   8        // Process 8 output rows at a time
#define COL_BLOCK_SIZE  256      // Process 256 columns at a time
#define PARALLEL_SIZE   8        // 8-way parallelism
```

### The Kernel: vec_dot_i2_i8_s_32W

```c
// Input:  x = activation (UINT8,  dequantized from INT8)
// Input:  y = weights    (INT8,   ternary -1/0/+1)
// Output: s = result (accumulated INT32)
//
// Dimensions: y has shape [K, N], x has shape [M, K]
// Computing: s[m] = x[m] × y[:, n]
```

### Step-by-Step Execution

#### Step 1: Unpack Weights

```c
// x contains packed UINT8 values, each holding 4 ternary weights
// We need to extract each weight as INT8 (-1, 0, +1)

// Original packed: [w0|w1|w2|w3] in one byte
// Extract using bit shifts:
uint8x16_t xq8_3 = vld1q_u8(x_row + ...);  // Load 16 bytes
uint8x16_t xq8_2 = vshrq_n_u8(xq8_3, 2);   // Shift right 2 bits
uint8x16_t xq8_1 = vshrq_n_u8(xq8_3, 4);   // Shift right 4 bits
uint8x16_t xq8_0 = vshrq_n_u8(xq8_3, 6);   // Shift right 6 bits

// Mask to get 2-bit values
xq8_0 = vandq_u8(xq8_0, mask);  // mask = 0b11
xq8_1 = vandq_u8(xq8_1, mask);
xq8_2 = vandq_u8(xq8_2, mask);
xq8_3 = vandq_u8(xq8_3, mask);

// Convert to INT8: 0→0, 1→1, 2→-1
int8x16_t q8_0 = vreinterpretq_s8_u8(xq8_0);
int8x16_t q8_1 = vreinterpretq_s8_u8(xq8_1);
// ... convert to signed
```

#### Step 2: Load Activations

```c
// y contains INT8 activations (already dequantized)
const int8x16_t yq8_0 = vld1q_s8(y + offset + 0);
const int8x16_t yq8_1 = vld1q_s8(y + offset + 16);
const int8x16_t yq8_2 = vld1q_s8(y + offset + 32);
const int8x16_t yq8_3 = vld1q_s8(y + offset + 48);
```

#### Step 3: Dot Product with NEON

**With DOTPROD extension (ARMv8.2+, e.g., A55, A75, A710):**
```c
// ARMv8.2-DOTPROD: Single instruction for multiply-accumulate
accu = vdotq_s32(accu, q8_0, yq8_0);  // 16 INT8 multiplies, accumulate 16 INT32
accu = vdotq_s32(accu, q8_1, yq8_1);
accu = vdotq_s32(accu, q8_2, yq8_2);
accu = vdotq_s32(accu, q8_3, yq8_3);
```

**Without DOTPROD (ARMv8.0, e.g., Cortex-A72 on Pi4):**
```c
// Use add/sub: ternary weights are -1, 0, +1
// +1 → vadd, -1 → vsub, 0 → skip
// This is 3x faster than multiply!
```

#### Step 4: Horizontal Sum

```c
// Sum all 4 lanes of the accumulator
int sumi = vaddlvq_s32(accu);  // ARM NEON instruction
s[row] = (float)sumi;          // Convert to FP32
```

## Data Layout

### Weight Matrix Y

```
Memory layout for ternary weights:

Each block: 64 weights → 32 bytes (packed INT2)
           + 4 bytes (scale factor)

Block 0:  [w0-w63] [scale0]  
Block 1:  [w64-w127] [scale1]
...

K dimension is divided into blocks of 64.
N dimension is contiguous INT8.
```

### Activation Matrix X

```
Memory layout:

Each value: 1 UINT8 (dequantized activation)
           → represents value in range [-scale, +scale]

Packed as 4 values per byte when packed in Q4 format.
In INT8 format: 1 byte per value.

Row-major layout: [x0, x1, x2, ..., xK-1]
```

## GEMM Tiling Strategy

```
Block sizes:
- ROW_BLOCK_SIZE = 8   (M dimension)
- COL_BLOCK_SIZE = 256 (N dimension)
- QK_I2_S = 64        (K dimension, block size)

For large matrices:
1. Process 8×256 tiles at a time (cache-friendly)
2. Within each tile, process K in blocks of 64
3. Accumulate results in INT32 accumulator
4. Apply scale factors at the end
```

## Scale Factors

```
Each 64-weight block has a scale:
  actual_weight[i] = packed_weight[i] * scale

After the INT32 dot product:
  result = int32_sum × scale
```

## Why This Is Fast

| Aspect | Traditional FP32 | BitNet |
|--------|-----------------|--------|
| Multiply | FP32 mul (~10 cycles) | INT8 dotprod (1 cycle) |
| Memory | 4 bytes/weight | 0.5 bytes/weight |
| Cache | More misses | Better locality |
| SIMD | 4 ops/cycle (FP32) | 16 ops/cycle (INT8) |

## Performance on Pi4 (Cortex-A72)

```
Cortex-A72: 4 cores @ 1.5GHz
DOTPROD: NO (ARMv8.0, not ARMv8.2)
L1 cache: 32KB per core
L2 cache: 512KB per core (shared)

Optimizations used:
- Add/sub instead of multiply (ternary weights are -1, 0, +1)
- INT8 arithmetic instead of FP32
- 3.2× memory compression from packing

Expected: 
- GEMM: ~2-5 GFLOPS (effective, no DOTPROD)
- Token generation: ~1-3 tokens/sec
```

## Performance with DOTPROD (ARMv8.2+)

```
With DOTPROD (e.g., Cortex-A55, A75, A710):
- vdotq_s32: 16 INT8 ops per cycle
- GEMM: ~10-20 GFLOPS (effective)
- Token generation: ~5-10 tokens/sec
```

## Integration with NeoGPU

The key functions to port:

1. `quantize_i2_s()` - Convert FP32 weights to ternary
2. `ggml_vec_dot_i2_i8_s_32W()` - The core GEMM kernel
3. Scale factor handling
4. KV cache management

These can be added to:
- `include/hs_ml.h` - ML types and declarations
- `src/hs_ml.c` - NEON-optimized implementations
