# NeoGPU ML API

## Overview

ML module for BitNet 1.58-bit LLM inference on ARM NEON.

## Compilation

Standard build (Pi4 / Cortex-A72):
    gcc -O3 -march=armv8-a -mtune=cortex-a72 -Iinclude -c src/hs_ml.c
    gcc -O3 -march=armv8-a -mtune=cortex-a72 -Iinclude -c src/hs_ml_ternary_mt.c

With DOTPROD (ARMv8.2+):
    gcc -O3 -march=armv8-a+simd+dotprod -mtune=cortex-a72 -Iinclude -c src/hs_ml.c

Link with -lpthread for multi-threading support.

## GEMM Functions

### hs_ml_gemm_int8 (Primary API)

    void hs_ml_gemm_int8(int32_t* C, 
                         const int8_t* A, 
                         const u8* B_ternary,
                         const float* B_scale,
                         u32 M, u32 N, u32 K);

Ternary GEMM: C = A x B

Params:
  C        - Output [M x N], INT32 accumulator
  A        - Activations [M x K], INT8
  B        - Weights [K x N], packed 2-bit ternary
  B_scale  - Scale factors (unused, reserved)
  M        - Output rows
  N        - Output cols
  K        - Hidden dimension

Runtime: Automatically selects DOTPROD or multi-threaded fallback.

### hs_ml_gemm_ternary_mt (Multi-threaded)

    void hs_ml_gemm_ternary_mt(int32_t* C,
                               const int8_t* A,
                               const u8* B_ternary,
                               u32 M, u32 N, u32 K,
                               int num_threads);

Explicit multi-threaded GEMM. num_threads: 0/1 = single, 2-4 = multi.

### hs_ml_gemm_ternary_optimal_threads

    int hs_ml_gemm_ternary_optimal_threads(u32 M, u32 N, u32 K);

Returns optimal thread count:
  - N < 256 or ops < 1M: 1 thread
  - ops < 10M: 2 threads  
  - ops >= 10M: 3 threads

## Data Format

### Ternary Weight Packing

64 weights -> 16 bytes (2 bits each)
4 weights per byte:
  bits [1:0] = weight 0
  bits [3:2] = weight 1
  bits [5:4] = weight 2
  bits [7:6] = weight 3

Encoding:
  00 = 0
  01 = +1
  10 = -1
  11 = reserved (treated as 0)

## Performance

### Pi4 (Cortex-A72, ARMv8.0)

Matrix Size              1 Thread    3 Threads
M=4, N=4096, K=4096      9.1 GOPS    21.9 GOPS
M=4, N=4096, K=11008    10.3 GOPS    24.8 GOPS
M=4, N=11008, K=4096     9.1 GOPS    22.9 GOPS
M=1, N=4096, K=4096      7.3 GOPS    14.7 GOPS

Starting from 0.29 GOPS -> 24.8 GOPS = 85x improvement

Key techniques:
1. Nibble LUT decoding (4x fewer lookups)
2. Activation deinterleaving (matches decode pattern)
3. 4-column blocking (amortizes setup)
4. Multi-threading (N-dimension parallel)

See docs/TERNARY_KERNEL_NOTES.md for full details.

## Testing

Build and run tests:
    gcc -O3 -march=armv8-a -Iinclude -o test_ml_gemm \
        tests/test_ml_gemm.c src/hs_ml.c src/hs_ml_ternary_mt.c -lpthread -lm
    ./test_ml_gemm

Red-team edge case tests:
    gcc -O3 -march=armv8-a -Iinclude -o test_redteam \
        tests/test_redteam.c src/hs_ml.c src/hs_ml_ternary_mt.c -lpthread -lm
    ./test_redteam
