# NeoGPU Ternary GEMM Kernel Optimization

## Summary

Optimized the BitNet ternary GEMM kernel for Raspberry Pi 4 (Cortex-A72, ARMv8.0).

**Final Performance:**
- Single-threaded: 7-10 GOPS (depending on matrix size)
- Multi-threaded (3 threads): **22-25 GOPS**
- Starting point was 0.29 GOPS (~85x total improvement)

## Hardware Constraints

- Cortex-A72 (ARMv8.0): No DOTPROD instruction
- Must use `vmlal_s8` (multiply-accumulate long) instead of `vdotq_s32`
- 4 cores but memory bandwidth limits scaling to 3 threads

## Key Optimizations

### 1. Nibble LUT Decoding (V18 kernel)
Instead of decoding 2-bit weights one at a time, use a 16-entry LUT indexed by nibble:
- Each nibble (4 bits) encodes 2 weights
- Single `vqtbl1q_s8` lookup decodes both weights
- 4x fewer LUT lookups than naive approach

### 2. Activation Deinterleaving  
Weights decode in stride-4 pattern. Deinterleave activations to match:
```
a[0,4,8,...], a[1,5,9,...], a[2,6,10,...], a[3,7,11,...]
```
Uses 4x `vuzpq_s8` per 64 activations.

### 3. 4-Column Blocking
Process 4 output columns simultaneously:
- Amortizes activation deinterleave cost (once per 4 columns)
- Better register utilization
- Better prefetch efficiency

### 4. Multi-Threading
Parallelize across N dimension:
- Each thread processes disjoint column ranges
- Main thread + 2 workers = 3 threads total
- 4 threads hits memory bandwidth wall

## Benchmark Results (M=4, N=4096, K=11008)

| Threads | GOPS | Speedup |
|---------|------|---------|
| 1 | 10.27 | 1.0x |
| 2 | 19.58 | 1.9x |
| 3 | 24.83 | 2.4x |
| 4 | 22.92 | 2.2x |

## Files

- `src/hs_ml.c`: Main ML implementation with integrated GEMM
- `src/hs_ml_ternary_mt.c`: Multi-threaded kernel
- `src/hs_ml_ternary_v18.c`: Best single-threaded kernel (reference)
- `include/hs_ml.h`: API definitions

## API

```c
// Auto-selects optimal thread count
void hs_ml_gemm_int8(int32_t* C, const int8_t* A, const u8* B_ternary,
                     const float* B_scale, u32 M, u32 N, u32 K);

// Explicit thread count (0 or 1 = single-threaded)
void hs_ml_gemm_ternary_mt(int32_t* C, const int8_t* A, const u8* B_ternary,
                           u32 M, u32 N, u32 K, int num_threads);

// Get optimal thread count for dimensions
int hs_ml_gemm_ternary_optimal_threads(u32 M, u32 N, u32 K);
```

## What Didn't Work

1. **Pre-packing weights**: Tried reorganizing weights to eliminate activation 
   deinterleaving. No speedup because the nibble LUT already produces the right
   pattern; the overhead is in the LUT lookups themselves.

2. **Add/subtract instead of multiply**: Attempted to use conditional add/sub
   instead of multiply-by-{-1,0,+1}. No improvement because `vmlal_s8` is
   already extremely efficient - multiply is essentially free.

## Future Work

1. **Pre-expanded INT8 weights**: Store weights as INT8 instead of 2-bit packed.
   4x memory cost but eliminates all decode overhead. Could reach 30+ GOPS.

2. **Tune for different Pi models**: Pi5 has different cache/memory characteristics.

3. **Integrate with KV cache**: Combine GEMM with attention for better locality.
