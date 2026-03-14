# Ternary GEMM Kernel Red-Team Analysis

## Summary
The V18 kernel passed all 16 red-team tests including edge cases,
boundary conditions, extreme values, and random stress testing.

## Performance
- Peak: 5.91 GOPS (small matrices)
- Sustained: 5.2-5.6 GOPS (large matrices)
- Original baseline: 0.29 GOPS
- Speedup: ~20x

## Known Limitations

### 1. K must be divisible by 4
The packed weight format stores 4 weights per byte.

### 2. INT16 Accumulation Safe
Each 64-element block contributes max 8128 to int16 accumulator,
well under 32767 limit.

### 3. B_scale Parameter Unused
Reserved for future dequantization support.

## Weight Encoding
- 00 = 0, 01 = +1, 10 = -1, 11 = 0
- Byte layout: bits[1:0]=pos0, bits[3:2]=pos1, bits[5:4]=pos2, bits[7:6]=pos3

## Files
- src/hs_ml.c: Integrated V18 kernel
- tests/test_redteam.c: Red-team test suite (16 tests)
