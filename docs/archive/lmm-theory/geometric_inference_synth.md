# SYNTHESIS: Geometric Inference for NeoGPU

## Architecture

Neural network inference reframed as **coordinate transformation through routing**.

```
┌─────────────────────────────────────────────────────────────┐
│                    GEOMETRIC INFERENCE                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   INPUT SPACE                    OUTPUT SPACE                │
│   ┌─────────┐                    ┌─────────┐                │
│   │ addr[0] │───[+1]────────────>│ addr[0] │                │
│   │ addr[1] │───[-1]────────────>│         │                │
│   │ addr[2] │─────────[0]        │ addr[1] │                │
│   │ addr[3] │───[+1]────────────>│         │                │
│   │   ...   │        ───────────>│   ...   │                │
│   └─────────┘                    └─────────┘                │
│                                                              │
│   address = dimension coordinate                             │
│   value = magnitude of excitation                            │
│   weight = routing decision {+1, -1, 0}                     │
│                                                              │
│   output[n] = Σ input[k] where route[k→n, +1]               │
│             - Σ input[k] where route[k→n, -1]               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Decisions

### 1. Routing Representation Hierarchy

Choose representation based on problem characteristics:

| Representation | Memory | Decode Cost | Use When |
|----------------|--------|-------------|----------|
| Binary (1-bit) | 1 bit/weight | 0 (identity) | Max throughput, quality allows |
| Ternary 2-bit | 2 bits/weight | LUT decode | Standard BitNet inference |
| Pre-expanded INT8 | 8 bits/weight | 0 (identity) | Small models, cached |
| Structured sparse | rule + indices | rule eval | Regular patterns exist |

**Decision**: Implement all four. Let model/hardware determine which to use.

### 2. Memory Layout = Geometric Layout

Address space IS coordinate space:
```
activation[k] stored at base_addr + k * sizeof(element)
```

Contiguous memory = contiguous dimensions = efficient sequential access.
This is not a choice - it is how hardware works. Align with it.

**Decision**: Always use contiguous layout for dense tensors.
For sparse: store indices + values, gather at runtime.

### 3. Ground State is Zero

Zero activations = ground state = no perturbation.
Do not treat zeros as "small values" - they are absence of signal.

**Decision**: Exploit sparsity when present. Skip zero contributions.
For ReLU networks, output sparsity is common (50%+). Track and exploit.

### 4. Binary as the Limit

Binary weights + binary activations = pure bit logic.
```
result = popcount(A XNOR W) for {-1,+1} encoding
       = popcount(A AND W) for {0,1} encoding
```

Throughput: limited only by memory bandwidth.
On Pi4: ~30 GOPS single-thread, scales poorly due to memory (already at limit).

**Decision**: Implement binary path for future models that support it.

## Implementation Spec

### File Structure

```
src/hs_ml_routing.h      # Routing abstraction header
src/hs_ml_routing.c      # Routing primitives
src/hs_ml_binary.c       # Binary GEMM (XNOR + popcount)
src/hs_ml_ternary_mt.c   # Ternary GEMM (existing, LUT decode)
src/hs_ml_sparse.c       # Sparse routing (future)
```

### Core Primitives

```c
/* Routing representations */
typedef enum {
    ROUTE_BINARY,      /* 1 bit per weight: {-1,+1} as {0,1} */
    ROUTE_TERNARY_2B,  /* 2 bits per weight: {-1,0,+1} as {10,00,01} */
    ROUTE_TERNARY_INT8,/* 8 bits per weight: {-1,0,+1} as int8_t */
    ROUTE_SPARSE,      /* (index, sign) pairs */
} RouteFormat;

/* Route a single layer */
void hs_ml_route(
    int32_t* output,           /* [M, N] accumulator */
    const void* input,         /* [M, K] activations */
    const void* weights,       /* [N, K] routing table */
    RouteFormat fmt,           /* how weights are encoded */
    uint32_t M, uint32_t N, uint32_t K
);

/* Binary-specific (maximum throughput) */
void hs_ml_route_binary(
    int32_t* output,
    const uint8_t* input_bits, /* [M, K/8] packed bits */
    const uint8_t* weight_bits,/* [N, K/8] packed bits */
    uint32_t M, uint32_t N, uint32_t K
);
```

### NEON Implementation Notes

**Ternary (existing):**
- vqtbl1q_s8: 4-bit → 8-bit LUT = 2 weights decoded per lookup
- vpadalq_s16: pairwise add-accumulate for dot product
- 4-column blocking to amortize activation load

**Binary (new):**
- veorq_u8: XOR for mismatch detection
- vmvnq_u8: NOT to get XNOR
- vcntq_u8: popcount per byte
- vpadalq_u8: accumulate counts

### Performance Targets

| Format | Single-Thread | 3-Thread | Memory |
|--------|---------------|----------|--------|
| Ternary 2-bit (current) | 10 GOPS | 25 GOPS | 4 MB (4K×4K) |
| Binary 1-bit | 30 GOPS | ~30 GOPS* | 2 MB (4K×4K) |

*Binary does not scale with threads because already memory-bound.

## Success Criteria

- [ ] Binary GEMM implemented and tested
- [ ] Binary achieves 30+ GOPS on M=4, N=4096, K=4096
- [ ] Routing abstraction supports format switching
- [ ] Documentation explains geometric interpretation
- [ ] No regression in ternary performance

## Explicit Handling of Major Tensions

### Tension 1: Efficiency vs Flexibility

**Resolution**: Abstraction layer with specialized implementations.
`hs_ml_route()` dispatches to optimal implementation based on format.
No runtime penalty for flexibility - format is known at model load time.

### Tension 2: New Concepts vs Working Code

**Resolution**: Theory documented separately from implementation.
`docs/theory/GEOMETRIC_INFERENCE.md` explains the concepts.
`src/hs_ml_*.c` implements without requiring reader to understand theory.
The code works regardless of how you think about it.

### Tension 3: Binary Quality vs Efficiency

**Resolution**: Not our problem to solve. We provide the infrastructure.
If a model is trained binary, we run it fast.
If a model is ternary, we run that fast too.
Model choice is upstream of inference implementation.

## What This Enables

1. **Unified framework** for binary, ternary, and future quantized models
2. **Clear path to hardware** - routing abstraction maps to FPGA/ASIC
3. **Principled sparsity** - zeros are ground state, not approximation
4. **Theoretical foundation** - why these operations are correct

## Next Steps

1. Implement `hs_ml_binary.c` with NEON XNOR+popcount
2. Add routing abstraction header
3. Benchmark binary vs ternary on same matrices
4. Document mapping from geometric theory to code
5. (Future) Sparse routing for attention patterns
