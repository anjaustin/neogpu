# NODES: Geometric Neural Network Inference

## Node 1: Ternary Weights Are Routing, Not Multiplication

Ternary weights {-1, 0, +1} do not multiply. They route:
- +1: route to positive accumulator
- -1: route to negative accumulator  
- 0: route to null

The "multiply" is a conceptual overhead. The operation is selection + summation.

Why it matters: Changes what optimal hardware looks like. Not ALU, but crossbar.

## Node 2: The LUT IS the Frozen Transform

Current implementation uses `vqtbl1q_s8(lut, nibble)` to decode weights.
This is not "decode then operate" - the LUT lookup IS the operation.
The nibble indexes into which transform to apply.

For binary weights, the transform is identity. No LUT needed. The bit IS the routing.

Why it matters: We cannot optimize decode further. We can only choose representations
where decode is trivial (binary) or unnecessary (pre-expanded).

## Node 3: Memory Address = Coordinate in State Space

Address 0x1000 + k = dimension k in the activation space.
Value at address = magnitude of excitation in that dimension.
Ground state = all zeros at all addresses.

Why it matters: Unifies memory layout with geometric meaning. The address IS the identity
of the dimension. No separate indexing needed.

Tension with practical memory: Non-contiguous access is slow. Geometry must align with
memory hierarchy for performance.

## Node 4: Weights as Routing Tables

A weight matrix is a routing table:
```
input_address -> (output_address, sign)
```

For dense weights: all inputs connect to each output with some sign.
For sparse weights: some connections are absent (weight = 0).

Inference = message passing: each input sends its value along its edges,
outputs sum incoming values with signs.

Why it matters: Reframes GEMM as graph traversal. Opens door to sparse routing,
structured patterns, hardware-native addressing.

## Node 5: Structured Routing Compresses Naturally

If the routing follows a pattern:
- Convolution: sliding window pattern
- Attention: query-key similarity pattern
- FFN: dense (all-to-all)

We do not store individual edges. We store the RULE that generates edges.
The rule is the compression.

Why it matters: Structured sparsity is not "optimization" - it is expressing that
the routing has regularity. The rule IS the minimal representation.

## Node 6: Shape = Relationship, Not Property

```
shape = activation_state relative_to weight_state
```

Shape is not intrinsic to data. It is the relationship between data and the
coordinate system that gives it meaning.

The weight DEFINES the coordinate system.
The activation is a point in that system.
Inference = expressing the point in successive coordinate systems.

Tension with "shape" as tensor dimensions: Conventional ML thinks of shape as [N,C,H,W].
This is storage layout, not geometric meaning.

## Node 7: Ground State and Perturbations

Ground state = zero activation everywhere.
Any non-zero = perturbation from ground.
Sparse activations = few perturbations = efficient.

Weights define how perturbations propagate: input perturbation -> output perturbation.
The network is a perturbation transfer function.

Why it matters: Zeros are not "small values" - they are absence of signal.
Sparsity is not approximation - it is exact representation of the ground state.

## Node 8: Hardware Implications

Optimal hardware for routing-based inference:
1. Address-based routing (not ALU multiply)
2. Dual accumulators (positive/negative)
3. Bit-level logic for binary (XNOR gates)
4. Gather-accumulate, not multiply-accumulate

Current CPUs/GPUs: designed for multiply-accumulate.
The mismatch is why we need optimization.

Tension: Existing hardware is what it is. We must work within it.
But: FPGA/ASIC could implement routing natively.

## Node 9: Binary as the Limit

Binary weights: 1 bit per routing decision.
Binary activations: 1 bit per dimension magnitude.

Binary x Binary = XNOR + popcount.
No decode, no multiply, pure logic.
Theoretical limit: ~100+ GOPS on Pi4 with current memory.

Tension: Model quality with binary. Research question, not engineering question.

## Summary of Tensions

1. Geometric meaning vs memory layout (Node 3 vs Node 8)
2. Frozen transform efficiency vs representation (Node 2)
3. Structured vs dense routing (Node 5)
4. Ideal hardware vs existing hardware (Node 8)
5. Binary efficiency vs model quality (Node 9)
