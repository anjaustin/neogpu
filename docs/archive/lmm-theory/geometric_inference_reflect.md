# REFLECT: Geometric Neural Network Inference

## Core Insight

**Neural network inference is coordinate transformation, not arithmetic.**

Weights define reference frames. Activations are positions in those frames.
Each layer transforms position from one frame to another.
The "compute" is resolving which frame element maps to which.

For ternary/binary weights, this transformation has only three/two possible
relationships per dimension: aligned, opposed, or orthogonal.

The minimal representation of this transformation is the routing table:
which inputs connect to which outputs, with what sign.

## Resolution: Geometric Meaning vs Memory Layout

These are not in tension - they are the same thing when designed correctly.

If we define:
- address = dimension index
- value = magnitude in that dimension
- contiguous addresses = adjacent dimensions

Then sequential memory access = processing adjacent dimensions.
The geometry aligns with the memory hierarchy when dimensions are contiguous.

For sparse patterns (many zeros), gather/scatter breaks this alignment.
Solution: Store only non-zero connections (sparse format) OR structure
the sparsity to maintain contiguity (structured sparsity).

## Resolution: Frozen Transform Efficiency

The transform cannot be eliminated - it IS the neural network.
But it can be made trivial:

| Format | Transform | Cost |
|--------|-----------|------|
| 2-bit ternary | LUT decode | 4 LUT ops per 8 weights |
| Binary | Identity | 0 ops |
| Pre-expanded INT8 | Identity | 0 ops, 8x memory |

For memory-bound problems (large models), 2-bit packed is optimal.
For compute-bound problems (small models, cached), pre-expanded wins.
For maximum throughput, binary is the limit.

The "frozen transform" insight: choose representation where the transform
is what you want to compute anyway.

## Resolution: Structured vs Dense Routing

Structured routing is not an optimization of dense routing.
It is a DIFFERENT routing pattern that happens to be describable by a rule.

Dense FFN: every input connects to every output. Routing = full matrix.
Convolution: local sliding window. Routing = small kernel + position.
Attention: learned similarity. Routing = query-key match.

The rule IS the compression. Storing the rule instead of the expanded
routing is not approximation - it is exact.

When routing has structure, exploit it. When it does not, accept density.

## Resolution: Ideal Hardware vs Existing Hardware

We cannot change hardware today. But we can:

1. Emulate routing on multiply-accumulate hardware (what we do now)
2. Map operations to existing primitives that match routing
   - NEON: vqtbl (table lookup) is a 16-way routing primitive
   - Binary: XNOR + popcount is native in most ISAs
3. Design for future hardware by specifying what ideal routing hardware looks like

Ideal routing hardware:
- Content-addressable memory for sparse gather
- Dual signed accumulators per output
- Bit-serial processing for binary
- Configurable crossbar for programmable routing

This is closer to FPGA than CPU. Reconfigurable routing fabric.

## Resolution: Binary Efficiency vs Model Quality

This is a training question, not inference question.

The inference path is clear:
- Binary weights + binary activations = bit logic only
- Throughput limited only by memory bandwidth
- On cached data: 100+ GOPS on Pi4

The training question: can networks learn useful functions with binary constraints?

Recent research (BitNet, 1-bit LLMs) says yes, with caveats.
Quality vs efficiency is a Pareto frontier, not a single point.

For NeoGPU: implement both paths. Let model choice determine which to use.

## What I Now Understand

1. **Multiply-accumulate is a leaky abstraction.** It works, but obscures what
   the network actually does: route activations based on learned patterns.

2. **The LUT decode in our ternary kernel is not overhead.** It IS the routing
   operation, expressed as table lookup. We cannot remove it without changing
   the representation.

3. **Binary is not a degradation of ternary.** It is a simpler routing scheme
   (2 states vs 3) that happens to map perfectly to hardware logic.

4. **Memory address as coordinate is not metaphor.** It is literally how the
   hardware works. The address bus routes data based on position.

5. **Ground state matters.** Zero activations are not "small" - they are absence.
   Sparse networks are not approximate - they exactly represent which dimensions
   are excited.

6. **The path forward is representation choice.** For each problem:
   - How much can routing be structured?
   - How sparse are activations?
   - What is the memory/compute tradeoff?
   - Can the model tolerate binary?

## Remaining Questions

1. Can we design a weight format that IS the routing table, with no decode?
   (Answer: binary is already this.)

2. Can attention be expressed as pure routing with no softmax?
   (Possibly: hard attention, or threshold-based selection.)

3. What does backprop look like in routing terms?
   (Gradient = which routes should change, and how.)

4. Is there a hardware-software co-design that makes routing native?
   (FPGA LUTs are exactly this. The question is programmability.)
