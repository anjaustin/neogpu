# Geometric Inference: Neural Networks as Coordinate Transforms

## Core Insight

Neural network inference is not arithmetic. It is **geometry**.

Weights do not multiply. They define **reference frames**. Activations are magnitudes 
that only have meaning relative to these frames.

## The Traditional View (Wrong)

```
output = weight * activation
```

This implies multiplication is fundamental. It is not.

## The Geometric View (Correct)

```
output = activation resolved in weights coordinate system
```

The weight defines what "aligned" means. The activation just exists. The operation 
resolves one relative to the other.

## Ternary Weights as Geometry

For ternary weights {-1, 0, +1}:

| Weight | Geometric Meaning |
|--------|-------------------|
| +1 | Activation is aligned with output direction |
| -1 | Activation is opposite to output direction |
| 0 | Activation is orthogonal (no projection) |

The "multiplication" is actually **routing**:
- +1: route activation to positive accumulator
- -1: route activation to negative accumulator
- 0: route to null (discard)

Final output = positive_accumulator - negative_accumulator

**There is no multiplication. Only selection and summation.**

## Binary Weights: Pure Geometry

Binary weights {-1, +1} reduce to two relationships:
- Same direction
- Opposite direction

The operation becomes: **XNOR + popcount**

This is not a simplification of multiplication. Multiplication was always 
an overcomplication of what is fundamentally a directional relationship.

## Memory Address as Coordinate

The memory address of an activation IS its coordinate in the state space:

```
address 0x1000: dimension 0
address 0x1001: dimension 1
...
address 0x1000 + k: dimension k
```

The value at address k is the magnitude of excitation in dimension k.

**Ground state**: All addresses contain zero. No excitation in any dimension.

**Activated state**: Non-zero values at addresses. Excitations from ground.

## Weights as Routing Tables

A weight matrix is not a matrix of numbers. It is a **routing table**:

```
input_address -> (output_address, sign)
```

For each input dimension k and output dimension n:
- weight[n,k] = +1: route input[k] to output[n] with positive sign
- weight[n,k] = -1: route input[k] to output[n] with negative sign
- weight[n,k] = 0: no route (dimensions are orthogonal)

**Inference is message passing:**
```
for each edge (in_addr, out_addr, sign):
    output[out_addr] += sign * input[in_addr]
```

## The Frozen Transform

Compression and decompression are coordinate transforms. If we **freeze** the 
transform, we do not decompress at runtime - we transform the operation instead.

```
Traditional:
  compressed_weights -> decompress -> operate -> result

Frozen transform:
  compressed_weights -> operate_in_compressed_space -> result
```

For ternary weights with LUT decode:
- The LUT **is** the frozen transform
- vqtbl1q_s8(lut, nibble) applies a precomputed function
- The nibble indexes which transformation to apply

For binary weights:
- **The transform is identity**
- The bit IS the routing decision
- No decode needed

## Shape as Relationship

```
shape = activation_state relative_to weight_state
```

Shape is not a property of data. It is a **relationship** between data and 
the coordinate system that gives it meaning.

The weight defines the coordinate system.
The activation is a point in that system.
The output is the activation re-expressed in a new system.

## Structured Routing

If the routing pattern (weight to address mapping) has structure, we do not need 
to store individual weights. We store the **rule** that generates the routing.

Examples:
- Convolution: routing follows a sliding window pattern
- Attention: routing follows a learned query-key relationship
- FFN: dense routing (all-to-all)

Structured sparsity = routing rules with regularities.

## Hardware Implications

Optimal hardware for this model:

1. **Address-based routing**: Hardware that routes values based on address patterns
2. **Signed accumulation**: Dual accumulators for positive/negative contributions
3. **Bit-level operations**: For binary weights/activations, pure logic gates suffice

The operation `output[n] = sum(sign[n,k] * input[k])` becomes:
```
output[n] = sum(input[k] where positive[n,k]) - sum(input[k] where negative[n,k])
```

This is gather-accumulate, not multiply-accumulate.

## Theoretical Minimum

For a dense ternary matrix [N x K]:
- Information content: N * K * log2(3) = 1.58 NK bits
- Optimal storage: 1.58 bits per weight
- Current 2-bit packing: 21% overhead

For binary weights:
- Information content: NK bits
- Storage: NK bits (optimal)

**You cannot compress below the information content without losing fidelity.**

## Conclusion

Neural networks are coordinate transform machines. Weights define geometry, not 
arithmetic. Activations are excitations from a ground state. Addresses are coordinates.

Inference is resolving where a point lies, given a sequence of reference frame changes.

The "AI" is in the geometry of the frames. The compute is just bookkeeping.

---

## Appendix: Address-Space Ground State

The memory address space itself can define the ground state of computation.

### Definition

Let memory be a function: `M: Address -> Value`

**Ground state**: M(a) = 0 for all addresses a

Any non-ground state is a **perturbation** from ground:
```
delta(a) = M(a) - ground(a) = M(a) - 0 = M(a)
```

The activation tensor is the set of all perturbations from ground.

### Weight as Delta-Transform

A weight matrix W defines how perturbations at input addresses propagate to 
perturbations at output addresses:

```
delta_out[n] = sum over k: W[n,k] * delta_in[k]
```

For ternary W[n,k] in {-1, 0, +1}:
```
delta_out[n] = sum(delta_in[k] where W[n,k]=+1) - sum(delta_in[k] where W[n,k]=-1)
```

This is **perturbation routing**: which input excitations contribute (positively 
or negatively) to each output excitation.

### Implications

1. **Zero activations carry no information** - they are ground state
2. **Sparse activations** = few perturbations from ground = efficient
3. **The address IS the identity** of the perturbation
4. **Weights are perturbation transfer functions**

### Hardware Ground State

In physical hardware:
- Ground = no current, no voltage differential, no charge
- Activation = current, voltage, charge (perturbation from ground)
- Weight = routing topology (fixed in silicon or configurable)

The weight is literally **wires** (connections) and **polarity** (inverters).

For ternary: wire (+1), inverted wire (-1), no wire (0).

This is not metaphor. It is the physical reality of what computation means.
