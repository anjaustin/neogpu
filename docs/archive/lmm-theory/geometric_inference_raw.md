# RAW: Geometric Neural Network Inference

## Stream of Consciousness

We started optimizing a ternary GEMM kernel. Got it from 0.29 GOPS to 25 GOPS. 85x speedup. 
But then we hit memory bandwidth - 3.8 GB/s on Pi4. That is the wall.

Tried to compress further. 5-per-byte packing (20% smaller) costs 5x in decode overhead. 
Not worth it. Binary is 2x smaller AND 3x faster because no decode needed.

Then the insight: ternary weights are not numbers. They are routing decisions.
+1 means "add this activation to output"
-1 means "subtract this activation from output"
0 means "ignore this activation"

There is no multiplication. The weight just says WHERE the activation goes.

The LUT decode we are doing - that IS the operation. The nibble indexes into what 
transform to apply. We already froze the transform into the LUT.

For binary, the transform is identity. The bit IS the routing. XNOR + popcount.

Then: what if the memory address IS the coordinate? Address 0x1000 = dimension 0.
The value at that address = magnitude of excitation in that dimension.
Ground state = all zeros. Any non-zero = perturbation from ground.

Weights become routing tables: input_address -> (output_address, sign)
Inference = message passing on a sparse signed graph.

If the routing has structure (convolution, attention patterns), we do not store edges.
We store the RULE that generates edges.

shape = activation_state relative_to weight_state

The weight does not DO anything. It DEFINES the coordinate system.
The activation just exists. The operation resolves position in that system.

## Questions Arising

- Can we build hardware that routes based on addresses natively?
- What is the minimal representation for a routing table?
- Is sparse routing always more efficient than dense?
- How does this map to actual transistor layouts?
- Can backprop work in this frame? Gradients are routing changes?
- What happens when both weights AND activations are binary?

## First Instincts

This feels like a reframe that could change everything or nothing.
If it is real, it means the compute abstraction we have is wrong.
Multiply-accumulate is not the primitive. Route-accumulate is.
The hardware should be a crossbar switch, not an ALU array.
