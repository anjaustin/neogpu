# LMM: BitNet 1.58b Inference on Pi4 (RAW)

## What is BitNet 1.58b?
- Microsoft research: "BitNet: Scaling 1-bit LLMs"
- 1.58 bits per weight (ternary: -1, 0, +1)
- 1.58B parameters
- Designed for CPU-efficient inference
- Paper: https://arxiv.org/abs/2402.17762

## Why Pi4?
- 4 cores @ 1.5GHz = ~6 GFLOPS per core
- NEON SIMD = 128-bit = 4 FP32 ops/cycle
- ~24 GFLOPS theoretical peak
- BitNet uses INT8/INT4 - can do 16 INT8 ops/cycle with dot products
- Could be 10-20 tokens/sec?

## The Challenge
- 1.58B params = 1.58GB (INT8) or ~800MB (INT4)
- Matrix multiplications: all layer weights × input
- KV cache management
- Context length
- Tokenizer

## What's the same as NeoGPU?
- Message-passing for async inference
- NEON for matrix ops (already have some)
- Could integrate into message queue

## What's different?
- LLM is VERY different from graphics
- State machine for decoding
- Vocabulary (50K+ tokens)
- No "frames" - just continuous text

## Questions
- Run via messages or standalone?
- How to handle memory?
- GGUF format or custom?
- Output: display? serial? network?
