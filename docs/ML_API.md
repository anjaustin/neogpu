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

## Routing Abstraction (hs_ml_routing.h / hs_ml_routing.c)

Unified dispatch for all weight formats:

    void hs_ml_route(s32* output, const void* input, const HSRouteDesc* route, u32 M);
    void hs_ml_route_mt(s32* output, const void* input, const HSRouteDesc* route, u32 M, int num_threads);
    int  hs_ml_route_optimal_threads(const HSRouteDesc* route, u32 M);

Supported formats (HSRouteFormat):
  - HS_ROUTE_BINARY:       XNOR + popcount, ~28 GOPS 1T
  - HS_ROUTE_TERNARY_2BIT: LUT decode + vmlal, ~10 GOPS 1T / ~24 GOPS 3T
  - HS_ROUTE_TERNARY_INT8: (stub)
  - HS_ROUTE_SPARSE:       (stub)

Thread-count heuristic (measured on Pi4 Cortex-A72):
  - ops < 3M: 1 thread
  - ops >= 3M: 3 threads
  - 2T never cleanly beats both 1T and 3T

Red-teamed: zero overhead vs direct kernel calls (<1% delta).

## Message-Passing Layer (hs_ml_msg.h / hs_ml_msg.c)

Inference as message routing (adapted from NeoGPU graphics fabric):

    MLSystem  - owns per-channel per-producer SPSC queues
    MLMsg     - 128-byte message (layer, op, format, dimensions, pointers)
    MLChannel - PREFILL (throughput), DECODE (latency), TELEM (droppable)

Key APIs:
    ml_sys_submit()         - enqueue a message
    ml_sys_step()           - drain channels in priority order, dispatch to kernels
    ml_sys_capture_start()  - record message stream for replay
    ml_sys_replay()         - deterministic replay

## Ternary Inference (hs_ml_infer.h / hs_ml_infer.c)

End-to-end transformer forward pass using ternary GEMM kernels.

### Model struct: HSMLTernary
  - Per-layer ternary weights (uint8_t* 2-bit packed + float* scales)
  - GQA support (num_heads, num_kv_heads, head_dim)
  - Tokenizer metadata (vocab, merges, BOS/EOS/PAD)
  - Pre-allocated scratch buffers (no hot malloc)

### Layer forward: hs_mlt_layer_forward()
  hidden -> RMSNorm -> quantize -> Q/K/V GEMM -> RoPE -> attention -> attn_sub_norm
  -> O GEMM -> residual -> RMSNorm -> quantize -> gate/up GEMM -> ReLU^2 * up
  -> ffn_sub_norm -> down GEMM -> residual

### Session API: HSMLTernarySession
    hs_mlt_session_init()   - allocate persistent KV caches
    hs_mlt_prefill()        - process prompt, fill caches
    hs_mlt_decode()         - one decode step, O(ctx*hd) attention
    hs_mlt_session_reset()  - clear caches for new sequence

### GGUF Loader: hs_mlt_load_gguf()
  Supports:
  - GGUF v2/v3
  - llama.* and bitnet-b1.58.* metadata prefixes
  - F32, F16, I8 tensor types (auto-converted to float32)
  - I2_S (type 36): native NEON kernel, no conversion to centered ternary
  - Tokenizer vocab/merges/BOS/EOS from GGUF metadata

### Native I2_S Kernel: i2s_proj_neon()
  ARM NEON kernel for raw I2_S weight format:
  - Group layout: group_idx=j/16, group_pos=j%16, MSB-first within groups
  - Centered correction: dot(raw, x) - sum(x) to map {0,1,2} -> {-1,0,+1}
  - Red-teamed: NEON matches scalar reference at all sizes (K=64..6912)

## Test Coverage

Total verified tests: 60+
  - Core redteam: 16/16
  - GEMM correctness: 6/6
  - Ternary kernel: 7/7
  - FFN + RMSNorm: 3/3
  - KV cache + attention + RoPE: 5/5
  - Tokenizer: 4/4
  - Message layer: 6/6
  - Inference: 22/22
  - Session: 19/19
  - GGUF loader: 17/17
  - Real 2B-4T: loads, finite logits, generates tokens
  - I2_S NEON vs scalar: 9/9
