# NeoGPU C Port - Feature Tracking

## Overview

Port of PicoGPU from Haxe/HashLink to ARM NEON-optimized C for maximum performance.

**Current Performance:** ~990K fps message layer, 177x speedup over original  
**Binary Size:** 73KB (vs 2MB+ HashLink)

---

## Feature Comparison Table

| Category | Feature | Haxe Original | C Port | Status | Notes |
|----------|--------|---------------|--------|--------|-------|
| **Math** | vec4 operations | ✅ | ✅ | ✅ Done | v4_add, sub, dot, cross, normalize, lerp |
| | mat4 operations | ✅ | ✅ | ✅ Done | m4_identity, multiply, translation, rotation, invert, look_at, perspective |
| | Quaternion | ✅ | ✅ | ✅ Done | v4_quat_axis_angle, m4_from_quat |
| | sqrt | ✅ | ✅ | ✅ Done | hs_sqrt |
| | abs | ✅ | ✅ | ✅ Done | hs_abs |
| | floor, ceil | ✅ | ✅ | ✅ Done | hs_floor, hs_ceil |
| | sin, cos, tan | ✅ | ✅ | ✅ Done | hs_sin, hs_cos, hs_tan |
| | exp | ✅ | ✅ | ✅ Done | hs_exp() |
| | log | ✅ | ✅ | ✅ Done | hs_log() |
| | pow | ✅ | ✅ | ✅ Done | hs_pow() |
| | atan2 | ✅ | ✅ | ✅ Done | hs_atan2() |
| | round | ✅ | ✅ | ✅ Done | hs_round() |
| | imin, imax | ✅ | ✅ | ✅ Done | hs_imin(), hs_imax() |
| **State** | OP_SET_SHADER | ✅ | ✅ | ✅ Done | |
| | OP_SET_PARAM | ✅ | ✅ | ✅ Done | Fixed alignment bug |
| | OP_SET_GLOBAL | ✅ | ✅ | ✅ Done | Fixed param index bug |
| | OP_SET_CAMERA | ✅ | ✅ | ✅ Done | |
| | OP_CULL | ✅ | ✅ | ✅ Done | 0=off, 1=back, -1=front |
| | OP_BLEND | ✅ | ✅ | ✅ Done | |
| | OP_DEPTH | ✅ | ✅ | ✅ Done | |
| | OP_COLOR_MASK | ✅ | ✅ | ✅ Done | |
| | OP_CLIP | ✅ | ✅ | ✅ Done | |
| | OP_STENCIL | ✅ | ✅ | ✅ Done | |
| | OP_STENCIL_FUNC | ✅ | ✅ | ✅ Done | |
| | OP_DEPTH_COMPARE | ✅ | ✅ | ✅ Done | |
| | alpha() | ✅ | ✅ | ✅ Done | Just added |
| | setTarget() | ✅ | ✅ | ✅ Done | Now supports depth buffer |
| **Buffers** | loadBuffer(index) | ✅ | ✅ | ✅ Done | 16 buffers supported |
| | getI32, setI32 | ✅ | ✅ | ✅ Done | |
| | getF32, setF32 | ✅ | ✅ | ✅ Done | |
| | getVec4, setVec4 | ✅ | ✅ | ✅ Done | |
| | getMat4, setMat4 | ✅ | ✅ | ✅ Done | |
| | getMat3x4 | ✅ | ✅ | ✅ Done | hs_buffer_set_mat3x4() |
| **Textures** | loadTexture(index) | ✅ | ✅ | ✅ Done | |
| | filter(bool) | ✅ | ✅ | ✅ Done | hs_gpu_texture_filter() |
| | wrap(bool) | ✅ | ✅ | ✅ Done | hs_gpu_texture_wrap() |
| | isDisposed() | ✅ | ❌ | ⚠️ N/A | Needs actual GPU integration |
| **System** | beginFrame | ✅ | ✅ | ✅ Done | hs_gpu_begin_frame() |
| | endFrame | ✅ | ✅ | ✅ Done | hs_gpu_end_frame() |
| **Storage** | loadStorage(name) | ✅ | ❌ | ⚠️ N/A | Requires filesystem |

---

## Completed Features

### Core Message System
- ✅ Message queue with 64-byte aligned structures
- ✅ 5 nodes: Shader, Buffer, Texture, Output, Sound
- ✅ Recording and replay functionality
- ✅ Overflow detection and handling
- ✅ Debug output via `HS_DEBUG=1` compile flag

### Math Library (NEON Optimized)
- ✅ All vec4 operations using ARM NEON SIMD
- ✅ All mat4 operations using ARM NEON SIMD
- ✅ Quaternion creation and conversion
- ✅ Helper functions: clamp, lerp, rnd

### Bug Fixes from Original
- ✅ `v4_cross` - Fixed incorrect shufflevector formula
- ✅ `m4_invert` - Fixed undefined variable references
- ✅ `OP_SET_PARAM` - Fixed unaligned f32 reads
- ✅ `OP_SET_GLOBAL` - Fixed wrong field check
- ✅ `OP_DRAW_INSTANCE` - Fixed field extraction
- ✅ `m4_look_at` - Fixed column/row orientation

---

## Porting Priority

### High Priority (Game Essentials)
1. Input system - dirX, dirY, button, mouseX/Y
2. Time functions - date(), time()
3. Render target with depth buffer
4. alpha() blend mode

### Medium Priority (Polish)
5. Texture filter/wrap modes
6. round, imin, imax math
7. error() and stop()

### Low Priority (Nice to Have)
8. exp, log, pow, atan2
9. loadStorage (needs filesystem)
10. beginFrame/endFrame hooks

---

## Building

```bash
# Normal build (no debug output)
make hs-build

# Debug build (verbose output)
gcc -DHS_DEBUG=1 -O3 ... -o hs_gpu_demo

# Run tests
./hs_gpu_demo

# Run benchmarks
./bench_full        # Full benchmark suite
./bench_full q      # Quiet mode
```

---

## Files

### Core Implementation
- `hs_core.h/c` - Message system, OpCodes, structs
- `hs_nodes.h/c` - Node message handlers
- `hs_gpu.h/c` - High-level GPU API
- `hs_math_neon.h/c` - NEON math library
- `hs_input.h` - Input/controls system
- `hs_buffer.h` - Buffer/Texture data types
- `hs_audio.h` - 4-channel audio system (48KHz)
- `hs_storage.h` - Persistent storage (16 slots × 256B)
- `hs_graphics.h` - GBM/EGL/GLES graphics backend

### Tests & Benchmarks
- `main.c` - Test suite (28 tests)
- `bench_full.c` - Comprehensive benchmarks
