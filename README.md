# NeoGPU - ARM NEON Optimized GPU

High-performance message-passing GPU layer with ARM NEON SIMD optimizations.

## Performance

- **Message Layer: 1,000,000+ fps** (single-core ARM Cortex-A72)
- **177x faster** than original HashLink version
- **73KB binary** (vs 2MB+ HashLink runtime)
- **270 FPS** on Raspberry Pi 4 (ray casting spheres test)

## New Features (vs Original)

- **Audio**: 4-channel sound at 48KHz (`hs_audio.h`)
- **Storage**: Persistent storage with file I/O (`hs_storage.h`)

## Quick Start

```bash
# Build
make build

# Run demo
make run
```

## Building for Different Targets

### Native ARM (Cortex-A72)
```bash
make build
```

### Cross-compile for Raspberry Pi
```bash
make CC=aarch64-linux-gnu-gcc STRIP=aarch64-linux-gnu-strip build
```

## Project Structure

```
src/
  hs_core.c       - Message queue, OpCodes, system core
  hs_nodes.c      - Node message handlers (Shader, Buffer, Texture, Output, Sound)
  hs_gpu.c        - High-level GPU API
  main.c          - Demo/test suite
  benchmark.c     - Benchmarks

include/
  hs_core.h       - Core header
  hs_nodes.h      - Node headers
  hs_gpu.h        - GPU API header
  hs_math_neon.h  - NEON-optimized vec4/mat4 math
  hs_buffer.h     - Buffer/Texture data types
  hs_input.h      - Input/controls system
  hs_audio.h      - 4-channel audio system (48KHz)
  hs_storage.h    - Persistent storage (16 slots × 256B)
  hs_graphics.h   - GBM/EGL/GLES graphics backend

tests/
  test_01_clear.c     - Clear screen test
  test_02_triangle.c   - Single triangle
  test_03_instancing.c - Hardware instancing
  test_04_blending.c   - Alpha blending
  test_05_cube3d.c     - 3D rotating cube
  test_06_raycast.c    - Ray casting spheres (270 FPS on Pi)
```

## API Reference

### GPU Setup
```c
HSGpu gpu;
hs_gpu_init(&gpu);
```

### Drawing
```c
hs_gpu_clear(&gpu, v4_make(0, 0, 0, 1));  // Clear with color
hs_gpu_set_shader(&gpu, 0);                 // Set shader
hs_gpu_set_param(&gpu, 0, v4_one());       // Set param
hs_gpu_set_camera(&gpu, m4_identity());     // Set camera
hs_gpu_load_buffer(&gpu, 0);                // Load buffer
hs_gpu_draw(&gpu, 0);                       // Draw
```

### State
```c
hs_gpu_cull(&gpu, 1);           // Back-face culling
hs_gpu_blend(&gpu, 1, 0);       // No blending
hs_gpu_alpha(&gpu, true);       // Alpha blending
hs_gpu_depth(&gpu, true);       // Enable depth test
hs_gpu_color_mask(&gpu, 0x0F);  // RGBA mask
hs_gpu_clip(&gpu, 0, 0, 640, 480);
```

### Textures
```c
hs_gpu_load_texture(&gpu, 0);
hs_gpu_set_target(&gpu, 0, 0);           // Texture + depth
hs_gpu_show_texture(&gpu, 0);
hs_gpu_texture_filter(&gpu, 0, true);    // Linear filtering
hs_gpu_texture_wrap(&gpu, 0, true);      // Repeat wrap
```

### Recording/Replay
```c
hs_gpu_start_recording(&gpu);
hs_gpu_frame_begin(&gpu);
// ... draw calls ...
hs_gpu_frame_end(&gpu);
hs_gpu_present(&gpu);
u32 msg_count = hs_gpu_stop_recording(&gpu);

// Replay later
hs_gpu_replay(&gpu, gpu.log_buffer, msg_count);
```

Notes:

- Recording is channel-filtered by default (P0: render channel only). See `HSSystem.record_mask` and `hs_set_record_mask()`.
- Use `hs_gpu_fence(&gpu, CHAN_RENDER, cid)` to request an apply-time fence result via `OP_RESULT`.
- Use `OP_QUERY_STATS` / `OP_QUERY_FABRIC` to retrieve runtime fabric stats via `OP_RESULT`.

### Input
```c
HSInput input;
hs_input_init(&input);

// Poll each frame
hs_input_tick(&input);

f32 dx = hs_dir_x(&input);    // -1 to 1
f32 dy = hs_dir_y(&input);    // -1 to 1
bool btn = hs_button(&input);
f32 mx = hs_mouse_x(&input);
f32 t = hs_time(&input);      // Frame time
```

### Audio (4 channels, 48KHz)
```c
HSAudio audio;
hs_audio_init(&audio);

// Set channel 0 to shader 1
hs_audio_set_channel(&audio, 0, 1);

// Get audio buffer for channel 0
float* buf = hs_audio_get_buffer(&audio, 0);
if (buf) {
    // Fill with samples (3000 samples per buffer)
    for (int i = 0; i < HS_AUDIO_BUFFER_SIZE; i++) {
        buf[i] = sinf(i * 0.01f);  // Simple sine wave
    }
}

// Advance to next buffer
hs_audio_advance(&audio, 0);

// Stop channel
hs_audio_stop(&audio, 0);
```

### Storage (16 slots × 256 bytes)
```c
HSStorage store;
hs_storage_init(&store);

// Load or create slot
HSStorageSlot* slot = hs_storage_load(&store, "savegame1");

// Read/write data
u8 val = hs_storage_get_u8(&store, "savegame1", 0);
hs_storage_set_u8(&store, "savegame1", 0, 42);

f32 fval = hs_storage_get_f32(&store, "savegame1", 4);
hs_storage_set_f32(&store, "savegame1", 4, 3.14f);

// Save to file
hs_storage_save(&store, "save.dat");

// Load from file
hs_storage_load_file(&store, "save.dat");
```

### Graphics (GBM/EGL/GLES for Pi display)
```c
HSGraphics gfx;
hs_graphics_init(&gfx);  // Opens DRM, GBM, EGL

// Create texture from buffer
HSTexture* tex = hs_graphics_create_texture(&gfx, 0, 256, 256, pixel_data);

// Check if texture is disposed
bool disposed = hs_graphics_texture_disposed(&gfx, 0);

// Clear and present
hs_graphics_clear(&gfx, 0.0f, 0.0f, 0.0f, 1.0f);
hs_graphics_present(&gfx);

// Cleanup
hs_graphics_finish(&gfx);
```

## Math Library

All math uses ARM NEON SIMD for maximum performance:

```c
vec4 v1 = v4_make(1, 2, 3, 4);
vec4 v2 = v4_make(5, 6, 7, 8);

vec4 sum = v4_add(v1, v2);
vec4 diff = v4_sub(v1, v2);
f32 dot = v4_dot(v1, v2);
vec4 cross = v4_cross(v1, v2);
vec4 norm = v4_normalize(v1);
vec4 lerp = v4_lerp(v1, v2, 0.5f);

mat4 m1 = m4_identity();
mat4 m2 = m4_translation(1, 2, 3);
mat4 m3 = m4_multiply(m1, m2);
mat4 inv = m4_invert(m3);
```

## Constants

```c
#define HS_FPS      60
#define HS_WIDTH    640
#define HS_HEIGHT   480
#define HS_MAX_NODES       16
#define HS_MAX_MSG_LOG     65536
#define HS_QUEUE_SIZE      256
```

## Debug Mode

Enable verbose debug output:

```bash
gcc -DHS_DEBUG=1 -O3 -Iinclude -c src/main.c -o main.o
...
```

## Testing

### Core Tests (x86/ARM)
```bash
make build && ./neogpu_demo
```

### Graphics Tests (Raspberry Pi with DRM/GBM/EGL/GLES)

One-liner to compile and run any test (replace XX with 01-06):
```bash
make build && gcc -O3 -march=armv8.2-a+fp16+simd -mtune=cortex-a72 -Iinclude -c tests/test_XX_raycast.c -o /tmp/test.o && gcc src/hs_core.o src/hs_gpu.o src/hs_nodes.o /tmp/test.o -o /tmp/test -lm -lGLESv2 -lgbm -ldrm -lEGL && sudo /tmp/test
```

**Available tests:**
| Test | Description |
|------|-------------|
| test_01_clear | Clear screen |
| test_02_triangle | Single triangle |
| test_03_instancing | Hardware instancing |
| test_04_blending | Alpha blending |
| test_05_cube3d | 3D rotating cube |
| test_06_raycast | Ray casting spheres (**270 FPS**) |

Run specific test:
```bash
# Test 06 (raycast)
make build && gcc -O3 -march=armv8.2-a+fp16+simd -mtune=cortex-a72 -Iinclude -c tests/test_06_raycast.c -o /tmp/test.o && gcc src/hs_core.o src/hs_gpu.o src/hs_nodes.o /tmp/test.o -o /tmp/test -lm -lGLESv2 -lgbm -ldrm -lEGL && sudo /tmp/test
```

Test results on Raspberry Pi 4:
- test_01-05: All working
- test_06 (raycast): **270 FPS**

## Features

| Feature | Status |
|---------|--------|
| Message queue (64-byte aligned) | ✅ |
| 5 nodes (Shader, Buffer, Texture, Output, Sound) | ✅ |
| Recording & replay | ✅ |
| Overflow detection | ✅ |
| All vec4/mat4 ops (NEON) | ✅ |
| Input system | ✅ |
| Alpha blending | ✅ |
| Render targets + depth | ✅ |
| Texture filter/wrap | ✅ |
| Audio system (4 channels, 48KHz) | ✅ |
| Storage (16 slots × 256B, file I/O) | ✅ |
| Graphics (GBM/EGL/GLES for Pi display) | ✅ |
| **Test suite** (6 graphics tests) | ✅ |

## Memory Usage

### Runtime Memory

| Component | Size |
|-----------|------|
| Full HSGpu (with buffers) | 1,055 KB |
| Core system only (no logs) | 32 KB |
| Per-frame bandwidth @60fps | 1.9 KB/frame |

### Binary Size

| Section | Size |
|--------|------|
| text (code) | 24 KB |
| data | 1 KB |
| bss (buffers at runtime) | 2.1 MB |
| **Total** | **~25 KB** (stripped) |

## Platform Requirements

- ARM Cortex-A72 (or compatible ARMv8.2-a+)
- GCC with NEON support
- Linux (tested)

## Original

This is a standalone C implementation inspired by [PicoGPU](https://github.com/ncannasse/picogpu) by Nicolas Cannasse.
