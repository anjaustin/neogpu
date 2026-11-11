# NeoGPU: PicoGPU Feature Port Plan

**Date**: 2026-04-01  
**Status**: P1 COMPLETE
**Purpose**: Complete NeoGPU feature parity with PicoGPU  
**Basis**: PicoGPU Haxe source in `pico/` directory

---

## Part A: Already Ported

| Feature | Status | Notes |
|---------|--------|-------|
| Message-passing architecture | ✅ | HS-OS style implemented in `hs_core.c` |
| Shader, Buffer, Texture, Output nodes | ✅ | `hs_nodes.c` |
| Sound node | ✅ | Basic implementation |
| Vertex/index buffers | ✅ | `hs_gpu.c`, `hs_backend_gles.c` |
| Blending, culling, depth | ✅ | GLES backend |
| Render targets | ✅ | `setTarget()` |
| Instanced drawing | ✅ | `drawInstance()` |
| Matrix/vector/quaternion math | ✅ | NEON-optimized in `hs_math_neon.h` |
| GBM/EGL/GLES2 backend | ✅ | Pi4 native |
| 4-channel audio | ✅ | Basic |
| Input streaming (evdev) | ✅ | `hs_input_stream.c` |
| Input API (high-level) | ✅ | `hs_input.h` - button(), dirX/Y(), mouseX/Y() |
| Persistent storage | ✅ | `hs_storage.h` - slot-based, file-backed |
| **Text Rendering** | ✅ | **GLES textured quad rendering** |
| **Pong + Scores** | ✅ | **Scores rendered on screen + persistent** |

---

## Part B: P1 Completed

### B.1 Text Rendering — COMPLETE

**PicoGPU**: Font atlas (`.fnt` + `.png`), `drawText()` API, UTF-8 support

**NeoGPU Implementation**:
- ✅ `hs_text.h` — Font structure, glyph data, GLES API
- ✅ `hs_text.c` — BMFont XML parser + GLES textured quad renderer
- ✅ `src/medodica_font.atlas` — 128x64 RGBA font atlas (32768 bytes)
- ✅ `src/medodica_font.fnt` — Font metrics from PicoGPU
- ✅ `hs_font_load()` — Loads .fnt + .atlas
- ✅ `hs_font_upload_texture()` — Uploads atlas to GL texture
- ✅ `hs_font_render_text()` — Renders text with textured quads

**Bug Fixed**: BMFont `rect` values (texture position) vs `offset` (render offset) were swapped.

---

### B.2 Input API — COMPLETE

**NeoGPU**: All implemented in `hs_input.h` and `hs_input_stream.c`

**Note**: Pong uses terminal stdin (appropriate for headless/dev). evdev integration available via `hs_input_stream.c`.

---

### B.3 Persistent Storage — COMPLETE

**NeoGPU**: `hs_storage.h` with binary file save/load

**Pong Integration**: High scores save to `/tmp/neogpu_pong_save.bin` on exit, loaded on startup.

---

### Priority 2 — Asset Pipeline

#### B.4 PNG App Distribution
**PicoGPU**: Apps save/load as 640x480 PNG with embedded zlib-compressed data (shaders, buffers, memory)

**Required**:
1. `hs_png.c` / `hs_png.h`
2. libpng for encoding/decoding
3. Embed app state as PNG ancillary chunks
4. `tools/neogpu_pack.c` — pack project → PNG
5. `tools/neogpu_unpack.c` — extract PNG → project
6. zlib compression for compact storage

**Dependencies**: `libpng-dev`, `zlib-dev`

---

#### B.5 WAV Audio Import
**PicoGPU**: Import WAV → auto-convert to 48KHz F32 buffer

**Required**:
1. `hs_audio_loader.c` / `hs_audio_loader.h`
2. WAV file parser (manual or libsndfile)
3. Resampling to 48KHz if needed (linear interpolation)
4. `hs_load_wav(filename)` → Buffer handle
5. Integrate into memory system

---

### Priority 3 — Sound

#### B.6 Shader-Driven Sound Synthesis
**PicoGPU**: Assign fragment shader to audio channel, shader writes `sound` variable (-1 to 1)

```haxe
setChannel(channel:Int, shader:Int)  // channel 0-3, shader index
// shader declares:
//   @global var time : Float
//   var sound : Float
// shader sets sound = sin(time * frequency)
```

**Current**: `hs_ml_audio.c` exists - verify state

**Required**:
1. Extend SoundNode to support synthesis shaders
2. Per-channel sound buffer generation
3. Double-buffered 48KHz playback via ALSA
4. `setChannel()` API
5. 4 independent channels

---

### Priority 4 — Nice to Have

#### B.7 Advanced Blend Modes
**PicoGPU Blend enum**: ONE, ZERO, SRC_ALPHA, ONE_MINUS_SRC_ALPHA, DST_ALPHA, ONE_MINUS_DST_ALPHA, etc.

**Verify**: GLES backend has complete blend state

**Files**: `src/hs_backend_gles.c`, `include/hs_graphics.h`

---

#### B.8 Stencil Operations
**PicoGPU**:
```haxe
stencil(op, fail, pass, front)
stencilFunc(comp, reference, readMask, writeMask)
```

**Verify**: GLES2 stencil implementation completeness

---

#### B.9 Clip Rectangle
**PicoGPU**: `clip(x, y, width, height)` — scissor test

**Required**: Implement via `glScissor()` in GLES backend

---

#### B.10 Code Editor / HScript
**PicoGPU**: Built-in editor with HScript interpreter for runtime code editing

**Decision**: Skip HScript interpreter

**Alternative**: `tools/neogpu_editor.c` — SDL2-based shader editor with hot-reload via IPC to running process

---

## Part C: Implementation Order

```
Phase 1 (Priority 1 — Critical):
  B.1  Text rendering       → enables UI, debug
  B.2  Input API            → enables interactivity  
  B.3  Storage              → enables save games

Phase 2 (Priority 2 — Asset Pipeline):
  B.4  PNG save/load         → app distribution
  B.5  WAV import            → audio assets

Phase 3 (Priority 3 — Sound):
  B.6  Shader sound          → synthesis audio

Phase 4 (Priority 4 — Polish):
  B.7  Blend modes           → verify completeness
  B.8  Stencil               → verify completeness
  B.9  Clip rectangle         → scissor test
  B.10 Editor (optional)      → dev UX
```

---

## Part D: New Files Created

```
include/
  hs_text.h              # Text rendering API ✅ NEW

src/
  hs_text.c              # BMFont parser + atlas loader ✅ NEW
  medodica_font.atlas    # 128x64 RGBA font atlas ✅ NEW
  stb_image.h            # Minimal PNG loader (incomplete) ⚠️

tools/
  neogpu_pack.c          # Pack project to PNG ❌
  neogpu_unpack.c        # Extract PNG to project ❌
  neogpu_editor.c        # Shader editor (optional) ❌

tests/
  test_text.c            # Font loading, text rendering ❌
  test_png.c             # Roundtrip save/load ❌
  test_audio_loader.c    # WAV import ❌
  test_sound.c           # 4-channel synthesis ❌
```

---

## Part E: Files to Modify

```
include/
  hs_input.h             # ✅ Already complete - needs Pong integration
  hs_audio.h             # Extend with setChannel ❌
  hs_storage.h           # ✅ Already complete - needs Pong integration

src/
  hs_nodes.c             # Add TextNode, extend SoundNode ❌
  hs_backend_gles.c      # Add text rendering (textured quads) ❌
  hs_gpu.c               # Add text drawing integration ❌
  hs_text.c              # Add GLES text rendering ❌

tools/
  neogpu_pong.c          # Integrate hs_input_stream + hs_storage ❌

Makefile                 # Add libpng, libsndfile deps ❌
```

---

## Part F: Build Dependencies

```bash
apt install libpng-dev zlib-dev libsndfile-dev
```

```makefile
# Add to Makefile
LIBS += -lpng -lz -lsndfile
CFLAGS += $(shell pkg-config --cflags libpng sndfile)
```

---

## Part G: Testing

| Feature | Test | Status |
|---------|------|--------|
| Text | `neogpu_pong` | ✅ Renders "P:X AI:Y" and "SCORE" on screen |
| Input | `neogpu_pong` | ✅ Terminal stdin (evdev available in hs_input_stream) |
| Storage | `neogpu_pong` | ✅ High scores persist to /tmp/neogpu_pong_save.bin |
| PNG | `test_png.c` | ❌ Not yet created |
| WAV | `test_audio_loader.c` | ❌ Not yet created |
| Sound | `test_sound.c` | ❌ Not yet created |
| Graphics | `test_06_raycast` | ✅ PASSES at 376 FPS |

---

## Part H: P1 Red-Team Findings — FIXED

### Issues Found and Resolved

1. **Text Rendering**: `hs_gpu_draw_text()` only logged, didn't render
   - **Fix**: Created `hs_text.c` with full GLES textured quad rendering
   - **Bug**: BMFont `rect` vs `offset` confusion caused wrong UV coordinates

2. **Pong Storage**: High scores not persisted
   - **Fix**: Added `load_scores()` / `save_scores()` to Pong
   - Saves to `/tmp/neogpu_pong_save.bin` with magic header

3. **Pong Input**: Terminal stdin inappropriate for GBM display
   - **Decision**: Keep terminal stdin for dev; evdev available via `hs_input_stream.c`

### Files Modified

- `tools/neogpu_pong.c` — Added score rendering and persistence
- `src/hs_text.c` — GLES text renderer (new)
- `include/hs_text.h` — Text API (new)
- `src/medodica_font.atlas` — Font atlas (new)
- `src/medodica_font.fnt` — Copied from pico/res/style/
