# NeoGPU Remediation Plan

**Date**: 2026-04-01  
**Status**: BROKEN - Multiple stubbed/unimplemented components

---

## Executive Summary

The NeoGPU message-passing substrate works. The GLES3 backend works for basic clear/draw. Everything else (text, shaders, audio, input, storage) is stubbed or broken. This plan addresses each gap.

---

## Gap Analysis

| Component | Status | Severity | File(s) |
|-----------|--------|----------|---------|
| Text rendering | BROKEN | HIGH | `hs_backend_gles.c` |
| Shader pipeline | BROKEN | HIGH | `hs_backend_gles.c`, `hs_nodes.c` |
| Audio output | STUB | HIGH | `hs_audio.h` |
| Input polling | ✅ IMPLEMENTED | HIGH | `hs_input_stream.h`, `hs_input_stream.c`, `hs_nodes.c` |
| Storage persistence | UNUSED | MEDIUM | `hs_storage.h` |
| V3D/QPU compute | INCOMPLETE | LOW | `test_v3d_tfu.c`, `qpu_asm.h` |

---

## 1. Text Rendering — IMPLEMENT

**Problem**: `hs_gpu_draw_text()` sends string payload through message layer, but GLES backend ignores it and draws a texture quad from slot 0. No font system exists.

**Required Work**:
1. Create `hs_font.h` with font loading and glyph rasterization
2. Implement bitmap font atlas generation (or stb_truetype subset)
3. Render text string to texture atlas on `hs_gpu_draw_text()`
4. Bind resulting texture and draw textured quad

**Implementation**:
```c
// New file: include/hs_font.h
typedef struct {
    HSGLTexture* atlas;      // Glyph texture atlas
    u8 glyphs[256];         // Glyph data
    u8 widths[256];         // Glyph widths
} HSFont;

bool hs_font_load(HSFont* font, const char* ttf_path, u32 size);
void hs_font_render_text(HSFont* font, const char* text, HSGLTexture* out);
```

**File**: `src/hs_font.c` (new)

---

## 2. Shader Pipeline — IMPLEMENT

**Problem**: `hs_backend_gles.c` uses ONE hardcoded color shader for ALL draws. `shader_node.current_shader` is stored but never read. No shader compilation, no program management.

**Required Work**:
1. Create shader program registry in GLES backend
2. Implement `hs_gpu_set_shader()` to compile/lookup programs
3. Modify `gles_execute()` to select program based on `current_shader`
4. Store compiled programs by shader ID

**Current broken code** (`hs_backend_gles.c:319`):
```c
glUseProgram(b->program);  // Always uses hardcoded b->program
```

**Fix**: Look up program by `frame->shader_state->current_shader`:
```c
GLuint prog = gles_get_program(b, frame->shader_state->current_shader);
glUseProgram(prog);
```

**Implementation**:
- Add `b->programs[16]` array to store compiled GL programs
- Add `hs_gpu_load_shader()` to compile and register GLSL source
- Modify backend to query `HSRenderCmd` for shader ID

---

## 3. Audio Output — IMPLEMENT

**Problem**: `hs_audio.h` provides buffer management but no actual audio driver. No ALSA, PulseAudio, or any playback.

**Required Work**:
1. Create `hs_audio_out.h` with ALSA PCM handle
2. Implement `hs_audio_init()` with snd_pcm_open()
3. Add audio thread that pulls from `HSAudio` buffers
4. Wire up `hs_audio_set_channel()` to actual ALSA channel

**Implementation**:
```c
// New file: include/hs_audio_out.h
#include <alsa/asoundlib.h>

typedef struct {
    snd_pcm_t* playback;
    pthread_t thread;
    HSAudio audio;
    bool running;
} HSAudioOut;

bool hs_audio_out_init(HSAudioOut* out, const char* device);
void* hs_audio_out_thread(void* arg);
void hs_audio_out_shutdown(HSAudioOut* out);
```

**File**: `src/hs_audio_out.c` (new)

---

## 4. Input Polling — IMPLEMENT

**Problem**: `hs_input.h` defines structures but no platform-specific event polling. No evdev, SDL, or any input source implementation.

**Required Work**:
1. Create `hs_input_evdev.h` for Linux input
2. Implement `hs_input_evdev_init()` to open `/dev/input/event*`
3. Poll and parse EV_KEY, EV_REL, EV_ABS events
4. Update `HSInput` struct fields

**Implementation**:
```c
// New file: include/hs_input_evdev.h
typedef struct {
    int fd;
    HSInput input;
} HSEvdev;

bool hs_evdev_init(HSEvdev* dev, const char* device);
bool hs_evdev_poll(HSEvdev* dev);
void hs_evdev_shutdown(HSEvdev* dev);
```

**File**: `src/hs_input_evdev.c` (new)

---

## 5. Storage Persistence — FIX

**Problem**: `hs_storage_save()` and `hs_storage_load_file()` exist but are never called. Game saves don't persist.

**Required Work**:
1. Call `hs_storage_load_file()` on game startup
2. Call `hs_storage_save()` on game exit or periodic autosave
3. Wire to `atexit()` or explicit save trigger

**Implementation**:
- In `neogpu_pong.c` main loop, add:
```c
atexit(() => hs_storage_save(&store, "save.dat"));
hs_storage_load_file(&store, "save.dat");
```

**Note**: This is an integration issue, not missing API.

---

## 6. V3D/QPU Compute — COMPLETE

**Problem**: `test_v3d_tfu.c` has "TODO: Implement actual TFU execution". VPM operations in `qpu_asm.h` are stubs.

**Required Work**:
1. Implement VPM (Video Processor Memory) load/store operations
2. Implement TFU (Texture Fetch Unit) operations
3. Add QPU program submission to V3D

**Files**:
- `include/qpu_asm.h` - Complete VPM operations
- `tests/test_v3d_tfu.c` - Implement TFU execution

---

## Implementation Order

```
Phase 1 (Foundation):
  1. Shader pipeline     → Enables actual rendering
  2. Text rendering      → Enables debug output, UI

Phase 2 (Interactivity):
  3. Input polling      ✅ IMPLEMENTED (with hotplug)
  4. Audio output       → Enables sound

Phase 3 (Polish):
  5. Storage persistence → Enables save games
  6. V3D/QPU compute    → Enables GPU ML inference
```

---

## Success Criteria

- [ ] `hs_gpu_draw_text("hello")` renders readable text
- [ ] `hs_gpu_set_shader(1)` selects different program
- [ ] Sound plays when `hs_audio_set_channel()` called
- [x] Arrow keys move paddle in Pong (needs integration)
- [ ] Game saves persist across restarts
- [ ] BitNet inference runs on V3D QPU (not just CPU)

---

## Files Created

| File | Purpose | Status |
|------|---------|--------|
| `include/hs_input_stream.h` | Input streaming API | ✅ Done |
| `src/hs_input_stream.c` | evdev reader + SPSC queue + hotplug | ✅ Done |
| `tests/test_input_stream.c` | Test utility | ✅ Done |
| `include/hs_nodes.h` | NODE_INPUT + InputState | ✅ Done |
| `src/hs_nodes.c` | input_node_* implementations | ✅ Done |

---

## Files to Create (Remaining)

| File | Purpose |
|------|---------|
| `src/hs_font.c` | Font atlas + text rendering |
| `include/hs_font.h` | Font API |
| `src/hs_audio_out.c` | ALSA playback driver |
| `include/hs_audio_out.h` | Audio output API |

---

## Files to Modify

| File | Change |
|------|--------|
| `src/hs_backend_gles.c` | Shader program registry, text rendering, shader selection |
| `src/hs_gpu.c` | Add `hs_gpu_load_shader()` |
| `include/hs_render.h` | Add shader ID to `HSRenderCmd` |
| `tools/neogpu_pong.c` | Integrate input stream, audio, storage |
