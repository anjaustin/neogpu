# NeoGPU Game Engine - Product Requirements Document

## 1. Overview

**Project:** NeoGPU - ARM NEON Optimized Game Engine  
**Goal:** Complete the message-passing abstraction layer into a fully functional game engine  
**Target Platforms:** Raspberry Pi 4 (primary), other ARM Linux devices  

## 2. Current State

### ✅ Implemented
| Component | Status | Details |
|-----------|--------|---------|
| Message Queue | ✅ | 1M+ ops/sec, 256 slots, 64B aligned |
| NEON Math | ✅ | vec4, mat4, splines (Bezier2/Bezier3, Catmull-Rom, Hermite) |
| GLES3/DRM Backend | ✅ | DRM/GBM/EGL init + basic helpers; used by graphics tests |
| Storage | ✅ | 16 slots × 256B, file I/O |
| Audio Buffers | ✅ | 4 channels @ 48KHz (buffer generation only) |
| Graphics Tests | ✅ | 6 direct-GLES tests (triangle, cube, raycast @ ~270 FPS on Pi4) |

### ⚠️ Stub / Incomplete
| Component | Status | Gap |
|-----------|--------|-----|
| Message → Backend Execution | Stub | Nodes mostly track state/log; not yet issuing real GLES draws via the message stream |
| Async Ops (Threading) | Stub | Background task queue exists (`hs_async`); task ownership/results and real work paths need completion |
| Mesh Renderer | Stub | No mesh data path + built-in shaders wired through message layer |
| Text Rendering | Stub | `hs_gpu_draw_text` logs only; no font load/raster/atlas |
| Audio Playback | Stub | Buffers exist; no ALSA/PulseAudio output device |
| Input Polling | Stub | `HSInput` exists; no evdev/SDL2 polling implementation |
| Window System (Desktop) | Stub | DRM path exists; no SDL2/GLFW window/context path |
| Game Loop | Missing | No standard update/render loop |

## 3. Requirements

### 3.0 Core Hardening (Priority: HIGH)

**Description:** Stabilize the message ABI, payload rules, recording/replay, and backend boundaries so engine features (mesh, text, audio out, real input) can be built without breaking determinism or portability.

**Requirements:**
- **Message ABI spec**: define destination node, argument encoding, payload layout/size, and valid ranges for every opcode.
- **Typed pack/unpack helpers** for all opcodes to avoid ad-hoc byte packing.
- **Unified payload allocation**: a single allocator/copy path used by all message producers; explicit per-op truncation/validation behavior.
- **Safe recording/replay**: capture mode deep-copies payload bytes so replay is self-contained; replay validates message stream (opcode, payload length, destination).
- **Backend boundary**: define a backend interface that consumes decoded node state / render intent; keep nodes as “decode + state update”, keep platform calls in the backend.
- **System op routing**: define a consistent home for `OP_ERROR`, `OP_TRACE`, `OP_STOP` (system node or system logging facility).
- **Validation + tests**: debug-time validator + unit tests for pack/unpack + replay determinism + malformed message rejection.

**Technical Approach:**
```c
// Examples (names TBD)
bool hs_validate_message(const HSSystem* sys, const Message* msg);
bool hs_payload_alloc_and_copy(HSSystem* sys, const void* data, u32 len, u16* out_idx, u32* out_len);

// Backend interface (interface only)
typedef struct HSBackend HSBackend;
```

### 3.1 Mesh Renderer (Priority: HIGH)

**Description:** Built-in shader and geometry system for rendering 3D/2D graphics.

**Requirements:**
- Built-in vertex shader (MVP transform, basic lighting)
- Built-in fragment shader (texture sampling, basic lighting)
- Mesh data structure (vertices, normals, UVs, indices)
- Render functions: `hs_mesh_draw()`, `hs_mesh_draw_instanced()`
- Support for: triangles, lines, points

**Technical Approach:**
```c
typedef struct {
    float* positions;    // x,y,z,w
    float* normals;      // x,y,z
    float* uvs;         // u,v
    float* colors;     // r,g,b,a
    u32*   indices;
    u32    vertex_count;
    u32    index_count;
    GLuint vbo;
    GLuint vao;
} HSMesh;
```

### 3.2 Font System (Priority: HIGH)

**Description:** TrueType font rendering for text display.

**Requirements:**
- Load TTF font file from storage
- Rasterize glyphs to texture atlas
- Render text with: size, color, alignment
- Support for: ASCII + extended characters
- Cache glyphs for performance

**Technical Approach:**
```c
typedef struct {
    GLuint atlas;           // Texture atlas
    u32    atlas_w;        // Atlas width
    u32    atlas_h;        // Atlas height
    u32    glyphs[256];    // Glyph data
    u8     loaded;
} HSFont;

void hs_font_load(HSFont* font, const char* ttf_path, u32 size);
void hs_font_draw(HSFont* font, const char* text, float x, float y, float size, u32 color);
```

### 3.3 Audio Playback (Priority: MEDIUM)

**Description:** Real-time audio output via ALSA/PulseAudio.

**Requirements:**
- Initialize ALSA/PulseAudio device
- Playback from HSAudio buffers
- Support for: 4 channels, 48KHz, 16-bit PCM
- Volume control per channel
- Low-latency mode (< 20ms)

**Technical Approach:**
```c
typedef struct {
    snd_pcm_t* playback_handle;
    u32        sample_rate;
    u32        channels;
    u8         initialized;
} HSAudioDevice;

esp_err_t hs_audio_init(HSAudioDevice* dev);
esp_err_t hs_audio_play(HSAudioDevice* dev, HSAudio* audio);
```

### 3.4 Input System (Priority: MEDIUM)

**Description:** Unified input from keyboard, mouse, gamepad.

**Requirements:**
- Poll evdev for Linux
- Support: keyboard, mouse, gamepad
- Map to standard game input (dir_x, dir_y, button1, button2)
- Hotplug support

**Technical Approach:**
```c
typedef struct {
    int   kbd_fd;     // /dev/input/eventX
    int   mouse_fd;
    int   pad_fd;
    HSInput state;
} HSInputDevice;

void hs_input_poll(HSInputDevice* dev, HSInput* out);
```

### 3.5 Window/Input Framework (Priority: MEDIUM)

**Description:** SDL2 integration for desktop portability.

**Requirements:**
- Create window with OpenGL ES context
- Handle window events (resize, close, etc.)
- Unified input via SDL2

**Technical Approach:**
```c
typedef struct {
    SDL_Window*   window;
    SDL_GLContext gl_context;
    HSInputDevice input;
    u32           width;
    u32           height;
} HSWindow;

esp_err_t hs_window_init(HSWindow* win, const char* title, u32 w, u32 h);
void hs_window_poll(HSWindow* win);
```

### 3.6 Game Loop (Priority: LOW)

**Description:** Standard game loop framework.

**Requirements:**
- Fixed timestep update
- Variable render
- Delta time calculation
- FPS counter

## 4. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      NeoGPU Game Engine                     │
├─────────────────────────────────────────────────────────────┤
│  Game Loop                                                  │
│  ├── Input Poll (evdev/SDL2)                               │
│  ├── Update (physics, AI, game logic)                     │
│  ├── Render                                                 │
│  │   ├── Clear                                             │
│  │   ├── Draw Meshes (NeoGPU messages → GLES3)             │
│  │   ├── Draw Text (font atlas)                            │
│  │   └── Present (EGL/DRM)                                │
│  └── Audio (ALSA)                                          │
├─────────────────────────────────────────────────────────────┤
│  Core Library (hs_*.h)                                      │
│  ├── hs_core.h     - Message queue, nodes                  │
│  ├── hs_math_neon.h - vec4, mat4, splines                 │
│  ├── hs_graphics.h - DRM/GBM/EGL init, textures            │
│  ├── hs_audio.h    - 4ch audio buffers                     │
│  ├── hs_storage.h  - Persistent storage                    │
│  ├── hs_input.h    - Input structs                         │
│  └── hs_mesh.h     - [NEW] Mesh renderer                   │
│  └── hs_font.h     - [NEW] Font system                     │
│  └── hs_audio_out.h - [NEW] ALSA playback                  │
│  └── hs_window.h   - [NEW] SDL2/DRM window                 │
├─────────────────────────────────────────────────────────────┤
│  Platform Layer                                             │
│  ├── Raspberry Pi: DRM/GBM/EGL + evdev + ALSA              │
│  ├── Desktop: SDL2 + OpenGL + PulseAudio                  │
│  └── Standalone: Custom implementation                     │
└─────────────────────────────────────────────────────────────┘
```

## 5. Milestones

### Milestone 0: Core Hardening
- [ ] Add `docs/MESSAGE_ABI.md` (per-op schema + ranges)
- [ ] Add opcode pack/unpack helpers (single source of truth)
- [ ] Add debug-time message validator (send/receive)
- [ ] Implement self-contained capture (deep-copy payload bytes)
- [ ] Implement replay validation (schema + destination)
- [ ] Define `include/hs_backend.h` backend interface
- [ ] Resolve routing for `OP_ERROR`, `OP_TRACE`, `OP_STOP`
- [ ] Add tests: pack/unpack round-trip + replay determinism + validator rejection

Related PRDs:
- `docs/PRD_COMM_FABRIC.md` (acks/results, structured errors, capture I/O, async completions, backpressure, introspection)

Execution Path:
- `include/hs_backend.h` defines the backend interface; the next major evolution is connecting the message-driven state + render commands to an execution backend (GLES).

### Milestone 1: Mesh Renderer
- [ ] Add `hs_mesh.h` with mesh structures
- [ ] Add built-in vertex shader (MVP)
- [ ] Add built-in fragment shader (textured)
- [ ] Implement `hs_mesh_draw()`
- [ ] Test with triangle/cube

### Milestone 2: Font System
- [ ] Add `hs_font.h` 
- [ ] Integrate FreeType or stb_truetype
- [ ] Implement glyph rasterization
- [ ] Implement text rendering
- [ ] Test with "Hello World"

### Milestone 3: Audio Playback
- [ ] Add `hs_audio_out.h`
- [ ] Integrate ALSA
- [ ] Playback from HSAudio buffers
- [ ] Test with tone generation

### Milestone 4: Input System
- [ ] Add `hs_input_evdev.h`
- [ ] Poll keyboard/mouse/gamepad
- [ ] Map to HSInput
- [ ] Test with game controls

### Milestone 5: Game Loop Demo
- [ ] Combine all systems
- [ ] Demo game with rendering + audio + input

## 6. Success Criteria

- [ ] 270+ FPS on Pi4 (existing benchmark)
- [ ] Replay is self-contained (no dependence on transient payload lifetime)
- [ ] All opcodes have an explicit, tested payload schema
- [ ] Debug builds catch malformed messages with clear errors
- [ ] Backend can be swapped without changing message ABI
- [ ] Render 3D mesh with texture
- [ ] Display text at 60 FPS
- [ ] Play 4-channel audio simultaneously
- [ ] Responsive input (< 16ms latency)
- [ ] Demo game runs smoothly

## 7. Out of Scope

- Physics engine (use Box2D or similar)
- Network/multiplayer
- Advanced shaders (user provides these)
- Mobile platforms (future work)
- Vulkan backend (GLES3 is sufficient)

---

**Author:** NeoGPU Team  
**Version:** 1.1  
**Date:** 2026-03-12
