# NeoGPU Input Streaming System

**Date**: 2026-04-01  
**Status**: IMPLEMENTED

---

## Overview

The input streaming system provides low-latency, lock-free input processing for NeoGPU applications. It uses a dedicated reader thread per evdev device, a cache-line-aligned SPSC queue, and double-buffered state for zero-contention game loop access.

## Architecture

```
/dev/input/event*  →  evdev reader thread  →  SPSC queue  →  process thread  →  HSInput state
      ↑                                                              ↓
   select()                                                    double-buffered
   (non-blocking)                                              HSInput*
```

### Key Components

1. **HSInputSpscQueue**: Lock-free single-producer single-consumer queue with cache-line padding to prevent false sharing
2. **HSEvdevDevice**: Per-device state including fd, thread handle, and queue reference
3. **HSInputStream**: Aggregator managing multiple devices, double-buffered state, and hotplug

## Features

- [x] Non-blocking evdev reads via `select()` with 10ms timeout
- [x] Lock-free SPSC queue (256 slots) with 64-byte cache line alignment
- [x] Dedicated process thread coalesces events at up to 120Hz
- [x] Double-buffered HSInput - swap on frame boundary, zero contention
- [x] Multi-device support (up to 8 evdev handles)
- [x] EVIOCGRAB for exclusive device access
- [x] Hotplug detection via inotify
- [x] Device capability query via EVIOCGBIT
- [x] NODE_INPUT integration with HSSystem

## Usage

### Basic Setup

```c
#include "hs_input_stream.h"

HSInputStream stream;
hs_input_stream_init(&stream);
hs_input_stream_add_device(&stream, "/dev/input/event0");
hs_input_stream_start(&stream);

// Game loop
while (game_running) {
    const HSInput* state = hs_input_stream_get_state(&stream);
    // use state->dir_x, state->button1, etc.
    hs_input_stream_swap_buffers(&stream);
    // render frame
}

hs_input_stream_shutdown(&stream);
```

### With Hotplug

```c
hs_input_stream_init(&stream);
hs_input_stream_enable_hotplug(&stream);  // Auto-detect new devices
hs_input_stream_start(&stream);
```

### Query Device Capabilities

```c
uint8_t key_bits[32];
hs_input_stream_get_device_caps(&stream, 0, NULL, 0, key_bits, sizeof(key_bits), NULL, 0);
```

## API Reference

### Initialization

```c
bool hs_input_stream_init(HSInputStream* stream);
bool hs_input_stream_add_device(HSInputStream* stream, const char* device_path);
bool hs_input_stream_start(HSInputStream* stream);
```

### Runtime

```c
const HSInput* hs_input_stream_get_state(const HSInputStream* stream);
void hs_input_stream_swap_buffers(HSInputStream* stream);
void hs_input_stream_pause(HSInputStream* stream);
void hs_input_stream_resume(HSInputStream* stream);
```

### Cleanup

```c
void hs_input_stream_stop(HSInputStream* stream);
void hs_input_stream_shutdown(HSInputStream* stream);
```

### Hotplug

```c
void hs_input_stream_enable_hotplug(HSInputStream* stream);
void hs_input_stream_disable_hotplug(HSInputStream* stream);
```

### Device Info

```c
bool hs_input_stream_get_device_caps(HSInputStream* stream, int device_idx,
                                     uint8_t* ev_bits, size_t ev_bits_size,
                                     uint8_t* key_bits, size_t key_bits_size,
                                     uint8_t* abs_bits, size_t abs_bits_size);
```

## HSInput State

```c
typedef struct {
    f32  dir_x, dir_y;     // Directional input [-1.0 .. 1.0]
    bool button1;           // Primary action button
    bool button2;           // Secondary action button
    bool key_left;          // Raw key state
    bool key_right;
    bool key_up;
    bool key_down;
    f32  mouse_x, mouse_y;  // Mouse position
    bool mouse_left;         // Mouse buttons
    bool mouse_right;
    f32  pad_x, pad_y;      // Gamepad axes
    bool pad_a;              // Gamepad buttons
    bool pad_b;
    u32  frame;             // Frame counter
    f64  elapsed;           // Seconds since reset
    f64  epoch;             // Unix timestamp
} HSInput;
```

## NODE_INPUT Integration

The input stream can be used as a NeoGPU node for message-passing integration:

```c
// In your HSSystem setup:
hs_register(&sys, &input_node);

// Send input events as messages:
Message msg = {
    .to = NODE_INPUT,
    .op = OP_INPUT_KEY,
    .payload_idx = ...,
};
hs_send(&sys, &msg);
```

## Build

```bash
gcc -O3 -march=armv8-a+simd -Iinclude \
    tests/test_input_stream.c src/hs_input_stream.c \
    -o test_input_stream -lpthread
```

## Files

| File | Purpose |
|------|---------|
| `include/hs_input_stream.h` | API definitions |
| `src/hs_input_stream.c` | Implementation |
| `tests/test_input_stream.c` | Test utility |

## Bug Fixes (2026-04-01)

All 13 issues from initial implementation fixed:

1. SPSC false sharing - 64-byte cache line padding
2. Dropped events untracked - per-thread counter + atomic add
3. Double-buffering broken - write to inactive buffer
4. Thread exit fd leak - proper cleanup on all paths
5. Failed start orphan threads - cleanup on failure
6. swap_buffers race - atomic exchange + clear old
7. Sticky buttons - memset on swap
8. No EVIOCGRAB - exclusive device access
9. Axis mapping conflicts - corrected ABS_ mappings
10. Polling inefficient - pthread_cond_t blocking
11. No HSSystem integration - NODE_INPUT with messages
12. No capability query - EVIOCGBIT wrapper
13. No hotplug - inotify thread
