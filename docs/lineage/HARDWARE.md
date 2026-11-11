# PicoGPU Hardware Documentation

## System Overview

This document describes the hardware capabilities of the system running the PicoGPU C Port.

---

## Hardware Platform

| Property | Value |
|----------|-------|
| **Board** | Raspberry Pi 4 Model B |
| **Revision** | 1.5 |
| **CPU** | Broadcom BCM2711 (Cortex-A72) |
| **CPU Cores** | 4 × ARM Cortex-A72 @ 1.5GHz |
| **RAM** | 8GB (7.6GB available) |
| **Storage** | MicroSD Card |

---

## Memory Configuration

| Pool | Size | Notes |
|------|------|-------|
| **Total RAM** | 7.6 GB | 8GB model |
| **Available** | 6.5 GB | |
| **ARM** | 948 MB | vcgencmd setting (may be outdated) |
| **GPU** | 76 MB | vcgencmd setting (may be dynamic) |

> Note: vcgencmd reports 948MB ARM / 76MB GPU but `free` shows 7.6GB total. GPU memory appears to be dynamically allocated from the full 8GB.

### Boot Parameters
```
vc_mem.mem_base=0x3ec00000
vc_mem.mem_size=0x40000000 (1GB)
coherent_pool=1M
```

---

## GPU (VideoCore VI)

### GPU Hardware
| Property | Value |
|----------|-------|
| **Chip** | Broadcom VideoCore VI |
| **Architecture** | V3D (VideoCore 3D) |
| **API Support** | OpenGL ES 3.0 / Vulkan 1.1 |
| **Shader Cores** | Unified shader architecture |

### Loaded Kernel Modules

| Module | Description |
|--------|-------------|
| `v3d` | VideoCore 3D renderer (OpenGL ES) |
| `vc4` | Display controller + HDMI/DSI output |
| `drm_display_helper` | DRM display helpers |
| `drm_shmem_helper` | DRM shared memory helpers |
| `gpu_sched` | GPU command scheduler |
| `cec` | HDMI CEC control |
| `drm_dma_helper` | DMA helper for DRM |

### DRM Devices

| Device | Path | Purpose |
|--------|------|---------|
| **card0** | `/dev/dri/card0` | V3D rendering |
| **card1** | `/dev/dri/card1` | VC4 display |
| **renderD128** | `/dev/dri/renderD128` | Render node |

### DRM Version
```
drm 1.1.0 20060810
```

---

## Display Output

### Connected Display

| Property | Value |
|----------|-------|
| **Port** | DSI-1 (Display Serial Interface) |
| **Status** | ✅ Connected |
| **Resolution** | 800 × 480 |
| **Refresh** | 60Hz (implied) |
| **Framebuffer** | fb0 (vc4drmfb) |
| **Driver** | vc4 |

### Available Ports

| Port | Status | Notes |
|------|--------|-------|
| **DSI-1** | ✅ Connected | Pi Touchscreen 7" |
| **HDMI-A-1** | ❌ Disconnected | |
| **HDMI-A-2** | ❌ Disconnected | |
| **Writeback-1** | ❌ Available | |

### Display Modes

| Source | Available Modes |
|--------|-----------------|
| DSI-1 | 800×480 |
| HDMI-A-1 | (none connected) |
| HDMI-A-2 | (none connected) |

---

## Audio

### Sound Hardware
| Property | Value |
|----------|-------|
| **Codec** | Broadcom HDMI / 3.5mm jack |
| **Driver** | snd_bcm2835 (ALSA) |

### Loaded Audio Modules
```
snd_soc_core
snd_soc_hdmi_codec (via vc4)
```

---

## PicoGPU Compatibility

### Resolution Match

| PicoGPU Target | Display Resolution | Match |
|----------------|-------------------|-------|
| 640 × 480 | 800 × 480 | ✅ Compatible (letterbox) |

### Memory Requirements

| Resource | PicoGPU Need | Available |
|----------|--------------|-----------|
| GPU Memory | 300KB (original) | 76 MB |
| Framebuffer | ~2.3 MB (800×480×4×2) | Within GPU mem |
| Message Buffers | ~1 MB | Within ARM mem |

### Rendering Path for PicoGPU

```
PicoGPU (message queue)
    ↓
OpenGL ES 3.0 (v3d driver)
    ↓
GBM (Graphics Buffer Manager)
    ↓
VC4 (display controller)
    ↓
DSI-1 → Touchscreen Display
```

---

## API Capabilities Available

### OpenGL ES 3.0 (via v3d)
- Vertex shaders
- Fragment shaders
- Instanced rendering
- Multiple render targets (MRT)
- Float textures
- Depth textures
- Stencil textures
- Framebuffer objects (FBO)

### Vulkan 1.1 (via v3d)
- Compute shaders
- Geometry shaders
- Tessellation
- SPIR-V support

### Display (via vc4)
- DRM/KMS
- GBM
- EGL
- Hardware overlays

---

## Potential PicoGPU Integration Points

### Option 1: OpenGL ES + EGL + GBM
```c
// Create GBM surface
struct gbm_device* gbm = gbm_create_device(fd);
struct gbm_surface* surface = gbm_surface_create(...);

// Create EGL context
EGLDisplay egl = eglGetDisplay(gbm);
eglInitialize(egl, ...);

// Render with GLES
glClear(GL_COLOR_BUFFER_BIT);

// Flip to display
drmModeSetCrtc(fd, ..., surface->bo->handle, ...);
```

### Option 2: Vulkan
```c
// Create Vulkan instance with VK_KHR_display
VkInstance instance;
vkEnumerateInstanceExtensionProperties(...);

// Get display mode
vkGetDisplayModePropertiesKHR(...);

// Render and present
vkQueuePresentKHR(queue, &presentInfo);
```

### Option 3: Simple Framebuffer
```c
// Direct framebuffer access
int fb = open("/dev/fb0", O_RDWR);
mmap(fb, 800*480*4, PROT_READ|PROT_WRITE, ...);

// Write pixels directly
// (simplest but no hardware acceleration)
```

---

## Benchmarks (Current PicoGPU Port)

| Metric | Value |
|--------|-------|
| Message Layer FPS | ~1,000,000 |
| Binary Size | 73 KB |
| Memory/Frame @60fps | 63 KB |

---

## References

- [Raspberry Pi VideoCore Docs](https://github.com/raspberrypi/linux/tree/rpi-5.15.y/drivers/gpu/drm/vc4)
- [V3D Driver](https://github.com/raspberrypi/linux/blob/rpi-5.15.y/drivers/gpu/drm/v3d)
- [VC4 Driver](https://github.com/raspberrypi/linux/blob/rpi-5.15.y/drivers/gpu/drm/vc4)
- [Mesa V3D](https://gitlab.freedesktop.org/mesa/mesa/-/tree/main/src/broadcom/v3d)
- [GBM API](https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/include/drm/gbm.h)
- [EGL + GBM Tutorial](https://github.com/eyelash/tutorials/blob/master/wayland-egl-gbm.md)

---

*Generated: March 2026*
