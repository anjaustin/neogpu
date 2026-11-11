# LMM: Next Step-Charge (NODES)

## Current Implementation Analysis

### Audio (hs_audio.h)
- 4 channels @ 48KHz
- Simple buffer structure
- **Not implemented**: actual audio playback
- Status: stub/placeholder

### Storage (hs_storage.h)  
- 16 slots × 256B
- File save/load
- **Not implemented**: actual use in codebase
- Status: stub/placeholder

## What Exists That Works
- Message fabric (8.5M msgs/sec)
- GLES rendering (414K fps)
- IPC (TCP + Unix)
- QoS channels (RT/RENDER/TELEM)

## Gap Analysis
The audio/storage headers are placeholders. What we actually have is a message-passing system, not a full graphics engine.

## What Would Be a Real Step-Charge?

### Option A: Audio Pipeline
- Integrate with ALSA or PulseAudio on Pi
- Real-time audio buffer management
- Message-driven audio processing

### Option B: Frame Timing
- Add vsync integration
- Precise frame timing measurement
- Deadline tracking

### Option C: Capture/Recording
- Screen capture to file
- Video encoding
- Replay system

### Option D: Network Display
- DRM/KMS direct access
- Multi-display support
- True framebuffer driver

## Most Impactful
Given the message-passing architecture, integrating audio into the message fabric would be the most "NeoGPU-like" approach - treat audio as another channel with QoS.
