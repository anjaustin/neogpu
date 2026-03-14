# LMM: Next Step-Charge (RAW)

## Current State
- GLES backend working well (414K fps)
- Message-passing fabric with QoS channels
- TCP/IP + Unix IPC
- 135 tests passing

## What's Missing for "Real-Time Graphics System"?

### 1. Audio Pipeline
- You mentioned "Audio" in features
- hs_audio.h exists but integrated?
- Real-time audio requires precise timing
- Need: audio thread, buffer management, latency control

### 2. Frame Timing / Scheduling
- Current: simple tick-based processing
- Missing: vsync, frame timing, deadline handling
- Real-time systems need predictable frame times

### 3. Storage
- hs_storage.h exists
- Need: reliable async storage for captures

### 4. Multi-Display
- Single display currently
- Multi-monitor support?

### 5. Windowing
- No windowing system
- Could add layer for window management

## Questions for NODES
- What's the priority?
- Audio first? Frame timing? Storage?
- What are the hard requirements?
