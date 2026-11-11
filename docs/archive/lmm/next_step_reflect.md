# LMM: Next Step-Charge (REFLECT)

## Edge Cases

### Audio on Pi4
- ALSA available but can be glitchy
- Need: low-latency configuration
- Fallback: disable audio if unavailable

### Audio/Graphics Sync
- Audio must match video frame timing
- Latency: need < 20ms for good experience
- Message fabric can help synchronize

### Storage Reliability
- Power loss = data loss with current impl
- Could add CRC/checkpoint
- Or just accept as "temp storage"

### Frame Timing on Pi4
- No vsync without compositor
- DRM pageflip timing available
- Need: measure actual display refresh

## Failure Modes

| Feature | Risk | Mitigation |
|---------|------|------------|
| Audio | ALSA glitch | Fallback, disable |
| Storage | Corruption | Checksum |
| Frame timing | No vsync | DRM pageflip |
| Capture | Disk I/O | Async, non-blocking |

## What Makes Sense for NeoGPU
Given this is a message-passing GPU layer, the "step-change" should:
1. Stay true to message-passing architecture
2. Leverage the QoS channels we built
3. Be implementable in reasonable time

Audio integrated into message fabric = best fit.
