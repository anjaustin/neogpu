# LMM: Next Step-Charge (SYNTHESIZE)

## Decision: Audio Pipeline with Message Fabric

The most "NeoGPU-native" step-change is to integrate audio into the message-passing architecture.

### Why This Makes Sense
1. Uses existing QoS channels (TELEM for audio?)
2. Message-driven audio processing
3. Matches the system's design philosophy
4. Achievable in reasonable time

## P0 Spec: Audio Pipeline

### 1. Audio Backend Interface
```c
typedef enum {
    HS_AUDIO_BACKEND_NONE,
    HS_AUDIO_BACKEND_ALSA,
    HS_AUDIO_BACKEND_DUMMY,
} HSAudioBackend;

typedef struct {
    HSAudioBackend backend;
    int sample_rate;
    int channels;
    int buffer_size;
    void* impl;  // backend-specific
} HSAudioDevice;
```

### 2. Audio Messages
- New opcode: OP_AUDIO_PLAY
- New opcode: OP_AUDIO_STOP
- Audio data as message payload
- Channel assignment: audio channel = QoS priority

### 3. Audio Thread
- Separate thread for playback
- Pulls from audio queue (high priority channel)
- Lock-free ring buffer to main thread

### 4. Integration with QoS
- Audio on dedicated RT channel?
- Or integrate into CHAN_RENDER (sync with video)

## Implementation Phases

### Phase 1: ALSA Backend
- List devices
- Open playback
- Basic tone generation

### Phase 2: Message Integration  
- Add audio ops to opcode list
- Create audio node handler
- Test sync with rendering

### Phase 3: Polish
- Buffer underrun handling
- Volume control
- Multiple audio streams

## Acceptance Criteria
- [ ] ALSA backend opens and plays sound
- [ ] Message triggers audio playback
- [ ] No frame drops during audio
- [ ] Graceful fallback if no audio
