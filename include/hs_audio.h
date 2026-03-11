#ifndef HS_AUDIO_H
#define HS_AUDIO_H

#include "hs_core.h"
#include <stdlib.h>
#include <string.h>

#define HS_AUDIO_CHANNELS    4
#define HS_AUDIO_BUFFER_SIZE 3000
#define HS_AUDIO_FREQ        48000

typedef struct {
    float  buffer[HS_AUDIO_BUFFER_SIZE];
    u32    buffer_count;
    u8     active;
    u8     shader_id;
} HSAudioChannel;

typedef struct {
    HSAudioChannel channels[HS_AUDIO_CHANNELS];
    u8            initialized;
} HSAudio;

static inline void hs_audio_init(HSAudio* audio) {
    memset(audio, 0, sizeof(HSAudio));
    audio->initialized = 1;
}

static inline void hs_audio_set_channel(HSAudio* audio, u8 channel, u8 shader_id) {
    if (channel >= HS_AUDIO_CHANNELS) return;
    audio->channels[channel].shader_id = shader_id;
    audio->channels[channel].active = (shader_id != 0xFF);
}

static inline float* hs_audio_get_buffer(HSAudio* audio, u8 channel) {
    if (channel >= HS_AUDIO_CHANNELS || !audio->channels[channel].active) {
        return NULL;
    }
    return audio->channels[channel].buffer;
}

static inline void hs_audio_advance(HSAudio* audio, u8 channel) {
    if (channel >= HS_AUDIO_CHANNELS) return;
    audio->channels[channel].buffer_count++;
}

static inline void hs_audio_stop(HSAudio* audio, u8 channel) {
    if (channel >= HS_AUDIO_CHANNELS) return;
    memset(audio->channels[channel].buffer, 0, sizeof(audio->channels[channel].buffer));
    audio->channels[channel].active = 0;
    audio->channels[channel].buffer_count = 0;
}

static inline void hs_audio_stop_all(HSAudio* audio) {
    for (int i = 0; i < HS_AUDIO_CHANNELS; i++) {
        hs_audio_stop(audio, i);
    }
}

#endif
