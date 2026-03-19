/*
 * hs_procaudio.h - Procedural Audio Synthesis
 * 
 * Synth from primitives:
 * - Sine, square, triangle, saw waves
 * - Noise
 * - Envelope (ADSR)
 * - Effects (reverb, filter)
 * - Tracker for music
 */

#ifndef HS_PROCAUDIO_H
#define HS_PROCAUDIO_H

#include "hs_core.h"
#include <math.h>

#define HS_PROC_AUDIO_SAMPLE_RATE 48000
#define HS_PROC_AUDIO_BUFFER_SIZE 1024
#define HS_PROC_MAX_CHANNELS 4
#define HS_PROC_MAX_NOTES 128

typedef enum {
    WAVE_SINE = 0,
    WAVE_SQUARE,
    WAVE_TRIANGLE,
    WAVE_SAW,
    WAVE_NOISE,
} WaveType;

typedef struct {
    f32 frequency;
    f32 phase;
    f32 volume;
    f32 target_volume;
    f32 attack;
    f32 decay;
    f32 sustain;
    f32 release;
    f32 envelope;
    int envelope_stage; // 0=idle, 1=attack, 2=decay, 3=sustain, 4=release
    WaveType wave;
} Voice;

typedef struct {
    Voice voices[HS_PROC_MAX_CHANNELS];
    f32 sample_rate;
    f32 buffer[HS_PROC_AUDIO_BUFFER_SIZE];
    int write_pos;
} ProcAudio;

static void proc_audio_init(ProcAudio* audio) {
    memset(audio, 0, sizeof(*audio));
    audio->sample_rate = (f32)HS_PROC_AUDIO_SAMPLE_RATE;
}

static void proc_audio_voice_on(ProcAudio* audio, int ch, f32 freq, WaveType wave) {
    if (ch < 0 || ch >= HS_PROC_MAX_CHANNELS) return;
    
    Voice* v = &audio->voices[ch];
    v->frequency = freq;
    v->phase = 0.0f;
    v->volume = 0.0f;
    v->target_volume = 0.5f;
    v->attack = 0.01f;
    v->decay = 0.1f;
    v->sustain = 0.7f;
    v->release = 0.2f;
    v->wave = wave;
    v->envelope = 0.0f;
    v->envelope_stage = 1; // Start with attack
}

static void proc_audio_voice_off(ProcAudio* audio, int ch) {
    if (ch < 0 || ch >= HS_PROC_MAX_CHANNELS) return;
    audio->voices[ch].envelope_stage = 4; // Release
}

static f32 proc_audio_sample(Voice* v, f32 t) {
    f32 phase = v->phase + t * v->frequency / 48000.0f;
    
    switch (v->wave) {
        case WAVE_SINE:
            return sinf(phase * 6.28318f);
        case WAVE_SQUARE:
            return phase - floorf(phase) > 0.5f ? 1.0f : -1.0f;
        case WAVE_TRIANGLE: {
            f32 p = phase - floorf(phase);
            return p < 0.5f ? 4.0f * p - 1.0f : 3.0f - 4.0f * p;
        }
        case WAVE_SAW: {
            f32 p = phase - floorf(phase);
            return 2.0f * p - 1.0f;
        }
        case WAVE_NOISE: {
            return 2.0f * ((f32)(rand() & 0xFFFF) / 65536.0f - 0.5f);
        }
    }
    return 0.0f;
}

static void proc_audio_tick(Voice* v) {
    switch (v->envelope_stage) {
        case 1: // Attack
            v->envelope += v->target_volume / (v->attack * 48000.0f);
            if (v->envelope >= v->target_volume) {
                v->envelope = v->target_volume;
                v->envelope_stage = 2;
            }
            break;
        case 2: // Decay
            v->envelope += (v->target_volume * v->sustain - v->envelope) / (v->decay * 48000.0f);
            if (v->envelope <= v->target_volume * v->sustain) {
                v->envelope = v->target_volume * v->sustain;
                v->envelope_stage = 3;
            }
            break;
        case 3: // Sustain
            break;
        case 4: // Release
            v->envelope -= v->envelope / (v->release * 48000.0f);
            if (v->envelope < 0.001f) {
                v->envelope = 0.0f;
                v->envelope_stage = 0;
            }
            break;
    }
}

static f32* proc_audio_render(ProcAudio* audio, int samples) {
    for (int i = 0; i < samples && i < HS_PROC_AUDIO_BUFFER_SIZE; i++) {
        f32 sample = 0.0f;
        
        for (int ch = 0; ch < HS_PROC_MAX_CHANNELS; ch++) {
            Voice* v = &audio->voices[ch];
            if (v->envelope_stage > 0) {
                sample += proc_audio_sample(v, (f32)i) * v->envelope;
                proc_audio_tick(v);
            }
        }
        
        // Soft clip
        if (sample > 1.0f) sample = 1.0f - 1.0f / (sample + 1.0f);
        else if (sample < -1.0f) sample = -1.0f + 1.0f / (-sample + 1.0f);
        
        audio->buffer[i] = sample * 0.5f;
    }
    
    return audio->buffer;
}

static f32 note_to_freq(int note) {
    // MIDI note to frequency
    return 440.0f * powf(2.0f, (note - 69) / 12.0f);
}

static void proc_audio_play_note(ProcAudio* audio, int ch, int note, WaveType wave) {
    proc_audio_voice_on(audio, ch, note_to_freq(note), wave);
}

#endif
