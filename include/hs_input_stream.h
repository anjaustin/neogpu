#ifndef HS_INPUT_STREAM_H
#define HS_INPUT_STREAM_H

#include "hs_core.h"
#include "hs_input.h"
#include <linux/input.h>
#include <stdatomic.h>
#include <pthread.h>

#define NODE_INPUT    7

#define HS_INPUT_STREAM_SPSC_SIZE 256

#define HS_INPUT_MAX_DEVICES 8

typedef enum {
    HS_INPUT_EVENT_KEY,
    HS_INPUT_EVENT_REL,
    HS_INPUT_EVENT_ABS,
} HSInputEventType;

typedef struct {
    u32 tv_sec;
    u32 tv_usec;
    u16 type;
    u16 code;
    s32 value;
} __attribute__((packed)) HSInputEvent;

typedef struct {
    HSInputEvent events[HS_INPUT_STREAM_SPSC_SIZE];
    HSAtomicCacheLine head;
    HSAtomicCacheLine tail;
} HSInputSpscQueue;

typedef struct {
    int fd;
    char path[256];
    atomic_bool active;
    pthread_t thread;
    HSInputSpscQueue* queue;
    _Atomic(uint64_t)* dropped_events;
} HSEvdevDevice;

typedef struct {
    HSInputSpscQueue queue;

    HSEvdevDevice devices[HS_INPUT_MAX_DEVICES];
    int device_count;

    HSInput state_a;
    HSInput state_b;
    _Atomic(HSInput*) active_state_ptr;

    pthread_t process_thread;
    pthread_t hotplug_thread;
    atomic_bool running;
    atomic_bool paused;
    bool hotplug_enabled;

    _Atomic(uint64_t) event_count;
    _Atomic(uint64_t) dropped_events;

    pthread_mutex_t queue_mutex;
    pthread_cond_t queue_not_empty;
    pthread_cond_t queue_not_full;
} HSInputStream;

void hs_input_spsc_init(HSInputSpscQueue* q);
bool hs_input_spsc_push(HSInputSpscQueue* q, const HSInputEvent* ev, uint64_t* dropped);
bool hs_input_spsc_pop(HSInputSpscQueue* q, HSInputEvent* ev);
bool hs_input_spsc_empty(const HSInputSpscQueue* q);
u32 hs_input_spsc_available(const HSInputSpscQueue* q);

bool hs_input_stream_init(HSInputStream* stream);
bool hs_input_stream_add_device(HSInputStream* stream, const char* device_path);
bool hs_input_stream_start(HSInputStream* stream);
void hs_input_stream_stop(HSInputStream* stream);
void hs_input_stream_shutdown(HSInputStream* stream);

void hs_input_stream_process_events(HSInputStream* stream);
const HSInput* hs_input_stream_get_state(const HSInputStream* stream);
void hs_input_stream_swap_buffers(HSInputStream* stream);

void hs_input_stream_pause(HSInputStream* stream);
void hs_input_stream_resume(HSInputStream* stream);

bool hs_input_stream_get_device_caps(HSInputStream* stream, int device_idx,
                                     uint8_t* ev_bits, size_t ev_bits_size,
                                     uint8_t* key_bits, size_t key_bits_size,
                                     uint8_t* abs_bits, size_t abs_bits_size);

void hs_input_stream_enable_hotplug(HSInputStream* stream);
void hs_input_stream_disable_hotplug(HSInputStream* stream);

#endif
