#include "hs_input_stream.h"
#include "hs_math_neon.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/select.h>
#include <sys/ioctl.h>
#include <sys/inotify.h>

void hs_input_spsc_init(HSInputSpscQueue* q) {
    memset(q, 0, sizeof(*q));
    atomic_init(&q->head.v, 0);
    atomic_init(&q->tail.v, 0);
}

bool hs_input_spsc_push(HSInputSpscQueue* q, const HSInputEvent* event, uint64_t* dropped) {
    u32 head = atomic_load_explicit(&q->head.v, memory_order_relaxed);
    u32 tail = atomic_load_explicit(&q->tail.v, memory_order_acquire);
    u32 next = (head + 1) % HS_INPUT_STREAM_SPSC_SIZE;
    if (next == tail) {
        if (dropped) (*dropped)++; /* SPSC: producer-owned counter, not shared. */
        return false;
    }
    q->events[head] = *event;
    atomic_store_explicit(&q->head.v, next, memory_order_release);
    return true;
}

bool hs_input_spsc_pop(HSInputSpscQueue* q, HSInputEvent* event) {
    u32 tail = atomic_load_explicit(&q->tail.v, memory_order_relaxed);
    u32 head = atomic_load_explicit(&q->head.v, memory_order_acquire);
    if (head == tail) return false;
    *event = q->events[tail];
    atomic_store_explicit(&q->tail.v, (tail + 1) % HS_INPUT_STREAM_SPSC_SIZE, memory_order_release);
    return true;
}

bool hs_input_spsc_empty(const HSInputSpscQueue* q) {
    return atomic_load_explicit(&q->head.v, memory_order_acquire) ==
           atomic_load_explicit(&q->tail.v, memory_order_acquire);
}

u32 hs_input_spsc_available(const HSInputSpscQueue* q) {
    u32 head = atomic_load_explicit(&q->head.v, memory_order_acquire);
    u32 tail = atomic_load_explicit(&q->tail.v, memory_order_acquire);
    return (head + HS_INPUT_STREAM_SPSC_SIZE - tail) % HS_INPUT_STREAM_SPSC_SIZE;
}

static void apply_event(HSInput* state, const HSInputEvent* event) {
    bool pressed = event->value != 0;
    if (event->type == EV_KEY) {
        switch (event->code) {
            case KEY_LEFT: state->key_left = pressed; break;
            case KEY_RIGHT: state->key_right = pressed; break;
            case KEY_UP: state->key_up = pressed; break;
            case KEY_DOWN: state->key_down = pressed; break;
            case KEY_SPACE: case KEY_ENTER: state->button1 = pressed; break;
            case KEY_DELETE: case KEY_BACKSPACE: state->button2 = pressed; break;
            case BTN_LEFT: state->mouse_left = pressed; break;
            case BTN_RIGHT: state->mouse_right = pressed; break;
            case BTN_A: case BTN_START: state->pad_a = pressed; break;
            case BTN_B: case BTN_SELECT: state->pad_b = pressed; break;
        }
    } else if (event->type == EV_REL) {
        if (event->code == REL_X) state->mouse_x = hs_clamp(state->mouse_x + event->value, 0, HS_WIDTH);
        if (event->code == REL_Y) state->mouse_y = hs_clamp(state->mouse_y + event->value, 0, HS_HEIGHT);
    } else if (event->type == EV_ABS) {
        /* Device-specific axis calibration is outside the generic event API. */
        if (event->code == ABS_X) state->mouse_x = hs_clamp(event->value / 65535.0f, 0, 1) * HS_WIDTH;
        if (event->code == ABS_Y) state->mouse_y = hs_clamp(event->value / 65535.0f, 0, 1) * HS_HEIGHT;
        if (event->code == ABS_RX) state->pad_x = hs_clamp(event->value / 32767.0f, -1, 1);
        if (event->code == ABS_RY) state->pad_y = hs_clamp(event->value / 32767.0f, -1, 1);
    }
}

static void drain_locked(HSInputStream* stream) {
    HSInputEvent event;
    bool changed = false;
    while (hs_input_spsc_pop(&stream->queue, &event)) {
        apply_event(&stream->state_a, &event);
        atomic_fetch_add_explicit(&stream->event_count, 1, memory_order_relaxed);
        changed = true;
    }
    if (changed) {
        HSInput* state = &stream->state_a;
        state->dir_x = hs_clamp((float)state->key_right - (float)state->key_left + state->pad_x, -1, 1);
        state->dir_y = hs_clamp((float)state->key_down - (float)state->key_up + state->pad_y, -1, 1);
        state->frame++;
        state->elapsed = (f64)state->frame / HS_FPS;
        state->epoch = (f64)time(NULL);
    }
}

bool hs_input_stream_submit(HSInputStream* stream, const HSInputEvent* event) {
    if (!stream || !stream->initialized || !event) return false;
    pthread_mutex_lock(&stream->queue_mutex);
    bool accepted = hs_input_spsc_push(&stream->queue, event, NULL);
    if (accepted) pthread_cond_signal(&stream->queue_not_empty);
    else atomic_fetch_add_explicit(&stream->dropped_events, 1, memory_order_relaxed);
    pthread_mutex_unlock(&stream->queue_mutex);
    return accepted;
}

static void* process_thread(void* arg) {
    HSInputStream* stream = arg;
    pthread_mutex_lock(&stream->queue_mutex);
    while (atomic_load_explicit(&stream->running, memory_order_acquire)) {
        while (atomic_load_explicit(&stream->running, memory_order_acquire) &&
               (hs_input_spsc_empty(&stream->queue) || atomic_load(&stream->paused))) {
            pthread_cond_wait(&stream->queue_not_empty, &stream->queue_mutex);
        }
        if (!atomic_load(&stream->running)) break;
        drain_locked(stream);
    }
    pthread_mutex_unlock(&stream->queue_mutex);
    return NULL;
}

static void* reader_thread(void* arg) {
    HSEvdevDevice* device = arg;
    while (atomic_load_explicit(&device->active, memory_order_acquire)) {
        fd_set readers;
        FD_ZERO(&readers);
        FD_SET(device->fd, &readers);
        struct timeval timeout = {0, 10000};
        int rc = select(device->fd + 1, &readers, NULL, NULL, &timeout);
        if (rc < 0) { if (errno == EINTR) continue; break; }
        if (!rc) continue;
        struct input_event events[64];
        ssize_t bytes = read(device->fd, events, sizeof(events));
        if (bytes < 0 && (errno == EINTR || errno == EAGAIN)) continue;
        if (bytes <= 0) break;
        for (size_t i = 0; i < (size_t)bytes / sizeof(events[0]); i++) {
            HSInputEvent event = {.tv_sec = events[i].time.tv_sec, .tv_usec = events[i].time.tv_usec,
                .type = events[i].type, .code = events[i].code, .value = events[i].value};
            hs_input_stream_submit(device->owner, &event);
        }
    }
    atomic_store(&device->active, false);
    return NULL;
}

static void stop_device(HSEvdevDevice* device) {
    atomic_store(&device->active, false);
    if (device->thread_started) pthread_join(device->thread, NULL);
    device->thread_started = false;
    if (device->fd >= 0) {
        ioctl(device->fd, EVIOCGRAB, 0);
        close(device->fd);
        device->fd = -1;
    }
}

bool hs_input_stream_add_device(HSInputStream* stream, const char* path) {
    if (!stream || !stream->initialized || !path || strlen(path) >= sizeof(stream->devices[0].path)) return false;
    pthread_mutex_lock(&stream->device_mutex);
    int slot = -1;
    for (int i = 0; i < HS_INPUT_MAX_DEVICES; i++) {
        if (stream->devices[i].fd >= 0 && strcmp(stream->devices[i].path, path) == 0) {
            pthread_mutex_unlock(&stream->device_mutex);
            return false;
        }
        if (slot < 0 && stream->devices[i].fd < 0) slot = i;
    }
    int fd = slot >= 0 ? open(path, O_RDONLY | O_NONBLOCK | O_CLOEXEC) : -1;
    if (fd < 0 || ioctl(fd, EVIOCGRAB, 1) < 0) {
        if (fd >= 0) close(fd);
        pthread_mutex_unlock(&stream->device_mutex);
        return false;
    }
    HSEvdevDevice* device = &stream->devices[slot];
    device->fd = fd;
    strcpy(device->path, path);
    device->owner = stream;
    if (slot >= stream->device_count) stream->device_count = slot + 1;
    bool ok = true;
    if (atomic_load(&stream->running)) {
        atomic_store(&device->active, true);
        ok = pthread_create(&device->thread, NULL, reader_thread, device) == 0;
        device->thread_started = ok;
        if (!ok) stop_device(device);
    }
    pthread_mutex_unlock(&stream->device_mutex);
    return ok;
}

static void* hotplug_thread(void* arg) {
    HSInputStream* stream = arg;
    int fd = stream->hotplug_fd;
    char buffer[4096] __attribute__((aligned(8)));
    while (atomic_load(&stream->running)) {
        fd_set readers;
        FD_ZERO(&readers); FD_SET(fd, &readers);
        struct timeval timeout = {0, 100000};
        int ready = select(fd + 1, &readers, NULL, NULL, &timeout);
        if (ready < 0) { if (errno == EINTR) continue; break; }
        if (!ready) continue;
        ssize_t length = read(fd, buffer, sizeof(buffer));
        if (length < 0) { if (errno == EAGAIN || errno == EINTR) continue; break; }
        for (size_t pos = 0; pos + sizeof(struct inotify_event) <= (size_t)length;) {
            struct inotify_event* event = (void*)(buffer + pos);
            size_t size = sizeof(*event) + event->len;
            if (size > (size_t)length - pos) break;
            if (event->len && strncmp(event->name, "event", 5) == 0) {
                char path[256];
                int n = snprintf(path, sizeof(path), "/dev/input/%s", event->name);
                if (n > 0 && (size_t)n < sizeof(path)) {
                    if (event->mask & IN_CREATE) hs_input_stream_add_device(stream, path);
                    if (event->mask & IN_DELETE) {
                        pthread_mutex_lock(&stream->device_mutex);
                        for (int i = 0; i < stream->device_count; i++)
                            if (strcmp(stream->devices[i].path, path) == 0) stop_device(&stream->devices[i]);
                        pthread_mutex_unlock(&stream->device_mutex);
                        pthread_mutex_lock(&stream->queue_mutex);
                        /* A removed device must not leave held buttons stuck. */
                        hs_input_init(&stream->state_a);
                        pthread_mutex_unlock(&stream->queue_mutex);
                    }
                }
            }
            pos += size;
        }
    }
    close(fd);
    return NULL;
}

bool hs_input_stream_init(HSInputStream* stream) {
    if (!stream) return false;
    memset(stream, 0, sizeof(*stream));
    stream->hotplug_fd = -1;
    hs_input_spsc_init(&stream->queue);
    hs_input_init(&stream->state_a);
    hs_input_init(&stream->state_b);
    atomic_init(&stream->active_state_ptr, &stream->state_a);
    atomic_init(&stream->running, false); atomic_init(&stream->paused, false);
    atomic_init(&stream->event_count, 0); atomic_init(&stream->dropped_events, 0);
    for (int i = 0; i < HS_INPUT_MAX_DEVICES; i++) {
        stream->devices[i].fd = -1;
        atomic_init(&stream->devices[i].active, false);
    }
    if (pthread_mutex_init(&stream->queue_mutex, NULL)) return false;
    if (pthread_mutex_init(&stream->device_mutex, NULL)) goto fail_queue;
    if (pthread_cond_init(&stream->queue_not_empty, NULL)) goto fail_device;
    if (pthread_cond_init(&stream->queue_not_full, NULL)) goto fail_empty;
    stream->initialized = true;
    return true;
fail_empty: pthread_cond_destroy(&stream->queue_not_empty);
fail_device: pthread_mutex_destroy(&stream->device_mutex);
fail_queue: pthread_mutex_destroy(&stream->queue_mutex);
    return false;
}

bool hs_input_stream_start(HSInputStream* stream) {
    if (!stream || !stream->initialized || atomic_load(&stream->running)) return false;
    atomic_store(&stream->running, true);
    if (pthread_create(&stream->process_thread, NULL, process_thread, stream)) {
        atomic_store(&stream->running, false); return false;
    }
    stream->process_started = true;
    bool ok = true;
    pthread_mutex_lock(&stream->device_mutex);
    for (int i = 0; i < stream->device_count; i++) {
        HSEvdevDevice* device = &stream->devices[i];
        if (device->fd < 0) continue;
        atomic_store(&device->active, true);
        if (pthread_create(&device->thread, NULL, reader_thread, device)) { ok = false; break; }
        device->thread_started = true;
    }
    pthread_mutex_unlock(&stream->device_mutex);
    if (ok && stream->hotplug_enabled) {
        int fd = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
        ok = fd >= 0 && inotify_add_watch(fd, "/dev/input", IN_CREATE | IN_DELETE) >= 0;
        if (ok) {
            stream->hotplug_fd = fd;
            ok = pthread_create(&stream->hotplug_thread, NULL, hotplug_thread, stream) == 0;
            stream->hotplug_started = ok;
        }
        if (!ok && fd >= 0) { close(fd); stream->hotplug_fd = -1; }
    }
    if (!ok) hs_input_stream_stop(stream);
    return ok;
}

void hs_input_stream_stop(HSInputStream* stream) {
    if (!stream || !stream->initialized) return;
    pthread_mutex_lock(&stream->queue_mutex);
    atomic_store(&stream->running, false);
    pthread_cond_broadcast(&stream->queue_not_empty);
    pthread_mutex_unlock(&stream->queue_mutex);
    if (stream->hotplug_started) pthread_join(stream->hotplug_thread, NULL);
    stream->hotplug_started = false;
    stream->hotplug_fd = -1;
    pthread_mutex_lock(&stream->device_mutex);
    for (int i = 0; i < stream->device_count; i++) stop_device(&stream->devices[i]);
    pthread_mutex_unlock(&stream->device_mutex);
    if (stream->process_started) pthread_join(stream->process_thread, NULL);
    stream->process_started = false;
}

void hs_input_stream_shutdown(HSInputStream* stream) {
    if (!stream || !stream->initialized) return;
    hs_input_stream_stop(stream);
    pthread_mutex_destroy(&stream->device_mutex);
    pthread_mutex_destroy(&stream->queue_mutex);
    pthread_cond_destroy(&stream->queue_not_empty);
    pthread_cond_destroy(&stream->queue_not_full);
    stream->initialized = false;
}

const HSInput* hs_input_stream_get_state(const HSInputStream* stream) {
    static _Thread_local HSInput snapshot;
    if (!stream || !stream->initialized) return NULL;
    pthread_mutex_t* lock = (pthread_mutex_t*)&stream->queue_mutex;
    pthread_mutex_lock(lock);
    snapshot = stream->state_a;
    pthread_mutex_unlock(lock);
    return &snapshot;
}

void hs_input_stream_process_events(HSInputStream* stream) {
    if (!stream || !stream->initialized) return;
    pthread_mutex_lock(&stream->queue_mutex);
    if (!atomic_load(&stream->paused)) drain_locked(stream);
    pthread_mutex_unlock(&stream->queue_mutex);
}

void hs_input_stream_swap_buffers(HSInputStream* stream) { hs_input_stream_process_events(stream); }

void hs_input_stream_pause(HSInputStream* stream) {
    if (!stream || !stream->initialized) return;
    pthread_mutex_lock(&stream->queue_mutex);
    atomic_store(&stream->paused, true);
    pthread_mutex_unlock(&stream->queue_mutex);
}

void hs_input_stream_resume(HSInputStream* stream) {
    if (!stream || !stream->initialized) return;
    pthread_mutex_lock(&stream->queue_mutex);
    atomic_store(&stream->paused, false);
    pthread_cond_broadcast(&stream->queue_not_empty);
    pthread_mutex_unlock(&stream->queue_mutex);
}

bool hs_input_stream_get_device_caps(HSInputStream* stream, int index,
    uint8_t* ev, size_t ev_size, uint8_t* key, size_t key_size, uint8_t* abs, size_t abs_size) {
    if (!stream || !stream->initialized) return false;
    pthread_mutex_lock(&stream->device_mutex);
    int fd = index >= 0 && index < stream->device_count ? stream->devices[index].fd : -1;
    bool ok = fd >= 0;
    if (ok && ev && ev_size) ok = ioctl(fd, EVIOCGBIT(0, ev_size), ev) >= 0;
    if (ok && key && key_size) ok = ioctl(fd, EVIOCGBIT(EV_KEY, key_size), key) >= 0;
    if (ok && abs && abs_size) ok = ioctl(fd, EVIOCGBIT(EV_ABS, abs_size), abs) >= 0;
    pthread_mutex_unlock(&stream->device_mutex);
    return ok;
}

void hs_input_stream_enable_hotplug(HSInputStream* stream) {
    if (stream && !atomic_load(&stream->running)) stream->hotplug_enabled = true;
}
void hs_input_stream_disable_hotplug(HSInputStream* stream) {
    if (stream && !atomic_load(&stream->running)) stream->hotplug_enabled = false;
}
