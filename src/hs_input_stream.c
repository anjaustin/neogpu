#include "hs_input_stream.h"
#include "hs_nodes.h"
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/select.h>
#include <dirent.h>
#include <sys/ioctl.h>
#include <sys/inotify.h>

#define EVDEV_BUFFER_SIZE 64
#define INOTIFY_EVENT_SIZE (sizeof(struct inotify_event))
#define INOTIFY_BUF_LEN (1024 * (INOTIFY_EVENT_SIZE + 16))

static inline u32 hs_ring_index(u32 pos, u32 size) {
    return pos & (size - 1);
}

void hs_input_spsc_init(HSInputSpscQueue* q) {
    memset(q, 0, sizeof(*q));
}

bool hs_input_spsc_push(HSInputSpscQueue* q, const HSInputEvent* ev, uint64_t* dropped) {
    u32 head = atomic_load_explicit(&q->head.v, memory_order_relaxed);
    u32 tail = atomic_load_explicit(&q->tail.v, memory_order_acquire);
    u32 next = hs_ring_index(head + 1, HS_INPUT_STREAM_SPSC_SIZE);

    if (next == tail) {
        if (dropped) {
            atomic_fetch_add_explicit(dropped, 1, memory_order_relaxed);
        }
        return false;
    }

    q->events[hs_ring_index(head, HS_INPUT_STREAM_SPSC_SIZE)] = *ev;
    atomic_store_explicit(&q->head.v, next, memory_order_release);
    return true;
}

bool hs_input_spsc_pop(HSInputSpscQueue* q, HSInputEvent* ev) {
    u32 tail = atomic_load_explicit(&q->tail.v, memory_order_relaxed);
    u32 head = atomic_load_explicit(&q->head.v, memory_order_acquire);

    if (head == tail) {
        return false;
    }

    *ev = q->events[hs_ring_index(tail, HS_INPUT_STREAM_SPSC_SIZE)];
    atomic_store_explicit(&q->tail.v, hs_ring_index(tail + 1, HS_INPUT_STREAM_SPSC_SIZE), memory_order_release);
    return true;
}

bool hs_input_spsc_empty(const HSInputSpscQueue* q) {
    u32 head = atomic_load_explicit(&q->head.v, memory_order_acquire);
    u32 tail = atomic_load_explicit(&q->tail.v, memory_order_acquire);
    return head == tail;
}

u32 hs_input_spsc_available(const HSInputSpscQueue* q) {
    u32 head = atomic_load_explicit(&q->head.v, memory_order_acquire);
    u32 tail = atomic_load_explicit(&q->tail.v, memory_order_acquire);
    return hs_ring_index(head, HS_INPUT_STREAM_SPSC_SIZE) - hs_ring_index(tail, HS_INPUT_STREAM_SPSC_SIZE);
}

static void apply_event_to_input(HSInput* state, const HSInputEvent* ev) {
    switch (ev->type) {
        case EV_KEY: {
            bool pressed = (ev->value != 0);
            switch (ev->code) {
                case KEY_LEFT:
                    state->key_left = pressed;
                    break;
                case KEY_RIGHT:
                    state->key_right = pressed;
                    break;
                case KEY_UP:
                    state->key_up = pressed;
                    break;
                case KEY_DOWN:
                    state->key_down = pressed;
                    break;
                case KEY_SPACE:
                case KEY_ENTER:
                    state->button1 = pressed;
                    break;
                case KEY_DELETE:
                case KEY_BACKSPACE:
                    state->button2 = pressed;
                    break;
                case BTN_LEFT:
                    state->mouse_left = pressed;
                    break;
                case BTN_RIGHT:
                    state->mouse_right = pressed;
                    break;
                case BTN_A:
                case BTN_START:
                    state->pad_a = pressed;
                    break;
                case BTN_B:
                case BTN_SELECT:
                    state->pad_b = pressed;
                    break;
            }
            break;
        }
        case EV_REL: {
            switch (ev->code) {
                case REL_X:
                    state->mouse_x += ev->value;
                    if (state->mouse_x < 0) state->mouse_x = 0;
                    if (state->mouse_x > HS_WIDTH) state->mouse_x = HS_WIDTH;
                    break;
                case REL_Y:
                    state->mouse_y += ev->value;
                    if (state->mouse_y < 0) state->mouse_y = 0;
                    if (state->mouse_y > HS_HEIGHT) state->mouse_y = HS_HEIGHT;
                    break;
                case REL_WHEEL:
                    break;
            }
            break;
        }
        case EV_ABS: {
            switch (ev->code) {
                case ABS_X:
                    state->mouse_x = ev->value / 65535.0f * HS_WIDTH;
                    break;
                case ABS_Y:
                    state->mouse_y = ev->value / 65535.0f * HS_HEIGHT;
                    break;
                case ABS_RX:
                    state->pad_x = ev->value / 32767.0f;
                    break;
                case ABS_RY:
                    state->pad_y = ev->value / 32767.0f;
                    break;
                case ABS_Z:
                case ABS_RZ:
                    break;
            }
            break;
        }
    }
}

static void compute_dir_from_keys(HSInput* state) {
    f32 dx = 0.0f, dy = 0.0f;
    if (state->key_left)  dx -= 1.0f;
    if (state->key_right) dx += 1.0f;
    if (state->key_up)    dy -= 1.0f;
    if (state->key_down)  dy += 1.0f;
    dx += state->pad_x;
    dy += state->pad_y;
    if (dx < -1.0f) dx = -1.0f;
    if (dx >  1.0f) dx =  1.0f;
    if (dy < -1.0f) dy = -1.0f;
    if (dy >  1.0f) dy =  1.0f;
    state->dir_x = dx;
    state->dir_y = dy;
}

static int open_evdev(const char* path) {
    int fd = open(path, O_RDONLY | O_NONBLOCK);
    if (fd < 0) return -1;
    if (ioctl(fd, EVIOCGRAB, 1) < 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static void* evdev_reader_thread(void* arg) {
    HSEvdevDevice* dev = (HSEvdevDevice*)arg;
    struct input_event evs[EVDEV_BUFFER_SIZE];
    uint64_t local_dropped = 0;

    while (atomic_load_explicit(&dev->active, memory_order_acquire)) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(dev->fd, &rfds);

        struct timeval tv = {0, 10000};
        int ret = select(dev->fd + 1, &rfds, NULL, NULL, &tv);

        if (ret < 0) {
            if (errno == EINTR) continue;
            atomic_store_explicit(&dev->active, false, memory_order_release);
            break;
        }
        if (ret == 0) continue;

        ssize_t n = read(dev->fd, evs, sizeof(evs));
        if (n < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;
            atomic_store_explicit(&dev->active, false, memory_order_release);
            break;
        }

        ssize_t count = n / sizeof(struct input_event);
        for (ssize_t i = 0; i < count; i++) {
            HSInputEvent ev = {
                .tv_sec = (u32)evs[i].time.tv_sec,
                .tv_usec = (u32)evs[i].time.tv_usec,
                .type = evs[i].type,
                .code = evs[i].code,
                .value = evs[i].value,
            };

            if (!hs_input_spsc_push(dev->queue, &ev, &local_dropped)) {
                local_dropped++;
            }
        }
    }

    if (local_dropped > 0 && dev->dropped_events) {
        atomic_fetch_add_explicit(dev->dropped_events, local_dropped, memory_order_relaxed);
    }

    return NULL;
}

static void* input_process_thread(void* arg) {
    HSInputStream* stream = (HSInputStream*)arg;
    HSInputEvent ev;

    while (atomic_load_explicit(&stream->running, memory_order_acquire)) {
        if (atomic_load_explicit(&stream->paused, memory_order_acquire)) {
            usleep(1000);
            continue;
        }

        pthread_mutex_lock(&stream->queue_mutex);
        while (hs_input_spsc_empty(&stream->queue)) {
            if (!atomic_load_explicit(&stream->running, memory_order_acquire)) {
                pthread_mutex_unlock(&stream->queue_mutex);
                goto exit_thread;
            }
            pthread_cond_wait(&stream->queue_not_empty, &stream->queue_mutex);
        }

        HSInput* state = atomic_load_explicit(&stream->active_state_ptr, memory_order_acquire);
        HSInput* write_target = (state == &stream->state_a) ? &stream->state_b : &stream->state_a;

        while (hs_input_spsc_pop(&stream->queue, &ev)) {
            apply_event_to_input(write_target, &ev);
            atomic_fetch_add_explicit(&stream->event_count, 1, memory_order_relaxed);
        }

        compute_dir_from_keys(write_target);
        write_target->frame++;
        write_target->elapsed = (f64)write_target->frame / (f64)HS_FPS;
        write_target->epoch = (f64)time(NULL);

        pthread_mutex_unlock(&stream->queue_mutex);
    }

exit_thread:
    return NULL;
}

static void* hotplug_thread(void* arg) {
    HSInputStream* stream = (HSInputStream*)arg;
    int inotify_fd = inotify_init();
    if (inotify_fd < 0) {
        return NULL;
    }

    int wd = inotify_add_watch(inotify_fd, "/dev/input", IN_CREATE | IN_DELETE);
    if (wd < 0) {
        close(inotify_fd);
        return NULL;
    }

    char buf[INOTIFY_BUF_LEN];
    while (atomic_load_explicit(&stream->running, memory_order_acquire)) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(inotify_fd, &rfds);
        struct timeval tv = {1, 0};
        int ret = select(inotify_fd + 1, &rfds, NULL, NULL, &tv);

        if (ret < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (ret == 0) continue;

        ssize_t len = read(inotify_fd, buf, sizeof(buf));
        if (len < 0) {
            if (errno == EINTR) continue;
            break;
        }

        ssize_t i = 0;
        while (i < len) {
            struct inotify_event* event = (struct inotify_event*)&buf[i];
            if (event->len > 0) {
                if (strncmp(event->name, "event", 5) == 0) {
                    char path[256];
                    snprintf(path, sizeof(path), "/dev/input/%s", event->name);

                    if (event->mask & IN_CREATE) {
                        if (stream->device_count < HS_INPUT_MAX_DEVICES) {
                            int fd = open_evdev(path);
                            if (fd >= 0) {
                                HSEvdevDevice* dev = &stream->devices[stream->device_count];
                                memset(dev, 0, sizeof(*dev));
                                dev->fd = fd;
                                dev->queue = &stream->queue;
                                dev->dropped_events = &stream->dropped_events;
                                strncpy(dev->path, path, sizeof(dev->path) - 1);
                                atomic_init(&dev->active, true);

                                pthread_t tid;
                                if (pthread_create(&tid, NULL, evdev_reader_thread, dev) == 0) {
                                    dev->thread = tid;
                                    stream->device_count++;
                                } else {
                                    close(fd);
                                }
                            }
                        }
                    } else if (event->mask & IN_DELETE) {
                    }
                }
            }
            i += INOTIFY_EVENT_SIZE + event->len;
        }
    }

    close(inotify_fd);
    return NULL;
}

bool hs_input_stream_init(HSInputStream* stream) {
    memset(stream, 0, sizeof(*stream));
    hs_input_spsc_init(&stream->queue);
    hs_input_init(&stream->state_a);
    hs_input_init(&stream->state_b);
    pthread_mutex_init(&stream->queue_mutex, NULL);
    pthread_cond_init(&stream->queue_not_empty, NULL);
    pthread_cond_init(&stream->queue_not_full, NULL);
    atomic_init(&stream->running, false);
    atomic_init(&stream->paused, false);
    atomic_init(&stream->event_count, 0);
    atomic_init(&stream->dropped_events, 0);
    atomic_init(&stream->active_state_ptr, &stream->state_a);
    stream->hotplug_enabled = false;
    return true;
}

bool hs_input_stream_add_device(HSInputStream* stream, const char* device_path) {
    if (stream->device_count >= HS_INPUT_MAX_DEVICES) {
        return false;
    }

    int fd = open_evdev(device_path);
    if (fd < 0) {
        return false;
    }

    HSEvdevDevice* dev = &stream->devices[stream->device_count];
    memset(dev, 0, sizeof(*dev));
    dev->fd = fd;
    dev->queue = &stream->queue;
    dev->dropped_events = &stream->dropped_events;
    strncpy(dev->path, device_path, sizeof(dev->path) - 1);
    atomic_init(&dev->active, false);

    stream->device_count++;
    return true;
}

bool hs_input_stream_start(HSInputStream* stream) {
    if (atomic_load_explicit(&stream->running, memory_order_acquire)) {
        return false;
    }

    for (int i = 0; i < stream->device_count; i++) {
        HSEvdevDevice* dev = &stream->devices[i];
        atomic_store_explicit(&dev->active, true, memory_order_relaxed);

        if (pthread_create(&dev->thread, NULL, evdev_reader_thread, dev) != 0) {
            atomic_store_explicit(&dev->active, false, memory_order_relaxed);
            for (int j = 0; j < i; j++) {
                HSEvdevDevice* prev = &stream->devices[j];
                atomic_store_explicit(&prev->active, false, memory_order_relaxed);
                if (prev->fd >= 0) {
                    ioctl(prev->fd, EVIOCGRAB, 0);
                    close(prev->fd);
                    prev->fd = -1;
                }
                pthread_join(prev->thread, NULL);
            }
            return false;
        }
    }

    atomic_store_explicit(&stream->running, true, memory_order_relaxed);

    if (pthread_create(&stream->process_thread, NULL, input_process_thread, stream) != 0) {
        atomic_store_explicit(&stream->running, false, memory_order_relaxed);
        for (int i = 0; i < stream->device_count; i++) {
            HSEvdevDevice* dev = &stream->devices[i];
            atomic_store_explicit(&dev->active, false, memory_order_relaxed);
            if (dev->fd >= 0) {
                ioctl(dev->fd, EVIOCGRAB, 0);
                close(dev->fd);
                dev->fd = -1;
            }
            pthread_join(dev->thread, NULL);
        }
        return false;
    }

    if (stream->hotplug_enabled) {
        if (pthread_create(&stream->hotplug_thread, NULL, hotplug_thread, stream) != 0) {
        }
    }

    return true;
}

void hs_input_stream_stop(HSInputStream* stream) {
    atomic_store_explicit(&stream->running, false, memory_order_relaxed);

    pthread_cond_broadcast(&stream->queue_not_empty);

    for (int i = 0; i < stream->device_count; i++) {
        HSEvdevDevice* dev = &stream->devices[i];
        atomic_store_explicit(&dev->active, false, memory_order_relaxed);
        if (dev->fd >= 0) {
            ioctl(dev->fd, EVIOCGRAB, 0);
            close(dev->fd);
            dev->fd = -1;
        }
    }

    for (int i = 0; i < stream->device_count; i++) {
        pthread_join(stream->devices[i].thread, NULL);
    }

    if (stream->hotplug_enabled) {
        pthread_join(stream->hotplug_thread, NULL);
    }

    pthread_join(stream->process_thread, NULL);
}

void hs_input_stream_shutdown(HSInputStream* stream) {
    hs_input_stream_stop(stream);
    pthread_mutex_destroy(&stream->queue_mutex);
    pthread_cond_destroy(&stream->queue_not_empty);
    pthread_cond_destroy(&stream->queue_not_full);
    memset(stream, 0, sizeof(*stream));
}

const HSInput* hs_input_stream_get_state(const HSInputStream* stream) {
    return atomic_load_explicit(&stream->active_state_ptr, memory_order_acquire);
}

void hs_input_stream_swap_buffers(HSInputStream* stream) {
    HSInput* old = atomic_exchange_explicit(
        &stream->active_state_ptr,
        (atomic_load(&stream->active_state_ptr) == &stream->state_a) ? &stream->state_b : &stream->state_a,
        memory_order_acq_rel
    );
    memset(old, 0, sizeof(HSInput));
}

void hs_input_stream_process_events(HSInputStream* stream) {
    (void)stream;
}

void hs_input_stream_pause(HSInputStream* stream) {
    atomic_store_explicit(&stream->paused, true, memory_order_relaxed);
}

void hs_input_stream_resume(HSInputStream* stream) {
    atomic_store_explicit(&stream->paused, false, memory_order_relaxed);
    pthread_cond_broadcast(&stream->queue_not_empty);
}

bool hs_input_stream_get_device_caps(HSInputStream* stream, int device_idx,
                                     uint8_t* ev_bits, size_t ev_bits_size,
                                     uint8_t* key_bits, size_t key_bits_size,
                                     uint8_t* abs_bits, size_t abs_bits_size) {
    if (device_idx < 0 || device_idx >= stream->device_count) {
        return false;
    }

    HSEvdevDevice* dev = &stream->devices[device_idx];
    if (dev->fd < 0) {
        return false;
    }

    if (ev_bits && ev_bits_size > 0) {
        ioctl(dev->fd, EVIOCGBIT(0, ev_bits_size), ev_bits);
    }
    if (key_bits && key_bits_size > 0) {
        ioctl(dev->fd, EVIOCGBIT(EV_KEY, key_bits_size), key_bits);
    }
    if (abs_bits && abs_bits_size > 0) {
        ioctl(dev->fd, EVIOCGBIT(EV_ABS, abs_bits_size), abs_bits);
    }

    return true;
}

void hs_input_stream_enable_hotplug(HSInputStream* stream) {
    stream->hotplug_enabled = true;
}

void hs_input_stream_disable_hotplug(HSInputStream* stream) {
    stream->hotplug_enabled = false;
}
