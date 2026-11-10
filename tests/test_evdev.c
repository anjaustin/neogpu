/* Privileged integration: exclusively uses virtual devices created by this test. */
#include "hs_input_stream.h"
#include <linux/uinput.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <glob.h>
#include <stdio.h>
#include <unistd.h>
#include <signal.h>

static double milliseconds(void) {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return time.tv_sec * 1000.0 + time.tv_nsec / 1e6;
}

static int virtual_device(char path[256]) {
    int fd = open("/dev/uinput", O_WRONLY | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) return -1;
    if (ioctl(fd, UI_SET_EVBIT, EV_KEY) || ioctl(fd, UI_SET_KEYBIT, KEY_LEFT) ||
        ioctl(fd, UI_SET_KEYBIT, KEY_RIGHT)) goto fail;
    struct uinput_setup setup = {.id = {.bustype = BUS_VIRTUAL, .vendor = 0x1234, .product = 0x5678}};
    snprintf(setup.name, sizeof(setup.name), "neogpu-p0-%d", (int)getpid());
    if (ioctl(fd, UI_DEV_SETUP, &setup) || ioctl(fd, UI_DEV_CREATE)) goto fail;
    char name[128];
    if (ioctl(fd, UI_GET_SYSNAME(sizeof(name)), name) < 0) goto fail;
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "/sys/class/input/%s/event*", name);
    double deadline = milliseconds() + 2000;
    do {
        glob_t entries = {0};
        if (glob(pattern, 0, NULL, &entries) == 0 && entries.gl_pathc) {
            const char* event = strrchr(entries.gl_pathv[0], '/');
            snprintf(path, 256, "/dev/input/%s", event + 1);
            globfree(&entries);
            if (access(path, F_OK) == 0) return fd;
        } else globfree(&entries);
        usleep(1000);
    } while (milliseconds() < deadline);
fail:
    ioctl(fd, UI_DEV_DESTROY);
    close(fd);
    return -1;
}

static bool emit(int fd, u16 code, int pressed) {
    struct input_event events[2] = {{.type = EV_KEY, .code = code, .value = pressed}, {.type = EV_SYN, .code = SYN_REPORT}};
    return write(fd, events, sizeof(events)) == sizeof(events);
}

static bool wait_keys(HSInputStream* stream, bool left, bool right) {
    double deadline = milliseconds() + 2000;
    do {
        const HSInput* state = hs_input_stream_get_state(stream);
        if (state->key_left == left && state->key_right == right) return true;
        usleep(1000);
    } while (milliseconds() < deadline);
    return false;
}

#define REQUIRE(condition) do { checks++; if (!(condition)) { \
    fprintf(stderr, "evdev:%d: %s\n", __LINE__, #condition); result = 1; goto cleanup; } } while (0)

int main(void) {
    alarm(15);
    int result = 0, checks = 0;
    int first = -1, second = -1, third = -1, fourth = -1;
    char first_path[256], second_path[256], third_path[256], fourth_path[256];
    static HSInputStream stream, manual, unplug;
    REQUIRE(hs_input_stream_init(&stream));
    first = virtual_device(first_path);
    REQUIRE(first >= 0);
    REQUIRE(hs_input_stream_add_device(&stream, first_path));
    hs_input_stream_enable_hotplug(&stream);
    REQUIRE(hs_input_stream_start(&stream));
    REQUIRE(emit(first, KEY_RIGHT, 1));
    REQUIRE(wait_keys(&stream, false, true));
    /* The watch exists before start returns; creating a second device tests hot-add. */
    second = virtual_device(second_path);
    REQUIRE(second >= 0);
    bool attached = false;
    double deadline = milliseconds() + 2000;
    do {
        pthread_mutex_lock(&stream.device_mutex);
        for (int i = 0; i < stream.device_count; i++) {
            if (!strcmp(stream.devices[i].path, second_path) && stream.devices[i].thread_started) attached = true;
        }
        pthread_mutex_unlock(&stream.device_mutex);
        if (!attached) usleep(1000);
    } while (!attached && milliseconds() < deadline);
    REQUIRE(attached);
    REQUIRE(emit(second, KEY_LEFT, 1));
    REQUIRE(wait_keys(&stream, true, true));
    REQUIRE(emit(first, KEY_RIGHT, 0));
    REQUIRE(wait_keys(&stream, true, false));
    ioctl(second, UI_DEV_DESTROY); close(second); second = -1;
    REQUIRE(wait_keys(&stream, false, false));
    hs_input_stream_shutdown(&stream);

    /* SYN_DROPPED: the partial packet is discarded and key state is re-read
     * from the kernel (EVIOCGKEY), so a press whose event was lost is seen. */
    REQUIRE(hs_input_stream_init(&manual));
    third = virtual_device(third_path);
    REQUIRE(third >= 0);
    REQUIRE(hs_input_stream_add_device(&manual, third_path));
    REQUIRE(emit(third, KEY_LEFT, 1));          /* kernel state: LEFT down; event never read */
    usleep(20000);
    struct input_event dropped[] = {
        {.type = EV_SYN, .code = SYN_DROPPED},
        {.type = EV_KEY, .code = KEY_RIGHT, .value = 1},  /* incomplete packet: must be discarded */
        {.type = EV_REL, .code = REL_X, .value = 50},     /* not covered by the key resync */
        {.type = EV_SYN, .code = SYN_REPORT},
    };
    hs_input_stream_device_events(&manual, 0, dropped, 4);
    hs_input_stream_process_events(&manual);
    const HSInput* snapshot = hs_input_stream_get_state(&manual);
    REQUIRE(snapshot->key_left == 1 && snapshot->key_right == 0 && snapshot->mouse_x == 0);
    REQUIRE(emit(third, KEY_LEFT, 0));          /* release also lost */
    usleep(20000);
    hs_input_stream_device_events(&manual, 0, dropped, 4);
    hs_input_stream_process_events(&manual);
    snapshot = hs_input_stream_get_state(&manual);
    REQUIRE(snapshot->key_left == 0 && snapshot->key_right == 0 && snapshot->mouse_x == 0);
    /* Framing events do not consume queue slots or advance the input frame. */
    uint64_t before = atomic_load(&manual.event_count);
    u32 frame = snapshot->frame;
    struct input_event framing[] = {{.type = EV_MSC, .code = MSC_SCAN, .value = 5}, {.type = EV_SYN, .code = SYN_REPORT}};
    hs_input_stream_device_events(&manual, 0, framing, 2);
    hs_input_stream_process_events(&manual);
    REQUIRE(atomic_load(&manual.event_count) == before && hs_input_stream_get_state(&manual)->frame == frame);
    hs_input_stream_shutdown(&manual);

    /* Unplug without hotplug: keys are released, and the slot is reclaimed
     * so a device can be added again (previously the fd stayed grabbed and
     * the path was rejected as a duplicate). */
    REQUIRE(hs_input_stream_init(&unplug));
    REQUIRE(hs_input_stream_add_device(&unplug, third_path));
    REQUIRE(hs_input_stream_start(&unplug));
    REQUIRE(emit(third, KEY_LEFT, 1));
    REQUIRE(wait_keys(&unplug, true, false));
    ioctl(third, UI_DEV_DESTROY); close(third); third = -1;
    REQUIRE(wait_keys(&unplug, false, false));
    deadline = milliseconds() + 2000;
    while (!atomic_load(&unplug.devices[0].failed) && milliseconds() < deadline) usleep(1000);
    REQUIRE(atomic_load(&unplug.devices[0].failed));
    fourth = virtual_device(fourth_path);
    REQUIRE(fourth >= 0);
    REQUIRE(hs_input_stream_add_device(&unplug, fourth_path));
    int open_slots = 0;
    for (int i = 0; i < HS_INPUT_MAX_DEVICES; i++) open_slots += unplug.devices[i].fd >= 0;
    REQUIRE(open_slots == 1 && !atomic_load(&unplug.devices[0].failed));
    REQUIRE(emit(fourth, KEY_RIGHT, 1));
    REQUIRE(wait_keys(&unplug, false, true));

    /* Unplug while our queue is full: the kernel's own key-up events are lost,
     * yet the key must not stay held once the consumer catches up. */
    hs_input_stream_pause(&unplug);
    for (int i = 0; i < 400; i++) {
        struct input_event toggles[2] = {{.type = EV_KEY, .code = KEY_LEFT, .value = i & 1},
                                         {.type = EV_SYN, .code = SYN_REPORT}};
        if (write(fourth, toggles, sizeof(toggles)) != sizeof(toggles)) break;
    }
    usleep(50000);
    REQUIRE(atomic_load(&unplug.dropped_events) > 0);   /* the queue really overflowed */
    ioctl(fourth, UI_DEV_DESTROY); close(fourth); fourth = -1;
    deadline = milliseconds() + 2000;
    while (!atomic_load(&unplug.devices[0].failed) && !atomic_load(&unplug.devices[1].failed) &&
           milliseconds() < deadline) usleep(1000);
    hs_input_stream_resume(&unplug);
    REQUIRE(wait_keys(&unplug, false, false));
    printf("Real evdev/uinput: %d checks passed, including two readers, hot-add/remove, "
           "SYN_DROPPED resync and unplug without hotplug\n", checks);
cleanup:
    hs_input_stream_shutdown(&stream);
    hs_input_stream_shutdown(&manual);
    hs_input_stream_shutdown(&unplug);
    if (third >= 0) { ioctl(third, UI_DEV_DESTROY); close(third); }
    if (fourth >= 0) { ioctl(fourth, UI_DEV_DESTROY); close(fourth); }
    if (second >= 0) { ioctl(second, UI_DEV_DESTROY); close(second); }
    if (first >= 0) { ioctl(first, UI_DEV_DESTROY); close(first); }
    return result;
}
