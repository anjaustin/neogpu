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
    int first = -1, second = -1;
    char first_path[256], second_path[256];
    HSInputStream stream = {0};
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
    printf("Real evdev/uinput: %d checks passed, including two readers and hot-add/remove\n", checks);
cleanup:
    hs_input_stream_shutdown(&stream);
    if (second >= 0) { ioctl(second, UI_DEV_DESTROY); close(second); }
    if (first >= 0) { ioctl(first, UI_DEV_DESTROY); close(first); }
    return result;
}
