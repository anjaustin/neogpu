#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <dirent.h>
#include "hs_input_stream.h"

static volatile int g_run = 1;
static void on_sig(int s) { (void)s; g_run = 0; }

static void list_evdev_devices(void) {
    DIR* d = opendir("/dev/input");
    if (!d) {
        fprintf(stderr, "Cannot open /dev/input\n");
        return;
    }

    struct dirent* ent;
    printf("Available input devices:\n");
    while ((ent = readdir(d)) != NULL) {
        if (strncmp(ent->d_name, "event", 5) == 0) {
            char path[256];
            snprintf(path, sizeof(path), "/dev/input/%s", ent->d_name);
            printf("  %s\n", path);
        }
    }
    closedir(d);
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);

    printf("=== NeoGPU Input Stream Test ===\n\n");

    list_evdev_devices();
    printf("\n");

    HSInputStream stream;
    if (!hs_input_stream_init(&stream)) {
        fprintf(stderr, "Failed to init input stream\n");
        return 1;
    }

    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            printf("Adding device: %s\n", argv[i]);
            if (!hs_input_stream_add_device(&stream, argv[i])) {
                fprintf(stderr, "Failed to add device: %s\n", argv[i]);
            }
        }
    } else {
        if (!hs_input_stream_add_device(&stream, "/dev/input/event0")) {
            fprintf(stderr, "Warning: Could not open /dev/input/event0 (need root?)\n");
            fprintf(stderr, "Usage: %s [device1] [device2] ...\n", argv[0]);
        }
    }

    if (!hs_input_stream_start(&stream)) {
        fprintf(stderr, "Failed to start input stream\n");
        hs_input_stream_shutdown(&stream);
        return 1;
    }

    printf("Input stream started. Press keys, move mouse, etc.\n");
    printf("Press Ctrl+C to exit.\n\n");

    int last_frame = 0;
    while (g_run) {
        const HSInput* state = hs_input_stream_get_state(&stream);
        u32 cur_frame = state->frame;

        if (cur_frame != last_frame) {
            printf("\rFrame %5u | dir=(%.2f, %.2f) | keys: %s%s%s%s | btn=%d | mouse=(%.0f, %.0f) | ev=%lu   ",
                cur_frame,
                state->dir_x, state->dir_y,
                state->key_left ? "L" : "_",
                state->key_right ? "R" : "_",
                state->key_up ? "U" : "_",
                state->key_down ? "D" : "_",
                state->button1,
                state->mouse_x, state->mouse_y,
                (unsigned long)atomic_load(&stream.event_count));
            fflush(stdout);

            hs_input_stream_swap_buffers(&stream);
            last_frame = cur_frame;
        }

        usleep(16000);
    }

    printf("\n\nShutting down...\n");
    hs_input_stream_shutdown(&stream);

    printf("Event count: %lu\n", (unsigned long)atomic_load(&stream.event_count));
    printf("Dropped events: %lu\n", (unsigned long)atomic_load(&stream.dropped_events));

    return 0;
}
