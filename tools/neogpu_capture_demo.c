/*
 * neogpu_capture_demo.c -- Message Capture & Replay Demo
 * 
 * Demonstrates:
 * 1. Message-passing GPU rendering via hs_gpu
 * 2. Message capture (HSCAP1 format)
 * 3. Message replay from capture
 * 4. Visual screenshot (PPM)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include "hs_gpu.h"
#include "hs_backend_gles.h"
#include "hs_graphics.h"

#define MAX_MSG_LOG 4096

static volatile bool g_run = true;
static void on_sig(int s) { (void)s; g_run = false; }


int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    printf("=== NeoGPU Capture & Replay Demo ===\n");

    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);

    static HSGpu gpu;
    hs_gpu_init(&gpu);

    HSBuffer* buf = hs_gpu_get_buffer(&gpu, 0);
    if (!hs_buffer_init(buf, 0, 256)) {
        fprintf(stderr, "Failed to init buffer bank 0\n");
        return 1;
    }
    
    float phase = 0.0f;
    hs_buffer_set_f32(buf, 0,  0.0f);
    hs_buffer_set_f32(buf, 1,  0.6f);
    hs_buffer_set_f32(buf, 2,  1.0f);
    hs_buffer_set_f32(buf, 3,  0.0f);
    hs_buffer_set_f32(buf, 4,  0.0f);
    hs_buffer_set_f32(buf, 5, -0.6f);
    hs_buffer_set_f32(buf, 6, -0.6f);
    hs_buffer_set_f32(buf, 7,  0.0f);
    hs_buffer_set_f32(buf, 8,  1.0f);
    hs_buffer_set_f32(buf, 9,  0.0f);
    hs_buffer_set_f32(buf, 10, 0.6f);
    hs_buffer_set_f32(buf, 11, -0.6f);
    hs_buffer_set_f32(buf, 12, 0.0f);
    hs_buffer_set_f32(buf, 13, 0.0f);
    hs_buffer_set_f32(buf, 14, 1.0f);
    buf->length = 15 * 4;

    HSBackend backend = hs_backend_gles_create();
    hs_gpu_attach_backend(&gpu, &backend);
    if (!gpu.backend) {
        fprintf(stderr, "Display backend unavailable (run as root with a display attached)\n");
        return 1;
    }

    printf("Recording 60 frames...\n");
    hs_gpu_start_recording(&gpu);
    
    for (int f = 0; f < 60 && g_run; f++) {
        phase += 0.1f;
        
        float r = 0.5f + 0.5f * sinf(phase);
        float g = 0.5f + 0.5f * sinf(phase + 2.0f);
        float b = 0.5f + 0.5f * sinf(phase + 4.0f);
        
        hs_gpu_begin_frame(&gpu);
        hs_gpu_clear(&gpu, v4_make(r * 0.2f, g * 0.2f, b * 0.3f, 1.0f));
        hs_gpu_draw(&gpu, 0);
        hs_gpu_process(&gpu);
        hs_gpu_end_frame(&gpu); /* presents; vsync paces the loop */
    }
    
    u32 log_count = hs_gpu_stop_recording(&gpu);
    printf("Captured %d messages\n", log_count);

    Message* msgs = gpu.log_buffer;
    
    static Message cap_msgs[MAX_MSG_LOG];
    static Payload cap_payloads[MAX_MSG_LOG];
    HSCapture cap;
    hs_capture_init(&cap, cap_msgs, cap_payloads, MAX_MSG_LOG);
    
    bool cap_ok = hs_capture_from_log(&gpu.system, msgs, log_count, &cap);
    printf("Capture from log: %s\n", cap_ok ? "OK" : "FAILED");
    
    /* Outputs go to the working directory: predictable /tmp names are unsafe
     * for a demo that runs as root. A stale file must not stand in for a
     * failed write, so remove it first. */
    const char* cap_path = "neogpu_capture.bin";
    const char* shot_path = "capture_demo.ppm";
    unlink(cap_path);
    bool write_ok = hs_capture_write_file(&cap, cap_path);
    printf("Write capture: %s (%s)\n", write_ok ? "OK" : "FAILED", cap_path);

    static Message replay_msgs[MAX_MSG_LOG];
    static Payload replay_payloads[MAX_MSG_LOG];
    HSCapture cap2;
    bool read_ok = hs_capture_read_file(&cap2, cap_path, replay_msgs, replay_payloads, MAX_MSG_LOG);
    printf("Read capture: %s\n", read_ok ? "OK" : "FAILED");

    /* The file round trip is the point: replay what was read back. */
    bool same = read_ok && cap2.count == cap.count;
    for (u32 i = 0; same && i < cap.count; i++) {
        same = memcmp(&cap.msgs[i], &cap2.msgs[i], sizeof(Message)) == 0 &&
               (!cap.msgs[i].payload_len ||
                memcmp(cap.payloads[cap.msgs[i].payload_idx].data, cap2.payloads[cap2.msgs[i].payload_idx].data,
                       cap.msgs[i].payload_len) == 0);
    }
    printf("Round trip identical: %s\n", same ? "YES" : "NO");

    printf("\n--- Replaying captured messages (from the file) ---\n");

    bool replay_ok = read_ok && hs_capture_replay(&gpu.system, &cap2);
    printf("Replay capture: %s\n", replay_ok ? "OK" : "FAILED");
    
    for (int f = 0; f < 30 && g_run; f++) {
        hs_gpu_begin_frame(&gpu);
        hs_gpu_clear(&gpu, v4_make(0.1f, 0.1f, 0.1f, 1.0f));
        hs_gpu_draw(&gpu, 0);
        if (f == 29) (void)hs_backend_gles_capture_next(&backend, shot_path);
        hs_gpu_process(&gpu);
        hs_gpu_end_frame(&gpu);
    }
    bool shot_ok = hs_backend_gles_capture_ok(&backend);
    hs_gpu_shutdown(&gpu);

    printf("\n=== Demo Complete ===\n");
    printf("Files created (current directory):\n");
    printf("  %s - Message capture (HSCAP1)\n", cap_path);
    printf("  %s - Screenshot (%s)\n", shot_path, shot_ok ? "written" : "FAILED");

    return cap_ok && write_ok && read_ok && same && replay_ok && shot_ok ? 0 : 1;
}
