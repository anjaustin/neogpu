#define _POSIX_C_SOURCE 200809L
#include "hs_fabric.h"
#include "hs_gpu.h"
#include "hs_backend_gles.h"
#include "hs_graphics.h"
#include "graphics_capture.h"
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
static HSFabric fabric;
static HSGpu gpu;
static atomic_uint phase, acknowledged[2];
typedef struct { u32 id; HSParticipant owner, original; } Actor;
static Actor actors[2];
static u32 expected_left, expected_right, pixel_checks, presentations;
static bool capture, verification_enabled;

/* Linker-only test interception, immediately before the existing GLES backend's
 * real eglSwapBuffers. No backend implementation/hook is changed for evidence. */
extern EGLBoolean __real_eglSwapBuffers(EGLDisplay display, EGLSurface surface);
EGLBoolean __wrap_eglSwapBuffers(EGLDisplay display, EGLSurface surface) {
    if (!verification_enabled) return __real_eglSwapBuffers(display, surface);
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    REQUIRE(viewport[2] > 0 && viewport[3] > 0);
    const u32 expected[2] = {expected_left, expected_right};
    for (u32 side = 0; side < 2; side++) {
        u8 pixel[4];
        glReadPixels(viewport[0] + viewport[2] * (side ? 3 : 1) / 4,
                     viewport[1] + viewport[3] / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
        REQUIRE(glGetError() == GL_NO_ERROR);
        for (u32 channel = 0; channel < 3; channel++) {
            u32 target = expected[side] == channel + 1 ? 255 : 0;
            REQUIRE(abs((int)pixel[channel] - (int)target) <= 4);
        }
        pixel_checks++;
    }
    if (capture) REQUIRE(test_capture_frame() == 0);
    presentations++;
    return __real_eglSwapBuffers(display, surface);
}

static void submit(Actor* actor, u64 sequence, bool visible) {
    HSParticipantFrame frame = {.sequence = sequence, .count = visible ? 1 : 0};
    frame.draws[0] = (HSRenderCmd){.op = HS_RC_DRAW, .a = (u8)actor->owner.slot};
    REQUIRE(hs_fabric_submit(&fabric, actor->owner, &frame) == HS_FABRIC_OK);
    memset(&frame, 0xff, sizeof(frame));
}

static void* participant(void* pointer) {
    Actor* actor = pointer;
    HSParticipantFrame private_draft = {0};
    for (u32 stage = 1; stage <= 6; stage++) {
        while (atomic_load_explicit(&phase, memory_order_acquire) < stage) sched_yield();
        if (stage == 1) submit(actor, 1, true);
        if (actor->id == 0) {
            if (stage == 2) {
                /* Private incomplete edit: never submitted, must not hide A. */
                private_draft.sequence = 2;
                private_draft.count = 0;
            }
            if (stage == 3) REQUIRE(private_draft.sequence == 2 && private_draft.count == 0);
            if (stage == 4) REQUIRE(hs_fabric_leave(&fabric, actor->owner) == HS_FABRIC_OK);
            if (stage == 5) {
                REQUIRE(hs_fabric_join(&fabric, &actor->owner) == HS_FABRIC_OK);
                REQUIRE(actor->owner.slot == actor->original.slot &&
                        actor->owner.generation > actor->original.generation);
                HSParticipantFrame stale = {.sequence = 2};
                REQUIRE(hs_fabric_submit(&fabric, actor->original, &stale) == HS_FABRIC_STALE);
            }
            if (stage == 6) submit(actor, 1, true);
        } else {
            if (stage == 2) submit(actor, 2, false);
            if (stage == 3) submit(actor, 3, true);
        }
        atomic_store_explicit(&acknowledged[actor->id], stage, memory_order_release);
    }
    return NULL;
}

static void vertices(u8 bank, f32 center, f32 red, f32 green, f32 blue) {
    HSBuffer* buffer = hs_gpu_get_buffer(&gpu, bank);
    if (!buffer->data) REQUIRE(hs_buffer_init(buffer, bank, 60));
    const f32 data[15] = {
        center, 0.6f, red, green, blue,
        center - 0.3f, -0.6f, red, green, blue,
        center + 0.3f, -0.6f, red, green, blue,
    };
    for (u32 i = 0; i < 15; i++) hs_buffer_set_f32(buffer, i, data[i]);
    buffer->length = sizeof(data);
}

int main(void) {
    alarm(30);
    REQUIRE(hs_fabric_init(&fabric));
    hs_gpu_init(&gpu);
    gpu.system.recording = false;
    for (u32 i = 0; i < 2; i++) {
        actors[i].id = i;
        REQUIRE(hs_fabric_join(&fabric, &actors[i].owner) == HS_FABRIC_OK);
        actors[i].original = actors[i].owner;
        atomic_init(&acknowledged[i], 0);
    }
    atomic_init(&phase, 0);
    vertices((u8)actors[0].owner.slot, -0.5f, 1, 0, 0);
    vertices((u8)actors[1].owner.slot, 0.5f, 0, 1, 0);
    HSBackend backend = hs_backend_gles_create();
    hs_gpu_attach_backend(&gpu, &backend);
    REQUIRE(gpu.backend != NULL);
    verification_enabled = true;
    pthread_t threads[2];
    for (u32 i = 0; i < 2; i++) REQUIRE(pthread_create(&threads[i], NULL, participant, &actors[i]) == 0);
    for (u32 stage = 1; stage <= 6; stage++) {
        atomic_store_explicit(&phase, stage, memory_order_release);
        for (u32 i = 0; i < 2; i++)
            while (atomic_load_explicit(&acknowledged[i], memory_order_acquire) < stage) sched_yield();
        while (hs_fabric_step(&fabric, HS_FABRIC_PARTICIPANTS)) { }
        /* The renderer alone replaces a bank, after its old owner retired. */
        if (stage == 5) vertices((u8)actors[0].owner.slot, -0.5f, 0, 0, 1);
        expected_left = stage <= 3 ? 1 : stage == 6 ? 3 : 0;
        expected_right = stage == 2 ? 0 : 2;
        char path[128];
        snprintf(path, sizeof(path), "tests/evidence-fabric-stage-%u.ppm", stage);
        REQUIRE(setenv("NEOGPU_CAPTURE", path, 1) == 0);
        for (u32 repeat = 0; repeat < 3; repeat++) {
            hs_gpu_begin_frame(&gpu);
            const f32 black[4] = {0, 0, 0, 1};
            REQUIRE(hs_fabric_compose(&fabric, &gpu.render, black));
            capture = repeat == 1;
            hs_gpu_end_frame(&gpu);
        }
        printf("{\"stage\":%u,\"expected_left_rgb_channel\":%u,\"expected_right_rgb_channel\":%u,"
               "\"pixel_checks_total\":%u,\"passed\":true}\n", stage, expected_left, expected_right, pixel_checks);
    }
    for (u32 i = 0; i < 2; i++) REQUIRE(pthread_join(threads[i], NULL) == 0);
    REQUIRE(fabric.applied == 5 && fabric.retired == 1);
    REQUIRE(presentations == 18 && pixel_checks == 36);
    hs_gpu_shutdown(&gpu);
    hs_fabric_destroy(&fabric);
    printf("{\"test\":\"participant_graphics\",\"applied\":5,\"retired\":1,\"presentations\":18,\"pixel_checks\":36,\"passed\":true}\n");
    return 0;
}
