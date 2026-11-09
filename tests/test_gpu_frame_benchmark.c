#define _GNU_SOURCE
#include "hs_gpu.h"
#include "comparison_transport.h"
#include <inttypes.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "%d: %s\n", __LINE__, #x); exit(1); } } while (0)
static HSGpu gpu;
#ifndef HS_GPU_PARTICIPANT_FRAMES
static HSFabric manual;
#define FABRIC manual
#else
#define FABRIC gpu.participants
#endif
static void* reference;
static const char* mode;
static bool is_reference, is_legacy;
static u32 producers, frames, draws;
static atomic_bool go;
static atomic_uint ready;
typedef struct __attribute__((aligned(64))) { u64 value; u8 padding[64 - sizeof(u64)]; } BackendCounter;
static BackendCounter backend_counter;
#define backend_calls backend_counter.value
typedef struct __attribute__((aligned(64))) { u32 id; HSParticipant owner; u64 accepted, retries; } Producer;
_Static_assert(sizeof(BackendCounter) == 64 && sizeof(Producer) % 64 == 0, "Observer counters must not share writer cache lines");
static Producer workers[HS_FABRIC_PARTICIPANTS];

static u64 ns(void) {
    struct timespec t; REQUIRE(clock_gettime(CLOCK_MONOTONIC, &t) == 0);
    return (u64)t.tv_sec * 1000000000ull + t.tv_nsec;
}
static void pin(u32 cpu) {
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
    REQUIRE(pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0);
}
static const HSParticipantFrame* view(u32 lane) {
    if (is_reference) {
        const HSParticipantFrame* frame = comparison_rigtorp.view(reference, lane);
        return frame->sequence ? frame : NULL;
    }
    return FABRIC.views[lane].active ? &FABRIC.views[lane].frame : NULL;
}
static HSFabricResult join(HSParticipant* owner) {
#ifdef HS_GPU_PARTICIPANT_FRAMES
    return hs_gpu_participant_join(&gpu, owner);
#else
    return hs_fabric_join(&manual, owner);
#endif
}
static bool submit(Producer* producer, const HSParticipantFrame* frame) {
    if (is_reference) return comparison_rigtorp.submit(reference, producer->id, frame);
#ifdef HS_GPU_PARTICIPANT_FRAMES
    HSFabricResult result = hs_gpu_participant_submit(&gpu, producer->owner, frame);
#else
    HSFabricResult result = hs_fabric_submit(&manual, producer->owner, frame);
#endif
    REQUIRE(result == HS_FABRIC_OK || result == HS_FABRIC_FULL || result == HS_FABRIC_BUSY);
    return result == HS_FABRIC_OK;
}
static void* produce(void* pointer) {
    Producer* producer = pointer;
    pin(1 + producer->id % 3);
    atomic_fetch_add_explicit(&ready, 1, memory_order_release);
    while (!atomic_load_explicit(&go, memory_order_acquire)) sched_yield();
    for (u32 sequence = 1; sequence <= frames; sequence++) {
        HSParticipantFrame input = {.sequence = sequence, .count = draws};
        for (u32 i = 0; i < draws; i++) input.draws[i] = (HSRenderCmd){.op = HS_RC_DRAW, .a = (u8)producer->id};
        while (!submit(producer, &input)) { producer->retries++; sched_yield(); }
        producer->accepted++;
    }
    return NULL;
}
static void execute(void* unused, const HSFrameContext* frame) {
    (void)unused;
    const HSRenderList* list = frame->render;
    REQUIRE(list && !list->overflow);
    const HSRenderCmd clear = {.op = HS_RC_CLEAR, .f3 = 1};
    if (is_legacy) {
        REQUIRE(list->count == draws + 1 && !memcmp(&list->cmds[0], &clear, sizeof(clear)));
        for (u32 i = 1; i <= draws; i++) {
            HSRenderCmd expected = {.op = HS_RC_DRAW};
            REQUIRE(!memcmp(&list->cmds[i], &expected, sizeof(expected)));
        }
    } else {
        REQUIRE(list->count >= 1 && !memcmp(&list->cmds[0], &clear, sizeof(clear)));
        u32 cursor = 1;
        for (u32 lane = 0; lane < producers; lane++) {
            const HSParticipantFrame* current = view(lane);
            if (!current) continue;
            REQUIRE(current->count == draws);
            for (u32 i = 0; i < draws; i++) {
                HSRenderCmd expected = {.op = HS_RC_DRAW, .a = (u8)lane};
                REQUIRE(cursor < list->count && !memcmp(&list->cmds[cursor++], &expected, sizeof(expected)));
            }
        }
        REQUIRE(cursor == list->count);
    }
    backend_calls++;
}
static const HSBackendOps ops = {.execute = execute};
static HSBackend backend = {.ops = &ops};
static void end_frame(void) {
    u64 before = backend_calls;
#ifdef HS_GPU_PARTICIPANT_FRAMES
    if (is_legacy) hs_gpu_end_frame(&gpu);
    else REQUIRE(hs_gpu_end_frame_checked(&gpu));
#else
    hs_gpu_end_frame(&gpu);
#endif
    REQUIRE(backend_calls == before + 1);
}
/* Equivalent external plumbing for the old runtime and the static reference.
 * The same list is restored after callback dispatch, as in the new runtime. */
static u32 append_manual(void) {
    u32 added = 0;
    for (u32 lane = 0; lane < producers; lane++) { const HSParticipantFrame* current = view(lane); if (current) added += current->count; }
    REQUIRE(gpu.render.count == 1 && added <= HS_RENDER_MAX_CMDS - 1);
    u32 cursor = 1;
    for (u32 lane = 0; lane < producers; lane++) {
        const HSParticipantFrame* current = view(lane);
        if (current) { memcpy(&gpu.render.cmds[cursor], current->draws, current->count * sizeof(HSRenderCmd)); cursor += current->count; }
    }
    gpu.render.count += added;
    return added;
}
static void empty_frame(void) {
    hs_gpu_begin_frame(&gpu);
    const HSRenderCmd clear = {.op = HS_RC_CLEAR, .f3 = 1};
    REQUIRE(hs_render_push(&gpu.render, &clear));
}
int main(int argc, char** argv) {
    REQUIRE(argc == 5); alarm(120);
    mode = argv[1]; producers = (u32)strtoul(argv[2], NULL, 10); frames = (u32)strtoul(argv[3], NULL, 10); draws = (u32)strtoul(argv[4], NULL, 10);
    REQUIRE(!strcmp(mode, "legacy") || !strcmp(mode, "native") || !strcmp(mode, "reference"));
    is_reference = !strcmp(mode, "reference"); is_legacy = !strcmp(mode, "legacy");
    REQUIRE(producers && producers <= HS_FABRIC_PARTICIPANTS && frames && frames <= 1000000 && draws && draws <= HS_FABRIC_DRAWS);
    pin(0); u64 init_start = ns(); hs_gpu_init(&gpu); gpu.system.recording = false;
#ifdef HS_GPU_PARTICIPANT_FRAMES
    REQUIRE(hs_gpu_participants_ready(&gpu));
#else
    if (!is_reference && !is_legacy) REQUIRE(hs_fabric_init(&manual));
#endif
    hs_gpu_attach_backend(&gpu, &backend);
    u64 init_ns = ns() - init_start;
    u64 accepted = 0, applied = 0, retries = 0, start;
    if (is_legacy) {
        REQUIRE(producers == 1);
        start = ns();
        for (u32 frame = 0; frame < frames; frame++) {
            hs_gpu_begin_frame(&gpu); hs_gpu_clear(&gpu, v4_make(0, 0, 0, 1));
            for (u32 i = 0; i < draws; i++) hs_gpu_draw(&gpu, 0);
            hs_gpu_process(&gpu); end_frame();
        }
        accepted = applied = frames;
    } else {
        if (is_reference) { reference = comparison_rigtorp.create(producers); REQUIRE(reference); }
        else for (u32 i = 0; i < producers; i++) { REQUIRE(join(&workers[i].owner) == HS_FABRIC_OK); REQUIRE(workers[i].owner.slot == i); }
        pthread_t threads[HS_FABRIC_PARTICIPANTS];
        u64 observed[HS_FABRIC_PARTICIPANTS] = {0};
        atomic_init(&go, false); atomic_init(&ready, 0);
        for (u32 i = 0; i < producers; i++) { workers[i].id = i; REQUIRE(pthread_create(&threads[i], NULL, produce, &workers[i]) == 0); }
        while (atomic_load_explicit(&ready, memory_order_acquire) < producers) sched_yield();
        start = ns(); atomic_store_explicit(&go, true, memory_order_release);
        while (applied < (u64)frames * producers) {
            empty_frame(); bool manual_path = is_reference;
#ifndef HS_GPU_PARTICIPANT_FRAMES
            manual_path = true;
#endif
            u32 added = 0;
            if (manual_path) {
                if (is_reference) comparison_rigtorp.step(reference);
                else hs_fabric_step(&FABRIC, HS_FABRIC_PARTICIPANTS);
                added = append_manual();
            }
            end_frame();
            if (manual_path) { REQUIRE(gpu.render.count == added + 1); gpu.render.count = 1; }
            REQUIRE(gpu.render.count == 1);
            for (u32 i = 0; i < producers; i++) {
                const HSParticipantFrame* current = view(i);
                if (!current || current->sequence == observed[i]) continue;
                REQUIRE(current->sequence == observed[i] + 1 && current->count == draws);
                observed[i]++; applied++;
            }
        }
        for (u32 i = 0; i < producers; i++) { REQUIRE(pthread_join(threads[i], NULL) == 0); REQUIRE(observed[i] == frames); accepted += workers[i].accepted; retries += workers[i].retries; }
        REQUIRE(accepted == applied);
        if (!is_reference) REQUIRE(FABRIC.applied == applied);
    }
    u64 elapsed = ns() - start;
    printf("{\"mode\":\"%s\",\"producers\":%u,\"draws\":%u,\"accepted\":%" PRIu64 ",\"applied\":%" PRIu64
           ",\"elapsed_ns\":%" PRIu64 ",\"retries\":%" PRIu64 ",\"backend_calls\":%" PRIu64 ",\"init_ns\":%" PRIu64 ",\"gpu_bytes\":%zu,\"fabric_bytes\":%zu,\"fabric_address_mod64\":%" PRIuPTR ",\"producer_counter_stride\":%zu,\"backend_counter_mod64\":%" PRIuPTR "}\n",
           mode, producers, draws, accepted, applied, elapsed, retries, backend_calls, init_ns, sizeof(HSGpu), sizeof(HSFabric), (uintptr_t)&FABRIC % 64, sizeof(Producer), (uintptr_t)&backend_counter % 64);
    if (reference) comparison_rigtorp.destroy(reference);
#ifndef HS_GPU_PARTICIPANT_FRAMES
    if (!is_reference && !is_legacy) hs_fabric_destroy(&manual);
#endif
    hs_gpu_shutdown(&gpu);
    return 0;
}
