#define _POSIX_C_SOURCE 200809L
#include "hs_gpu.h"
#include <errno.h>
#include <sched.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "%d: %s\n", __LINE__, #x); exit(1); } } while (0)
static HSGpu gpu;
static HSRenderList seen, saved;
static u32 executes, ends;
static bool reenter, rejected;
static int fail_slot = -1;
static bool initialized[HS_FABRIC_PARTICIPANTS];
static pthread_mutex_t race_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t race_changed = PTHREAD_COND_INITIALIZER;
static bool pause_retire, retire_reached, rejoined;
static HSParticipant replacement;
static bool composing_churn;

extern u32 __real_hs_fabric_step(HSFabric*, u32);
u32 __wrap_hs_fabric_step(HSFabric* fabric, u32 budget) {
    u32 events = __real_hs_fabric_step(fabric, budget);
    REQUIRE(pthread_mutex_lock(&race_lock) == 0);
    if (pause_retire && fabric == &gpu.participants) {
        REQUIRE(events == 1 && fabric->retired == 1);
        retire_reached = true; REQUIRE(pthread_cond_broadcast(&race_changed) == 0);
        while (!rejoined) REQUIRE(pthread_cond_wait(&race_changed, &race_lock) == 0);
        pause_retire = false;
    }
    REQUIRE(pthread_mutex_unlock(&race_lock) == 0);
    return events;
}

extern int __real_pthread_mutex_init(pthread_mutex_t*, const pthread_mutexattr_t*);
extern int __real_pthread_mutex_destroy(pthread_mutex_t*);
int __wrap_pthread_mutex_init(pthread_mutex_t* mutex, const pthread_mutexattr_t* attr) {
    int slot = -1;
    for (u32 i = 0; i < HS_FABRIC_PARTICIPANTS; i++) if (mutex == &gpu.participants.slots[i].mutex) slot = (int)i;
    if (slot == fail_slot && slot >= 0) { fail_slot = -1; return EAGAIN; }
    int result = __real_pthread_mutex_init(mutex, attr);
    if (slot >= 0 && !result) { REQUIRE(!initialized[slot]); initialized[slot] = true; }
    return result;
}
int __wrap_pthread_mutex_destroy(pthread_mutex_t* mutex) {
    for (u32 i = 0; i < HS_FABRIC_PARTICIPANTS; i++) if (mutex == &gpu.participants.slots[i].mutex) {
        REQUIRE(initialized[i]); initialized[i] = false;
    }
    return __real_pthread_mutex_destroy(mutex);
}
static void execute(void* unused, const HSFrameContext* frame) {
    (void)unused;
    REQUIRE(frame && frame->render && frame->render->count <= HS_RENDER_MAX_CMDS);
    seen = *frame->render; executes++;
    if (composing_churn) {
        REQUIRE(seen.count >= 4 && seen.cmds[0].op == HS_RC_CLEAR && seen.cmds[1].op == HS_RC_DRAW && seen.cmds[1].a == 7);
        u32 cursor = 2;
        for (u32 slot = 0; slot < HS_FABRIC_PARTICIPANTS; slot++) {
            const HSParticipantView* view = &gpu.participants.views[slot];
            if (!view->active) continue;
            REQUIRE(view->owner.slot == slot && view->frame.sequence == 1 && view->frame.count <= HS_FABRIC_DRAWS);
            const HSRenderCmd expected = {.op = HS_RC_DRAW, .a = slot};
            for (u32 i = 0; i < view->frame.count; i++) {
                REQUIRE(cursor < seen.count && !memcmp(&seen.cmds[cursor++], &expected, sizeof(expected)));
            }
        }
        REQUIRE(cursor + 2 == seen.count && seen.cmds[cursor].op == HS_RC_FRAME_END && seen.cmds[cursor + 1].op == HS_RC_PRESENT);
    }
    if (reenter) rejected = !hs_gpu_end_frame_checked(&gpu);
}
static void end(void* unused, const HSFrameContext* frame) { (void)unused; (void)frame; ends++; }
static const HSBackendOps ops = {.execute = execute, .end_frame = end};
static HSBackend backend = {.ops = &ops};
static void init(void) {
    hs_gpu_init(&gpu); executes = ends = 0; reenter = rejected = false;
    REQUIRE(hs_gpu_participants_ready(&gpu)); hs_gpu_attach_backend(&gpu, &backend);
}
static void destroy_gpu(void) {
    hs_gpu_shutdown(&gpu);
    for (u32 i = 0; i < HS_FABRIC_PARTICIPANTS; i++) REQUIRE(!initialized[i]);
}
static HSParticipant join(void) {
    HSParticipant owner;
    REQUIRE(hs_gpu_participant_join(&gpu, &owner) == HS_FABRIC_OK);
    return owner;
}
static HSParticipantFrame frame(HSParticipant owner, u32 sequence, u32 draws) {
    HSParticipantFrame input = {.sequence = sequence, .count = draws};
    for (u32 i = 0; i < draws; i++) input.draws[i] = (HSRenderCmd){.op = HS_RC_DRAW, .a = (u8)owner.slot};
    return input;
}
static void mixed_retention(void) {
    init(); HSParticipant owner = join();
    HSParticipantFrame input = frame(owner, 1, 3);
    REQUIRE(hs_gpu_participant_submit(&gpu, owner, &input) == HS_FABRIC_OK);
    hs_gpu_begin_frame(&gpu);
    hs_gpu_frame_begin(&gpu); hs_gpu_clear(&gpu, v4_make(0, 0, 0, 1));
    hs_gpu_draw(&gpu, 7); hs_gpu_frame_end(&gpu); hs_gpu_present(&gpu);
    hs_gpu_process(&gpu);
    REQUIRE(executes == 0 && gpu.participants.applied == 1);
    saved = gpu.render;
    REQUIRE(saved.count == 5 && hs_gpu_end_frame_checked(&gpu));
    REQUIRE(seen.count == 8 && seen.cmds[0].op == HS_RC_FRAME_BEGIN && seen.cmds[1].op == HS_RC_CLEAR);
    REQUIRE(seen.cmds[2].a == 7);
    for (u32 i = 3; i < 6; i++) REQUIRE(seen.cmds[i].op == HS_RC_DRAW && seen.cmds[i].a == owner.slot);
    REQUIRE(seen.cmds[6].op == HS_RC_FRAME_END && seen.cmds[7].op == HS_RC_PRESENT);
    REQUIRE(gpu.render.count == saved.count && !memcmp(gpu.render.cmds, saved.cmds, saved.count * sizeof(HSRenderCmd)));
    REQUIRE(hs_gpu_end_frame_checked(&gpu) && seen.count == 8); /* no duplicate retained draws */
    input = frame(owner, 2, 0);
    REQUIRE(hs_gpu_participant_submit(&gpu, owner, &input) == HS_FABRIC_OK);
    REQUIRE(hs_gpu_end_frame_checked(&gpu) && seen.count == saved.count);
    REQUIRE(hs_gpu_participant_leave(&gpu, owner) == HS_FABRIC_OK);
    REQUIRE(hs_gpu_end_frame_checked(&gpu) && atomic_load(&gpu.participant_count) == 0);
    REQUIRE(hs_gpu_participant_submit(&gpu, owner, &input) == HS_FABRIC_STALE);
    HSParticipant next = join();
    REQUIRE(next.slot == owner.slot && next.generation == owner.generation + 1);
    input = frame(next, 1, 1);
    REQUIRE(hs_gpu_participant_submit(&gpu, next, &input) == HS_FABRIC_OK);
    reenter = true;
    REQUIRE(hs_gpu_end_frame_checked(&gpu) && rejected && !gpu.frame_executing);
    REQUIRE(executes == ends);
    destroy_gpu();
    puts("{\"test\":\"default_mixed_order_retention_zero_draw_rejoin_reentry\",\"passed\":true}");
}
static void bounded_drain(void) {
    init(); HSParticipant owner = join();
    hs_gpu_begin_frame(&gpu);
    for (u32 i = 1; i <= HS_FABRIC_QUEUE; i++) {
        HSParticipantFrame input = frame(owner, i, 1);
        REQUIRE(hs_gpu_participant_submit(&gpu, owner, &input) == HS_FABRIC_OK);
    }
    HSParticipantFrame extra = frame(owner, HS_FABRIC_QUEUE + 1, 1);
    REQUIRE(hs_gpu_participant_submit(&gpu, owner, &extra) == HS_FABRIC_FULL);
    REQUIRE(hs_gpu_participant_leave(&gpu, owner) == HS_FABRIC_OK);
    for (u32 i = 1; i <= HS_FABRIC_QUEUE; i++) {
        REQUIRE(hs_gpu_end_frame_checked(&gpu));
        REQUIRE(gpu.participants.applied == i && gpu.participants.retired == 0);
        REQUIRE(gpu.participants.views[owner.slot].frame.sequence == i && seen.count == 1);
    }
    REQUIRE(hs_gpu_end_frame_checked(&gpu) && seen.count == 0 && gpu.participants.retired == 1);
    REQUIRE(atomic_load(&gpu.participant_count) == 0);
    owner = join(); extra = frame(owner, 1, 0);
    REQUIRE(hs_gpu_participant_submit(&gpu, owner, &extra) == HS_FABRIC_OK);
    REQUIRE(hs_gpu_step(&gpu) >= 1 && gpu.participants.applied == HS_FABRIC_QUEUE + 1);
    destroy_gpu();
    puts("{\"test\":\"default_bounded_frame_service_fifo_full_draining_retirement\",\"passed\":true}");
}
static void overflow(void) {
    init(); HSParticipant owner = join();
    HSParticipantFrame input = frame(owner, 1, 1);
    REQUIRE(hs_gpu_participant_submit(&gpu, owner, &input) == HS_FABRIC_OK);
    for (u32 i = 0; i < HS_RENDER_MAX_CMDS; i++) {
        HSRenderCmd command = {.op = HS_RC_DRAW, .a = 7};
        REQUIRE(hs_render_push(&gpu.render, &command));
    }
    saved = gpu.render;
    REQUIRE(!hs_gpu_end_frame_checked(&gpu) && gpu.render.overflow && !executes && !ends);
    REQUIRE(gpu.render.count == saved.count && !memcmp(gpu.render.cmds, saved.cmds, saved.count * sizeof(HSRenderCmd)));
    REQUIRE(gpu.participants.applied == 1);
    hs_gpu_begin_frame(&gpu);
    REQUIRE(hs_gpu_end_frame_checked(&gpu) && seen.count == 1 && gpu.render.count == 0);
    gpu.render.count = HS_RENDER_MAX_CMDS + 1;
    REQUIRE(!hs_gpu_end_frame_checked(&gpu));
    hs_gpu_begin_frame(&gpu);
    REQUIRE(hs_gpu_end_frame_checked(&gpu));
    hs_gpu_begin_frame(&gpu);
    for (u32 i = 0; i < HS_RENDER_MAX_CMDS - 1; i++) {
        HSRenderCmd marker = {.op = i % 2 ? HS_RC_PRESENT : HS_RC_FRAME_END};
        REQUIRE(hs_render_push(&gpu.render, &marker));
    }
    saved = gpu.render;
    REQUIRE(hs_gpu_end_frame_checked(&gpu) && seen.count == HS_RENDER_MAX_CMDS);
    REQUIRE(seen.cmds[0].op == HS_RC_DRAW && gpu.render.count == saved.count);
    REQUIRE(!memcmp(gpu.render.cmds, saved.cmds, saved.count * sizeof(HSRenderCmd)));
    destroy_gpu();
    puts("{\"test\":\"default_overflow_no_partial_presentation_retained_recovery\",\"passed\":true}");
}
static void independent_messages(void) {
    init(); HSParticipant owner = join();
    HSParticipantFrame input = frame(owner, 1, 1);
    REQUIRE(hs_gpu_participant_submit(&gpu, owner, &input) == HS_FABRIC_OK);
    hs_gpu_begin_frame(&gpu); hs_gpu_draw(&gpu, 7);
    /* End-frame must not silently become a legacy message-queue drain. */
    REQUIRE(hs_gpu_end_frame_checked(&gpu) && seen.count == 1 && seen.cmds[0].a == owner.slot);
    hs_gpu_process(&gpu);
    REQUIRE(hs_gpu_end_frame_checked(&gpu) && seen.count == 2);
    REQUIRE(seen.cmds[0].a == 7 && seen.cmds[1].a == owner.slot);
    destroy_gpu();
    puts("{\"test\":\"default_preserves_explicit_legacy_message_drain\",\"passed\":true}");
}
static void legacy_fast_path(void) {
    init(); reenter = true;
    REQUIRE(hs_gpu_end_frame_checked(&gpu) && rejected && executes == 1 && ends == 1);
    REQUIRE(!atomic_load(&gpu.participant_count) && !atomic_load(&gpu.participant_limit));
    gpu.render.overflow = true;
    REQUIRE(!hs_gpu_end_frame_checked(&gpu) && executes == 1 && ends == 1);
    hs_gpu_end_frame(&gpu); REQUIRE(executes == 1 && ends == 1);
    hs_gpu_begin_frame(&gpu); hs_gpu_end_frame(&gpu);
    REQUIRE(executes == 2 && ends == 2 && !gpu.frame_executing && gpu.participants.applied == 0);
    destroy_gpu();
    puts("{\"test\":\"default_idle_fast_path_checked_void_overflow_reentry\",\"passed\":true}");
}
static void high_slots(void) {
    init(); HSParticipant owners[HS_FABRIC_PARTICIPANTS];
    const u32 last = HS_FABRIC_PARTICIPANTS - 1;
    for (u32 i = 0; i < HS_FABRIC_PARTICIPANTS; i++) {
        owners[i] = join(); REQUIRE(owners[i].slot == i);
        REQUIRE(atomic_load(&gpu.participant_limit) == i + 1);
    }
    HSParticipantFrame input = frame(owners[last], 1, HS_FABRIC_DRAWS);
    REQUIRE(hs_gpu_participant_submit(&gpu, owners[last], &input) == HS_FABRIC_OK);
    for (u32 i = 0; i < last; i++) REQUIRE(hs_gpu_participant_leave(&gpu, owners[i]) == HS_FABRIC_OK);
    REQUIRE(hs_gpu_end_frame_checked(&gpu) && atomic_load(&gpu.participant_count) == 1);
    REQUIRE(seen.count == HS_FABRIC_DRAWS);
    for (u32 i = 0; i < seen.count; i++) REQUIRE(seen.cmds[i].a == last);
    REQUIRE(hs_gpu_participant_leave(&gpu, owners[last]) == HS_FABRIC_OK);
    REQUIRE(hs_gpu_end_frame_checked(&gpu) && atomic_load(&gpu.participant_count) == 0);
    REQUIRE(atomic_load(&gpu.participant_limit) == HS_FABRIC_PARTICIPANTS);
    owners[0] = join(); input = frame(owners[0], 1, 1);
    REQUIRE(hs_gpu_participant_submit(&gpu, owners[0], &input) == HS_FABRIC_OK);
    REQUIRE(hs_gpu_end_frame_checked(&gpu) && seen.count == 1 && seen.cmds[0].a == 0);
    destroy_gpu();
    puts("{\"test\":\"default_sparse_high_slot_monotonic_bound_and_reuse\",\"passed\":true}");
}
static void* replace_during_retire(void* unused) {
    (void)unused;
    REQUIRE(pthread_mutex_lock(&race_lock) == 0);
    while (!retire_reached) REQUIRE(pthread_cond_wait(&race_changed, &race_lock) == 0);
    REQUIRE(pthread_mutex_unlock(&race_lock) == 0);
    replacement = join(); REQUIRE(replacement.slot == 0);
    HSParticipantFrame input = frame(replacement, 1, 1);
    REQUIRE(hs_gpu_participant_submit(&gpu, replacement, &input) == HS_FABRIC_OK);
    REQUIRE(hs_gpu_participant_leave(&gpu, replacement) == HS_FABRIC_OK);
    REQUIRE(atomic_load(&gpu.participant_count) == 2); /* old decrement is paused */
    REQUIRE(pthread_mutex_lock(&race_lock) == 0);
    rejoined = true; REQUIRE(pthread_cond_broadcast(&race_changed) == 0);
    REQUIRE(pthread_mutex_unlock(&race_lock) == 0);
    return NULL;
}
static void retire_rejoin_race(void) {
    init(); HSParticipant owner = join();
    REQUIRE(hs_gpu_participant_leave(&gpu, owner) == HS_FABRIC_OK);
    REQUIRE(pthread_mutex_lock(&race_lock) == 0);
    pause_retire = true; retire_reached = rejoined = false;
    REQUIRE(pthread_mutex_unlock(&race_lock) == 0);
    pthread_t thread; REQUIRE(pthread_create(&thread, NULL, replace_during_retire, NULL) == 0);
    REQUIRE(hs_gpu_end_frame_checked(&gpu));
    REQUIRE(pthread_join(thread, NULL) == 0);
    REQUIRE(replacement.generation == owner.generation + 1 && atomic_load(&gpu.participant_count) == 1);
    REQUIRE(hs_gpu_end_frame_checked(&gpu) && gpu.participants.applied == 1 && seen.count == 1);
    REQUIRE(hs_gpu_end_frame_checked(&gpu) && gpu.participants.retired == 2 && atomic_load(&gpu.participant_count) == 0);
    destroy_gpu();
    puts("{\"test\":\"default_rejoin_between_retirement_and_live_count_decrement\",\"passed\":true}");
}
static void invalid_wrappers(void) {
    init(); HSParticipant owner = join(), extra;
    HSParticipantFrame input = frame(owner, 1, 1);
    REQUIRE(hs_gpu_participant_join(NULL, &extra) == HS_FABRIC_INVALID);
    REQUIRE(hs_gpu_participant_join(&gpu, NULL) == HS_FABRIC_INVALID);
    REQUIRE(hs_gpu_participant_submit(NULL, owner, &input) == HS_FABRIC_INVALID);
    REQUIRE(hs_gpu_participant_submit(&gpu, owner, NULL) == HS_FABRIC_INVALID);
    REQUIRE(hs_gpu_participant_leave(NULL, owner) == HS_FABRIC_INVALID);
    HSParticipant bad = {.slot = HS_FABRIC_PARTICIPANTS, .generation = owner.generation};
    REQUIRE(hs_gpu_participant_submit(&gpu, bad, &input) == HS_FABRIC_INVALID);
    REQUIRE(hs_gpu_participant_leave(&gpu, bad) == HS_FABRIC_INVALID);
    input.count = HS_FABRIC_DRAWS + 1;
    REQUIRE(hs_gpu_participant_submit(&gpu, owner, &input) == HS_FABRIC_INVALID);
    input = frame(owner, 2, 1);
    REQUIRE(hs_gpu_participant_submit(&gpu, owner, &input) == HS_FABRIC_SEQUENCE);
    REQUIRE(atomic_load(&gpu.participant_count) == 1 && atomic_load(&gpu.participant_limit) == 1);
    input = frame(owner, 1, 1);
    REQUIRE(hs_gpu_participant_submit(&gpu, owner, &input) == HS_FABRIC_OK);
    REQUIRE(hs_gpu_participant_leave(&gpu, owner) == HS_FABRIC_OK);
    REQUIRE(hs_gpu_participant_leave(&gpu, owner) == HS_FABRIC_CLOSING);
    input.sequence = 2;
    REQUIRE(hs_gpu_participant_submit(&gpu, owner, &input) == HS_FABRIC_CLOSING);
    hs_gpu_process(&gpu);
    REQUIRE(atomic_load(&gpu.participant_count) == 0);
    REQUIRE(hs_gpu_participant_leave(&gpu, owner) == HS_FABRIC_STALE);
    REQUIRE(atomic_load(&gpu.participant_count) == 0);
    destroy_gpu();
    REQUIRE(hs_gpu_participant_submit(&gpu, owner, &input) == HS_FABRIC_INVALID);
    REQUIRE(hs_gpu_participant_leave(&gpu, owner) == HS_FABRIC_INVALID);
    puts("{\"test\":\"default_invalid_wrapper_calls_do_not_change_accounting\",\"passed\":true}");
}
static void composition_model(void) {
    init(); HSParticipant owners[HS_FABRIC_PARTICIPANTS];
    u32 counts[HS_FABRIC_PARTICIPANTS];
    for (u32 slot = 0; slot < HS_FABRIC_PARTICIPANTS; slot++) owners[slot] = join();
    u32 seed = 0x42c0ffee, successful = 0, overflows = 0;
    for (u32 trial = 1; trial <= 1024; trial++) {
        u32 added = 0;
        for (u32 slot = 0; slot < HS_FABRIC_PARTICIPANTS; slot++) {
            seed = seed * 1664525u + 1013904223u;
            counts[slot] = trial == 1 ? HS_FABRIC_DRAWS : seed % (HS_FABRIC_DRAWS + 1);
            HSParticipantFrame input = frame(owners[slot], trial, counts[slot]);
            REQUIRE(hs_gpu_participant_submit(&gpu, owners[slot], &input) == HS_FABRIC_OK);
            added += counts[slot];
        }
        hs_gpu_process(&gpu); hs_gpu_begin_frame(&gpu);
        u32 suffix = trial % 7;
        u32 legacy = HS_RENDER_MAX_CMDS - added;
        if (trial % 3 == 0) legacy++;
        else if (trial % 3 == 1) legacy--;
        REQUIRE(legacy >= suffix && legacy <= HS_RENDER_MAX_CMDS);
        for (u32 i = 0; i < legacy; i++) {
            HSRenderCmd command = {.op = i < legacy - suffix ? HS_RC_SET_CLIP : (i % 2 ? HS_RC_PRESENT : HS_RC_FRAME_END), .x = i, .y = trial};
            REQUIRE(hs_render_push(&gpu.render, &command));
        }
        if (trial % 17 == 0) gpu.render.overflow = true;
        saved = gpu.render; u32 before = executes;
        bool expected = !saved.overflow && legacy + added <= HS_RENDER_MAX_CMDS;
        REQUIRE(hs_gpu_end_frame_checked(&gpu) == expected);
        REQUIRE(gpu.render.count == saved.count && !memcmp(gpu.render.cmds, saved.cmds, legacy * sizeof(HSRenderCmd)));
        if (!expected) { REQUIRE(executes == before && gpu.render.overflow); overflows++; continue; }
        REQUIRE(executes == before + 1 && seen.count == legacy + added);
        u32 prefix = legacy - suffix;
        REQUIRE(!memcmp(seen.cmds, saved.cmds, prefix * sizeof(HSRenderCmd)));
        u32 cursor = prefix;
        for (u32 slot = 0; slot < HS_FABRIC_PARTICIPANTS; slot++) {
            HSRenderCmd expected_draw = {.op = HS_RC_DRAW, .a = slot};
            for (u32 i = 0; i < counts[slot]; i++) REQUIRE(!memcmp(&seen.cmds[cursor++], &expected_draw, sizeof(expected_draw)));
        }
        REQUIRE(!memcmp(&seen.cmds[cursor], &saved.cmds[prefix], suffix * sizeof(HSRenderCmd)));
        REQUIRE(hs_gpu_end_frame_checked(&gpu) && gpu.render.count == saved.count); successful++;
    }
    REQUIRE(successful && overflows && gpu.participants.applied == 1024 * HS_FABRIC_PARTICIPANTS);
    printf("{\"test\":\"default_independent_composition_model\",\"trials\":1024,\"successful\":%u,\"overflow\":%u,\"passed\":true}\n", successful, overflows);
    destroy_gpu();
}
static void busy_retirement(void) {
    init(); HSParticipant owner = join();
    HSParticipantFrame input = frame(owner, 1, 2);
    REQUIRE(hs_gpu_participant_submit(&gpu, owner, &input) == HS_FABRIC_OK);
    REQUIRE(hs_gpu_end_frame_checked(&gpu) && seen.count == 2);
    REQUIRE(hs_gpu_participant_leave(&gpu, owner) == HS_FABRIC_OK);
    /* A zero-progress process call must not be mistaken for completed leave.
     * Holding this test-only lock forces the consumer's retirement trylock. */
    REQUIRE(pthread_mutex_lock(&gpu.participants.slots[owner.slot].mutex) == 0);
    REQUIRE(hs_gpu_process(&gpu) == 0 && atomic_load(&gpu.participant_count) == 1);
    REQUIRE(hs_gpu_end_frame_checked(&gpu) && seen.count == 2 && gpu.participants.retired == 0);
    REQUIRE(pthread_mutex_unlock(&gpu.participants.slots[owner.slot].mutex) == 0);
    REQUIRE(hs_gpu_end_frame_checked(&gpu) && seen.count == 0 && atomic_load(&gpu.participant_count) == 0);
    REQUIRE(gpu.participants.retired == 1);
    destroy_gpu();
    puts("{\"test\":\"default_busy_retirement_zero_progress_preserves_retention\",\"passed\":true}");
}
enum { CYCLES = 100 };
static atomic_uint finished, accepted;
static void* churn(void* unused) {
    (void)unused;
    for (u32 i = 0; i < CYCLES; i++) {
        HSParticipant owner; HSFabricResult result;
        while ((result = hs_gpu_participant_join(&gpu, &owner)) != HS_FABRIC_OK) {
            REQUIRE(result == HS_FABRIC_BUSY || result == HS_FABRIC_FULL); sched_yield();
        }
        HSParticipantFrame input = frame(owner, 1, composing_churn ? i % (HS_FABRIC_DRAWS + 1) : 1);
        while ((result = hs_gpu_participant_submit(&gpu, owner, &input)) != HS_FABRIC_OK) {
            REQUIRE(result == HS_FABRIC_BUSY); sched_yield();
        }
        atomic_fetch_add(&accepted, 1);
        while ((result = hs_gpu_participant_leave(&gpu, owner)) != HS_FABRIC_OK) {
            REQUIRE(result == HS_FABRIC_BUSY); sched_yield();
        }
    }
    atomic_fetch_add_explicit(&finished, 1, memory_order_release);
    return NULL;
}
static void concurrent(bool compose) {
    init(); atomic_init(&finished, 0); atomic_init(&accepted, 0);
    composing_churn = compose;
    pthread_t threads[HS_FABRIC_PARTICIPANTS];
    for (u32 i = 0; i < HS_FABRIC_PARTICIPANTS; i++) REQUIRE(pthread_create(&threads[i], NULL, churn, NULL) == 0);
    while (atomic_load_explicit(&finished, memory_order_acquire) < HS_FABRIC_PARTICIPANTS || atomic_load(&gpu.participant_count)) {
        if (!compose) { hs_gpu_step(&gpu); continue; }
        hs_gpu_begin_frame(&gpu);
        const HSRenderCmd commands[] = {{.op = HS_RC_CLEAR}, {.op = HS_RC_DRAW, .a = 7}, {.op = HS_RC_FRAME_END}, {.op = HS_RC_PRESENT}};
        for (u32 i = 0; i < 4; i++) REQUIRE(hs_render_push(&gpu.render, &commands[i]));
        REQUIRE(hs_gpu_end_frame_checked(&gpu) && gpu.render.count == 4);
        REQUIRE(!memcmp(gpu.render.cmds, commands, sizeof(commands)));
    }
    for (u32 i = 0; i < HS_FABRIC_PARTICIPANTS; i++) REQUIRE(pthread_join(threads[i], NULL) == 0);
    REQUIRE(atomic_load(&accepted) == CYCLES * HS_FABRIC_PARTICIPANTS);
    REQUIRE(gpu.participants.applied == atomic_load(&accepted) && gpu.participants.retired == gpu.participants.applied);
    composing_churn = false;
    destroy_gpu();
    printf("{\"test\":\"default_concurrent_join_submit_leave_count_reclamation\",\"compose\":%s,\"accepted\":800,\"passed\":true}\n", compose ? "true" : "false");
}
static void init_failures(void) {
    for (int slot = 0; slot < HS_FABRIC_PARTICIPANTS; slot++) {
        fprintf(stderr, "[expected fault injection] participant mutex %d\n", slot);
        fail_slot = slot; hs_gpu_init(&gpu);
        REQUIRE(!hs_gpu_participants_ready(&gpu));
        HSParticipant owner;
        REQUIRE(hs_gpu_participant_join(&gpu, &owner) == HS_FABRIC_INVALID);
        hs_gpu_attach_backend(&gpu, &backend);
        hs_gpu_begin_frame(&gpu); hs_gpu_clear(&gpu, v4_make(0, 0, 0, 1)); hs_gpu_process(&gpu);
        REQUIRE(hs_gpu_end_frame_checked(&gpu) && seen.count == 1);
        destroy_gpu(); hs_gpu_shutdown(&gpu);
    }
    REQUIRE(!hs_gpu_participants_ready(NULL) && !hs_gpu_end_frame_checked(NULL));
    puts("{\"test\":\"default_all_partial_init_failures_legacy_survival_cleanup\",\"passed\":true}");
}
int main(void) {
    alarm(60);
    mixed_retention(); bounded_drain(); overflow(); independent_messages(); legacy_fast_path(); high_slots(); retire_rejoin_race(); invalid_wrappers(); composition_model(); busy_retirement(); concurrent(false); concurrent(true); init_failures();
    REQUIRE(offsetof(HSGpu, participants_ready) / 64 != offsetof(HSGpu, frame_executing) / 64);
    printf("{\"gpu_bytes\":%zu,\"fabric_bytes\":%zu,\"ready_offset\":%zu,\"guard_offset\":%zu,\"fabric_offset\":%zu}\n",
           sizeof(HSGpu), sizeof(HSFabric), offsetof(HSGpu, participants_ready), offsetof(HSGpu, frame_executing), offsetof(HSGpu, participants));
    return 0;
}
