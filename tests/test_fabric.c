#define _POSIX_C_SOURCE 200809L
#include "hs_fabric.h"
#include <inttypes.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
static HSFabric fabric;
static HSRenderList render;
static const f32 black[4] = {0, 0, 0, 1};

static u64 ns(void) {
    struct timespec time;
    REQUIRE(clock_gettime(CLOCK_MONOTONIC, &time) == 0);
    return (u64)time.tv_sec * 1000000000ull + time.tv_nsec;
}

static HSParticipantFrame frame(HSParticipant owner, u64 sequence, u32 count) {
    HSParticipantFrame result = {.sequence = sequence, .count = count};
    for (u32 i = 0; i < count && i < HS_FABRIC_DRAWS; i++)
        result.draws[i] = (HSRenderCmd){.op = HS_RC_DRAW, .a = (u8)owner.slot};
    return result;
}

static void fifo_and_lifetime(void) {
    REQUIRE(hs_fabric_init(&fabric));
    HSParticipant owner, next;
    REQUIRE(hs_fabric_join(&fabric, &owner) == HS_FABRIC_OK);
    HSParticipantFrame input;
    for (u32 i = 1; i <= HS_FABRIC_QUEUE; i++) {
        input = frame(owner, i, i);
        REQUIRE(hs_fabric_submit(&fabric, owner, &input) == HS_FABRIC_OK);
        memset(&input, 0xff, sizeof(input)); /* No caller memory may be retained. */
    }
    input = frame(owner, HS_FABRIC_QUEUE + 1, 1);
    REQUIRE(hs_fabric_submit(&fabric, owner, &input) == HS_FABRIC_FULL);
    REQUIRE(hs_fabric_step(&fabric, 1) == 1);
    REQUIRE(fabric.views[owner.slot].frame.sequence == 1);
    REQUIRE(hs_fabric_submit(&fabric, owner, &input) == HS_FABRIC_OK);
    REQUIRE(hs_fabric_leave(&fabric, owner) == HS_FABRIC_OK);
    REQUIRE(hs_fabric_submit(&fabric, owner, &input) == HS_FABRIC_CLOSING);
    for (u32 i = 2; i <= HS_FABRIC_QUEUE + 1; i++) {
        REQUIRE(hs_fabric_step(&fabric, 1) == 1);
        REQUIRE(fabric.views[owner.slot].active);
        REQUIRE(fabric.views[owner.slot].frame.sequence == i);
        REQUIRE(fabric.views[owner.slot].frame.count == (i <= HS_FABRIC_QUEUE ? i : 1));
        for (u32 j = 0; j < fabric.views[owner.slot].frame.count; j++)
            REQUIRE(fabric.views[owner.slot].frame.draws[j].op == HS_RC_DRAW);
    }
    REQUIRE(fabric.applied == 5 && fabric.retired == 0);
    REQUIRE(hs_fabric_step(&fabric, 1) == 1);
    REQUIRE(!fabric.views[owner.slot].active && fabric.retired == 1);
    REQUIRE(hs_fabric_join(&fabric, &next) == HS_FABRIC_OK);
    REQUIRE(next.slot == owner.slot && next.generation == owner.generation + 1);
    REQUIRE(hs_fabric_submit(&fabric, owner, &input) == HS_FABRIC_STALE);
    REQUIRE(hs_fabric_leave(&fabric, owner) == HS_FABRIC_STALE);
    input = frame(next, 1, 1);
    REQUIRE(hs_fabric_submit(&fabric, next, &input) == HS_FABRIC_OK);
    REQUIRE(hs_fabric_step(&fabric, 1) == 1);
    for (u32 i = 0; i < 100; i++) {
        REQUIRE(hs_fabric_step(&fabric, 8) == 0);
        REQUIRE(hs_fabric_compose(&fabric, &render, black));
        REQUIRE(render.count == 5 && render.cmds[2].a == next.slot);
        REQUIRE(fabric.views[next.slot].frame.sequence == 1);
    }
    hs_fabric_destroy(&fabric);
    printf("{\"test\":\"fifo_copy_retirement_stale_handle_retention\",\"passed\":true}\n");
}

static void fairness_and_isolation(void) {
    REQUIRE(hs_fabric_init(&fabric));
    HSParticipant owners[HS_FABRIC_PARTICIPANTS], excess;
    for (u32 i = 0; i < HS_FABRIC_PARTICIPANTS; i++) {
        REQUIRE(hs_fabric_join(&fabric, &owners[i]) == HS_FABRIC_OK);
        for (u32 sequence = 1; sequence <= HS_FABRIC_QUEUE; sequence++) {
            HSParticipantFrame input = frame(owners[i], sequence, HS_FABRIC_DRAWS);
            REQUIRE(hs_fabric_submit(&fabric, owners[i], &input) == HS_FABRIC_OK);
        }
    }
    REQUIRE(hs_fabric_join(&fabric, &excess) == HS_FABRIC_FULL);
    REQUIRE(hs_fabric_step(&fabric, 0) == 0);
    for (u32 round = 1; round <= HS_FABRIC_QUEUE; round++) {
        for (u32 i = 0; i < HS_FABRIC_PARTICIPANTS; i++) {
            REQUIRE(hs_fabric_step(&fabric, 1) == 1);
            REQUIRE(fabric.views[i].frame.sequence == round);
        }
    }
    REQUIRE(hs_fabric_compose(&fabric, &render, black));
    REQUIRE(render.count == HS_FABRIC_PARTICIPANTS * HS_FABRIC_DRAWS + 4);
    REQUIRE(!render.overflow);

    /* White-box held critical-section fixture, not a killed thread/process. */
    REQUIRE(pthread_mutex_lock(&fabric.slots[0].mutex) == 0);
    HSParticipantFrame blocked = frame(owners[0], 5, 0);
    REQUIRE(hs_fabric_submit(&fabric, owners[0], &blocked) == HS_FABRIC_BUSY);
    REQUIRE(hs_fabric_leave(&fabric, owners[0]) == HS_FABRIC_BUSY);
    for (u32 i = 5; i < 105; i++) {
        HSParticipantFrame input = frame(owners[1], i, i % 2);
        REQUIRE(hs_fabric_submit(&fabric, owners[1], &input) == HS_FABRIC_OK);
        REQUIRE(hs_fabric_step(&fabric, 1) == 1);
        REQUIRE(fabric.views[1].frame.sequence == i);
        REQUIRE(fabric.views[0].frame.sequence == 4);
        REQUIRE(hs_fabric_compose(&fabric, &render, black));
    }
    REQUIRE(pthread_mutex_unlock(&fabric.slots[0].mutex) == 0);
    REQUIRE(hs_fabric_submit(&fabric, owners[0], &blocked) == HS_FABRIC_OK);
    REQUIRE(hs_fabric_step(&fabric, 1) == 1);
    REQUIRE(fabric.views[0].frame.sequence == 5 && fabric.views[0].frame.count == 0);
    hs_fabric_destroy(&fabric);
    printf("{\"test\":\"budget_one_fairness_held_slot_isolation_max_composition\",\"passed\":true}\n");
}

static void rejection_and_churn(void) {
    REQUIRE(hs_fabric_init(&fabric));
    for (u32 cycle = 0; cycle < 96; cycle++) {
        HSParticipant owner;
        REQUIRE(hs_fabric_join(&fabric, &owner) == HS_FABRIC_OK);
        REQUIRE(owner.slot == 0 && owner.generation == cycle + 1);
        HSParticipantFrame input = frame(owner, 0, 1);
        REQUIRE(hs_fabric_submit(&fabric, owner, &input) == HS_FABRIC_SEQUENCE);
        input.sequence = 1;
        input.draws[0].op = HS_RC_CLEAR;
        REQUIRE(hs_fabric_submit(&fabric, owner, &input) == HS_FABRIC_INVALID);
        input.draws[0].op = HS_RC_DRAW;
        input.draws[0].a = 1;
        REQUIRE(hs_fabric_submit(&fabric, owner, &input) == HS_FABRIC_INVALID);
        input.draws[0].a = 0;
        input.draws[0].payload_len = 1;
        REQUIRE(hs_fabric_submit(&fabric, owner, &input) == HS_FABRIC_INVALID);
        input = frame(owner, 1, HS_FABRIC_DRAWS + 1);
        REQUIRE(hs_fabric_submit(&fabric, owner, &input) == HS_FABRIC_INVALID);
        input = frame(owner, 1, 1);
        REQUIRE(hs_fabric_submit(&fabric, owner, &input) == HS_FABRIC_OK);
        REQUIRE(hs_fabric_submit(&fabric, owner, &input) == HS_FABRIC_SEQUENCE);
        REQUIRE(hs_fabric_leave(&fabric, owner) == HS_FABRIC_OK);
        REQUIRE(hs_fabric_step(&fabric, 8) == 1); /* accepted frame first */
        REQUIRE(hs_fabric_step(&fabric, 8) == 1); /* retirement next */
        REQUIRE(hs_fabric_submit(&fabric, owner, &input) == HS_FABRIC_STALE);
    }
    REQUIRE(fabric.applied == 96 && fabric.retired == 96);
    fabric.slots[0].generation = UINT64_MAX; /* Exhausted generations never wrap. */
    HSParticipant owner;
    REQUIRE(hs_fabric_join(&fabric, &owner) == HS_FABRIC_OK && owner.slot == 1);
    fabric.slots[owner.slot].admitted = UINT64_MAX; /* Sequence exhaustion fixture. */
    HSParticipantFrame exhausted = frame(owner, 0, 1);
    REQUIRE(hs_fabric_submit(&fabric, owner, &exhausted) == HS_FABRIC_SEQUENCE);
    REQUIRE(hs_fabric_submit(&fabric, (HSParticipant){99, 1}, NULL) == HS_FABRIC_INVALID);
    REQUIRE(hs_fabric_compose(&fabric, &render, black));
    REQUIRE(render.count == 4);
    REQUIRE(!hs_fabric_compose(&fabric, NULL, black));
    hs_fabric_destroy(&fabric);
    printf("{\"test\":\"validation_and_96_join_leave_cycles\",\"passed\":true}\n");
}

typedef struct { HSParticipant owner; u64 accepted, retries; } Producer;
static atomic_bool go;
static atomic_uint ready;
#define FRAMES 5000

static void* produce(void* pointer) {
    Producer* producer = pointer;
    atomic_fetch_add_explicit(&ready, 1, memory_order_release);
    while (!atomic_load_explicit(&go, memory_order_acquire)) sched_yield();
    for (u64 i = 1; i <= FRAMES; i++) {
        HSParticipantFrame input = frame(producer->owner, i, (u32)(i % (HS_FABRIC_DRAWS + 1)));
        for (;;) {
            HSFabricResult result = hs_fabric_submit(&fabric, producer->owner, &input);
            if (result == HS_FABRIC_OK) { producer->accepted++; break; }
            REQUIRE(result == HS_FABRIC_FULL || result == HS_FABRIC_BUSY);
            producer->retries++;
            sched_yield();
        }
        memset(&input, 0xff, sizeof(input));
    }
    return NULL;
}

static void concurrent(void) {
    REQUIRE(hs_fabric_init(&fabric));
    pthread_t threads[HS_FABRIC_PARTICIPANTS];
    Producer producers[HS_FABRIC_PARTICIPANTS] = {0};
    u64 observed[HS_FABRIC_PARTICIPANTS] = {0};
    atomic_init(&go, false); atomic_init(&ready, 0);
    for (u32 i = 0; i < HS_FABRIC_PARTICIPANTS; i++) {
        REQUIRE(hs_fabric_join(&fabric, &producers[i].owner) == HS_FABRIC_OK);
        REQUIRE(pthread_create(&threads[i], NULL, produce, &producers[i]) == 0);
    }
    while (atomic_load_explicit(&ready, memory_order_acquire) < HS_FABRIC_PARTICIPANTS) sched_yield();
    u64 start = ns(), retries = 0;
    atomic_store_explicit(&go, true, memory_order_release);
    while (fabric.applied < FRAMES * HS_FABRIC_PARTICIPANTS) {
        REQUIRE(ns() - start < 20000000000ull);
        hs_fabric_step(&fabric, 1);
        for (u32 i = 0; i < HS_FABRIC_PARTICIPANTS; i++) {
            const HSParticipantView* view = &fabric.views[i];
            if (!view->active || view->frame.sequence == observed[i]) continue;
            REQUIRE(view->frame.sequence == observed[i] + 1);
            REQUIRE(view->frame.count == view->frame.sequence % (HS_FABRIC_DRAWS + 1));
            REQUIRE(view->owner.generation == producers[i].owner.generation);
            for (u32 j = 0; j < view->frame.count; j++)
                REQUIRE(view->frame.draws[j].op == HS_RC_DRAW && view->frame.draws[j].a == i);
            observed[i]++;
        }
    }
    for (u32 i = 0; i < HS_FABRIC_PARTICIPANTS; i++) {
        REQUIRE(pthread_join(threads[i], NULL) == 0);
        REQUIRE(producers[i].accepted == FRAMES && observed[i] == FRAMES);
        retries += producers[i].retries;
    }
    u64 elapsed = ns() - start;
    REQUIRE(hs_fabric_compose(&fabric, &render, black));
    printf("{\"test\":\"concurrent_fifo_delivery\",\"participants\":%u,\"accepted\":%" PRIu64
           ",\"applied\":%" PRIu64 ",\"elapsed_ns\":%" PRIu64 ",\"retries\":%" PRIu64
           ",\"fabric_bytes\":%zu,\"passed\":true}\n", HS_FABRIC_PARTICIPANTS,
           (uint64_t)fabric.applied, (uint64_t)fabric.applied, (uint64_t)elapsed,
           (uint64_t)retries, sizeof(fabric));
    hs_fabric_destroy(&fabric);
}

#define CHURN_CYCLES 64
static atomic_uint finished;
static void* churn_participant(void* unused) {
    (void)unused;
    atomic_fetch_add_explicit(&ready, 1, memory_order_release);
    while (!atomic_load_explicit(&go, memory_order_acquire)) sched_yield();
    for (u32 i = 0; i < CHURN_CYCLES; i++) {
        HSParticipant owner;
        HSFabricResult result;
        while ((result = hs_fabric_join(&fabric, &owner)) != HS_FABRIC_OK) {
            REQUIRE(result == HS_FABRIC_BUSY || result == HS_FABRIC_FULL);
            sched_yield();
        }
        HSParticipantFrame input = frame(owner, 1, 1);
        while ((result = hs_fabric_submit(&fabric, owner, &input)) != HS_FABRIC_OK) {
            REQUIRE(result == HS_FABRIC_BUSY);
            sched_yield();
        }
        while ((result = hs_fabric_leave(&fabric, owner)) != HS_FABRIC_OK) {
            REQUIRE(result == HS_FABRIC_BUSY);
            sched_yield();
        }
    }
    atomic_fetch_add_explicit(&finished, 1, memory_order_release);
    return NULL;
}

static void concurrent_lifecycle(void) {
    REQUIRE(hs_fabric_init(&fabric));
    pthread_t threads[HS_FABRIC_PARTICIPANTS];
    atomic_init(&go, false); atomic_init(&ready, 0); atomic_init(&finished, 0);
    for (u32 i = 0; i < HS_FABRIC_PARTICIPANTS; i++)
        REQUIRE(pthread_create(&threads[i], NULL, churn_participant, NULL) == 0);
    while (atomic_load_explicit(&ready, memory_order_acquire) < HS_FABRIC_PARTICIPANTS) sched_yield();
    u64 start = ns();
    atomic_store_explicit(&go, true, memory_order_release);
    while (atomic_load_explicit(&finished, memory_order_acquire) < HS_FABRIC_PARTICIPANTS ||
           fabric.retired < HS_FABRIC_PARTICIPANTS * CHURN_CYCLES) {
        REQUIRE(ns() - start < 20000000000ull);
        hs_fabric_step(&fabric, HS_FABRIC_PARTICIPANTS);
    }
    for (u32 i = 0; i < HS_FABRIC_PARTICIPANTS; i++) {
        REQUIRE(pthread_join(threads[i], NULL) == 0);
        REQUIRE(!fabric.views[i].active);
    }
    REQUIRE(fabric.applied == HS_FABRIC_PARTICIPANTS * CHURN_CYCLES);
    REQUIRE(fabric.retired == fabric.applied);
    printf("{\"test\":\"concurrent_join_submit_leave\",\"accepted\":%" PRIu64
           ",\"applied\":%" PRIu64 ",\"retired\":%" PRIu64 ",\"passed\":true}\n",
           (uint64_t)fabric.applied, (uint64_t)fabric.applied, (uint64_t)fabric.retired);
    hs_fabric_destroy(&fabric);
}

int main(void) {
    alarm(30);
    fifo_and_lifetime(); fairness_and_isolation(); rejection_and_churn(); concurrent(); concurrent_lifecycle();
    return 0;
}
