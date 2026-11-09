#define _POSIX_C_SOURCE 200809L
#include "hs_fabric.h"
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef HS_FABRIC_SPLIT_PUBLICATION
#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "%d: %s\n", __LINE__, #x); exit(1); } } while (0)
static HSFabric fabric;
static HSParticipant owner;
static atomic_uint ticket;
enum { FRAMES = 20000, WRITERS = 4 };

static HSParticipantFrame frame(u32 sequence) {
    HSParticipantFrame result = {.sequence = sequence, .count = 1};
    result.draws[0] = (HSRenderCmd){.op = HS_RC_DRAW, .a = (u8)owner.slot};
    return result;
}

static void wrap_and_reuse(void) {
    REQUIRE(hs_fabric_init(&fabric));
    REQUIRE(hs_fabric_join(&fabric, &owner) == HS_FABRIC_OK);
    HSParticipantSlot* slot = &fabric.slots[owner.slot];
    /* Quiescent white-box seed, not a supported caller mutation. */
    __atomic_store_n(&slot->read_index, UINT32_MAX - 2, __ATOMIC_RELAXED);
    __atomic_store_n(&slot->write_index, UINT32_MAX - 2, __ATOMIC_RELAXED);
    for (u32 round = 0; round < 64; round++) {
        for (u32 j = 1; j <= HS_FABRIC_QUEUE; j++) {
            HSParticipantFrame input = frame(round * HS_FABRIC_QUEUE + j);
            REQUIRE(hs_fabric_submit(&fabric, owner, &input) == HS_FABRIC_OK);
        }
        HSParticipantFrame excess = frame((round + 1) * HS_FABRIC_QUEUE + 1);
        REQUIRE(hs_fabric_submit(&fabric, owner, &excess) == HS_FABRIC_FULL);
        for (u32 j = 1; j <= HS_FABRIC_QUEUE; j++) {
            REQUIRE(hs_fabric_step(&fabric, 1) == 1);
            REQUIRE(fabric.views[owner.slot].frame.sequence == round * HS_FABRIC_QUEUE + j);
        }
    }
    u32 cursor = __atomic_load_n(&slot->read_index, __ATOMIC_RELAXED);
    REQUIRE(cursor == (u32)(UINT32_MAX - 2 + 64u * HS_FABRIC_QUEUE));
    REQUIRE(hs_fabric_leave(&fabric, owner) == HS_FABRIC_OK);
    REQUIRE(hs_fabric_step(&fabric, 1) == 1);
    HSParticipant previous = owner;
    REQUIRE(hs_fabric_join(&fabric, &owner) == HS_FABRIC_OK);
    REQUIRE(owner.slot == previous.slot && owner.generation == previous.generation + 1);
    REQUIRE(__atomic_load_n(&slot->read_index, __ATOMIC_RELAXED) == cursor);
    REQUIRE(__atomic_load_n(&slot->write_index, __ATOMIC_RELAXED) == cursor);
    HSParticipantFrame input = frame(1);
    REQUIRE(hs_fabric_submit(&fabric, previous, &input) == HS_FABRIC_STALE);
    REQUIRE(hs_fabric_submit(&fabric, owner, &input) == HS_FABRIC_OK);
    /* Published entries are independent of a subsequently held producer lock. */
    REQUIRE(pthread_mutex_lock(&slot->mutex) == 0);
    REQUIRE(hs_fabric_step(&fabric, 1) == 1);
    REQUIRE(fabric.views[owner.slot].owner.generation == owner.generation);
    REQUIRE(hs_fabric_leave(&fabric, owner) == HS_FABRIC_BUSY);
    REQUIRE(pthread_mutex_unlock(&slot->mutex) == 0);
    hs_fabric_destroy(&fabric);
    puts("{\"test\":\"cursor_wrap_and_generation_reuse\",\"passed\":true}");
}

static void* submit(void* unused) {
    (void)unused;
    for (;;) {
        u32 sequence = atomic_fetch_add_explicit(&ticket, 1, memory_order_relaxed) + 1;
        if (sequence > FRAMES) break;
        HSParticipantFrame input = frame(sequence);
        HSFabricResult result;
        while ((result = hs_fabric_submit(&fabric, owner, &input)) != HS_FABRIC_OK) {
            REQUIRE(result == HS_FABRIC_BUSY || result == HS_FABRIC_FULL || result == HS_FABRIC_SEQUENCE);
            sched_yield();
        }
    }
    return NULL;
}

static void shared_owner(void) {
    REQUIRE(hs_fabric_init(&fabric));
    REQUIRE(hs_fabric_join(&fabric, &owner) == HS_FABRIC_OK);
    atomic_init(&ticket, 0);
    pthread_t threads[WRITERS];
    for (u32 i = 0; i < WRITERS; i++) REQUIRE(pthread_create(&threads[i], NULL, submit, NULL) == 0);
    for (u32 sequence = 1; sequence <= FRAMES; sequence++) {
        while (!hs_fabric_step(&fabric, 1)) { }
        REQUIRE(fabric.views[owner.slot].frame.sequence == sequence);
        REQUIRE(fabric.views[owner.slot].frame.count == 1);
        REQUIRE(fabric.views[owner.slot].frame.draws[0].op == HS_RC_DRAW);
    }
    for (u32 i = 0; i < WRITERS; i++) REQUIRE(pthread_join(threads[i], NULL) == 0);
    REQUIRE(hs_fabric_leave(&fabric, owner) == HS_FABRIC_OK);
    REQUIRE(hs_fabric_step(&fabric, 1) == 1);
    REQUIRE(fabric.applied == FRAMES && fabric.retired == 1);
    hs_fabric_destroy(&fabric);
    puts("{\"test\":\"four_submitters_one_owner_fifo\",\"frames\":20000,\"passed\":true}");
}

int main(void) { alarm(30); wrap_and_reuse(); shared_owner(); return 0; }
#else
int main(void) { puts("Split-publication candidate required"); return 77; }
#endif
