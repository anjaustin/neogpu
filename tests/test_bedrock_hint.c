#define _POSIX_C_SOURCE 200809L
#include "hs_fabric.h"
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "%d: %s\n", __LINE__, #x); exit(1); } } while (0)
static HSFabric fabric;
static HSParticipant owner;
static atomic_uint observed;
enum { FRAMES = 50000 };

static void* publish(void* unused) {
    (void)unused;
    for (u32 sequence = 1; sequence <= FRAMES; sequence++) {
        while (atomic_load_explicit(&observed, memory_order_acquire) != sequence - 1) sched_yield();
        HSParticipantFrame frame = {.sequence = sequence}; /* zero draws still require application */
        HSFabricResult result;
        while ((result = hs_fabric_submit(&fabric, owner, &frame)) != HS_FABRIC_OK) {
            REQUIRE(result == HS_FABRIC_BUSY);
            sched_yield();
        }
    }
    return NULL;
}

int main(void) {
    alarm(30);
    REQUIRE(hs_fabric_init(&fabric));
    /* Retirement alone must publish work, even with no submitted frame. */
    for (u32 cycle = 0; cycle < 1024; cycle++) {
        HSParticipant previous = owner;
        REQUIRE(hs_fabric_join(&fabric, &owner) == HS_FABRIC_OK);
        if (cycle) REQUIRE(hs_fabric_leave(&fabric, previous) == HS_FABRIC_STALE);
        REQUIRE(hs_fabric_step(&fabric, 8) == 0);
        REQUIRE(hs_fabric_leave(&fabric, owner) == HS_FABRIC_OK);
        REQUIRE(hs_fabric_step(&fabric, 1) == 1);
        REQUIRE(!fabric.views[owner.slot].active);
    }
    REQUIRE(hs_fabric_join(&fabric, &owner) == HS_FABRIC_OK);
    atomic_init(&observed, 0);
    pthread_t thread;
    REQUIRE(pthread_create(&thread, NULL, publish, NULL) == 0);
    for (u32 sequence = 1; sequence <= FRAMES; sequence++) {
        while (!hs_fabric_step(&fabric, 1)) { }
        REQUIRE(fabric.views[owner.slot].frame.sequence == sequence);
        REQUIRE(fabric.views[owner.slot].frame.count == 0);
        /* Let the producer republish immediately after each drain. */
        atomic_store_explicit(&observed, sequence, memory_order_release);
    }
    REQUIRE(pthread_join(thread, NULL) == 0);
    REQUIRE(hs_fabric_leave(&fabric, owner) == HS_FABRIC_OK);
    REQUIRE(hs_fabric_step(&fabric, 1) == 1);
    REQUIRE(fabric.applied == FRAMES && fabric.retired == 1025);
    REQUIRE(hs_fabric_step(&fabric, 8) == 0);
    hs_fabric_destroy(&fabric);
    puts("{\"test\":\"empty_retirement_and_drain_republication\",\"frames\":50000,\"passed\":true}");
    return 0;
}
