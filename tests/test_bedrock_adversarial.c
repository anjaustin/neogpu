#define _POSIX_C_SOURCE 200809L
#include "hs_fabric.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "%d: %s\n", __LINE__, #x); exit(1); } } while (0)
static HSFabric fabric;
static HSParticipant owner;
static pthread_mutex_t control = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t changed = PTHREAD_COND_INITIALIZER;
static unsigned armed, reached;
static bool resume;
static u32 step_result;
static HSFabricResult submit_result;

/* Called only by generated diagnostic source, never by production. */
void bedrock_adversarial_hook(unsigned stage) {
    REQUIRE(pthread_mutex_lock(&control) == 0);
    if (stage == armed) {
        reached = stage;
        REQUIRE(pthread_cond_broadcast(&changed) == 0);
        while (!resume) REQUIRE(pthread_cond_wait(&changed, &control) == 0);
    }
    REQUIRE(pthread_mutex_unlock(&control) == 0);
}
static void arm(unsigned stage) {
    REQUIRE(pthread_mutex_lock(&control) == 0);
    armed = stage; reached = 0; resume = false;
    REQUIRE(pthread_mutex_unlock(&control) == 0);
}
static void wait_for_hook(void) {
    REQUIRE(pthread_mutex_lock(&control) == 0);
    while (!reached) REQUIRE(pthread_cond_wait(&changed, &control) == 0);
    REQUIRE(pthread_mutex_unlock(&control) == 0);
}
static void release_hook(void) {
    REQUIRE(pthread_mutex_lock(&control) == 0);
    resume = true; armed = 0;
    REQUIRE(pthread_cond_broadcast(&changed) == 0);
    REQUIRE(pthread_mutex_unlock(&control) == 0);
}
static HSParticipantFrame frame(u32 sequence) {
    HSParticipantFrame input = {.sequence = sequence, .count = 1};
    input.draws[0] = (HSRenderCmd){.op = HS_RC_DRAW, .a = (u8)owner.slot};
    return input;
}
static void init(void) {
    REQUIRE(hs_fabric_init(&fabric));
    REQUIRE(hs_fabric_join(&fabric, &owner) == HS_FABRIC_OK && owner.slot == 0);
}
static void* step(void* unused) { (void)unused; step_result = hs_fabric_step(&fabric, 1); return NULL; }
static void* submit(void* unused) {
    (void)unused;
    HSParticipantFrame input = frame(1);
    submit_result = hs_fabric_submit(&fabric, owner, &input);
    return NULL;
}

static void unpublished(void) {
    init(); arm(1);
    pthread_t thread;
    REQUIRE(pthread_create(&thread, NULL, submit, NULL) == 0);
    wait_for_hook(); /* Producer has copied, but not published, its frame. */
    REQUIRE(hs_fabric_step(&fabric, 1) == 0 && fabric.applied == 0);
    REQUIRE(hs_fabric_leave(&fabric, owner) == HS_FABRIC_BUSY);
    release_hook(); REQUIRE(pthread_join(thread, NULL) == 0);
    REQUIRE(submit_result == HS_FABRIC_OK);
    REQUIRE(hs_fabric_leave(&fabric, owner) == HS_FABRIC_OK);
    REQUIRE(hs_fabric_step(&fabric, 1) == 1 && fabric.applied == 1);
    REQUIRE(hs_fabric_step(&fabric, 1) == 1 && fabric.retired == 1);
    hs_fabric_destroy(&fabric);
}
static void stale_empty(void) {
    init(); arm(2);
    pthread_t thread;
    REQUIRE(pthread_create(&thread, NULL, step, NULL) == 0);
    wait_for_hook(); /* Consumer has already sampled empty cursors. */
    HSParticipantFrame input = frame(1);
    REQUIRE(hs_fabric_submit(&fabric, owner, &input) == HS_FABRIC_OK);
    REQUIRE(hs_fabric_leave(&fabric, owner) == HS_FABRIC_OK);
    release_hook(); REQUIRE(pthread_join(thread, NULL) == 0);
    REQUIRE(step_result == 0 && fabric.retired == 0 && fabric.applied == 0);
    REQUIRE(hs_fabric_step(&fabric, 1) == 1 && fabric.applied == 1);
    REQUIRE(fabric.views[owner.slot].frame.sequence == 1);
    REQUIRE(hs_fabric_step(&fabric, 1) == 1 && fabric.retired == 1);
    hs_fabric_destroy(&fabric);
}
static void held_storage(void) {
    init();
    for (u32 i = 1; i <= HS_FABRIC_QUEUE; i++) {
        HSParticipantFrame input = frame(i);
        REQUIRE(hs_fabric_submit(&fabric, owner, &input) == HS_FABRIC_OK);
    }
    arm(3);
    pthread_t thread;
    REQUIRE(pthread_create(&thread, NULL, step, NULL) == 0);
    wait_for_hook(); /* Consumer copied, but has not released ring storage. */
    HSParticipantFrame excess = frame(HS_FABRIC_QUEUE + 1);
    REQUIRE(hs_fabric_submit(&fabric, owner, &excess) == HS_FABRIC_FULL);
    REQUIRE(hs_fabric_leave(&fabric, owner) == HS_FABRIC_OK);
    release_hook(); REQUIRE(pthread_join(thread, NULL) == 0);
    REQUIRE(step_result == 1 && fabric.views[owner.slot].frame.sequence == 1);
    for (u32 i = 2; i <= HS_FABRIC_QUEUE; i++) {
        REQUIRE(hs_fabric_step(&fabric, 1) == 1);
        REQUIRE(fabric.views[owner.slot].frame.sequence == i);
    }
    REQUIRE(fabric.applied == HS_FABRIC_QUEUE && fabric.retired == 0);
    REQUIRE(hs_fabric_step(&fabric, 1) == 1 && fabric.retired == 1);
    hs_fabric_destroy(&fabric);
}
int main(void) {
    alarm(20);
    unpublished(); stale_empty(); held_storage();
    puts("{\"test\":\"unpublished_stale_empty_held_storage\",\"passed\":true}");
    return 0;
}
