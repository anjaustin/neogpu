/* Architectural probes: false hypotheses are findings, not harness failures. */
#include "hs_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
static HSGpu gpu;
static u32 seen[8192], count;

static int observe(Node* node) {
    Message message;
    int applied = 0;
    while (mq_pop(&node->inbox, &message)) {
        REQUIRE(count < sizeof(seen) / sizeof(seen[0]));
        seen[count++] = message.cid;
        applied++;
    }
    return applied;
}

static void setup(void) {
    hs_gpu_init(&gpu);
    gpu.system.recording = false;
    gpu.system.render_list = NULL;
    gpu.shader_node.process_fn = observe;
    count = 0;
}

static void send_id(u32 id, HSChannel channel) {
    Message message = {.to = NODE_SHADER, .from = NODE_CPU, .op = OP_SET_SHADER,
                       .cid = id, .channel = channel};
    REQUIRE(hs_try_send(&gpu.system, &message, NULL, 0));
}

typedef struct { u32 first, number; } Batch;
static void* batch(void* pointer) {
    Batch* work = pointer;
    for (u32 i = 0; i < work->number; i++) send_id(work->first + i, CHAN_RENDER);
    return NULL;
}

static void joined_batch(u32 first, u32 number) {
    pthread_t thread;
    Batch work = {first, number};
    REQUIRE(pthread_create(&thread, NULL, batch, &work) == 0);
    REQUIRE(pthread_join(thread, NULL) == 0);
}

static void drain(u32 expected) {
    for (u32 i = 0; i < 8192 && count < expected; i++) hs_step(&gpu.system);
    REQUIRE(count == expected);
    REQUIRE(gpu.system.dropped_queue_full == 0);
}

static void ordering(void) {
    setup();
    hs_set_channel_budget(&gpu.system, CHAN_RENDER, 1);
    for (u32 i = 0; i <= HS_SPSC_SIZE; i++) send_id(i, CHAN_RENDER);
    REQUIRE(atomic_load(&gpu.system.mpsc_ok[CHAN_RENDER]) == 1);
    hs_step(&gpu.system); /* Free one SPSC slot while the older fallback waits. */
    send_id(HS_SPSC_SIZE + 1, CHAN_RENDER);
    drain(HS_SPSC_SIZE + 2);
    u32 inversions = 0;
    for (u32 i = 1; i < count; i++) inversions += seen[i] < seen[i - 1];
    printf("{\"probe\":\"same_producer_fifo\",\"admitted\":%u,\"applied\":%u,"
           "\"inversions\":%u,\"last_two\":[%u,%u],\"hypothesis_supported\":%s}\n",
           HS_SPSC_SIZE + 2, count, inversions, seen[count - 2], seen[count - 1],
           inversions ? "false" : "true");
    hs_gpu_shutdown(&gpu);
}

static void controls(void) {
    setup();
    for (u32 i = 0; i < 64; i++) send_id(i, CHAN_RENDER);
    drain(64);
    for (u32 i = 0; i < count; i++) REQUIRE(seen[i] == i);
    printf("{\"probe\":\"fifo_without_fallback_control\",\"applied\":64,\"inversions\":0}\n");
    hs_gpu_shutdown(&gpu);
}

static void fairness(void) {
    setup();
    hs_set_channel_budget(&gpu.system, CHAN_RENDER, 4);
    joined_batch(100, 8);
    joined_batch(200, 8);
    hs_step(&gpu.system);
    u32 first = 0, second = 0;
    for (u32 i = 0; i < count; i++) { first += seen[i] < 200; second += seen[i] >= 200; }
    drain(16);
    printf("{\"probe\":\"both_producers_serviced_first_step\",\"budget\":4,"
           "\"first_producer\":%u,\"second_producer\":%u,\"total_applied\":%u,"
           "\"hypothesis_supported\":%s}\n", first, second, count, first && second ? "true" : "false");
    hs_gpu_shutdown(&gpu);
}

static void churn(void) {
    setup();
    const u32 participants = HS_MAX_PRODUCERS * 3;
    for (u32 i = 0; i < participants; i++) { joined_batch(i, 1); hs_step(&gpu.system); }
    drain(participants);
    u32 fallback = atomic_load(&gpu.system.mpsc_ok[CHAN_RENDER]);
    printf("{\"probe\":\"departed_producer_lane_reclamation\",\"sequential_threads\":%u,"
           "\"lane_capacity\":%u,\"registration_counter\":%u,\"fallback_admissions\":%u,"
           "\"applied\":%u,\"hypothesis_supported\":%s}\n", participants, HS_MAX_PRODUCERS,
           atomic_load(&gpu.system.producer_count), fallback, count, fallback ? "false" : "true");
    hs_gpu_shutdown(&gpu);
}

static void stalled_reservation(void) {
    setup();
    /* Register main as a real lane owner before forcing subsequent registrations
     * down fallback. These are explicit white-box fixture operations, not sends. */
    send_id(10, CHAN_RT);
    atomic_store(&gpu.system.producer_count, HS_MAX_PRODUCERS);
    HSSubmitQueue* queue = &gpu.system.submit[CHAN_RENDER];
    u32 reserved = atomic_fetch_add(&queue->enqueue_pos, 1);
    REQUIRE(reserved == 0);
    joined_batch(20, 1); /* Published fallback item behind the unpublished head. */
    send_id(30, CHAN_RENDER); /* Independent SPSC lane. */
    hs_step(&gpu.system);
    u32 progressed = count;
    bool rt = false, lane = false, fallback = false;
    for (u32 i = 0; i < count; i++) { rt |= seen[i] == 10; lane |= seen[i] == 30; fallback |= seen[i] == 20; }
    HSSubmitSlot* slot = &queue->slots[0];
    slot->msg = (Message){.to = NODE_SHADER, .from = NODE_CPU, .op = OP_SET_SHADER,
                         .cid = 40, .channel = CHAN_RENDER};
    slot->payload_len = 0;
    atomic_store_explicit(&slot->seq, 1, memory_order_release);
    drain(4);
    printf("{\"probe\":\"stalled_reservation\",\"fixture\":\"synthetic_unpublished_head\","
           "\"before_release_applied\":%u,\"rt_progress\":%s,\"spsc_progress\":%s,"
           "\"fallback_peer_progress\":%s,\"after_release_applied\":%u,"
           "\"fallback_isolation_supported\":%s}\n", progressed, rt ? "true" : "false",
           lane ? "true" : "false", fallback ? "true" : "false", count, fallback ? "true" : "false");
    hs_gpu_shutdown(&gpu);
}

static void frame_message(u8 op, u8 to, u16 resource, u32 owner) {
    Message message = {.op = op, .to = to, .from = NODE_CPU, .cid = owner,
                       .payload_idx = resource, .channel = CHAN_RENDER};
    REQUIRE(hs_try_send(&gpu.system, &message, NULL, 0));
}

static void frames(void) {
    setup();
    gpu.system.render_list = &gpu.render;
    frame_message(OP_FRAME_BEGIN, NODE_SYSTEM, 0, 1);
    frame_message(OP_DRAW, NODE_BUFFER, 1, 1);
    frame_message(OP_FRAME_BEGIN, NODE_SYSTEM, 0, 2);
    frame_message(OP_DRAW, NODE_BUFFER, 2, 2);
    frame_message(OP_FRAME_END, NODE_SYSTEM, 0, 2);
    frame_message(OP_FRAME_END, NODE_SYSTEM, 0, 1);
    hs_step(&gpu.system);
    u32 a = 0, b = 0;
    for (u32 i = 0; i < gpu.render.count; i++) {
        if (gpu.render.cmds[i].op != HS_RC_DRAW) continue;
        a += gpu.render.cmds[i].a == 1;
        b += gpu.render.cmds[i].a == 2;
    }
    REQUIRE(!gpu.render.overflow);
    printf("{\"probe\":\"interleaved_frame_contributions\",\"messages_admitted\":6,"
           "\"render_commands\":%u,\"owner_a_draws\":%u,\"owner_b_draws\":%u,"
           "\"evidence_boundary\":\"decoded_list_not_pixels\",\"hypothesis_supported\":%s}\n",
           gpu.render.count, a, b, a == 1 && b == 1 ? "true" : "false");
    hs_gpu_shutdown(&gpu);

    setup();
    gpu.system.render_list = &gpu.render;
    frame_message(OP_FRAME_BEGIN, NODE_SYSTEM, 0, 1);
    frame_message(OP_DRAW, NODE_BUFFER, 1, 1);
    frame_message(OP_DRAW, NODE_BUFFER, 2, 2);
    frame_message(OP_FRAME_END, NODE_SYSTEM, 0, 1);
    hs_step(&gpu.system);
    a = b = 0;
    for (u32 i = 0; i < gpu.render.count; i++) {
        if (gpu.render.cmds[i].op != HS_RC_DRAW) continue;
        a += gpu.render.cmds[i].a == 1;
        b += gpu.render.cmds[i].a == 2;
    }
    REQUIRE(a == 1 && b == 1 && !gpu.render.overflow);
    printf("{\"probe\":\"single_frame_two_contributions_control\",\"owner_a_draws\":1,\"owner_b_draws\":1}\n");
    hs_gpu_shutdown(&gpu);
}

int main(void) {
    alarm(30);
    controls(); ordering(); fairness(); churn(); stalled_reservation(); frames();
    return 0;
}
