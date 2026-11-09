#define _GNU_SOURCE
#include "comparison_transport.h"
#include <errno.h>
#include <inttypes.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
#define STRIDE 128
static const ComparisonOps* ops;
static void* context;
static u32 participants, frames, draws;
static atomic_bool go;
static atomic_uint ready;
typedef struct { atomic_uint_fast64_t attempt, admission; u64 application; } Sample;
typedef struct { u32 id; u64 accepted, retries; Sample* samples; } Producer;

static u64 ns(void) {
    struct timespec time;
    REQUIRE(clock_gettime(CLOCK_MONOTONIC, &time) == 0);
    return (u64)time.tv_sec * 1000000000ull + time.tv_nsec;
}
static void pin(u32 cpu) {
    cpu_set_t set;
    CPU_ZERO(&set); CPU_SET(cpu, &set);
    REQUIRE(pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0);
}
static void* produce(void* pointer) {
    Producer* producer = pointer;
    pin(1 + producer->id % 3);
    atomic_fetch_add_explicit(&ready, 1, memory_order_release);
    while (!atomic_load_explicit(&go, memory_order_acquire)) sched_yield();
    for (u32 i = 1; i <= frames; i++) {
        HSParticipantFrame input = {.sequence = i, .count = draws};
        for (u32 j = 0; j < draws; j++)
            input.draws[j] = (HSRenderCmd){.op = HS_RC_DRAW, .a = (u8)producer->id};
        Sample* sample = i % STRIDE == 0 ? &producer->samples[i / STRIDE - 1] : NULL;
        if (sample) atomic_store_explicit(&sample->attempt, ns(), memory_order_release);
        while (!ops->submit(context, producer->id, &input)) { producer->retries++; sched_yield(); }
        if (sample) atomic_store_explicit(&sample->admission, ns(), memory_order_release);
        producer->accepted++;
    }
    return NULL;
}
static u32 number(const char* text, u32 maximum) {
    REQUIRE(text && text[0] && text[0] != '-');
    errno = 0; char* end;
    unsigned long value = strtoul(text, &end, 10);
    REQUIRE(!errno && !*end && value > 0 && value <= maximum);
    return (u32)value;
}
static void selftest(void) {
    const ComparisonOps* implementations[] = {&comparison_native, &comparison_rigtorp};
    for (u32 implementation = 0; implementation < 2; implementation++) {
        const ComparisonOps* current = implementations[implementation];
        void* test = current->create(1);
        REQUIRE(test);
        HSParticipantFrame input = {.count = 1};
        input.draws[0] = (HSRenderCmd){.op = HS_RC_DRAW};
        for (u32 i = 1; i <= HS_FABRIC_QUEUE; i++) {
            input.sequence = i;
            REQUIRE(current->submit(test, 0, &input));
        }
        input.sequence = HS_FABRIC_QUEUE + 1;
        REQUIRE(!current->submit(test, 0, &input));
        REQUIRE(current->step(test) == 1 && current->view(test, 0)->sequence == 1);
        REQUIRE(current->submit(test, 0, &input));
        memset(&input, 0xff, sizeof(input));
        for (u32 i = 2; i <= HS_FABRIC_QUEUE + 1; i++) {
            REQUIRE(current->step(test) == 1);
            const HSParticipantFrame* retained = current->view(test, 0);
            REQUIRE(retained->sequence == i && retained->count == 1 && retained->draws[0].op == HS_RC_DRAW);
        }
        REQUIRE(current->step(test) == 0);
        current->destroy(test);
        printf("{\"record\":\"selftest\",\"implementation\":\"%s\",\"passed\":true}\n", current->name);
    }
}
int main(int argc, char** argv) {
    if (argc == 2 && !strcmp(argv[1], "--selftest")) { selftest(); return 0; }
    REQUIRE(argc == 5);
    ops = !strcmp(argv[1], "neogpu") ? &comparison_native : !strcmp(argv[1], "rigtorp") ? &comparison_rigtorp : NULL;
    REQUIRE(ops);
    participants = number(argv[2], HS_FABRIC_PARTICIPANTS);
    frames = number(argv[3], 1000000);
    draws = number(argv[4], HS_FABRIC_DRAWS);
    alarm(90);
    pin(0);
    context = ops->create(participants);
    REQUIRE(context);
    Producer producers[HS_FABRIC_PARTICIPANTS] = {0};
    pthread_t threads[HS_FABRIC_PARTICIPANTS];
    u64 observed[HS_FABRIC_PARTICIPANTS] = {0};
    atomic_init(&go, false); atomic_init(&ready, 0);
    for (u32 i = 0; i < participants; i++) {
        producers[i].id = i;
        producers[i].samples = calloc(frames / STRIDE + 1, sizeof(Sample));
        REQUIRE(producers[i].samples);
        for (u32 j = 0; j < frames / STRIDE; j++) {
            atomic_init(&producers[i].samples[j].attempt, 0);
            atomic_init(&producers[i].samples[j].admission, 0);
        }
        REQUIRE(pthread_create(&threads[i], NULL, produce, &producers[i]) == 0);
    }
    while (atomic_load_explicit(&ready, memory_order_acquire) < participants) sched_yield();
    u64 start = ns(), applied = 0;
    atomic_store_explicit(&go, true, memory_order_release);
    while (applied < (u64)frames * participants) {
        u32 events = ops->step(context), observed_events = 0;
        for (u32 i = 0; i < participants; i++) {
            const HSParticipantFrame* view = ops->view(context, i);
            if (!view->sequence || view->sequence == observed[i]) continue;
            REQUIRE(view->sequence == observed[i] + 1 && view->count == draws);
            for (u32 j = 0; j < draws; j++) {
                HSRenderCmd expected = {.op = HS_RC_DRAW, .a = (u8)i};
                REQUIRE(memcmp(&view->draws[j], &expected, sizeof(expected)) == 0);
            }
            if (view->sequence % STRIDE == 0)
                producers[i].samples[view->sequence / STRIDE - 1].application = ns();
            observed[i]++;
            observed_events++;
        }
        REQUIRE(events == observed_events);
        applied += events;
    }
    u64 accepted = 0, retries = 0;
    for (u32 i = 0; i < participants; i++) {
        REQUIRE(pthread_join(threads[i], NULL) == 0);
        REQUIRE(observed[i] == frames && producers[i].accepted == frames);
        accepted += producers[i].accepted; retries += producers[i].retries;
    }
    u64 elapsed = ns() - start;
    for (u32 i = 0; i < participants; i++) {
        for (u32 j = 0; j < frames / STRIDE; j++) {
            Sample* sample = &producers[i].samples[j];
            u64 attempt = atomic_load(&sample->attempt), admission = atomic_load(&sample->admission);
            REQUIRE(attempt && admission >= attempt && sample->application >= attempt);
            printf("{\"record\":\"sample\",\"producer\":%u,\"sequence\":%u,\"attempt_ns\":%" PRIu64
                   ",\"admission_return_ns\":%" PRIu64 ",\"applied_ns\":%" PRIu64 "}\n",
                   i, (j + 1) * STRIDE, (uint64_t)attempt, (uint64_t)admission, (uint64_t)sample->application);
        }
        free(producers[i].samples);
    }
    printf("{\"record\":\"summary\",\"implementation\":\"%s\",\"participants\":%u,\"draws\":%u,"
           "\"accepted\":%" PRIu64 ",\"applied\":%" PRIu64 ",\"elapsed_ns\":%" PRIu64
           ",\"retries\":%" PRIu64 ",\"frame_bytes\":%zu,\"command_bytes\":%zu}\n",
           ops->name, participants, draws, (uint64_t)accepted, (uint64_t)applied,
           (uint64_t)elapsed, (uint64_t)retries, sizeof(HSParticipantFrame), sizeof(HSRenderCmd));
    ops->destroy(context);
    return 0;
}
