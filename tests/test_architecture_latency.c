#define _POSIX_C_SOURCE 200809L
#include "hs_gpu.h"
#include <errno.h>
#include <inttypes.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define SAMPLES 1000
#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
static HSGpu gpu;
static atomic_bool go, stop, done, entered;
static atomic_uint_fast64_t attempted[SAMPLES], admitted[SAMPLES];
static u64 applied[SAMPLES], background_applied, rt_applied, background_sent;
static u64 scheduled[SAMPLES];
static u64 retries, slow_start, slow_end;
static bool slow_mode;

static u64 ns(void) {
    struct timespec time;
    REQUIRE(clock_gettime(CLOCK_MONOTONIC, &time) == 0);
    return (u64)time.tv_sec * 1000000000ull + time.tv_nsec;
}

static void sleep_until(u64 deadline) {
    struct timespec time = {.tv_sec = deadline / 1000000000ull, .tv_nsec = deadline % 1000000000ull};
    int result;
    do { result = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &time, NULL); } while (result == EINTR);
    REQUIRE(result == 0);
}

static int observe(Node* node) {
    Message message;
    int count = 0;
    while (mq_pop(&node->inbox, &message)) {
        if (message.channel == CHAN_RT) {
            REQUIRE(message.cid < SAMPLES && applied[message.cid] == 0);
            applied[message.cid] = ns();
            rt_applied++;
        } else {
            background_applied++;
            if (slow_mode && !atomic_load(&entered)) {
                slow_start = ns();
                atomic_store_explicit(&entered, true, memory_order_release);
                sleep_until(slow_start + 5000000ull);
                slow_end = ns();
            }
        }
        count++;
    }
    return count;
}

static void* interactive(void* unused) {
    (void)unused;
    while (!atomic_load_explicit(&go, memory_order_acquire)) sched_yield();
    if (slow_mode) while (!atomic_load_explicit(&entered, memory_order_acquire)) sched_yield();
    u64 start = ns();
    u32 samples = slow_mode ? 1 : SAMPLES;
    for (u32 i = 0; i < samples; i++) {
        scheduled[i] = start + (u64)i * 1000000ull;
        if (!slow_mode) sleep_until(scheduled[i]);
        Message message = {.op = OP_SET_SHADER, .to = NODE_SHADER, .from = NODE_CPU,
                           .cid = i, .channel = CHAN_RT};
        atomic_store_explicit(&attempted[i], ns(), memory_order_release);
        while (!hs_try_send(&gpu.system, &message, NULL, 0)) { retries++; sched_yield(); }
        atomic_store_explicit(&admitted[i], ns(), memory_order_release);
    }
    atomic_store_explicit(&done, true, memory_order_release);
    return NULL;
}

static void* background(void* unused) {
    (void)unused;
    while (!atomic_load_explicit(&go, memory_order_acquire)) sched_yield();
    while (!atomic_load_explicit(&stop, memory_order_acquire)) {
        Message message = {.op = OP_SET_SHADER, .to = NODE_SHADER, .from = NODE_CPU,
                           .channel = CHAN_RENDER};
        if (hs_try_send(&gpu.system, &message, NULL, 0)) background_sent++;
        else sched_yield();
    }
    return NULL;
}

static int compare(const void* a, const void* b) {
    u64 x = *(const u64*)a, y = *(const u64*)b;
    return (x > y) - (x < y);
}

static void run(const char* name, bool load, bool slow) {
    hs_gpu_init(&gpu);
    gpu.system.recording = false;
    gpu.system.render_list = NULL;
    gpu.shader_node.process_fn = observe;
    slow_mode = slow;
    background_sent = background_applied = rt_applied = retries = slow_start = slow_end = 0;
    atomic_init(&go, false); atomic_init(&stop, false);
    atomic_init(&done, false); atomic_init(&entered, false);
    for (u32 i = 0; i < SAMPLES; i++) {
        atomic_init(&attempted[i], 0); atomic_init(&admitted[i], 0); applied[i] = 0;
    }
    pthread_t rt_thread, bg_thread;
    REQUIRE(pthread_create(&rt_thread, NULL, interactive, NULL) == 0);
    if (load) REQUIRE(pthread_create(&bg_thread, NULL, background, NULL) == 0);
    if (slow) {
        Message message = {.op = OP_SET_SHADER, .to = NODE_SHADER, .from = NODE_CPU, .channel = CHAN_RENDER};
        REQUIRE(hs_try_send(&gpu.system, &message, NULL, 0));
        background_sent = 1;
    }
    u64 start = ns();
    atomic_store_explicit(&go, true, memory_order_release);
    u32 samples = slow ? 1 : SAMPLES;
    while (!atomic_load_explicit(&done, memory_order_acquire) || rt_applied < samples) {
        REQUIRE(ns() - start < 10000000000ull);
        hs_step(&gpu.system);
    }
    atomic_store_explicit(&stop, true, memory_order_release);
    REQUIRE(pthread_join(rt_thread, NULL) == 0);
    if (load) REQUIRE(pthread_join(bg_thread, NULL) == 0);
    for (u32 i = 0; i < 10000 && background_applied < background_sent; i++) hs_step(&gpu.system);
    REQUIRE(background_applied == background_sent && rt_applied == samples);
    REQUIRE(gpu.system.dropped_queue_full == 0);
    u64 duration = ns() - start, values[SAMPLES], admission_values[SAMPLES];
    u32 applied_before_admission_return = 0;
    for (u32 i = 0; i < samples; i++) {
        u64 attempt = atomic_load(&attempted[i]), admit = atomic_load(&admitted[i]);
        REQUIRE(attempt && applied[i] >= attempt && admit >= attempt);
        values[i] = applied[i] - attempt;
        admission_values[i] = admit - attempt;
        applied_before_admission_return += applied[i] < admit;
        printf("{\"record\":\"sample\",\"workload\":\"%s\",\"sample\":%u,"
               "\"scheduled_ns\":%" PRIu64 ",\"attempt_ns\":%" PRIu64 ",\"admission_return_ns\":%" PRIu64 ",\"applied_ns\":%" PRIu64 "}\n",
               name, i, (uint64_t)scheduled[i], (uint64_t)attempt, (uint64_t)admit, (uint64_t)applied[i]);
    }
    qsort(values, samples, sizeof(values[0]), compare);
    qsort(admission_values, samples, sizeof(values[0]), compare);
    /* Nearest-rank percentiles, reported in integer nanoseconds. */
    printf("{\"record\":\"summary\",\"workload\":\"%s\",\"samples\":%u,"
           "\"rt_applied\":%" PRIu64 ",\"rt_retries\":%" PRIu64 ",\"background_admitted\":%" PRIu64 ","
           "\"background_applied\":%" PRIu64 ",\"duration_ns\":%" PRIu64 ","
           "\"attempt_to_apply_p50_ns\":%" PRIu64 ",\"attempt_to_apply_p95_ns\":%" PRIu64 ","
           "\"attempt_to_apply_p99_ns\":%" PRIu64 ",\"attempt_to_apply_max_ns\":%" PRIu64 ","
           "\"admission_max_ns\":%" PRIu64 ",\"applied_before_admission_return\":%u,"
           "\"slow_handler_ns\":%" PRIu64 "}\n",
           name, samples, (uint64_t)rt_applied, (uint64_t)retries, (uint64_t)background_sent,
           (uint64_t)background_applied, (uint64_t)duration,
           (uint64_t)values[(samples * 50 + 99) / 100 - 1],
           (uint64_t)values[(samples * 95 + 99) / 100 - 1],
           (uint64_t)values[(samples * 99 + 99) / 100 - 1], (uint64_t)values[samples - 1],
           (uint64_t)admission_values[samples - 1], applied_before_admission_return, (uint64_t)(slow_end - slow_start));
    hs_gpu_shutdown(&gpu);
}

int main(int argc, char** argv) {
    alarm(40);
    REQUIRE(argc == 1 || (argc == 2 && !strcmp(argv[1], "--reverse")));
    if (argc == 2) run("render_flood", true, false);
    run("unloaded", false, false);
    if (argc == 1) run("render_flood", true, false);
    run("slow_render_handler", false, true);
    return 0;
}
