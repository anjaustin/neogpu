/*
 * Test ML Message-Passing Layer
 *
 * NOTE: MLSystem uses blocking channels by default, requiring a separate
 * consumer thread. These tests use non-blocking for single-thread tests,
 * and a real producer/consumer thread pair for the multi-producer test.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <math.h>

#include "hs_ml_msg.h"

static double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000.0 + ts.tv_nsec / 1000.0;
}

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  [FAIL] %s: %s\n", __func__, msg); \
        return 1; \
    } \
} while(0)

#define TEST_PASS() do { \
    printf("  [PASS] %s\n", __func__); \
    return 0; \
} while(0)

/*============================================================================
 * Test: SPSC Queue
 *============================================================================*/

static int test_spsc_basic(void) {
    MLSpscQueue* q = malloc(sizeof(MLSpscQueue));
    ml_spsc_init(q);

    TEST_ASSERT(ml_spsc_empty(q), "should be empty after init");

    MLMsg msg = {0};
    msg.M = 42; msg.N = 100;
    TEST_ASSERT(ml_spsc_push(q, &msg), "push should succeed");
    TEST_ASSERT(!ml_spsc_empty(q), "should not be empty after push");

    MLMsg out = {0};
    TEST_ASSERT(ml_spsc_pop(q, &out), "pop should succeed");
    TEST_ASSERT(out.M == 42, "M should match");
    TEST_ASSERT(out.N == 100, "N should match");
    TEST_ASSERT(ml_spsc_empty(q), "should be empty after pop");
    TEST_ASSERT(!ml_spsc_pop(q, &out), "pop from empty should fail");

    free(q);
    TEST_PASS();
}

static int test_spsc_fifo_order(void) {
    MLSpscQueue* q = malloc(sizeof(MLSpscQueue));
    ml_spsc_init(q);

    /* Fill completely */
    for (u32 i = 0; i < ML_QUEUE_SIZE; i++) {
        MLMsg msg = {0};
        msg.cid = i;
        TEST_ASSERT(ml_spsc_push(q, &msg), "push should succeed");
    }
    TEST_ASSERT(!ml_spsc_push(q, &(MLMsg){0}), "push to full should fail");

    /* Drain and verify FIFO */
    for (u32 i = 0; i < ML_QUEUE_SIZE; i++) {
        MLMsg out;
        TEST_ASSERT(ml_spsc_pop(q, &out), "pop should succeed");
        TEST_ASSERT(out.cid == i, "FIFO order must be preserved");
    }

    free(q);
    TEST_PASS();
}

/*============================================================================
 * Test: System lifecycle
 *============================================================================*/

static int test_system_init(void) {
    MLSystem* sys = malloc(sizeof(MLSystem));
    ml_sys_init(sys);

    TEST_ASSERT(atomic_load(&sys->running), "should be running");
    TEST_ASSERT(atomic_load(&sys->producer_count) == 0, "no producers yet");

    int p0 = ml_sys_register_producer(sys);
    int p1 = ml_sys_register_producer(sys);
    TEST_ASSERT(p0 == 0, "first producer id == 0");
    TEST_ASSERT(p1 == 1, "second producer id == 1");
    TEST_ASSERT((int)atomic_load(&sys->producer_count) == 2, "count == 2");

    ml_sys_free(sys);
    free(sys);
    TEST_PASS();
}

/*============================================================================
 * Test: Submit + step (single-threaded, non-blocking channel)
 *============================================================================*/

static int test_submit_and_step(void) {
    MLSystem* sys = malloc(sizeof(MLSystem));
    ml_sys_init(sys);
    /* Use non-blocking so submit never waits */
    ml_sys_set_blocking(sys, ML_CHAN_PREFILL, false);

    int pid = ml_sys_register_producer(sys);
    TEST_ASSERT(pid >= 0, "producer registration ok");

    u32 M = 4, N = 64, K = 128;
    int8_t*   A = aligned_alloc(64, M * K);
    uint8_t*  W = aligned_alloc(64, N * K / 4);
    int32_t*  C = aligned_alloc(64, M * N * sizeof(int32_t));
    memset(A, 1, M * K);
    memset(W, 0x55, N * K / 4);
    memset(C, 0, M * N * sizeof(int32_t));

    bool ok = ml_sys_submit_gemm(sys, pid,
                                  ML_LAYER_FFN_GATE, 0,
                                  ML_CHAN_PREFILL, HS_ROUTE_TERNARY_2BIT,
                                  A, W, C, M, N, K);
    TEST_ASSERT(ok, "submit should succeed");

    u32 processed = ml_sys_step(sys);
    TEST_ASSERT(processed == 1, "should process 1 message");

    const MLLayerStats* st = ml_sys_layer_stats(sys, ML_LAYER_FFN_GATE);
    TEST_ASSERT(st->messages_processed == 1, "layer stats: 1 message");
    TEST_ASSERT(st->total_ops == (u64)M * N * K, "layer stats: ops match");

    free(A); free(W); free(C);
    ml_sys_free(sys);
    free(sys);
    TEST_PASS();
}

/*============================================================================
 * Test: Capture / replay
 *============================================================================*/

static int test_capture_replay(void) {
    MLSystem* sys = malloc(sizeof(MLSystem));
    ml_sys_init(sys);
    ml_sys_set_blocking(sys, ML_CHAN_PREFILL, false);

    int pid = ml_sys_register_producer(sys);

    MLMsg* cap = malloc(64 * sizeof(MLMsg));
    ml_sys_capture_start(sys, cap, 64);

    MLMsg f1 = ml_msg_fence(ML_CHAN_PREFILL, 1);
    MLMsg f2 = ml_msg_fence(ML_CHAN_PREFILL, 2);
    ml_sys_submit(sys, pid, &f1);
    ml_sys_submit(sys, pid, &f2);
    ml_sys_step(sys);

    u32 n = ml_sys_capture_stop(sys);
    TEST_ASSERT(n == 2, "captured 2 messages");
    TEST_ASSERT(cap[0].cid == 1, "first cid == 1");
    TEST_ASSERT(cap[1].cid == 2, "second cid == 2");

    ml_sys_reset_stats(sys);
    TEST_ASSERT(ml_sys_replay(sys, cap, n), "replay ok");

    free(cap);
    ml_sys_free(sys);
    free(sys);
    TEST_PASS();
}

/*============================================================================
 * Test: Multi-producer / single-consumer (separate threads)
 *============================================================================*/

typedef struct {
    MLSystem* sys;
    int pid;
    int count;       /* messages to send */
    int channel;
} ProdArg;

static void* producer_thread(void* arg) {
    ProdArg* pa = (ProdArg*)arg;
    /* Blocking channel: submit will wait if full */
    for (int i = 0; i < pa->count; i++) {
        MLMsg msg = ml_msg_fence((MLChannel)pa->channel, (u32)(pa->pid * 10000 + i));
        while (!ml_sys_submit(pa->sys, pa->pid, &msg)) {
            /* non-blocking: spin */
        }
    }
    return NULL;
}

static int test_multi_producer(void) {
    MLSystem* sys = malloc(sizeof(MLSystem));
    ml_sys_init(sys);
    /* Non-blocking so producers spin rather than deadlock */
    for (int ch = 0; ch < ML_CHAN_COUNT; ch++)
        ml_sys_set_blocking(sys, (MLChannel)ch, false);

    const int NP = 2, MSGS = 500;
    pthread_t thr[2];
    ProdArg   arg[2];

    for (int i = 0; i < NP; i++) {
        arg[i].sys     = sys;
        arg[i].pid     = ml_sys_register_producer(sys);
        arg[i].count   = MSGS;
        arg[i].channel = ML_CHAN_PREFILL;
        pthread_create(&thr[i], NULL, producer_thread, &arg[i]);
    }

    int total = 0;
    while (total < NP * MSGS)
        total += (int)ml_sys_step(sys);

    for (int i = 0; i < NP; i++) pthread_join(thr[i], NULL);
    TEST_ASSERT(total == NP * MSGS, "all messages processed");

    ml_sys_free(sys);
    free(sys);
    TEST_PASS();
}

/*============================================================================
 * Benchmark: message routing overhead
 *============================================================================*/

static void bench_message_throughput(void) {
    printf("\nBenchmark: Message routing overhead\n");

    MLSystem* sys = malloc(sizeof(MLSystem));
    ml_sys_init(sys);
    ml_sys_set_blocking(sys, ML_CHAN_PREFILL, false);
    ml_sys_set_budget(sys, ML_CHAN_PREFILL, 100000);

    int pid = ml_sys_register_producer(sys);

    /* Drip-feed so the queue never fills */
    const int BATCH = ML_QUEUE_SIZE / 2;
    const int TOTAL = 10000;
    int sent = 0, processed = 0;

    double t0 = get_time_us();
    while (sent < TOTAL || processed < TOTAL) {
        /* Enqueue a batch */
        for (int i = 0; i < BATCH && sent < TOTAL; i++, sent++) {
            MLMsg msg = ml_msg_fence(ML_CHAN_PREFILL, sent);
            ml_sys_submit(sys, pid, &msg);
        }
        /* Drain */
        processed += (int)ml_sys_step(sys);
    }
    double us = get_time_us() - t0;

    printf("  %d messages: %.2f us total  (%.3f us/msg  %.1f M/s)\n",
           TOTAL, us, us / TOTAL, TOTAL / us);

    ml_sys_free(sys);
    free(sys);
}

/*============================================================================
 * Benchmark: GEMM through message layer
 *============================================================================*/

static void bench_gemm_messages(void) {
    printf("\nBenchmark: GEMM via message layer (M=4 N=4096 K=4096)\n");

    MLSystem* sys = malloc(sizeof(MLSystem));
    ml_sys_init(sys);
    ml_sys_set_blocking(sys, ML_CHAN_PREFILL, false);
    ml_sys_set_budget(sys, ML_CHAN_PREFILL, 256);

    int pid = ml_sys_register_producer(sys);

    u32 M = 4, N = 4096, K = 4096;
    int8_t*  A = aligned_alloc(64, M * K);
    uint8_t* W = aligned_alloc(64, N * K / 4);
    int32_t* C = aligned_alloc(64, M * N * sizeof(int32_t));
    for (size_t i = 0; i < M * K; i++) A[i] = (rand() % 3) - 1;
    for (size_t i = 0; i < N * K / 4; i++) W[i] = rand() & 0xFF;

    /* warmup */
    ml_sys_submit_gemm(sys, pid, ML_LAYER_FFN_GATE, 0,
                       ML_CHAN_PREFILL, HS_ROUTE_TERNARY_2BIT,
                       A, W, C, M, N, K);
    ml_sys_step(sys);

    const int ITERS = 10;
    double t0 = get_time_us();
    for (int i = 0; i < ITERS; i++) {
        ml_sys_submit_gemm(sys, pid, ML_LAYER_FFN_GATE, 0,
                           ML_CHAN_PREFILL, HS_ROUTE_TERNARY_2BIT,
                           A, W, C, M, N, K);
        ml_sys_step(sys);
    }
    double ms = (get_time_us() - t0) / (ITERS * 1000.0);
    double gops = (double)M * N * K / (ms * 1e6);

    printf("  %.2f ms/iter  %.1f GOPS\n", ms, gops);
    printf("  channel: %s  format: ternary-2bit\n",
           ml_channel_name(ML_CHAN_PREFILL));

    const MLLayerStats* st = ml_sys_layer_stats(sys, ML_LAYER_FFN_GATE);
    printf("  messages dispatched: %llu  avg: %.2f ms\n",
           (unsigned long long)st->messages_processed,
           st->total_ns / ((double)st->messages_processed * 1e6));

    free(A); free(W); free(C);
    ml_sys_free(sys);
    free(sys);
}

/*============================================================================
 * main
 *============================================================================*/

int main(void) {
    printf("ML Message-Passing Layer Tests\n");
    printf("==============================\n\n");

    int fail = 0;

    printf("SPSC Queue:\n");
    fail += test_spsc_basic();
    fail += test_spsc_fifo_order();

    printf("\nSystem:\n");
    fail += test_system_init();
    fail += test_submit_and_step();
    fail += test_capture_replay();

    printf("\nConcurrency:\n");
    fail += test_multi_producer();

    bench_message_throughput();
    bench_gemm_messages();

    printf("\n==============================\n");
    printf("%s\n", fail == 0 ? "All tests passed." : "FAILURES DETECTED.");
    return fail;
}
