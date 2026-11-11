/* Deterministic regressions for release-blocking audit findings. */
#include "hs_gpu.h"
#include "hs_msg.h"
#include "hs_storage.h"
#include "hs_ipc.h"
#include "hs_async.h"
#include "hs_input_stream.h"
#include "hs_text.h"
#include <pthread.h>
#include <sched.h>
#include <errno.h>
#include <dirent.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static unsigned checks, failures;
#define CHECK(expr) do { checks++; if (!(expr)) { failures++; \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr); } } while (0)

static void test_codecs(void) {
    u8 data[96] = {0}, bytes[4] = {0};
    u32 a = 0, b = 0;
    hs_pack_set_record_mask(data, 6);
    CHECK(hs_unpack_set_record_mask(data, 4, &a) && a == 6);
    for (u32 n = 0; n < 4; n++) CHECK(!hs_unpack_set_record_mask(data, n, &a));
    hs_pack_stencil(data, 1, 2, 3, 4);
    CHECK(hs_unpack_u8x4(data, 4, bytes) && bytes[3] == 4);
    for (u32 n = 0; n < 4; n++) CHECK(!hs_unpack_u8x4(data, n, bytes));
    hs_pack_stencil_func(data, 4, 3, 2, 1);
    CHECK(hs_unpack_u8x4(data, 4, bytes) && bytes[0] == 4 && bytes[3] == 1);
    hs_pack_u32x2(data, 7, 9);
    CHECK(hs_unpack_u32x2(data, 8, &a, &b) && a == 7 && b == 9);
    hs_pack_async_done(data, 123, 3, 4, 1);
    u8 type = 0, slot = 0, success = 0;
    CHECK(hs_unpack_async_done(data, 8, &a, &type, &slot, &success) &&
          a == 123 && type == 3 && slot == 4 && success == 1);

    /* Every decoder is checked at each shorter length as well as its packed size. */
#define LENGTHS(size, expression) do { \
    for (u32 length = 0; length <= (size); length++) CHECK((expression) == (length == (size))); \
} while (0)
    hs_pack_u32x2(data, 7, 9);
    LENGTHS(8, hs_unpack_u32x2(data, length, &a, &b));
    CHECK(a == 7 && b == 9);
    hs_pack_async_done(data, 123, 3, 4, 1);
    LENGTHS(8, hs_unpack_async_done(data, length, &a, &type, &slot, &success));
    CHECK(a == 123 && type == 3 && slot == 4 && success == 1);
    f32 floats[4]; u16 shorts[4]; u8 x, y; u32 count;
    hs_pack_set_param(data, 3, v4_one());
    LENGTHS(20, hs_unpack_set_param(data, length, &a, floats));
    CHECK(a == 3 && floats[3] == 1);
    hs_pack_u8x2(data, 3, 4);
    LENGTHS(2, hs_unpack_u8x2(data, length, &x, &y));
    CHECK(x == 3 && y == 4);
    hs_pack_u16x4(data, 1, 2, 3, 4);
    LENGTHS(8, hs_unpack_u16x4(data, length, shorts));
    CHECK(shorts[3] == 4);
    hs_pack_depth_compare(data, 3, true);
    CHECK(hs_unpack_u8x2(data, 2, &x, &y) && x == 3 && y == 1);
    hs_pack_draw_instance(data, 1, 2, 123456);
    LENGTHS(8, hs_unpack_draw_instance(data, length, &x, &y, &count));
    CHECK(x == 1 && y == 2 && count == 123456);
    f32 rgba[4] = {1, 2, 3, 4}; memcpy(data, rgba, sizeof(rgba));
    LENGTHS(16, hs_unpack_clear_color(data, length, floats));
    CHECK(floats[3] == 4);
    hs_pack_clear_ds(data, 0.5f, 7);
    LENGTHS(8, hs_unpack_clear_ds(data, length, &floats[0], &x));
    CHECK(floats[0] == 0.5f && x == 7);
    hs_pack_error_ex(data, 1, 2, 3, 4, 5, 6, 7, 8, "error");
    u8 op, to, from, stage;
    LENGTHS(52, hs_unpack_error_ex(data, length, &a, &op, &to, &from, &stage, &b));
    CHECK(a == 1 && op == 2 && to == 3 && from == 4 && stage == 5 && b == 6);
    hs_pack_result_fence(data, 123, 2);
    LENGTHS(8, hs_unpack_result_fence(data, length, &a, &x));
    CHECK(a == 123 && x == 2);
    hs_pack_set_chan_budget(data, 2, 456);
    LENGTHS(8, hs_unpack_set_chan_budget(data, length, &x, &a));
    CHECK(x == 2 && a == 456);
    hs_pack_set_block_policy(data, 2, 1);
    LENGTHS(2, hs_unpack_set_block_policy(data, length, &x, &y));
    CHECK(x == 2 && y == 1);
    u32 budgets[3], dropped[4], flags, producers;
    hs_pack_result_system_stats(data, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
    LENGTHS(64, hs_unpack_result_system_stats(data, length, &a, &b, &count, budgets, dropped, &producers, &flags));
    CHECK(a == 1 && b == 2 && count == 3 && budgets[2] == 6 && dropped[3] == 10 && producers == 11 && flags == 12);
    u32 channels[3] = {1, 2, 3}, out[6][3];
    hs_pack_result_fabric(data, channels, channels, channels, channels, channels, channels, 7, 8);
    LENGTHS(96, hs_unpack_result_fabric(data, length, out[0], out[1], out[2], out[3], out[4], out[5], &a, &b));
    CHECK(out[5][2] == 3 && a == 7 && b == 8);
    u8 truncated_fabric[64] = {0};
    CHECK(!hs_unpack_result_fabric(truncated_fabric, sizeof(truncated_fabric),
                                  out[0], out[1], out[2], out[3], out[4], out[5], &a, &b));
#undef LENGTHS
}

static void test_bounds(void) {
    HSBuffer buf = {0};
    CHECK(hs_buffer_init(&buf, 0, 64));
    const u32 invalid[] = {16, 0x3fffffffu, 0x40000000u, UINT32_MAX};
    for (unsigned i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++) {
        memset(buf.data, 0, buf.capacity);
        hs_buffer_set_f32(&buf, invalid[i], 9);
        hs_buffer_set_i32(&buf, invalid[i], 9);
        hs_buffer_set_vec4(&buf, invalid[i], v4_one());
        hs_buffer_set_mat4(&buf, invalid[i], m4_identity());
        hs_buffer_set_mat3x4(&buf, invalid[i], m4_identity());
        for (u32 j = 0; j < buf.capacity; j++) CHECK(buf.data[j] == 0);
        CHECK(hs_buffer_get_f32(&buf, invalid[i]) == 0);
        CHECK(hs_buffer_get_i32(&buf, invalid[i]) == 0);
        CHECK(vgetq_lane_f32(hs_buffer_get_vec4(&buf, invalid[i]), 0) == 0);
    }
    hs_buffer_set_f32(&buf, 15, 42);
    CHECK(hs_buffer_get_f32(&buf, 15) == 42);
    volatile u32 just_outside = 13; /* exercise the runtime guard, not constant folding */
    hs_buffer_set_vec4(&buf, just_outside, v4_one());
    CHECK(hs_buffer_get_f32(&buf, 15) == 42);
    hs_buffer_free(&buf);
    hs_buffer_set_i32(&buf, 0, 9);
    CHECK(hs_buffer_get_i32(&buf, 0) == 0);
    CHECK(hs_buffer_get_f32(NULL, 0) == 0);
    HSStorage store;
    hs_storage_init(&store);
    hs_storage_set_f32(&store, "test", 252, 42);
    CHECK(hs_storage_get_f32(&store, "test", 252) == 42);
    hs_storage_set_f32(&store, "test", UINT32_MAX, 9);
    CHECK(hs_storage_get_f32(&store, "test", UINT32_MAX) == 0);
    CHECK(hs_storage_get_f32(&store, "test", 252) == 42);
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static HSGpu gpu;
static unsigned applied_burst;
static int count_shader(Node* node) {
    int count = shader_node_process(node);
    applied_burst += (unsigned)count;
    return count;
}

static void test_actor_handoff(void) {
    hs_clear(&gpu.system);
    gpu.system.recording = false;
    gpu.shader_node.process_fn = count_shader;
    applied_burst = 0;
    for (u32 i = 0; i < 1024; i++) {
        Message message = {.to = NODE_SHADER, .from = NODE_CPU, .op = OP_SET_SHADER,
                           .payload_idx = (u16)(i % 256), .channel = CHAN_RENDER};
        CHECK(hs_try_send(&gpu.system, &message, NULL, 0));
    }
    hs_step(&gpu.system);
    hs_step(&gpu.system);
    CHECK(applied_burst == 1024);
    CHECK(gpu.system.dropped_queue_full == 0);
    gpu.shader_node.process_fn = shader_node_process;
}

static void test_capture(void) {
    hs_gpu_init(&gpu);
    CHECK(gpu.system.log_payloads == NULL);
    for (u32 i = 0; i < HS_MAX_PAYLOADS + 17; i++) {
        hs_gpu_clear(&gpu, v4_make((float)i, 0, 0, 1));
        hs_gpu_process(&gpu);
    }
    static Message messages[HS_MAX_PAYLOADS + 17];
    static Payload payloads[HS_MAX_PAYLOADS + 17];
    HSCapture capture;
    hs_capture_init(&capture, messages, payloads, HS_MAX_PAYLOADS + 17);
    CHECK(hs_capture_from_log(&gpu.system, gpu.log_buffer, gpu.system.log_head, &capture));
    CHECK(capture.count == HS_MAX_PAYLOADS + 17);
    CHECK(gpu.system.owns_log_payloads && gpu.system.log_payloads != NULL);
    for (u32 i = 0; i < capture.count; i++) {
        float color;
        memcpy(&color, payloads[i].data, sizeof(color));
        CHECK(color == (float)i);
    }
    CHECK(hs_capture_replay(&gpu.system, &capture));
    CHECK(payloads[0].data[0] == 0);
    capture.count = 1;
    CHECK(!hs_capture_write_file(&capture, "/dev/full"));
    capture.count = capture.capacity + 1;
    CHECK(!hs_capture_write_file(&capture, "tests/must-not-be-created.cap"));
    CHECK(access("tests/must-not-be-created.cap", F_OK) != 0);
}

static void test_ipc_shutdown(void) {
    HSIpcServer invalid;
    CHECK(!hs_ipc_start(&invalid, NULL, NULL));
    hs_ipc_stop(&invalid);
    CHECK(!invalid.thread_started && invalid.listen_fd == -1);
    CHECK(!hs_ipc_start_tcp(&invalid, NULL, NULL, 0));
    hs_ipc_stop(&invalid);
    char path[108];
    snprintf(path, sizeof(path), "tests/p0-ipc-%d.sock", (int)getpid());
    for (int i = 0; i < 20; i++) {
        HSIpcServer server = {0};
        CHECK(hs_ipc_start(&server, &gpu.system, path));
        int fd = socket(AF_UNIX, SOCK_STREAM, 0);
        struct sockaddr_un address = {.sun_family = AF_UNIX};
        strcpy(address.sun_path, path);
        CHECK(fd >= 0 && connect(fd, (struct sockaddr*)&address, sizeof(address)) == 0);
        CHECK(send(fd, "x", 1, MSG_NOSIGNAL) == 1);
        usleep(1000);
        double start = now_ms();
        hs_ipc_stop(&server);
        double elapsed = now_ms() - start;
        CHECK(elapsed < 1000);
        CHECK(!server.thread_started && server.listen_fd == -1);
        hs_ipc_stop(&server);
        close(fd);
    }
    FILE* f = fopen(path, "wb");
    CHECK(f != NULL);
    if (f) fclose(f);
    HSIpcServer failed = {0};
    CHECK(!hs_ipc_start(&failed, &gpu.system, path));
    hs_ipc_stop(&failed);
    CHECK(access(path, F_OK) == 0); /* failed bind must preserve user files */
    unlink(path);
    CHECK(!hs_ipc_start_tcp(&failed, &gpu.system, "not-an-address", 12345));
    hs_ipc_stop(&failed);
    /* No address means loopback, never the wildcard. */
    for (int variant = 0; variant < 2; variant++) {
        HSIpcServer tcp = {0};
        CHECK(hs_ipc_start_tcp(&tcp, &gpu.system, variant ? "" : NULL, 0));
        struct sockaddr_in bound;
        socklen_t length = sizeof(bound);
        CHECK(getsockname(tcp.listen_fd, (struct sockaddr*)&bound, &length) == 0 &&
              bound.sin_addr.s_addr == htonl(INADDR_LOOPBACK));
        hs_ipc_stop(&tcp);
    }
}

typedef struct { HSAsync* async; const char* directory; unsigned first, accepted; } AsyncSubmit;

static void* save_batch(void* pointer) {
    AsyncSubmit* arg = pointer;
    for (unsigned i = arg->first; i < arg->first + 8; i++) {
        char path[128];
        snprintf(path, sizeof(path), "%s/%u", arg->directory, i);
        if (hs_async_save_file(arg->async, path, &i, sizeof(i))) arg->accepted++;
    }
    return NULL;
}

static void test_async_ownership(void) {
    HSAsync async;
    hs_async_init(&async, NULL);
    CHECK(async.initialized);
    char directory[] = "tests/async-p0-XXXXXX";
    CHECK(mkdtemp(directory) != NULL);
    pthread_t threads[4];
    AsyncSubmit args[4];
    for (unsigned i = 0; i < 4; i++) {
        args[i] = (AsyncSubmit){.async = &async, .directory = directory, .first = i * 8};
        CHECK(pthread_create(&threads[i], NULL, save_batch, &args[i]) == 0);
    }
    for (unsigned i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
        CHECK(args[i].accepted == 8);
    }
    double deadline = now_ms() + 2000;
    unsigned done = 0;
    do {
        done = 0;
        pthread_mutex_lock(&async.mutex);
        for (unsigned i = 0; i < HS_ASYNC_MAX_PENDING; i++) done += async.tasks[i].done != 0;
        pthread_mutex_unlock(&async.mutex);
        if (done < HS_ASYNC_MAX_PENDING) usleep(1000);
    } while (done < HS_ASYNC_MAX_PENDING && now_ms() < deadline);
    CHECK(done == HS_ASYNC_MAX_PENDING);
    char extra[128];
    snprintf(extra, sizeof(extra), "%s/extra", directory);
    CHECK(!hs_async_save_file(&async, extra, "x", 1));
    for (unsigned i = 0; i < 32; i++) {
        char path[128];
        snprintf(path, sizeof(path), "%s/%u", directory, i);
        FILE* f = fopen(path, "rb");
        CHECK(f != NULL);
        unsigned value = UINT32_MAX;
        if (f) { CHECK(fread(&value, sizeof(value), 1, f) == 1); fclose(f); }
        CHECK(value == i);
        unlink(path);
    }
    CHECK(hs_async_process(&async));
    CHECK(hs_async_save_file(&async, extra, "done", 4));
    hs_async_shutdown(&async); /* drains, rather than abandoning queued saves */
    FILE* f = fopen(extra, "rb");
    char data[4];
    CHECK(f != NULL);
    if (f) { CHECK(fread(data, 1, 4, f) == 4 && memcmp(data, "done", 4) == 0); fclose(f); }
    unlink(extra); rmdir(directory);
    hs_async_shutdown(&async);
    hs_async_init(&async, NULL);
    errno = 0;
    CHECK(!hs_async_load_texture(&async, 0, "unsupported") && errno == ENOTSUP);
    CHECK(hs_async_save_file(&async, "/dev/full", "cannot flush", 12));
    deadline = now_ms() + 2000;
    bool failed_write = false;
    do {
        pthread_mutex_lock(&async.mutex);
        failed_write = async.tasks[0].done && !async.tasks[0].success;
        pthread_mutex_unlock(&async.mutex);
        if (!failed_write) usleep(1000);
    } while (!failed_write && now_ms() < deadline);
    CHECK(failed_write);
    CHECK(hs_async_process(&async));
    hs_async_shutdown(&async);
    hs_async_init(&async, NULL);
    hs_async_shutdown(&async); /* initialized but never started */
}

typedef struct { HSInputStream* stream; unsigned sent; } InputSubmit;

static void* input_batch(void* pointer) {
    InputSubmit* arg = pointer;
    HSInputEvent event = {.type = EV_REL, .code = REL_X, .value = 1};
    while (arg->sent < 1000) {
        if (hs_input_stream_submit(arg->stream, &event)) arg->sent++;
        else sched_yield();
    }
    return NULL;
}

static bool wait_input(HSInputStream* stream, uint64_t count) {
    double deadline = now_ms() + 2000;
    while (atomic_load(&stream->event_count) < count && now_ms() < deadline) usleep(1000);
    return atomic_load(&stream->event_count) == count;
}

static void test_input_ownership(void) {
    HSInputSpscQueue queue;
    hs_input_spsc_init(&queue);
    HSInputEvent event = {0};
    for (int i = 0; i < 200; i++) CHECK(hs_input_spsc_push(&queue, &event, NULL));
    for (int i = 0; i < 200; i++) CHECK(hs_input_spsc_pop(&queue, &event));
    for (int i = 0; i < 100; i++) CHECK(hs_input_spsc_push(&queue, &event, NULL));
    CHECK(hs_input_spsc_available(&queue) == 100);
    HSInputStream stream;
    CHECK(hs_input_stream_init(&stream));
    CHECK(hs_input_stream_start(&stream));
    usleep(10000); /* let the empty-queue consumer enter its wait */
    event = (HSInputEvent){.type = EV_KEY, .code = KEY_RIGHT, .value = 1};
    CHECK(hs_input_stream_submit(&stream, &event));
    CHECK(wait_input(&stream, 1));
    for (int i = 0; i < 10; i++) {
        const HSInput* snapshot = hs_input_stream_get_state(&stream);
        CHECK(snapshot && snapshot->key_right && snapshot->dir_x == 1);
        hs_input_stream_swap_buffers(&stream);
    }
    hs_input_stream_pause(&stream);
    event.value = 0;
    CHECK(hs_input_stream_submit(&stream, &event));
    usleep(10000);
    CHECK(atomic_load(&stream.event_count) == 1);
    hs_input_stream_resume(&stream);
    CHECK(wait_input(&stream, 2));
    CHECK(!hs_input_stream_get_state(&stream)->key_right);
    pthread_t threads[4];
    InputSubmit args[4];
    double start = now_ms();
    for (unsigned i = 0; i < 4; i++) {
        args[i] = (InputSubmit){.stream = &stream};
        CHECK(pthread_create(&threads[i], NULL, input_batch, &args[i]) == 0);
    }
    for (unsigned i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
        CHECK(args[i].sent == 1000);
    }
    CHECK(wait_input(&stream, 4002));
    printf("Input MPSC: 4000 events in %.3f ms, %llu admission retries\n", now_ms() - start,
           (unsigned long long)atomic_load(&stream.dropped_events));
    hs_input_stream_stop(&stream);
    hs_input_stream_stop(&stream);
    hs_input_stream_shutdown(&stream);
    hs_input_stream_shutdown(&stream);
}

static unsigned open_descriptors(void) {
    DIR* directory = opendir("/proc/self/fd");
    if (!directory) return UINT32_MAX;
    unsigned count = 0;
    struct dirent* entry;
    while ((entry = readdir(directory))) if (entry->d_name[0] != '.') count++;
    closedir(directory);
    return count;
}

/* ---- 2026-11-13 audit regressions ---- */

/* Place an empty Vyukov ring at an arbitrary position: slot for pos p holds p. */
static void seed_submit_ring(HSSubmitQueue* q, u32 pos) {
    for (u32 k = 0; k < HS_SUBMIT_SIZE; k++) {
        u32 p = pos + k;
        atomic_store(&q->slots[p & (HS_SUBMIT_SIZE - 1)].seq, p);
    }
    atomic_store(&q->enqueue_pos, pos);
    atomic_store(&q->dequeue_pos, pos);
}

static void fresh_gpu(void) {
    hs_gpu_shutdown(&gpu);
    hs_gpu_init(&gpu);
    gpu.system.recording = false;
}

static void test_submit_ring_wrap(void) {
    fresh_gpu();
    /* Exhaust producer ids so this thread's sends take the MPMC submit ring. */
    atomic_store(&gpu.system.producer_count, HS_MAX_PRODUCERS);
    atomic_fetch_add(&gpu.system.producer_epoch, 1);
    HSSubmitQueue* q = &gpu.system.submit[CHAN_RENDER];
    ShaderState* shader = &gpu.shader_state;
    Message m = {.to = NODE_SHADER, .from = NODE_CPU, .op = OP_SET_SHADER,
                 .payload_idx = 7, .channel = CHAN_RENDER};

    /* Empty ring parked on the last u32 position: dequeue must see "empty"
     * (the intptr_t difference used to be +2^32 here and spun forever). */
    seed_submit_ring(q, 0xFFFFFFFFu);
    hs_step(&gpu.system);
    CHECK(atomic_load(&q->dequeue_pos) == 0xFFFFFFFFu);
    CHECK(hs_try_send(&gpu.system, &m, NULL, 0));
    hs_step(&gpu.system);
    CHECK(atomic_load(&q->dequeue_pos) == 0u && shader->current_shader == 7);
    hs_step(&gpu.system);
    CHECK(atomic_load(&q->dequeue_pos) == 0u);

    /* Full ring straddling the wrap: admission must refuse, never wait. */
    seed_submit_ring(q, 0xFFFFFFFFu - 10);
    u32 accepted = 0;
    for (u32 i = 0; i < HS_SUBMIT_SIZE; i++) {
        m.payload_idx = (u16)(i & 0xFF);
        accepted += hs_try_send(&gpu.system, &m, NULL, 0) ? 1u : 0u;
    }
    CHECK(accepted == HS_SUBMIT_SIZE);
    unsigned full_before = atomic_load(&gpu.system.submit_full[CHAN_RENDER]);
    CHECK(!hs_try_send(&gpu.system, &m, NULL, 0));
    CHECK(atomic_load(&gpu.system.submit_full[CHAN_RENDER]) == full_before + 1);
    for (int i = 0; i < 4; i++) hs_step(&gpu.system);
    CHECK(atomic_load(&q->dequeue_pos) == atomic_load(&q->enqueue_pos));
    CHECK(atomic_load(&q->dequeue_pos) == (u32)(0xFFFFFFFFu - 10 + HS_SUBMIT_SIZE));
    fresh_gpu();
}

static bool control_result(u32 cid, u32* old_value, u32* new_value) {
    SystemState* st = &gpu.system_state_data;
    if (st->last_result_cid != cid) return false;
    return hs_unpack_u32x2(gpu.system.payloads[st->last_result_payload_idx].data,
                           st->last_result_len, old_value, new_value);
}

static void test_budget_floor(void) {
    fresh_gpu();
    CHECK(!hs_set_channel_budget(&gpu.system, CHAN_RT, 0));
    CHECK(!hs_set_channel_budget(&gpu.system, CHAN_RENDER, 0));
    CHECK(gpu.system.chan_budget[CHAN_RT] == 4096 && gpu.system.chan_budget[CHAN_RENDER] == 16384);
    CHECK(hs_set_channel_budget(&gpu.system, CHAN_TELEM, HS_CHAN_BUDGET_MIN));
    CHECK(gpu.system.chan_budget[CHAN_TELEM] == HS_CHAN_BUDGET_MIN);

    /* The control op (what neogpu_tool set-budget 1 0 sends) is rejected too. */
    u8 p[8];
    hs_pack_set_chan_budget(p, (u8)CHAN_RT, 0);
    Message m = {.to = NODE_SYSTEM, .from = NODE_CPU, .op = OP_SET_CHAN_BUDGET, .cid = 77, .channel = CHAN_RT};
    CHECK(hs_send_with_payload(&gpu.system, &m, p, sizeof(p)));
    hs_step(&gpu.system);
    hs_step(&gpu.system);
    u32 old_value = 0, new_value = 0;
    CHECK(control_result(77, &old_value, &new_value) && old_value == 0xFFFFFFFFu && new_value == 0xFFFFFFFFu);
    CHECK(gpu.system.chan_budget[CHAN_RT] == 4096);

    /* RT still drains afterwards, and a valid change is applied and reported. */
    hs_pack_set_chan_budget(p, (u8)CHAN_RT, 64);
    m.cid = 78;
    CHECK(hs_send_with_payload(&gpu.system, &m, p, sizeof(p)));
    hs_step(&gpu.system);
    hs_step(&gpu.system);
    CHECK(control_result(78, &old_value, &new_value) && old_value == 4096 && new_value == 64);
    CHECK(gpu.system.chan_budget[CHAN_RT] == 64);
}

/* Routing uses the inline hs_message_conforms(); it must give exactly the
 * verdict of hs_validate_message() for every opcode, destination, length,
 * payload index and string terminator. */
static void test_conforms_matches_validate(void) {
    fresh_gpu();
    HSSystem* sys = &gpu.system;
    for (u32 op = 0; op < OP_COUNT; op++) {
        const HSOpSpec* spec = &hs_op_spec_table[op];
        if (spec->kind == HS_PAYLOAD_NONE) CHECK(spec->min_len == 0 && spec->max_len == 0);
        if (spec->kind == HS_PAYLOAD_FIXED) CHECK(spec->min_len == spec->max_len);
        CHECK(spec->min_len <= spec->max_len && spec->max_len <= HS_PAYLOAD_SIZE);
    }
    u32 mismatches = 0, accepted = 0, rejected = 0;
    const u32 lengths[] = {0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 19, 20, 21, 51, 52, 53, 63, 64, 65,
                           HS_PAYLOAD_SIZE - 1, HS_PAYLOAD_SIZE, HS_PAYLOAD_SIZE + 1, 65535};
    const u32 indices[] = {0, 1, sys->payload_capacity - 1, sys->payload_capacity, 65535};
    for (u32 op = 0; op <= OP_COUNT + 1 && op < 256; op++)
        for (u32 to = 0; to < 256; to++)
            for (u32 l = 0; l < sizeof(lengths) / sizeof(lengths[0]); l++)
                for (u32 i = 0; i < sizeof(indices) / sizeof(indices[0]); i++)
                    for (u32 term = 0; term < 2; term++) {
                        Message m = {.to = (u8)to, .op = (u8)op, .payload_idx = (u16)indices[i],
                                     .payload_len = lengths[l]};
                        if (m.payload_len && m.payload_idx < sys->payload_capacity && m.payload_len <= HS_PAYLOAD_SIZE)
                            sys->payloads[m.payload_idx].data[m.payload_len - 1] = term ? 0 : 'x';
                        bool full = hs_validate_message(sys, &m, NULL);
                        mismatches += full != hs_message_conforms(sys, &m);
                        accepted += full; rejected += !full;
                    }
    CHECK(mismatches == 0);
    CHECK(accepted > 0 && rejected > 0);
    CHECK(!hs_message_conforms(NULL, &(Message){0}) && !hs_message_conforms(sys, NULL));
}

/* The schema gate sits in routing: nodes no longer check, so every message
 * a node pops must conform. Each GPU node's process function is wrapped to
 * check its whole inbox before the node runs. */
static NodeProcessFn gate_inner[256];
static u32 gate_seen, gate_nonconforming;

static int gate_check(Node* node) {
    const MessageQueue* q = &node->inbox;
    for (u32 k = 0; k < q->count; k++) {
        gate_seen++;
        gate_nonconforming += !hs_message_conforms(node->sys, &q->msgs[(q->head + k) % HS_QUEUE_SIZE]);
    }
    return gate_inner[node->id](node);
}

static void gate_install(void) {
    Node* nodes[] = {&gpu.shader_node, &gpu.buffer_node, &gpu.texture_node,
                     &gpu.output_node, &gpu.sound_node, &gpu.system_node};
    for (u32 i = 0; i < sizeof(nodes) / sizeof(nodes[0]); i++) {
        gate_inner[nodes[i]->id] = nodes[i]->process_fn;
        nodes[i]->process_fn = gate_check;
    }
    gate_seen = gate_nonconforming = 0;
}

static u32 gate_rng = 0x2545F491u;
static u32 gate_rand(void) {
    gate_rng = gate_rng * 1664525u + 1013904223u;
    return gate_rng >> 8;
}

static void test_gate_at_routing(void) {
    /* 1. Random messages through every untrusted entry point. Ops that
     * reconfigure the system, end frames or print are left out. */
    fresh_gpu();
    gate_install();
    const u8 to_nodes[] = {NODE_SHADER, NODE_BUFFER, NODE_TEXTURE, NODE_OUTPUT, NODE_SOUND, NODE_SYSTEM};
    u32 sent = 0;
    for (u32 i = 0; i < 20000; i++) {
        u8 op;
        do {
            op = (u8)(gate_rand() % (OP_COUNT + 2));
        } while (op == OP_STOP || op == OP_SET_CHAN_BUDGET || op == OP_SET_BLOCK_POLICY ||
                 op == OP_SET_RECORD_MASK || op == OP_FRAME_BEGIN || op == OP_FRAME_END ||
                 op == OP_PRESENT || op == OP_FENCE || op == OP_ERROR || op == OP_TRACE ||
                 op == OP_ERROR_EX);
        u32 r = gate_rand();
        Message m = {.to = (r & 15) ? to_nodes[r % 6] : (u8)(r >> 4), .from = NODE_CPU, .op = op,
                     .flags = (u8)((gate_rand() & 3) ? (gate_rand() & 0x7F) : HS_MSGF_ACK),
                     .cid = i + 1000};
        u8 bytes[HS_PAYLOAD_SIZE];
        for (u32 b = 0; b < sizeof(bytes); b++) bytes[b] = (u8)gate_rand();
        u32 len = gate_rand() % (HS_PAYLOAD_SIZE + 1);
        if (len && (gate_rand() & 1)) bytes[len - 1] = 0;
        bool ok;
        switch (i % 3) {
            case 0: /* references the payload ring directly */
                m.payload_idx = (u16)(gate_rand() % (gpu.system.payload_capacity + 2));
                m.payload_len = (gate_rand() & 1) ? len : gate_rand() % 4;
                if (m.payload_len && m.payload_idx < gpu.system.payload_capacity && m.payload_len <= HS_PAYLOAD_SIZE)
                    memcpy(gpu.system.payloads[m.payload_idx].data, bytes, m.payload_len);
                ok = hs_send(&gpu.system, &m);
                break;
            case 1: ok = hs_send_with_payload(&gpu.system, &m, bytes, len); break;
            default: ok = hs_try_send(&gpu.system, &m, bytes, len); break;
        }
        sent += ok;
        if ((i & 31) == 31) hs_step(&gpu.system);
    }
    hs_step(&gpu.system);
    hs_step(&gpu.system);
    CHECK(sent > 19000);
    CHECK(gate_nonconforming == 0);
    CHECK(gate_seen > 1000 && gpu.system.rejected_invalid > 1000);
    printf("Routing gate fuzz: %u sent, %u reached nodes, %u refused\n",
           sent, gate_seen, gpu.system.rejected_invalid);

    /* 2. Every hs_gpu_* helper sends trusted, conforming messages (HS_VERIFY_TRUSTED
     * builds abort otherwise). */
    fresh_gpu();
    gate_install();
    char long_text[300];
    memset(long_text, 'a', sizeof(long_text) - 1);
    long_text[sizeof(long_text) - 1] = 0;
    for (int pass = 0; pass < 2; pass++) {
        gpu.system.validate_on_send = pass == 1; /* gate trusted sends too */
        hs_gpu_set_shader(&gpu, 1);
        hs_gpu_set_param(&gpu, 2, v4_make(1, 2, 3, 4));
        hs_gpu_set_global(&gpu, 1, m4_identity());
        hs_gpu_set_camera(&gpu, m4_identity());
        hs_gpu_cull(&gpu, 1);
        hs_gpu_blend(&gpu, 1, 2);
        hs_gpu_alpha(&gpu, true);
        hs_gpu_depth(&gpu, true);
        hs_gpu_color_mask(&gpu, 0xF);
        hs_gpu_clip(&gpu, 1, 2, 3, 4);
        hs_gpu_stencil(&gpu, 1, 3, 4, 1);
        hs_gpu_stencil_func(&gpu, 5, 6, 0x7F, 0x3F);
        hs_gpu_depth_compare(&gpu, 1, true);
        hs_gpu_load_buffer(&gpu, 0);
        hs_gpu_draw(&gpu, 0);
        hs_gpu_draw_instance(&gpu, 0, 1, 3);
        hs_gpu_draw_text(&gpu, "gate");
        hs_gpu_draw_text(&gpu, "");
        hs_gpu_draw_text(&gpu, long_text);
        hs_gpu_load_texture(&gpu, 0);
        hs_gpu_set_target(&gpu, 0, 0);
        hs_gpu_show_texture(&gpu, 0);
        hs_gpu_texture_filter(&gpu, 0, true);
        hs_gpu_texture_wrap(&gpu, 0, true);
        hs_gpu_clear(&gpu, v4_make(0, 0, 0, 1));
        hs_gpu_clear_ds(&gpu, 1.0f, 0);
        hs_gpu_set_channel(&gpu, 0, 1);
        hs_gpu_trace(&gpu, "routing gate: helper trace");
        hs_gpu_frame_begin(&gpu);
        hs_gpu_frame_end(&gpu);
        hs_gpu_present(&gpu);
        CHECK(hs_gpu_fence(&gpu, CHAN_RENDER, 77));
        hs_gpu_stop(&gpu);
        hs_step(&gpu.system);
        hs_step(&gpu.system);
    }
    gpu.system.validate_on_send = false;
    CHECK(gate_nonconforming == 0 && gate_seen >= 2 * 28);
    CHECK(gpu.system.rejected_invalid == 0 && gpu.system.dropped_error_ex == 0);
    SystemState* st = &gpu.system_state_data;
    CHECK(st->error_count == 0);
    /* Both stencil helpers sent 8 bytes for a 4-byte schema, so the node
     * gate refused them and they never applied. */
    CHECK(gpu.shader_state.stencil_op == 1 && gpu.shader_state.stencil_fail == 3 &&
          gpu.shader_state.stencil_pass == 4 && gpu.shader_state.stencil_front == 1);
    CHECK(gpu.shader_state.stencil_compare == 5 && gpu.shader_state.stencil_ref == 6 &&
          gpu.shader_state.stencil_read_mask == 0x7F && gpu.shader_state.stencil_write_mask == 0x3F);

    /* 3. The trusted mark lives in the queue slot and is rewritten on every
     * enqueue: fill the producer lane and the submit ring with trusted
     * messages, then reuse every slot for malformed untrusted ones. */
    fresh_gpu();
    gate_install();
    const u32 slots = HS_SPSC_SIZE + HS_SUBMIT_SIZE;
    u32 queued = 0;
    for (u32 i = 0; i < slots; i++) {
        Message ok_msg = {.to = NODE_SHADER, .from = NODE_CPU, .op = OP_SET_SHADER, .payload_idx = (u16)(i & 0xFF)};
        queued += hs_send_trusted(&gpu.system, &ok_msg, NULL, 0);
    }
    CHECK(queued == slots);
    CHECK(atomic_load(&gpu.system.mpsc_ok[CHAN_RENDER]) == HS_SUBMIT_SIZE);
    hs_step(&gpu.system);
    queued = 0;
    for (u32 i = 0; i < slots; i++) {
        Message bad = {.to = NODE_SHADER, .from = NODE_CPU, .op = OP_SET_SHADER, .payload_len = 1};
        queued += hs_send(&gpu.system, &bad);
    }
    CHECK(queued == slots);
    hs_step(&gpu.system);
    CHECK(gate_nonconforming == 0 && gpu.system.rejected_invalid == slots);

#ifndef HS_VERIFY_TRUSTED
    /* Trusted sends really skip the gate (that is the saving): a
     * non-conforming trusted message reaches the node, through the producer
     * lane and through the submit ring. HS_VERIFY_TRUSTED builds abort (6). */
    fresh_gpu();
    gate_install();
    queued = 0;
    for (u32 i = 0; i < HS_SPSC_SIZE + 1; i++) {
        Message bad = {.to = NODE_SHADER, .from = NODE_CPU, .op = OP_SET_SHADER, .payload_len = 1};
        queued += hs_send_trusted(&gpu.system, &bad, NULL, 0);
    }
    CHECK(queued == HS_SPSC_SIZE + 1 && atomic_load(&gpu.system.mpsc_ok[CHAN_RENDER]) == 1);
    hs_step(&gpu.system);
    CHECK(gate_nonconforming == HS_SPSC_SIZE + 1 && gpu.system.rejected_invalid == 0);
#endif

    /* 4. The system node used to print OP_TRACE/OP_ERROR strings without any
     * check: an unterminated string is now refused before it is read. */
    fresh_gpu();
    gate_install();
    st = &gpu.system_state_data;
    const u8 unterminated[5] = {'h', 'e', 'l', 'l', 'o'};
    Message trace = {.to = NODE_SYSTEM, .from = NODE_CPU, .op = OP_TRACE, .flags = HS_MSGF_ACK, .cid = 21};
    CHECK(hs_send_with_payload(&gpu.system, &trace, unterminated, sizeof(unterminated)));
    hs_step(&gpu.system);
    hs_step(&gpu.system);
    CHECK(gpu.system.rejected_invalid == 1 && gate_nonconforming == 0);
    CHECK(st->last_ack_cid == 21 && st->last_ack_status == HS_ACK_INVALID && st->last_ack_op == OP_TRACE);

    /* 5. Replay applies the gate: a malformed message in a capture is
     * refused (HS_ACK_INVALID) and the rest replays; with validate_on_send
     * the replay fails instead. */
    fresh_gpu();
    gate_install();
    st = &gpu.system_state_data;
    ShaderState* shader = &gpu.shader_state;
    Message msgs[3] = {
        {.to = NODE_SHADER, .from = NODE_CPU, .op = OP_SET_SHADER, .flags = HS_MSGF_ACK, .cid = 31,
         .payload_idx = 5, .channel = CHAN_RENDER},
        {.to = NODE_SHADER, .from = NODE_CPU, .op = OP_CLIP, .flags = HS_MSGF_ACK, .cid = 32,
         .channel = CHAN_RENDER},
        {.to = NODE_SHADER, .from = NODE_CPU, .op = OP_SET_SHADER, .flags = HS_MSGF_ACK, .cid = 33,
         .payload_idx = 6, .channel = CHAN_RENDER},
    };
    static Payload capture_payloads[3];
    HSCapture cap;
    hs_capture_init(&cap, msgs, capture_payloads, 3);
    cap.count = 3;
    u32 acks = st->ack_count;
    CHECK(hs_capture_replay(&gpu.system, &cap));
    CHECK(gpu.system.rejected_invalid == 1 && gate_nonconforming == 0 && shader->current_shader == 6);
    hs_step(&gpu.system);
    hs_step(&gpu.system);
    CHECK(st->ack_count == acks + 3 && st->last_ack_cid == 33 && st->last_ack_status == HS_ACK_OK);
    gpu.system.validate_on_send = true;
    CHECK(!hs_capture_replay(&gpu.system, &cap));
    gpu.system.validate_on_send = false;

#ifdef HS_VERIFY_TRUSTED
    /* 6. A trusted send that does not conform is a runtime bug: abort. */
    fflush(NULL);
    pid_t child = fork();
    if (child == 0) {
        if (!freopen("/dev/null", "w", stderr)) _exit(3);
        Message bad = {.to = NODE_SHADER, .from = NODE_CPU, .op = OP_SET_SHADER, .payload_len = 1};
        (void)hs_send_trusted(&gpu.system, &bad, NULL, 0);
        hs_step(&gpu.system);
        _exit(0);
    }
    int status = 0;
    CHECK(child > 0 && waitpid(child, &status, 0) == child);
    CHECK(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
#endif
}

/* hs_step skips nodes with empty inboxes, and hs_gpu_process stops after a
 * step that leaves nothing behind instead of running a confirming empty step.
 * Render commands are validated before they are written into the list. */
static NodeProcessFn step_inner[256];
static u32 step_calls[256];
static bool step_relay;
static int step_count(Node* node) {
    step_calls[node->id]++;
    if (step_relay && node->id == NODE_SOUND) {
        /* A later node sends to one that has already run in this step. */
        step_relay = false;
        Message r = {.to = NODE_SHADER, .from = NODE_SOUND, .op = OP_SET_SHADER, .payload_idx = 5,
                     .channel = CHAN_RENDER};
        CHECK(hs_send(node->sys, &r));
    }
    return step_inner[node->id](node);
}

static void test_step_work(void) {
    fresh_gpu();
    memset(step_calls, 0, sizeof(step_calls));
    for (u8 i = 0; i < gpu.system.node_count; i++) {
        Node* n = gpu.system.nodes[i];
        if (!n->process_fn) continue;
        step_inner[n->id] = n->process_fn;
        n->process_fn = step_count;
    }
    CHECK(hs_step(&gpu.system) == 0 && !gpu.system.step_pending);
    for (u32 i = 0; i < 256; i++) CHECK(step_calls[i] == 0);

    /* One message: only its node runs, and the ack it sends to the system
     * node is applied within the same hs_gpu_process call. */
    SystemState* st = &gpu.system_state_data;
    u32 acks = st->ack_count;
    Message m = {.to = NODE_SHADER, .from = NODE_CPU, .op = OP_SET_SHADER, .payload_idx = 3,
                 .flags = HS_MSGF_ACK, .cid = 901, .channel = CHAN_RENDER};
    CHECK(hs_send(&gpu.system, &m));
    u32 steps = hs_gpu_process(&gpu);
    CHECK(steps >= 1 && steps <= 2 && !gpu.system.step_pending);
    CHECK(gpu.shader_state.current_shader == 3);
    CHECK(st->ack_count == acks + 1 && st->last_ack_cid == 901 && st->last_ack_status == HS_ACK_OK);
    CHECK(step_calls[NODE_SHADER] == 1 && step_calls[NODE_BUFFER] == 0 && step_calls[NODE_TEXTURE] == 0 &&
          step_calls[NODE_OUTPUT] == 0 && step_calls[NODE_SOUND] == 0);
    CHECK(step_calls[NODE_SYSTEM] == 1);
    CHECK(hs_gpu_process(&gpu) == 0);

    /* A message left in an inbox by a node that ran later is applied too. */
    u32 shader_calls = step_calls[NODE_SHADER];
    step_relay = true;
    hs_gpu_set_channel(&gpu, 0, 1);
    CHECK(hs_step(&gpu.system) > 0 && gpu.system.step_pending && gpu.shader_state.current_shader == 3);
    CHECK(hs_gpu_process(&gpu) == 1 && !gpu.system.step_pending);
    CHECK(gpu.shader_state.current_shader == 5 && step_calls[NODE_SHADER] == shader_calls + 1);
    step_relay = true;
    hs_gpu_set_channel(&gpu, 0, 1);
    CHECK(hs_gpu_process(&gpu) == 2 && gpu.shader_state.current_shader == 5 && !step_relay);

    /* A channel stopped by its budget is pending; one call drains it all. */
    CHECK(hs_set_channel_budget(&gpu.system, CHAN_RENDER, 4));
    hs_gpu_begin_frame(&gpu);
    for (u32 i = 0; i < 50; i++) hs_gpu_draw(&gpu, (u8)(i & 7));
    CHECK(hs_step(&gpu.system) > 0 && gpu.system.step_pending && gpu.render.count == 4);
    CHECK(hs_gpu_process(&gpu) == 12 && !gpu.system.step_pending);
    CHECK(gpu.render.count == 50 && !gpu.render.overflow);
    for (u32 i = 0; i < 50; i++) {
        HSRenderCmd want = {.op = HS_RC_DRAW, .a = (u8)(i & 7)};
        CHECK(!memcmp(&gpu.render.cmds[i], &want, sizeof(want)));
    }
    CHECK(hs_set_channel_budget(&gpu.system, CHAN_RENDER, 16384));

    /* Commands are written whole: every field not set by the op is zero. */
    hs_gpu_begin_frame(&gpu);
    memset(gpu.render.cmds, 0xA5, 4 * sizeof(HSRenderCmd));
    hs_gpu_clip(&gpu, 1, 2, 3, 4);
    hs_gpu_clear(&gpu, v4_make(0.25f, 0.5f, 0.75f, 1));
    hs_gpu_process(&gpu);
    HSRenderCmd clip = {.op = HS_RC_SET_CLIP, .x = 1, .y = 2, .payload_idx = 3, .payload_len = 4};
    HSRenderCmd clear = {.op = HS_RC_CLEAR, .f0 = 0.25f, .f1 = 0.5f, .f2 = 0.75f, .f3 = 1};
    CHECK(gpu.render.count == 2 && !memcmp(&gpu.render.cmds[0], &clip, sizeof(clip)) &&
          !memcmp(&gpu.render.cmds[1], &clear, sizeof(clear)));

    /* OP_FRAME_BEGIN resets the list and is then its first command. */
    hs_gpu_frame_begin(&gpu);
    hs_gpu_process(&gpu);
    HSRenderCmd begin = {.op = HS_RC_FRAME_BEGIN};
    CHECK(gpu.render.count == 1 && !memcmp(&gpu.render.cmds[0], &begin, sizeof(begin)));

    /* A full list: non-render messages do not mark overflow; a draw does,
     * and nothing is written past the end. */
    gpu.render.count = HS_RENDER_MAX_CMDS;
    hs_gpu_set_shader(&gpu, 2);
    hs_gpu_process(&gpu);
    CHECK(gpu.render.count == HS_RENDER_MAX_CMDS && !gpu.render.overflow && gpu.shader_state.current_shader == 2);
    hs_gpu_draw(&gpu, 1);
    hs_gpu_process(&gpu);
    CHECK(gpu.render.count == HS_RENDER_MAX_CMDS && gpu.render.overflow);
    hs_gpu_frame_begin(&gpu);
    hs_gpu_process(&gpu);
    CHECK(gpu.render.count == 1 && !gpu.render.overflow);

    for (u8 i = 0; i < gpu.system.node_count; i++) {
        Node* n = gpu.system.nodes[i];
        if (n->process_fn == step_count) n->process_fn = step_inner[n->id];
    }
}

/* The producer caches the consumer's head and re-reads it only when its lane
 * looks full: a drained lane is reused without spilling to the submit ring,
 * and an undrained one still never overwrites unconsumed slots. */
static void test_spsc_head_cache(void) {
    fresh_gpu();
    u32 spilled = atomic_load(&gpu.system.mpsc_ok[CHAN_RENDER]);
    for (u32 round = 0; round < 4; round++) {
        for (u32 i = 0; i < HS_SPSC_SIZE; i++) hs_gpu_set_shader(&gpu, (u8)(round * 16 + (i & 15)));
        CHECK(atomic_load(&gpu.system.mpsc_ok[CHAN_RENDER]) == spilled);
        hs_gpu_process(&gpu);
        CHECK(gpu.shader_state.current_shader == round * 16 + 15);
    }
    for (u32 i = 0; i < HS_SPSC_SIZE + 1; i++) hs_gpu_set_shader(&gpu, (u8)(i == HS_SPSC_SIZE ? 99 : 1));
    CHECK(atomic_load(&gpu.system.mpsc_ok[CHAN_RENDER]) == spilled + 1);
    hs_gpu_process(&gpu);
    CHECK(gpu.shader_state.current_shader == 99); /* lanes drain before the submit ring */
    /* After hs_clear the lane starts empty again. */
    for (u32 i = 0; i < 10; i++) hs_gpu_set_shader(&gpu, 2);
    hs_clear(&gpu.system);
    spilled = atomic_load(&gpu.system.mpsc_ok[CHAN_RENDER]);
    for (u32 i = 0; i < HS_SPSC_SIZE; i++) hs_gpu_set_shader(&gpu, 3);
    CHECK(atomic_load(&gpu.system.mpsc_ok[CHAN_RENDER]) == spilled);
    hs_gpu_process(&gpu);
    CHECK(gpu.shader_state.current_shader == 3);
}

static void test_invalid_messages_nacked(void) {
    fresh_gpu();
    SystemState* st = &gpu.system_state_data;
    ShaderState* shader = &gpu.shader_state;
    u8 blend[2] = {3, 4};
    Message m = {.to = NODE_SHADER, .from = NODE_CPU, .op = OP_BLEND, .flags = HS_MSGF_ACK,
                 .cid = 5, .channel = CHAN_RENDER};
    u8 src_before = shader->blend_src, dst_before = shader->blend_dst;
    u32 acks = st->ack_count;
    CHECK(hs_send_with_payload(&gpu.system, &m, blend, 1)); /* short payload */
    hs_step(&gpu.system);
    hs_step(&gpu.system);
    CHECK(st->ack_count == acks + 1 && st->last_ack_cid == 5);
    CHECK(st->last_ack_status == HS_ACK_INVALID && st->last_ack_op == OP_BLEND);
    CHECK(shader->blend_src == src_before && shader->blend_dst == dst_before);

    m.cid = 6;
    CHECK(hs_send_with_payload(&gpu.system, &m, blend, sizeof(blend)));
    hs_step(&gpu.system);
    hs_step(&gpu.system);
    CHECK(st->last_ack_cid == 6 && st->last_ack_status == HS_ACK_OK);
    CHECK(shader->blend_src == 3 && shader->blend_dst == 4);

    /* Payload-free message that requires a payload is also rejected. */
    Message clip = {.to = NODE_SHADER, .from = NODE_CPU, .op = OP_CLIP, .flags = HS_MSGF_ACK,
                    .cid = 7, .channel = CHAN_RENDER};
    CHECK(hs_send(&gpu.system, &clip));
    hs_step(&gpu.system);
    hs_step(&gpu.system);
    CHECK(st->last_ack_cid == 7 && st->last_ack_status == HS_ACK_INVALID);

    /* Ops without a decoder (fixed 64-byte matrix) used to skip silently and
     * ACK success on a short payload; the schema gate rejects them. */
    float camera[16] = {2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1};
    float view_before = shader->camera_view_proj[0];
    Message cam = {.to = NODE_SHADER, .from = NODE_CPU, .op = OP_SET_CAMERA, .flags = HS_MSGF_ACK,
                   .cid = 10, .channel = CHAN_RENDER};
    CHECK(hs_send_with_payload(&gpu.system, &cam, camera, 10));
    hs_step(&gpu.system);
    hs_step(&gpu.system);
    CHECK(st->last_ack_cid == 10 && st->last_ack_status == HS_ACK_INVALID);
    CHECK(shader->camera_view_proj[0] == view_before);
    cam.cid = 11;
    CHECK(hs_send_with_payload(&gpu.system, &cam, camera, sizeof(camera)));
    hs_step(&gpu.system);
    hs_step(&gpu.system);
    CHECK(st->last_ack_cid == 11 && st->last_ack_status == HS_ACK_OK && shader->camera_view_proj[0] == 2.0f);

    /* Input opcodes have a schema now: with validation on, they are accepted
     * (previously rejected as "bad destination"), and NaN axes are zeroed. */
    static Node input;
    static InputState input_state;
    memset(&input, 0, sizeof(input));
    input.id = NODE_INPUT;
    input.sys = &gpu.system;
    input.state = &input_state;
    input_node_init(&input);
    hs_register(&gpu.system, &input);
    gpu.system.validate_on_send = true;
    u8 key[3] = {105, 0, 1};
    Message k = {.to = NODE_INPUT, .from = NODE_CPU, .op = OP_INPUT_KEY, .flags = HS_MSGF_ACK,
                 .cid = 8, .channel = CHAN_RT};
    CHECK(hs_send_with_payload(&gpu.system, &k, key, sizeof(key)));
    u32 nan_bits = 0x7FC00000u, inf_bits = 0x7F800000u;
    u8 axes[8];
    memcpy(axes, &nan_bits, 4);
    memcpy(axes + 4, &inf_bits, 4);
    Message g = {.to = NODE_INPUT, .from = NODE_CPU, .op = OP_INPUT_GAMEPAD, .flags = HS_MSGF_ACK,
                 .cid = 9, .channel = CHAN_RT};
    CHECK(hs_send_with_payload(&gpu.system, &g, axes, sizeof(axes)));
    hs_step(&gpu.system);
    hs_step(&gpu.system);
    CHECK(input_state.base.key_left == 1);
    CHECK(input_state.base.pad_x == 0.0f && input_state.base.pad_y == 0.0f);
    CHECK(st->last_ack_cid == 9 && st->last_ack_status == HS_ACK_OK);
    fresh_gpu();
}

static void test_storage_names_and_files(void) {
    static HSStorage store, copy;
    hs_storage_init(&store);
    char long_name[64];
    memset(long_name, 'n', sizeof(long_name) - 1);
    long_name[sizeof(long_name) - 1] = 0;
    CHECK(hs_storage_load(&store, long_name) == NULL);       /* no silent truncation */
    long_name[HS_STORAGE_NAME_MAX - 1] = 0;                  /* 31 chars: fits */
    hs_storage_set_u8(&store, long_name, 3, 42);
    CHECK(hs_storage_get_u8(&store, long_name, 3) == 42);
    unsigned used = 0;
    for (int i = 0; i < HS_STORAGE_SLOTS; i++) used += store.slots[i].loaded;
    CHECK(used == 1);                                        /* one slot, reused */
    char path[96];
    snprintf(path, sizeof(path), "tests/p0-storage-%d.bin", (int)getpid());
    CHECK(hs_storage_save(&store, path));
    hs_storage_init(&copy);
    CHECK(hs_storage_load_file(&copy, path) && hs_storage_get_u8(&copy, long_name, 3) == 42);
    FILE* f = fopen(path, "r+b");                           /* truncate: must be rejected */
    CHECK(f && ftruncate(fileno(f), 100) == 0);
    if (f) fclose(f);
    hs_storage_init(&copy);
    CHECK(!hs_storage_load_file(&copy, path));
    CHECK(!copy.slots[0].loaded);
    CHECK(!hs_storage_save(&store, "/dev/full"));
    unlink(path);
}

static void test_font_glyph_codes(void) {
    static HSFont font;
    CHECK(hs_font_load(&font, "src/medodica_font"));
    /* code="0".."9" are the digit characters, not code points 0..9. */
    for (int c = '0'; c <= '9'; c++) CHECK(font.glyphs[c].advance > 0 && font.glyphs[c].width > 0);
    for (int c = 0; c <= 9; c++) CHECK(font.glyphs[c].advance == 0);
    /* Entities decode to their characters. */
    CHECK(font.glyphs['"'].advance > 0 && font.glyphs['&'].advance > 0 && font.glyphs['<'].advance > 0);
    CHECK(font.glyphs['A'].width > 0 && font.glyphs[' '].advance > 0);
    /* An atlas larger than the fixed buffer is refused, not copied. */
    static u8 big[(HS_FONT_ATLAS_W * 2) * HS_FONT_ATLAS_H * 4];
    CHECK(!hs_font_load_fnt(&font, NULL, big, HS_FONT_ATLAS_W * 2, HS_FONT_ATLAS_H));
}

/* Wire format mirrors src/hs_ipc.c (packed). */
typedef struct { u32 magic; u16 version; u16 type; u32 len; u32 cid; } __attribute__((packed)) WireHdr;
typedef struct { u8 op; u8 flags; u16 reserved; u32 payload_len; } __attribute__((packed)) WireReq;
typedef struct { u32 status; u32 result_op; u32 result_len; } __attribute__((packed)) WireResp;

static bool ipc_set_budget(int fd, u32 cid, u8 channel, u32 budget, u32* status, u32* old_value, u32* new_value) {
    u8 frame[sizeof(WireHdr) + sizeof(WireReq) + 8];
    WireHdr h = {.magic = 0x4E474950u, .version = 1, .type = 1, .len = sizeof(WireReq) + 8, .cid = cid};
    WireReq r = {.op = OP_SET_CHAN_BUDGET, .payload_len = 8};
    memcpy(frame, &h, sizeof(h));
    memcpy(frame + sizeof(h), &r, sizeof(r));
    hs_pack_set_chan_budget(frame + sizeof(h) + sizeof(r), channel, budget);
    if (send(fd, frame, sizeof(frame), MSG_NOSIGNAL) != (ssize_t)sizeof(frame)) return false;
    u8 reply[sizeof(WireHdr) + sizeof(WireResp) + 96];
    size_t got = 0, want = sizeof(WireHdr) + sizeof(WireResp);
    while (got < want) {
        ssize_t n = recv(fd, reply + got, sizeof(reply) - got, 0);
        if (n <= 0) return false;
        got += (size_t)n;
        if (got >= sizeof(WireHdr)) {
            WireHdr rh;
            memcpy(&rh, reply, sizeof(rh));
            want = sizeof(WireHdr) + rh.len;
            if (want > sizeof(reply)) return false;
        }
    }
    WireResp resp;
    memcpy(&resp, reply + sizeof(WireHdr), sizeof(resp));
    *status = resp.status;
    *old_value = *new_value = 0;
    if (resp.status == 0 && resp.result_len == 8)
        hs_unpack_u32x2(reply + sizeof(WireHdr) + sizeof(WireResp), 8, old_value, new_value);
    return true;
}

static atomic_bool stepping;
static void* step_loop(void* unused) {
    (void)unused;
    while (atomic_load(&stepping)) { hs_step(&gpu.system); usleep(200); }
    return NULL;
}

static void test_ipc_late_result_not_misattributed(void) {
    fresh_gpu();
    char path[108];
    snprintf(path, sizeof(path), "tests/p0-late-%d.sock", (int)getpid());
    HSIpcServer server = {0};
    CHECK(hs_ipc_start(&server, &gpu.system, path));
    struct stat sb;
    CHECK(stat(path, &sb) == 0 && (sb.st_mode & 0777) == 0600);
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un address = {.sun_family = AF_UNIX};
    strcpy(address.sun_path, path);
    CHECK(fd >= 0 && connect(fd, (struct sockaddr*)&address, sizeof(address)) == 0);
    u32 status = 0, old_value = 0, new_value = 0;
    /* Nobody steps: request 1 times out but stays queued and runs later. */
    CHECK(ipc_set_budget(fd, 1, (u8)CHAN_RENDER, 111, &status, &old_value, &new_value));
    CHECK(status != 0);
    /* Request 2 (same client cid and op). With an RT budget of 1, request 1
     * runs a full step before request 2, so its late result is published
     * while request 2 is waiting. The reply must describe request 2. */
    CHECK(hs_set_channel_budget(&gpu.system, CHAN_RT, 1));
    atomic_store(&stepping, true);
    pthread_t stepper;
    CHECK(pthread_create(&stepper, NULL, step_loop, NULL) == 0);
    CHECK(ipc_set_budget(fd, 1, (u8)CHAN_RENDER, 222, &status, &old_value, &new_value));
    CHECK(status == 0 && old_value == 111 && new_value == 222);
    atomic_store(&stepping, false);
    pthread_join(stepper, NULL);
    close(fd);
    hs_ipc_stop(&server);
    fresh_gpu();
}

/* accept() failing with EMFILE (descriptor exhaustion) is transient: the IPC
 * thread must keep serving once descriptors are available again. */
static void test_ipc_accept_survives_emfile(void) {
    fresh_gpu();
    char path[108];
    snprintf(path, sizeof(path), "tests/p0-emfile-%d.sock", (int)getpid());
    HSIpcServer server = {0};
    CHECK(hs_ipc_start(&server, &gpu.system, path));
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un address = {.sun_family = AF_UNIX};
    strcpy(address.sun_path, path);
    struct rlimit saved, tight;
    CHECK(getrlimit(RLIMIT_NOFILE, &saved) == 0);
    tight = saved;
    /* The limit bounds descriptor numbers: capping it at the lowest free
     * number makes every new descriptor (the server's accept) fail EMFILE. */
    int lowest_free = dup(0);
    CHECK(lowest_free >= 0);
    close(lowest_free);
    tight.rlim_cur = (rlim_t)lowest_free;
    CHECK(setrlimit(RLIMIT_NOFILE, &tight) == 0);
    int probe = dup(0);
    CHECK(probe < 0 && errno == EMFILE);  /* the condition really holds */
    if (probe >= 0) close(probe);
    CHECK(fd >= 0 && connect(fd, (struct sockaddr*)&address, sizeof(address)) == 0); /* queued in backlog */
    usleep(150000);                        /* server's accept fails with EMFILE meanwhile */
    CHECK(setrlimit(RLIMIT_NOFILE, &saved) == 0);
    atomic_store(&stepping, true);
    pthread_t stepper;
    CHECK(pthread_create(&stepper, NULL, step_loop, NULL) == 0);
    struct timeval timeout = {.tv_sec = 2};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    u32 status = 99, old_value = 0, new_value = 0;
    CHECK(ipc_set_budget(fd, 3, (u8)CHAN_RENDER, 333, &status, &old_value, &new_value));
    CHECK(status == 0 && new_value == 333);
    atomic_store(&stepping, false);
    pthread_join(stepper, NULL);
    close(fd);
    hs_ipc_stop(&server);
    fresh_gpu();
}

static void test_lifecycle(void) {
    static HSGpu instance;
    for (unsigned i = 0; i < 100; i++) {
        hs_gpu_init(&instance);
        CHECK(instance.system.log_payloads == NULL);
        hs_gpu_clear(&instance, v4_make((float)i, 0, 0, 1));
        hs_gpu_process(&instance);
        CHECK(instance.system.log_head == 1 && instance.system.owns_log_payloads);
        hs_gpu_shutdown(&instance);
        CHECK(instance.system.log_payloads == NULL && !instance.system.lock_inited);
        hs_gpu_shutdown(&instance);
    }
}

int main(void) {
    alarm(30); /* A deadlock is a failing test, not an indefinitely hung job. */
    unsigned descriptor_baseline = open_descriptors();
    printf("Measured layouts: HSGpu=%zu, Payload=%zu, lazy journal capacity=%zu bytes\n",
           sizeof(HSGpu), sizeof(Payload), (size_t)HS_MAX_MSG_LOG * sizeof(Payload));
    test_codecs();
    test_bounds();
    test_capture();
    test_actor_handoff();
    test_ipc_shutdown();
    test_async_ownership();
    test_input_ownership();
    test_submit_ring_wrap();
    test_budget_floor();
    test_conforms_matches_validate();
    test_gate_at_routing();
    test_step_work();
    test_spsc_head_cache();
    test_invalid_messages_nacked();
    test_storage_names_and_files();
    test_font_glyph_codes();
    test_ipc_late_result_not_misattributed();
    test_ipc_accept_survives_emfile();
    test_lifecycle();
    hs_gpu_shutdown(&gpu);
    CHECK(open_descriptors() == descriptor_baseline);
    printf("P0 regressions: %u checks, %u failures\n", checks, failures);
    return failures ? 1 : 0;
}
