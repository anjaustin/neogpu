/* Deterministic regressions for release-blocking audit findings. */
#include "hs_gpu.h"
#include "hs_msg.h"
#include "hs_storage.h"
#include "hs_ipc.h"
#include "hs_async.h"
#include "hs_input_stream.h"
#include <sched.h>
#include <errno.h>
#include <dirent.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>
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
    test_lifecycle();
    hs_gpu_shutdown(&gpu);
    CHECK(open_descriptors() == descriptor_baseline);
    printf("P0 regressions: %u checks, %u failures\n", checks, failures);
    return failures ? 1 : 0;
}
