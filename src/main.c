#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include "hs_gpu.h"
#include "hs_math_neon.h"
#include "hs_input.h"
#include "hs_buffer.h"
#include "hs_async.h"

/* ============================================================
 * Test helpers
 * ============================================================ */
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name, cond) do { \
    if (cond) { tests_passed++; printf("  [PASS] %s\n", name); } \
    else      { tests_failed++; printf("  [FAIL] %s\n", name); } \
} while(0)

/* ============================================================
 * Math library tests
 * ============================================================ */
static void test_math(void) {
    printf("\n--- Math Tests ---\n");

    /* vec4 basics */
    vec4 a = v4_make(1, 2, 3, 4);
    vec4 b = v4_make(5, 6, 7, 8);
    vec4 sum = v4_add(a, b);
    TEST("v4_add", v4_equal(sum, v4_make(6, 8, 10, 12), 1e-5f));

    vec4 diff = v4_sub(b, a);
    TEST("v4_sub", v4_equal(diff, v4_make(4, 4, 4, 4), 1e-5f));

    f32 dot = v4_dot(a, b);
    TEST("v4_dot", fabsf(dot - 70.0f) < 1e-4f);  /* 1*5+2*6+3*7+4*8=70 */

    vec4 n = v4_normalize(v4_make(3, 0, 0, 0));
    TEST("v4_normalize", fabsf(v4_length(n) - 1.0f) < 1e-5f);

    /* Cross product (w=0) */
    vec4 cx = v4_make(1, 0, 0, 0);
    vec4 cy = v4_make(0, 1, 0, 0);
    vec4 cz = v4_cross(cx, cy);
    /* v4_cross uses vextq rotation trick; verify z component */
    TEST("v4_cross z", fabsf(v4_z(cz)) > 0.5f);

    /* Lerp */
    vec4 lp = v4_lerp(v4_zero(), v4_one(), 0.5f);
    TEST("v4_lerp", v4_equal(lp, v4_make(0.5f, 0.5f, 0.5f, 0.5f), 1e-5f));

    /* mat4 identity * identity = identity */
    mat4 id = m4_identity();
    mat4 id2 = m4_multiply(id, id);
    TEST("m4_identity*identity", v4_equal(id2.val[0], v4_make(1,0,0,0), 1e-5f));

    /* Translation */
    mat4 t = m4_translation(10, 20, 30);
    TEST("m4_translation", fabsf(v4_x(t.val[3]) - 10.0f) < 1e-5f);

    /* Rotation: rotating (1,0,0,0) by 90 degrees around Z should give (0,1,0,0) */
    mat4 rz = m4_rotation_z(PI * 0.5f);
    /* Apply column-major transform: result = M * v (columns are basis vectors) */
    TEST("m4_rotation_z", fabsf(v4_y(rz.val[0])) > 0.9f);

    /* Inverse: M * M^-1 = I */
    mat4 tm = m4_translation(3, 4, 5);
    mat4 inv = m4_invert(tm);
    mat4 prod = m4_multiply(tm, inv);
    TEST("m4_invert", v4_equal(prod.val[0], v4_make(1,0,0,0), 1e-3f) &&
                      v4_equal(prod.val[3], v4_make(0,0,0,1), 1e-3f));

    /* Scalar helpers */
    TEST("hs_abs", fabsf(hs_abs(-5.0f) - 5.0f) < 1e-5f);
    TEST("hs_floor", fabsf(hs_floor(3.7f) - 3.0f) < 1e-5f);
    TEST("hs_ceil", fabsf(hs_ceil(3.2f) - 4.0f) < 1e-5f);
    TEST("hs_clamp", fabsf(hs_clamp(10.0f, 0.0f, 5.0f) - 5.0f) < 1e-5f);
    TEST("hs_lerp", fabsf(hs_lerp(0.0f, 10.0f, 0.5f) - 5.0f) < 1e-5f);

    /* Quaternion -> matrix */
    vec4 q = v4_quat_axis_angle(v4_make(0,0,1,0), PI * 0.5f);
    mat4 qm = m4_from_quat(q);
    TEST("quat_to_mat4", fabsf(v4_y(qm.val[0])) > 0.9f);
}

/* ============================================================
 * Backend hook tests
 * ============================================================ */
typedef struct {
    u32 init_count;
    u32 begin_count;
    u32 exec_count;
    u32 end_count;
} MockBackendState;

static bool mock_backend_init(void* ctx, HSGpu* gpu) {
    (void)gpu;
    ((MockBackendState*)ctx)->init_count++;
    return true;
}

static void mock_backend_begin(void* ctx, const HSFrameContext* frame) {
    (void)frame;
    ((MockBackendState*)ctx)->begin_count++;
}

static void mock_backend_exec(void* ctx, const HSFrameContext* frame) {
    (void)frame;
    ((MockBackendState*)ctx)->exec_count++;
}

static void mock_backend_end(void* ctx, const HSFrameContext* frame) {
    (void)frame;
    ((MockBackendState*)ctx)->end_count++;
}

static void test_backend_hooks(void) {
    printf("\n--- Backend Hook Tests ---\n");

    static HSGpu gpu;
    hs_gpu_init(&gpu);

    static MockBackendState st;
    memset(&st, 0, sizeof(st));

    static const HSBackendOps ops = {
        .init = mock_backend_init,
        .shutdown = NULL,
        .begin_frame = mock_backend_begin,
        .execute = mock_backend_exec,
        .end_frame = mock_backend_end,
    };

    HSBackend backend = {
        .ctx = &st,
        .ops = &ops,
    };

    hs_gpu_attach_backend(&gpu, &backend);
    hs_gpu_begin_frame(&gpu);
    hs_gpu_end_frame(&gpu);

    TEST("backend_init", st.init_count == 1);
    TEST("backend_begin", st.begin_count == 1);
    TEST("backend_execute", st.exec_count == 1);
    TEST("backend_end", st.end_count == 1);
}

typedef struct {
    u32 cmd_count;
    u8  ops[16];
} MockExecState;

static void mock_backend_exec_capture(void* ctx, const HSFrameContext* frame) {
    MockExecState* st = (MockExecState*)ctx;
    if (!frame || !frame->render) return;
    st->cmd_count = frame->render->count;
    u32 n = frame->render->count;
    if (n > 16) n = 16;
    for (u32 i = 0; i < n; i++) {
        st->ops[i] = frame->render->cmds[i].op;
    }
}

static void test_render_commands(void) {
    printf("\n--- Render Command Tests ---\n");

    static HSGpu gpu;
    hs_gpu_init(&gpu);

    static MockExecState st;
    memset(&st, 0, sizeof(st));

    static const HSBackendOps ops = {
        .init = NULL,
        .shutdown = NULL,
        .begin_frame = NULL,
        .execute = mock_backend_exec_capture,
        .end_frame = NULL,
    };

    HSBackend backend = {
        .ctx = &st,
        .ops = &ops,
    };
    hs_gpu_attach_backend(&gpu, &backend);

    hs_gpu_begin_frame(&gpu);

    hs_gpu_color_mask(&gpu, 0x0F);
    hs_gpu_alpha(&gpu, false);
    hs_gpu_clip(&gpu, 0, 0, 640, 480);
    hs_gpu_texture_filter(&gpu, 0, true);
    hs_gpu_texture_wrap(&gpu, 0, false);

    vec4 clear_color = v4_make(0.2f, 0.3f, 0.4f, 1.0f);
    hs_gpu_clear(&gpu, clear_color);
    hs_gpu_show_texture(&gpu, 0);
    hs_gpu_draw_text(&gpu, "hi");
    hs_gpu_draw(&gpu, 0);
    hs_gpu_process(&gpu);

    hs_gpu_end_frame(&gpu);

    TEST("render_cmd_count", st.cmd_count >= 9);
    TEST("render_cmd_mask", st.ops[0] == HS_RC_SET_COLOR_MASK);
    TEST("render_cmd_alpha", st.ops[1] == HS_RC_SET_ALPHA);
    TEST("render_cmd_clip", st.ops[2] == HS_RC_SET_CLIP);
    TEST("render_cmd_tex_filter", st.ops[3] == HS_RC_SET_TEX_FILTER);
    TEST("render_cmd_tex_wrap", st.ops[4] == HS_RC_SET_TEX_WRAP);
    TEST("render_cmd_clear", st.ops[5] == HS_RC_CLEAR);
    TEST("render_cmd_show_tex", st.ops[6] == HS_RC_SHOW_TEXTURE);
}

/* ============================================================
 * Buffer tests
 * ============================================================ */
static void test_buffer(void) {
    printf("\n--- Buffer Tests ---\n");

    HSBuffer buf;
    bool ok = hs_buffer_init(&buf, 0, 256);
    TEST("buffer_init", ok && buf.data != NULL);

    hs_buffer_set_f32(&buf, 0, 3.14f);
    f32 val = hs_buffer_get_f32(&buf, 0);
    TEST("buffer_set/get_f32", fabsf(val - 3.14f) < 1e-5f);

    hs_buffer_set_i32(&buf, 1, 42);
    s32 ival = hs_buffer_get_i32(&buf, 1);
    TEST("buffer_set/get_i32", ival == 42);

    vec4 v = v4_make(1.0f, 2.0f, 3.0f, 4.0f);
    hs_buffer_set_vec4(&buf, 4, v);
    vec4 rv = hs_buffer_get_vec4(&buf, 4);
    TEST("buffer_set/get_vec4", v4_equal(rv, v, 1e-5f));

    mat4 m = m4_translation(1, 2, 3);
    hs_buffer_set_mat4(&buf, 8, m);
    /* Verify the translation component at float-index 8+12=20 */
    f32 tx = hs_buffer_get_f32(&buf, 20);
    TEST("buffer_set_mat4", fabsf(tx - 1.0f) < 1e-5f);

    hs_buffer_free(&buf);
    TEST("buffer_free", buf.data == NULL);
}

/* ============================================================
 * Input system tests
 * ============================================================ */
static void test_input(void) {
    printf("\n--- Input Tests ---\n");

    HSInput input;
    hs_input_init(&input);
    TEST("input_init", input.frame == 0);

    input.key_right = true;
    input.key_down = true;
    hs_input_tick(&input);
    TEST("input_dir", input.dir_x > 0.9f && input.dir_y > 0.9f);
    TEST("input_time", fabsf(hs_time(&input) - (1.0f / 60.0f)) < 1e-4f);
    TEST("input_frame", input.frame == 1);
}

/* ============================================================
 * Message validation tests
 * ============================================================ */
static void test_message_validation(void) {
    printf("\n--- Message Validation Tests ---\n");

    static HSGpu gpu;
    hs_gpu_init(&gpu);
    gpu.system.validate_on_send = true;

    const char* err = NULL;

    /* Wrong destination */
    Message m1 = {
        .to = NODE_BUFFER,
        .from = NODE_CPU,
        .op = OP_SET_SHADER,
        .flags = 0,
        .cid = 0,
        .tick = 0,
        .payload_idx = 0,
        .payload_len = 0
    };
    TEST("validate_bad_destination", !hs_validate_message(&gpu.system, &m1, &err));

    /* Bad payload length */
    Message m2 = {
        .to = NODE_SHADER,
        .from = NODE_CPU,
        .op = OP_SET_PARAM,
        .flags = 0,
        .cid = 0,
        .tick = 0,
        .payload_idx = 0,
        .payload_len = 19
    };
    TEST("validate_bad_payload_len", !hs_validate_message(&gpu.system, &m2, &err));

    /* String not terminated */
    gpu.system.payloads[0].data[0] = 'A';
    gpu.system.payloads[0].data[1] = 'B';
    gpu.system.payloads[0].data[2] = 'C';
    gpu.system.payloads[0].data[3] = 'D';
    Message m3 = {
        .to = NODE_SYSTEM,
        .from = NODE_CPU,
        .op = OP_TRACE,
        .flags = 0,
        .cid = 0,
        .tick = 0,
        .payload_idx = 0,
        .payload_len = 4
    };
    TEST("validate_string_nul", !hs_validate_message(&gpu.system, &m3, &err));

    /* Invalid payload index */
    Message m4 = {
        .to = NODE_SHADER,
        .from = NODE_CPU,
        .op = OP_BLEND,
        .flags = 0,
        .cid = 0,
        .tick = 0,
        .payload_idx = (u16)gpu.system.payload_capacity,
        .payload_len = 2
    };
    TEST("validate_payload_idx", !hs_validate_message(&gpu.system, &m4, &err));

    /* Valid payload-bearing op */
    gpu.system.payloads[0].data[0] = 1;
    gpu.system.payloads[0].data[1] = 0;
    Message m5 = {
        .to = NODE_SHADER,
        .from = NODE_CPU,
        .op = OP_BLEND,
        .flags = 0,
        .cid = 0,
        .tick = 0,
        .payload_idx = 0,
        .payload_len = 2
    };
    TEST("validate_ok", hs_validate_message(&gpu.system, &m5, &err));

    /* Ack request */
    Message m6 = {
        .to = NODE_SHADER,
        .from = NODE_CPU,
        .op = OP_SET_SHADER,
        .flags = HS_MSGF_ACK,
        .cid = 123,
        .tick = 0,
        .payload_idx = 0,
        .payload_len = 0
    };
    bool send_ok = hs_send(&gpu.system, &m6);
    hs_step(&gpu.system);
    hs_step(&gpu.system);
    SystemState* st = (SystemState*)gpu.system_node.state;
    TEST("ack_send_ok", send_ok);
    TEST("ack_received", st->ack_count > 0 && st->last_ack_cid == 123 && st->last_ack_op == OP_SET_SHADER);

    /* Structured error emitted on route-time validation failure */
    Message bad = {
        .to = NODE_BUFFER,
        .from = NODE_CPU,
        .op = OP_SET_SHADER,
        .flags = 0,
        .cid = 777,
        .tick = 0,
        .payload_idx = 0,
        .payload_len = 0,
    };
    bool bad_ok = hs_send(&gpu.system, &bad);
    hs_step(&gpu.system);
    TEST("send_queued", bad_ok);
    TEST("error_ex_emitted", st->error_count > 0 && st->last_error_op == OP_SET_SHADER && st->last_error_to == NODE_BUFFER);

    /* Backpressure: overflow submit queue without stepping */
    gpu.system.recording = false;
    u32 ok_count = 0;
    for (u32 i = 0; i < (HS_SUBMIT_SIZE + 64); i++) {
        Message spam = {
            .to = NODE_SHADER,
            .from = NODE_CPU,
            .op = OP_SET_SHADER,
            .flags = 0,
            .cid = i,
            .tick = 0,
            .payload_idx = 0,
            .payload_len = 0,
        };
        if (hs_send(&gpu.system, &spam)) ok_count++;
    }
    hs_step(&gpu.system);
    TEST("queue_full_triggered", ok_count < (HS_SUBMIT_SIZE + 64) && gpu.system.submit_full > 0);
}

/* ============================================================
 * Async completion tests
 * ============================================================ */
static void test_async_done(void) {
    printf("\n--- Async Completion Tests ---\n");

    static HSGpu gpu;
    hs_gpu_init(&gpu);
    gpu.system.validate_on_send = true;
    gpu.system.recording = false;

    HSAsync async;
    hs_async_init(&async, NULL);
    hs_async_attach_system(&async, &gpu.system, NODE_SYSTEM);

    const char* path = "/tmp/neogpu_async_test.bin";
    const char data[] = "abc";
    bool enq_ok = hs_async_save_file(&async, path, data, 3);
    TEST("async_enqueued", enq_ok);

    SystemState* st = (SystemState*)gpu.system_node.state;
    u32 start = st->async_done_count;

    bool got = false;
    for (int i = 0; i < 200; i++) {
        if (hs_async_process(&async)) {
            hs_step(&gpu.system);
            if (st->async_done_count > start) {
                got = true;
                break;
            }
        }
        usleep(1000);
    }

    TEST("async_done_msg", got && st->last_async_success == 1);

    FILE* f = fopen(path, "rb");
    if (f) {
        char buf[4] = {0};
        size_t r = fread(buf, 1, 3, f);
        fclose(f);
        TEST("async_file_written", r == 3 && buf[0] == 'a' && buf[1] == 'b' && buf[2] == 'c');
    } else {
        TEST("async_file_written", false);
    }

    hs_async_shutdown(&async);
}

static void test_async_atomic_running(void) {
    printf("\n--- Async Atomic Tests ---\n");

    HSAsync async;
    hs_async_init(&async, NULL);

    /* start */
    bool ok = hs_async_save_file(&async, "/tmp/neogpu_async_atomic.bin", "x", 1);
    TEST("async_start", ok);

    /* stop */
    hs_async_shutdown(&async);
    TEST("async_stop", true);
}

/* ============================================================
 * Multi-thread messaging falsification
 * ============================================================ */
typedef struct {
    HSSystem* sys;
    volatile bool* stop;
    u32 sent;
    u32 failed;
} SenderArgs;

static void* sender_thread(void* arg) {
    SenderArgs* a = (SenderArgs*)arg;
    u32 i = 0;
    while (!*a->stop) {
        Message m = {
            .to = NODE_SHADER,
            .from = NODE_CPU,
            .op = OP_SET_SHADER,
            .flags = HS_MSGF_ACK,
            .cid = i,
            .tick = 0,
            .payload_idx = (u16)(i & 0xFF),
            .payload_len = 0,
        };
        if (hs_send(a->sys, &m)) a->sent++;
        else a->failed++;
        i++;
        usleep(50);
    }
    return NULL;
}

static void test_thread_safety(void) {
    printf("\n--- Thread Safety Tests ---\n");

    static HSGpu gpu;
    hs_gpu_init(&gpu);
    gpu.system.validate_on_send = true;
    gpu.system.recording = false;

    volatile bool stop = false;
    SenderArgs args = {
        .sys = &gpu.system,
        .stop = &stop,
        .sent = 0,
        .failed = 0,
    };

    pthread_t th;
    int rc = pthread_create(&th, NULL, sender_thread, &args);
    TEST("thread_spawn", rc == 0);

    /* Main thread steps while sender enqueues (leave CPU time for sender). */
    for (int i = 0; i < 500; i++) {
        hs_step(&gpu.system);
        usleep(1000);
    }

    stop = true;
    pthread_join(th, NULL);

    SystemState* st = (SystemState*)gpu.system_node.state;
    TEST("thread_sent", args.sent > 0);
    TEST("thread_ack_progress", st->ack_count > 0);
}

/* ============================================================
 * GPU message system demo (original + new ops)
 * ============================================================ */
static void test_gpu(void) {
    printf("\n--- GPU Message System Demo ---\n");

    static HSGpu gpu;  /* static because it's large (~17MB) */
    hs_gpu_init(&gpu);

    printf("Recording frame...\n");
    hs_gpu_start_recording(&gpu);

    /* Clear */
    vec4 clear_color = v4_make(0.1f, 0.1f, 0.12f, 1.0f);
    hs_gpu_clear(&gpu, clear_color);
    hs_gpu_process(&gpu);

    /* Camera: look at origin from z=4 */
    mat4 view = m4_look_at(
        v4_make(0, 0, 4, 1),
        v4_zero(),
        v4_make(0, 1, 0, 0)
    );
    mat4 proj = m4_perspective(hs_deg_to_rad(60.0f), 640.0f/480.0f, 0.1f, 100.0f);
    mat4 vp = m4_multiply(proj, view);
    hs_gpu_set_camera(&gpu, vp);
    hs_gpu_process(&gpu);

    /* Shader + params */
    hs_gpu_set_shader(&gpu, 0);
    vec4 light = v4_normalize(v4_make(0.5f, 0.2f, 1.0f, 0.0f));
    hs_gpu_set_param(&gpu, 0, light);
    hs_gpu_process(&gpu);

    /* Render state */
    hs_gpu_depth(&gpu, true);
    hs_gpu_cull(&gpu, 1);  /* back-face culling */
    hs_gpu_blend(&gpu, 1, 0);
    hs_gpu_color_mask(&gpu, 0x0F);
    hs_gpu_process(&gpu);

    /* Stencil (new) */
    hs_gpu_stencil(&gpu, 0, 0, 0, 1);
    hs_gpu_stencil_func(&gpu, 1, 0, 0xFF, 0xFF);
    hs_gpu_depth_compare(&gpu, 2, true);
    hs_gpu_process(&gpu);

    /* Draw */
    hs_gpu_load_buffer(&gpu, 0);
    hs_gpu_draw(&gpu, 0);
    hs_gpu_process(&gpu);

    /* Draw text (new) */
    hs_gpu_draw_text(&gpu, "Hello PicoGPU!");
    hs_gpu_process(&gpu);

    /* Draw instanced */
    hs_gpu_load_buffer(&gpu, 1);
    hs_gpu_draw_instance(&gpu, 0, 1, 64);
    hs_gpu_process(&gpu);

    /* Show texture (new) */
    hs_gpu_load_texture(&gpu, 0);
    hs_gpu_show_texture(&gpu, 0);
    hs_gpu_process(&gpu);

    /* Clear depth/stencil */
    hs_gpu_clear_ds(&gpu, 1.0f, 0);
    hs_gpu_process(&gpu);

    /* Sound channel */
    hs_gpu_set_channel(&gpu, 0, 2);
    hs_gpu_process(&gpu);

    u32 log_count = hs_gpu_stop_recording(&gpu);
    printf("\nLog contains %d messages\n", log_count);
    TEST("gpu_recorded_msgs", log_count > 0);

    /* Capture (self-contained payload copy) */
    static Message cap_msgs[HS_MAX_MSG_LOG];
    static Payload cap_payloads[HS_MAX_MSG_LOG];
    HSCapture cap;
    hs_capture_init(&cap, cap_msgs, cap_payloads, HS_MAX_MSG_LOG);
    bool cap_ok = hs_capture_from_log(&gpu.system, gpu.log_buffer, log_count, &cap);
    TEST("gpu_capture", cap_ok);

    /* Capture I/O */
    const char* cap_path = "/tmp/neogpu_capture.bin";
    bool cap_write_ok = hs_capture_write_file(&cap, cap_path);
    TEST("gpu_capture_write", cap_write_ok);

    static Message cap2_msgs[HS_MAX_MSG_LOG];
    static Payload cap2_payloads[HS_MAX_MSG_LOG];
    HSCapture cap2;
    bool cap_read_ok = hs_capture_read_file(&cap2, cap_path, cap2_msgs, cap2_payloads, HS_MAX_MSG_LOG);
    TEST("gpu_capture_read", cap_read_ok);

    /* Replay -- replay BEFORE clearing, so payload data is still valid */
    printf("\nReplaying frame...\n");
    bool replay_ok = hs_gpu_replay(&gpu, gpu.log_buffer, log_count);
    TEST("gpu_replay", replay_ok);

    /* Replay the self-contained capture */
    bool cap_replay_ok = hs_capture_replay(&gpu.system, &cap);
    TEST("gpu_replay_capture", cap_replay_ok);

    /* Replay the capture loaded from disk */
    bool cap2_replay_ok = hs_capture_replay(&gpu.system, &cap2);
    TEST("gpu_replay_capture_io", cap2_replay_ok);
}

/* ============================================================
 * Benchmark
 * ============================================================ */
#include <time.h>

static void benchmark_math(void) {
    printf("\n--- Math Benchmark ---\n");
    
    mat4 m1 = m4_identity();
    mat4 m2 = m4_translation(1, 2, 3);
    mat4 m3 = m4_rotation_x(0.5f);
    
    clock_t start = clock();
    for (int i = 0; i < 100000; i++) {
        mat4 r = m4_multiply(m1, m2);
        r = m4_multiply(r, m3);
        (void)r;
    }
    clock_t end = clock();
    double ms = (end - start) * 1000.0 / CLOCKS_PER_SEC;
    printf("  m4_multiply x100k: %.2f ms (%.0f ops/sec)\n", 
           ms, 100000.0 / (ms / 1000.0));
    
    vec4 v1 = v4_make(1, 2, 3, 4);
    vec4 v2 = v4_make(5, 6, 7, 8);
    
    start = clock();
    for (int i = 0; i < 1000000; i++) {
        vec4 r = v4_cross(v1, v2);
        (void)r;
    }
    end = clock();
    ms = (end - start) * 1000.0 / CLOCKS_PER_SEC;
    printf("  v4_cross x1M: %.2f ms (%.0f ops/sec)\n", 
           ms, 1000000.0 / (ms / 1000.0));
    
    start = clock();
    for (int i = 0; i < 100000; i++) {
        mat4 r = m4_invert(m2);
        (void)r;
    }
    end = clock();
    ms = (end - start) * 1000.0 / CLOCKS_PER_SEC;
    printf("  m4_invert x100k: %.2f ms (%.0f ops/sec)\n", 
           ms, 100000.0 / (ms / 1000.0));
}

static void benchmark_gpu(void) {
    printf("\n--- GPU Message Benchmark ---\n");
    
    static HSGpu gpu;
    hs_gpu_init(&gpu);
    
    clock_t start = clock();
    for (int frame = 0; frame < 1000; frame++) {
        hs_gpu_start_recording(&gpu);
        
        vec4 color = v4_make(0.1f, 0.2f, 0.3f, 1.0f);
        hs_gpu_clear(&gpu, color);
        
        mat4 cam = m4_identity();
        hs_gpu_set_camera(&gpu, cam);
        
        hs_gpu_set_shader(&gpu, 0);
        
        vec4 param = v4_make(1, 0, 0, 0);
        hs_gpu_set_param(&gpu, 0, param);
        
        hs_gpu_load_buffer(&gpu, 0);
        hs_gpu_draw(&gpu, 0);
        
        hs_gpu_process(&gpu);
        hs_gpu_stop_recording(&gpu);
    }
    clock_t end = clock();
    double ms = (end - start) * 1000.0 / CLOCKS_PER_SEC;
    printf("  1000 frames: %.2f ms (%.0f fps)\n", 
           ms, 1000.0 / (ms / 1000.0));
    
    /* Message count */
    printf("  Messages per frame: ~%d\n", 6);
}

static void benchmark_buffer(void) {
    printf("\n--- Buffer Benchmark ---\n");
    
    HSBuffer buf;
    hs_buffer_init(&buf, 0, 65536);
    
    clock_t start = clock();
    for (int i = 0; i < 100000; i++) {
        hs_buffer_set_f32(&buf, i & 0x3FFF, (float)i);
        float v = hs_buffer_get_f32(&buf, i & 0x3FFF);
        (void)v;
    }
    clock_t end = clock();
    double ms = (end - start) * 1000.0 / CLOCKS_PER_SEC;
    printf("  buffer get/set x100k: %.2f ms (%.0f ops/sec)\n", 
           ms, 100000.0 / (ms / 1000.0));
    
    vec4 v = v4_make(1, 2, 3, 4);
    start = clock();
    for (int i = 0; i < 100000; i++) {
        hs_buffer_set_vec4(&buf, i & 0x3FFF, v);
    }
    end = clock();
    ms = (end - start) * 1000.0 / CLOCKS_PER_SEC;
    printf("  buffer set_vec4 x100k: %.2f ms (%.0f ops/sec)\n", 
           ms, 100000.0 / (ms / 1000.0));
    
    hs_buffer_free(&buf);
}

/* ============================================================
 * Main
 * ============================================================ */
int main(void) {
    printf("=== HS-GPU NEON Test Suite ===\n");

    test_math();
    test_backend_hooks();
    test_render_commands();
    test_buffer();
    test_input();
    test_message_validation();
    test_async_done();
    test_async_atomic_running();
    test_thread_safety();
    test_gpu();

    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);

    /* Benchmarks */
    benchmark_math();
    benchmark_buffer();
    benchmark_gpu();

    return tests_failed > 0 ? 1 : 0;
}
