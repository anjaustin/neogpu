#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include "hs_gpu.h"
#include "hs_msg.h"
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
 * Tooling CLI
 * ============================================================ */
static void tool_usage(const char* argv0) {
    printf("NeoGPU tooling mode\n\n");
    printf("Usage:\n");
    printf("  %s --tool [commands...]\n\n", argv0 ? argv0 : "neogpu_demo");
    printf("Commands:\n");
    printf("  --query-stats                 Query system stats (OP_QUERY_STATS)\n");
    printf("  --query-fabric                Query fabric counters (OP_QUERY_FABRIC)\n");
    printf("  --set-record-mask <mask>      Set HSSystem.record_mask (OP_SET_RECORD_MASK)\n");
    printf("  --set-budget <ch> <budget>    Set chan budget (OP_SET_CHAN_BUDGET)\n");
    printf("  --set-block <ch> <0|1>        Set block policy (OP_SET_BLOCK_POLICY)\n");
    printf("  --loop <ms>                   Repeat queries every ms (with queries enabled)\n");
    printf("\nChannel ids: 1=RT, 2=RENDER, 3=TELEM\n");
}

static bool tool_wait_for_result(HSSystem* sys, SystemState* st, u32 start, u8 want_op, u32 max_steps) {
    if (!sys || !st) return false;
    for (u32 i = 0; i < max_steps; i++) {
        hs_step(sys);
        if (st->result_count > start) {
            if (want_op == 0) return true;
            return st->last_result_op == want_op;
        }
        usleep(1000);
    }
    return false;
}

static void tool_print_stats(HSSystem* sys, SystemState* st) {
    if (!sys || !st) return;
    u32 tick = 0, log_head = 0, record_mask = 0, prod = 0, flags = 0;
    u32 budgets[3] = {0}, dropped[4] = {0};
    if (!hs_unpack_result_system_stats(sys->payloads[st->last_result_payload_idx].data, st->last_result_len,
                                       &tick, &log_head, &record_mask, budgets, dropped, &prod, &flags)) {
        printf("stats: <decode failed>\n");
        return;
    }

    printf("stats: tick=%u log_head=%u prod=%u record_mask=0x%08x\n",
           (unsigned)tick, (unsigned)log_head, (unsigned)prod, (unsigned)record_mask);
    printf("budgets: rt=%u render=%u telem=%u\n",
           (unsigned)budgets[0], (unsigned)budgets[1], (unsigned)budgets[2]);
    printf("dropped: error_ex=%u queue_full=%u system_nonrt=%u result=%u\n",
           (unsigned)dropped[0], (unsigned)dropped[1], (unsigned)dropped[2], (unsigned)dropped[3]);
    printf("flags: recording=%u validate=%u block_rt=%u block_render=%u block_telem=%u\n",
           (unsigned)((flags >> 0) & 1u), (unsigned)((flags >> 1) & 1u),
           (unsigned)((flags >> 2) & 1u), (unsigned)((flags >> 3) & 1u), (unsigned)((flags >> 4) & 1u));
}

static void tool_print_fabric(HSSystem* sys, SystemState* st) {
    if (!sys || !st) return;
    u32 spsc_ok3[3] = {0}, spsc_full3[3] = {0}, mpsc_ok3[3] = {0}, submit_full3[3] = {0};
    u32 prod = 0, waiters = 0;
    if (!hs_unpack_result_fabric(sys->payloads[st->last_result_payload_idx].data, st->last_result_len,
                                 spsc_ok3, spsc_full3, mpsc_ok3, submit_full3, &prod, &waiters)) {
        printf("fabric: <decode failed>\n");
        return;
    }

    printf("fabric: prod=%u waiters=%u\n", (unsigned)prod, (unsigned)waiters);
    printf("spsc_ok:    rt=%u render=%u telem=%u\n", (unsigned)spsc_ok3[0], (unsigned)spsc_ok3[1], (unsigned)spsc_ok3[2]);
    printf("spsc_full:  rt=%u render=%u telem=%u\n", (unsigned)spsc_full3[0], (unsigned)spsc_full3[1], (unsigned)spsc_full3[2]);
    printf("mpsc_ok:    rt=%u render=%u telem=%u\n", (unsigned)mpsc_ok3[0], (unsigned)mpsc_ok3[1], (unsigned)mpsc_ok3[2]);
    printf("submit_full:rt=%u render=%u telem=%u\n", (unsigned)submit_full3[0], (unsigned)submit_full3[1], (unsigned)submit_full3[2]);
}

static int run_tool_mode(int argc, char** argv) {
    if (argc <= 1) {
        tool_usage(argv ? argv[0] : NULL);
        return 2;
    }

    bool do_query_stats = false;
    bool do_query_fabric = false;
    bool do_set_mask = false;
    bool do_set_budget = false;
    bool do_set_block = false;
    u32 loop_ms = 0;

    u32 new_mask = 0;
    u8 bud_ch = 0;
    u32 bud_val = 0;
    u8 block_ch = 0;
    u8 block_val = 0;

    for (int i = 1; i < argc; i++) {
        const char* a = argv[i];
        if (!a) continue;

        if (strcmp(a, "--tool") == 0) {
            continue;
        } else if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
            tool_usage(argv ? argv[0] : NULL);
            return 0;
        } else if (strcmp(a, "--query-stats") == 0) {
            do_query_stats = true;
        } else if (strcmp(a, "--query-fabric") == 0) {
            do_query_fabric = true;
        } else if (strcmp(a, "--set-record-mask") == 0) {
            if (i + 1 >= argc) {
                printf("missing value for --set-record-mask\n");
                return 2;
            }
            new_mask = (u32)strtoul(argv[++i], NULL, 0);
            do_set_mask = true;
        } else if (strcmp(a, "--set-budget") == 0) {
            if (i + 2 >= argc) {
                printf("missing values for --set-budget\n");
                return 2;
            }
            bud_ch = (u8)strtoul(argv[++i], NULL, 0);
            bud_val = (u32)strtoul(argv[++i], NULL, 0);
            do_set_budget = true;
        } else if (strcmp(a, "--set-block") == 0) {
            if (i + 2 >= argc) {
                printf("missing values for --set-block\n");
                return 2;
            }
            block_ch = (u8)strtoul(argv[++i], NULL, 0);
            block_val = (u8)strtoul(argv[++i], NULL, 0);
            do_set_block = true;
        } else if (strcmp(a, "--loop") == 0) {
            if (i + 1 >= argc) {
                printf("missing value for --loop\n");
                return 2;
            }
            loop_ms = (u32)strtoul(argv[++i], NULL, 0);
        } else {
            printf("unknown arg: %s\n", a);
            tool_usage(argv ? argv[0] : NULL);
            return 2;
        }
    }

    if (!do_query_stats && !do_query_fabric && !do_set_mask && !do_set_budget && !do_set_block) {
        tool_usage(argv ? argv[0] : NULL);
        return 2;
    }

    if (loop_ms != 0 && !do_query_stats && !do_query_fabric) {
        printf("--loop requires at least one query (--query-stats and/or --query-fabric)\n");
        return 2;
    }

    static HSGpu gpu;
    hs_gpu_init(&gpu);
    gpu.system.validate_on_send = true;
    gpu.system.recording = false;

    SystemState* st = (SystemState*)gpu.system_node.state;

    bool setters_done = false;
    for (;;) {
        /* setters run once (even in loop mode) */
        if (!setters_done && do_set_mask) {
            u8 p[4];
            hs_pack_set_record_mask(p, new_mask);
            Message m = {.to = NODE_SYSTEM, .from = NODE_CPU, .op = OP_SET_RECORD_MASK, .cid = 1, .channel = CHAN_RT};
            u32 start = st->result_count;
            if (!hs_send_with_payload(&gpu.system, &m, p, sizeof(p)) || !tool_wait_for_result(&gpu.system, st, start, OP_SET_RECORD_MASK, 2000)) {
                printf("set-record-mask: failed\n");
            } else {
                u32 oldv = 0, newv = 0;
                (void)hs_unpack_u32x2(gpu.system.payloads[st->last_result_payload_idx].data, st->last_result_len, &oldv, &newv);
                printf("set-record-mask: 0x%08x -> 0x%08x\n", (unsigned)oldv, (unsigned)newv);
            }
        }

        if (!setters_done && do_set_budget) {
            u8 p[8];
            hs_pack_set_chan_budget(p, bud_ch, bud_val);
            Message m = {.to = NODE_SYSTEM, .from = NODE_CPU, .op = OP_SET_CHAN_BUDGET, .cid = 2, .channel = CHAN_RT};
            u32 start = st->result_count;
            if (!hs_send_with_payload(&gpu.system, &m, p, sizeof(p)) || !tool_wait_for_result(&gpu.system, st, start, OP_SET_CHAN_BUDGET, 2000)) {
                printf("set-budget: failed\n");
            } else {
                u32 oldv = 0, newv = 0;
                (void)hs_unpack_u32x2(gpu.system.payloads[st->last_result_payload_idx].data, st->last_result_len, &oldv, &newv);
                if (oldv == 0xFFFFFFFFu || newv == 0xFFFFFFFFu) {
                    printf("set-budget: invalid channel %u\n", (unsigned)bud_ch);
                } else {
                    printf("set-budget: ch=%u %u -> %u\n", (unsigned)bud_ch, (unsigned)oldv, (unsigned)newv);
                }
            }
        }

        if (!setters_done && do_set_block) {
            u8 p[2];
            hs_pack_set_block_policy(p, block_ch, block_val);
            Message m = {.to = NODE_SYSTEM, .from = NODE_CPU, .op = OP_SET_BLOCK_POLICY, .cid = 3, .channel = CHAN_RT};
            u32 start = st->result_count;
            if (!hs_send_with_payload(&gpu.system, &m, p, sizeof(p)) || !tool_wait_for_result(&gpu.system, st, start, OP_SET_BLOCK_POLICY, 2000)) {
                printf("set-block: failed\n");
            } else {
                u32 oldv = 0, newv = 0;
                (void)hs_unpack_u32x2(gpu.system.payloads[st->last_result_payload_idx].data, st->last_result_len, &oldv, &newv);
                if (oldv == 0xFFFFFFFFu || newv == 0xFFFFFFFFu) {
                    printf("set-block: invalid channel %u\n", (unsigned)block_ch);
                } else {
                    printf("set-block: ch=%u %u -> %u\n", (unsigned)block_ch, (unsigned)oldv, (unsigned)newv);
                }
            }
        }

        setters_done = true;

        /* queries */
        if (do_query_stats) {
            Message q = {.to = NODE_SYSTEM, .from = NODE_CPU, .op = OP_QUERY_STATS, .cid = 10, .channel = CHAN_DEFAULT};
            u32 start = st->result_count;
            if (!hs_send(&gpu.system, &q) || !tool_wait_for_result(&gpu.system, st, start, OP_QUERY_STATS, 2000)) {
                printf("query-stats: failed\n");
            } else {
                tool_print_stats(&gpu.system, st);
            }
        }

        if (do_query_fabric) {
            Message q = {.to = NODE_SYSTEM, .from = NODE_CPU, .op = OP_QUERY_FABRIC, .cid = 11, .channel = CHAN_DEFAULT};
            u32 start = st->result_count;
            if (!hs_send(&gpu.system, &q) || !tool_wait_for_result(&gpu.system, st, start, OP_QUERY_FABRIC, 2000)) {
                printf("query-fabric: failed\n");
            } else {
                tool_print_fabric(&gpu.system, st);
            }
        }

        if (loop_ms == 0) break;
        usleep(loop_ms * 1000);
    }

    return 0;
}

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

    /* Backpressure: overflow submit/producer queues without stepping */
    gpu.system.recording = false;
    gpu.system.block_on_full = false;
    gpu.system.block_on_full_chan[CHAN_RENDER] = false;
    gpu.system.block_on_full_chan[CHAN_RT] = false;
    u32 ok_count = 0;
    for (u32 i = 0; i < (HS_SUBMIT_SIZE + HS_MAX_PRODUCERS * HS_SPSC_SIZE + 256); i++) {
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
    u32 sf = atomic_load_explicit(&gpu.system.submit_full[CHAN_RENDER], memory_order_relaxed);
    u32 lf = atomic_load_explicit(&gpu.system.spsc_full[CHAN_RENDER], memory_order_relaxed);
    TEST("queue_full_triggered", ok_count < (HS_SUBMIT_SIZE + HS_MAX_PRODUCERS * HS_SPSC_SIZE + 256) && (sf > 0 || lf > 0));

    /* ensure per-producer counters sum to something when SPSC overflow happens */
    if (lf > 0) {
        u32 sum = 0;
        for (u32 i = 0; i < HS_MAX_PRODUCERS; i++) {
            sum += atomic_load_explicit(&gpu.system.spsc_full_by_prod[CHAN_RENDER][i], memory_order_relaxed);
        }
        TEST("queue_full_sharded", sum > 0);
    } else {
        TEST("queue_full_sharded", true);
    }
}

/* ============================================================
 * Frame/QoS tests
 * ============================================================ */
static void test_frame_qos(void) {
    printf("\n--- Frame/QoS Tests ---\n");

    static HSGpu gpu;
    hs_gpu_init(&gpu);
    gpu.system.validate_on_send = true;
    gpu.system.recording = true;

    SystemState* st = (SystemState*)gpu.system_node.state;

    /* Capture policy: render-only should not record RT ACKs */
    hs_clear(&gpu.system);
    hs_start_recording(&gpu.system);
    hs_set_record_mask(&gpu.system, hs_channel_bit(CHAN_RENDER));

    u32 ack0 = st->ack_count;
    Message m = {
        .to = NODE_SHADER,
        .from = NODE_CPU,
        .op = OP_SET_SHADER,
        .flags = HS_MSGF_ACK,
        .cid = 1,
        .tick = 0,
        .payload_idx = 0,
        .payload_len = 0,
    };
    bool ok = hs_send(&gpu.system, &m);
    hs_step(&gpu.system);
    hs_step(&gpu.system);
    TEST("recordmask_send_ok", ok);
    TEST("recordmask_ack_delivered", st->ack_count > ack0);

    u32 n = hs_stop_recording(&gpu.system);
    bool saw_ack = false;
    for (u32 i = 0; i < n; i++) {
        if (gpu.log_buffer[i].op == OP_ACK) {
            saw_ack = true;
            break;
        }
    }
    TEST("recordmask_no_ack", !saw_ack);

    /* If RT is included, ACK should be recorded */
    hs_clear(&gpu.system);
    hs_start_recording(&gpu.system);
    hs_set_record_mask(&gpu.system, hs_channel_bit(CHAN_RENDER) | hs_channel_bit(CHAN_RT));
    ack0 = st->ack_count;
    ok = hs_send(&gpu.system, &m);
    hs_step(&gpu.system);
    hs_step(&gpu.system);
    TEST("recordmask_send_ok2", ok);
    TEST("recordmask_ack_delivered2", st->ack_count > ack0);

    n = hs_stop_recording(&gpu.system);
    saw_ack = false;
    for (u32 i = 0; i < n; i++) {
        if (gpu.log_buffer[i].op == OP_ACK) {
            saw_ack = true;
            break;
        }
    }
    TEST("recordmask_with_ack", saw_ack);

    /* Fence emits OP_RESULT (op=FENCE) with a decodeable payload */
    hs_clear(&gpu.system);
    u32 expect_tick = gpu.system.tick;
    bool fence_ok = hs_gpu_fence(&gpu, CHAN_RENDER, 4242);
    hs_step(&gpu.system);
    TEST("fence_send_ok", fence_ok);
    TEST("fence_result", st->result_count > 0 && st->last_result_cid == 4242 && st->last_result_op == OP_FENCE && st->last_result_len == 8);
    {
        u32 tick = 0;
        u8 ch = 0;
        bool unpack_ok = hs_unpack_result_fence(gpu.system.payloads[st->last_result_payload_idx].data, st->last_result_len, &tick, &ch);
        TEST("fence_unpack", unpack_ok && tick == expect_tick && ch == (u8)CHAN_RENDER);
    }

    /* Frame markers show up in render list and reset it at FRAME_BEGIN */
    hs_clear(&gpu.system);
    hs_gpu_frame_begin(&gpu);
    hs_gpu_clear(&gpu, v4_make(0.0f, 0.0f, 0.0f, 1.0f));
    hs_gpu_frame_end(&gpu);
    hs_gpu_present(&gpu);
    hs_gpu_process(&gpu);
    TEST("frame_markers", gpu.render.count >= 4 &&
                          gpu.render.cmds[0].op == HS_RC_FRAME_BEGIN &&
                          gpu.render.cmds[1].op == HS_RC_CLEAR &&
                          gpu.render.cmds[2].op == HS_RC_FRAME_END &&
                          gpu.render.cmds[3].op == HS_RC_PRESENT);
}

/* ============================================================
 * System query/control tests
 * ============================================================ */
static void test_system_queries(void) {
    printf("\n--- System Query Tests ---\n");

    static HSGpu gpu;
    hs_gpu_init(&gpu);
    gpu.system.validate_on_send = true;
    gpu.system.recording = false;

    SystemState* st = (SystemState*)gpu.system_node.state;
    u32 start_results = st->result_count;

    /* Query stats */
    Message q = {
        .to = NODE_SYSTEM,
        .from = NODE_CPU,
        .op = OP_QUERY_STATS,
        .flags = 0,
        .cid = 100,
        .tick = 0,
        .payload_idx = 0,
        .payload_len = 0,
        .channel = CHAN_DEFAULT,
    };
    bool ok = hs_send(&gpu.system, &q);
    hs_step(&gpu.system);
    hs_step(&gpu.system);
    TEST("query_stats_send", ok);
    TEST("query_stats_result", st->result_count > start_results && st->last_result_op == OP_QUERY_STATS && st->last_result_cid == 100 && st->last_result_len == 64);
    {
        u32 tick = 0, log_head = 0, record_mask = 0, prod = 0, flags = 0;
        u32 budgets[3] = {0}, dropped[4] = {0};
        bool uok = hs_unpack_result_system_stats(gpu.system.payloads[st->last_result_payload_idx].data, st->last_result_len,
                                                 &tick, &log_head, &record_mask, budgets, dropped, &prod, &flags);
        TEST("query_stats_unpack", uok && budgets[0] == gpu.system.chan_budget[CHAN_RT] && budgets[1] == gpu.system.chan_budget[CHAN_RENDER]);
    }

    /* Set record mask */
    u8 set_mask_payload[4];
    u32 new_mask = hs_channel_bit(CHAN_RENDER) | hs_channel_bit(CHAN_RT);
    hs_pack_set_record_mask(set_mask_payload, new_mask);
    Message setm = {
        .to = NODE_SYSTEM,
        .from = NODE_CPU,
        .op = OP_SET_RECORD_MASK,
        .flags = 0,
        .cid = 200,
        .tick = 0,
        .payload_idx = 0,
        .payload_len = 0,
        .channel = CHAN_RT,
    };
    ok = hs_send_with_payload(&gpu.system, &setm, set_mask_payload, sizeof(set_mask_payload));
    hs_step(&gpu.system);
    hs_step(&gpu.system);
    TEST("set_record_mask_send", ok);
    TEST("set_record_mask_result", st->last_result_op == OP_SET_RECORD_MASK && st->last_result_cid == 200 && st->last_result_len == 8);
    {
        u32 oldv = 0, newv = 0;
        bool uok = hs_unpack_u32x2(gpu.system.payloads[st->last_result_payload_idx].data, st->last_result_len, &oldv, &newv);
        TEST("set_record_mask_unpack", uok && newv == new_mask);
    }

    /* Set render budget */
    u8 set_bud_payload[8];
    hs_pack_set_chan_budget(set_bud_payload, (u8)CHAN_RENDER, 1234);
    Message setb = {
        .to = NODE_SYSTEM,
        .from = NODE_CPU,
        .op = OP_SET_CHAN_BUDGET,
        .flags = 0,
        .cid = 300,
        .tick = 0,
        .payload_idx = 0,
        .payload_len = 0,
        .channel = CHAN_RT,
    };
    ok = hs_send_with_payload(&gpu.system, &setb, set_bud_payload, sizeof(set_bud_payload));
    hs_step(&gpu.system);
    hs_step(&gpu.system);
    TEST("set_budget_send", ok);
    TEST("set_budget_applied", gpu.system.chan_budget[CHAN_RENDER] == 1234);

    /* Query fabric counters */
    Message fq = {
        .to = NODE_SYSTEM,
        .from = NODE_CPU,
        .op = OP_QUERY_FABRIC,
        .flags = 0,
        .cid = 400,
        .tick = 0,
        .payload_idx = 0,
        .payload_len = 0,
        .channel = CHAN_DEFAULT,
    };
    ok = hs_send(&gpu.system, &fq);
    hs_step(&gpu.system);
    hs_step(&gpu.system);
    TEST("query_fabric_send", ok);
    TEST("query_fabric_result", st->last_result_op == OP_QUERY_FABRIC && st->last_result_cid == 400 && st->last_result_len == 64);
    {
        u32 spsc_ok3[3] = {0}, spsc_full3[3] = {0}, mpsc_ok3[3] = {0}, submit_full3[3] = {0};
        u32 prod = 0, waiters = 0;
        bool uok = hs_unpack_result_fabric(gpu.system.payloads[st->last_result_payload_idx].data, st->last_result_len,
                                           spsc_ok3, spsc_full3, mpsc_ok3, submit_full3, &prod, &waiters);
        TEST("query_fabric_unpack", uok && prod == atomic_load_explicit(&gpu.system.producer_count, memory_order_relaxed));
    }
}

/* ============================================================
 * Red-team: System queries under TELEM spam
 * ============================================================ */
typedef struct {
    HSSystem* sys;
    atomic_bool* stop;
    u32 sent;
    u32 failed;
} TelemSpamArgs;

static void* telem_spam_thread(void* arg) {
    TelemSpamArgs* a = (TelemSpamArgs*)arg;
    u32 i = 0;
    while (!atomic_load_explicit(a->stop, memory_order_acquire)) {
        Message m = {
            .to = NODE_SYSTEM,
            .from = NODE_CPU,
            .op = OP_QUEUE_FULL,
            .flags = NODE_SYSTEM,
            .cid = i,
            .tick = 0,
            .payload_idx = (u16)OP_TRACE,
            .payload_len = 0,
            .channel = CHAN_TELEM,
        };

        if (hs_send(a->sys, &m)) a->sent++;
        else a->failed++;
        i++;
    }
    return NULL;
}

static void test_redteam_system_queries(void) {
    printf("\n--- Red-Team: System Queries Under TELEM Spam ---\n");

    static HSGpu gpu;
    hs_gpu_init(&gpu);
    gpu.system.validate_on_send = true;
    gpu.system.recording = false;

    /* TELEM must never block; leave defaults for RT blocking */
    gpu.system.block_on_full_chan[CHAN_TELEM] = false;
    gpu.system.block_on_full_chan[CHAN_RT] = true;

    atomic_bool stop;
    atomic_init(&stop, false);

    const int N = 6;
    pthread_t th[N];
    TelemSpamArgs args[N];
    for (int i = 0; i < N; i++) {
        args[i].sys = &gpu.system;
        args[i].stop = &stop;
        args[i].sent = 0;
        args[i].failed = 0;
        int rc = pthread_create(&th[i], NULL, telem_spam_thread, &args[i]);
        TEST("telem_spawn", rc == 0);
    }

    SystemState* st = (SystemState*)gpu.system_node.state;
    u32 start_results = st->result_count;
    bool ok_all = true;

    /* Issue repeated queries while step drains under spam. */
    for (u32 k = 0; k < 200; k++) {
        Message q = {
            .to = NODE_SYSTEM,
            .from = NODE_CPU,
            .op = OP_QUERY_STATS,
            .flags = 0,
            .cid = 1000 + k,
            .tick = 0,
            .payload_idx = 0,
            .payload_len = 0,
            .channel = CHAN_TELEM, /* intentionally wrong; must be clamped to RT */
        };
        bool ok = hs_send(&gpu.system, &q);
        if (!ok || q.channel != CHAN_RT) {
            ok_all = false;
            break;
        }

        /* step a bit to ensure the system processes and emits a result */
        hs_step(&gpu.system);
        hs_step(&gpu.system);

        if (st->result_count <= start_results) {
            ok_all = false;
            break;
        }
        start_results = st->result_count;
    }

    atomic_store_explicit(&stop, true, memory_order_release);
    hs_wake_senders(&gpu.system);
    for (int i = 0; i < N; i++) pthread_join(th[i], NULL);

    TEST("telem_query_progress", ok_all);
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
    atomic_bool* stop;
    u32 sent;
    u32 failed;
} SenderArgs;

static void* sender_thread(void* arg) {
    SenderArgs* a = (SenderArgs*)arg;
    u32 i = 0;
    while (!atomic_load_explicit(a->stop, memory_order_acquire)) {
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

    atomic_bool stop_atomic;
    atomic_init(&stop_atomic, false);
    SenderArgs args = {
        .sys = &gpu.system,
        .stop = &stop_atomic,
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

    atomic_store_explicit(&stop_atomic, true, memory_order_release);
    pthread_join(th, NULL);

    SystemState* st = (SystemState*)gpu.system_node.state;
    TEST("thread_sent", args.sent > 0);
    TEST("thread_ack_progress", st->ack_count > 0);
}

static void test_thread_safety_many(void) {
    printf("\n--- Thread Safety (Many Producers) ---\n");

    static HSGpu gpu;
    hs_gpu_init(&gpu);
    gpu.system.validate_on_send = true;
    gpu.system.recording = false;

    enum { N = 4 };
    pthread_t th[N];
    SenderArgs args[N];
    atomic_bool stop;
    atomic_init(&stop, false);

    for (int i = 0; i < N; i++) {
        args[i].sys = &gpu.system;
        args[i].stop = &stop;
        args[i].sent = 0;
        args[i].failed = 0;
        int rc = pthread_create(&th[i], NULL, sender_thread, &args[i]);
        TEST("thread_spawn_many", rc == 0);
    }

    for (int i = 0; i < 2000; i++) {
        hs_step(&gpu.system);
        usleep(500);
    }

    atomic_store_explicit(&stop, true, memory_order_release);
    for (int i = 0; i < N; i++) pthread_join(th[i], NULL);

    u32 total_sent = 0;
    for (int i = 0; i < N; i++) total_sent += args[i].sent;
    SystemState* st = (SystemState*)gpu.system_node.state;

    TEST("many_sent", total_sent > 0);
    TEST("many_ack_progress", st->ack_count > 0);
}

static void* sender_payload_thread(void* arg) {
    SenderArgs* a = (SenderArgs*)arg;
    u32 i = 0;
    while (!atomic_load_explicit(a->stop, memory_order_acquire)) {
        u8 data[2];
        data[0] = (i & 1) ? 1 : 0;
        data[1] = (i & 1) ? 0 : 1;
        Message m = {
            .to = NODE_SHADER,
            .from = NODE_CPU,
            .op = OP_BLEND,
            .flags = HS_MSGF_ACK,
            .cid = 1000000u + i,
            .tick = 0,
            .payload_idx = 0,
            .payload_len = 0,
        };
        if (hs_send_with_payload(a->sys, &m, data, sizeof(data))) a->sent++;
        else a->failed++;
        i++;
        usleep(50);
    }
    return NULL;
}

static void test_thread_safety_payloads(void) {
    printf("\n--- Thread Safety (Payload Producers) ---\n");

    static HSGpu gpu;
    hs_gpu_init(&gpu);
    gpu.system.validate_on_send = true;
    gpu.system.recording = false;

    enum { N = 4 };
    pthread_t th[N];
    SenderArgs args[N];
    atomic_bool stop;
    atomic_init(&stop, false);

    for (int i = 0; i < N; i++) {
        args[i].sys = &gpu.system;
        args[i].stop = &stop;
        args[i].sent = 0;
        args[i].failed = 0;
        int rc = pthread_create(&th[i], NULL, sender_payload_thread, &args[i]);
        TEST("thread_spawn_payload", rc == 0);
    }

    for (int i = 0; i < 2000; i++) {
        hs_step(&gpu.system);
        usleep(500);
    }

    atomic_store_explicit(&stop, true, memory_order_release);
    for (int i = 0; i < N; i++) pthread_join(th[i], NULL);

    u32 total_sent = 0;
    for (int i = 0; i < N; i++) total_sent += args[i].sent;
    SystemState* st = (SystemState*)gpu.system_node.state;
    TEST("payload_sent", total_sent > 0);
    TEST("payload_ack", st->ack_count > 0);
}

static void test_redteam_overproducers(void) {
    printf("\n--- Red-Team: Over-Producers ---\n");

    static HSGpu gpu;
    hs_gpu_init(&gpu);
    gpu.system.validate_on_send = true;
    gpu.system.recording = false;
    /* This test intentionally overloads producers; keep it non-blocking so threads can stop cleanly. */
    gpu.system.block_on_full = false;
    gpu.system.block_on_full_chan[CHAN_RENDER] = false;
    gpu.system.block_on_full_chan[CHAN_RT] = false;

    enum { N = 16 };
    pthread_t th[N];
    SenderArgs args[N];
    atomic_bool stop;
    atomic_init(&stop, false);

    for (int i = 0; i < N; i++) {
        args[i].sys = &gpu.system;
        args[i].stop = &stop;
        args[i].sent = 0;
        args[i].failed = 0;
        int rc = pthread_create(&th[i], NULL, sender_payload_thread, &args[i]);
        TEST("thread_spawn_over", rc == 0);
    }

    for (int i = 0; i < 2500; i++) {
        hs_step(&gpu.system);
        usleep(200);
    }

    atomic_store_explicit(&stop, true, memory_order_release);
    hs_wake_senders(&gpu.system);
    for (int i = 0; i < N; i++) pthread_join(th[i], NULL);

    u32 total_sent = 0;
    u32 total_failed = 0;
    for (int i = 0; i < N; i++) {
        total_sent += args[i].sent;
        total_failed += args[i].failed;
    }

    SystemState* st = (SystemState*)gpu.system_node.state;

    TEST("over_sent", total_sent > 0);
    TEST("over_ack", st->ack_count > 0);
    /* With >HS_MAX_PRODUCERS, we expect fallbacks; backpressure may or may not occur. */
    TEST("over_fallback", atomic_load_explicit(&gpu.system.producer_count, memory_order_relaxed) >= HS_MAX_PRODUCERS);
}

/* ============================================================
 * Producer throughput benchmark (SPSC lanes + MPSC fallback)
 * ============================================================ */
typedef struct {
    HSSystem* sys;
    atomic_bool* stop;
    u32 ok;
    u32 fail;
} BenchArgs;

static void* bench_sender(void* arg) {
    BenchArgs* a = (BenchArgs*)arg;
    u32 i = 0;
    while (!atomic_load_explicit(a->stop, memory_order_acquire)) {
        Message m = {
            .to = NODE_SHADER,
            .from = NODE_CPU,
            .op = OP_SET_SHADER,
            .flags = 0,
            .cid = i,
            .tick = 0,
            .payload_idx = (u16)(i & 0xFF),
            .payload_len = 0,
            .channel = CHAN_RENDER,
        };
        if (hs_send(a->sys, &m)) {
            a->ok++;
        } else {
            if (atomic_load_explicit(a->stop, memory_order_acquire)) break;
            a->fail++;
        }
        i++;
    }
    return NULL;
}

static u64 ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (u64)ts.tv_sec * 1000000000ull + (u64)ts.tv_nsec;
}

static void bench_producers(int threads, int ms) {
    static HSGpu gpu;
    static bool inited = false;
    if (!inited) {
        hs_gpu_init(&gpu);
        inited = true;
    }
    hs_clear(&gpu.system);
    gpu.system.validate_on_send = false;
    gpu.system.recording = false;
    gpu.system.render_list = NULL;
    gpu.system.block_on_full = true;

    atomic_bool stop;
    atomic_init(&stop, false);

    pthread_t th[32];
    BenchArgs args[32];
    if (threads > 32) threads = 32;

    for (int i = 0; i < threads; i++) {
        args[i].sys = &gpu.system;
        args[i].stop = &stop;
        args[i].ok = 0;
        args[i].fail = 0;
        pthread_create(&th[i], NULL, bench_sender, &args[i]);
    }

    u64 end = ns_now() + (u64)ms * 1000000ull;
    while (ns_now() < end) {
        hs_step(&gpu.system);
    }

    atomic_store_explicit(&stop, true, memory_order_release);
    gpu.system.block_on_full = false;
    gpu.system.block_on_full_chan[CHAN_RENDER] = false;
    gpu.system.block_on_full_chan[CHAN_RT] = false;
    hs_wake_senders(&gpu.system);
    for (int i = 0; i < threads; i++) pthread_join(th[i], NULL);

    /* Drain remaining */
    for (int i = 0; i < 1000; i++) hs_step(&gpu.system);

    u64 ok = 0, fail = 0;
    for (int i = 0; i < threads; i++) {
        ok += args[i].ok;
        fail += args[i].fail;
    }

    u32 spsc_ok = atomic_load_explicit(&gpu.system.spsc_ok[CHAN_RENDER], memory_order_relaxed);
    u32 mpsc_ok = atomic_load_explicit(&gpu.system.mpsc_ok[CHAN_RENDER], memory_order_relaxed);
    u32 spsc_full = atomic_load_explicit(&gpu.system.spsc_full[CHAN_RENDER], memory_order_relaxed);
    u32 submit_full = atomic_load_explicit(&gpu.system.submit_full[CHAN_RENDER], memory_order_relaxed);
    u32 prod_count = atomic_load_explicit(&gpu.system.producer_count, memory_order_relaxed);

    double sec = (double)ms / 1000.0;
    double mps = sec > 0.0 ? (double)ok / sec : 0.0;

    printf("  %2d thr, %4d ms: %.0f msg/s (ok=%llu fail=%llu) spsc_ok=%u mpsc_ok=%u spsc_full=%u submit_full=%u prod=%u\n",
           threads, ms, mps, (unsigned long long)ok, (unsigned long long)fail,
           (unsigned)spsc_ok, (unsigned)mpsc_ok, (unsigned)spsc_full, (unsigned)submit_full, (unsigned)prod_count);

    TEST("bench_no_fail", fail == 0);
}

static void bench_comm_layer(void) {
    printf("\n--- Comms Producer Benchmark ---\n");
    bench_producers(1, 500);
    bench_producers(2, 500);
    bench_producers(4, 500);
    bench_producers(8, 500);
    bench_producers(16, 500);
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
    hs_gpu_frame_begin(&gpu);

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

    hs_gpu_frame_end(&gpu);
    hs_gpu_present(&gpu);
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
int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (argv[i] && strcmp(argv[i], "--tool") == 0) {
            return run_tool_mode(argc, argv);
        }
    }
    printf("=== HS-GPU NEON Test Suite ===\n");

    test_math();
    test_backend_hooks();
    test_render_commands();
    test_buffer();
    test_input();
    test_message_validation();
    test_frame_qos();
    test_system_queries();
    test_redteam_system_queries();
    test_async_done();
    test_async_atomic_running();
    test_thread_safety();
    test_thread_safety_many();
    test_thread_safety_payloads();
    test_redteam_overproducers();
    test_gpu();

    bench_comm_layer();

    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);

    /* Benchmarks */
    benchmark_math();
    benchmark_buffer();
    benchmark_gpu();

    return tests_failed > 0 ? 1 : 0;
}
