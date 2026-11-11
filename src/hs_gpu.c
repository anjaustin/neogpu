#include "hs_gpu.h"
#include "hs_msg.h"
#include <stdio.h>
#include <string.h>
#include <stddef.h>

_Static_assert(offsetof(HSGpu, participants) % 64 == 0, "Embedded fabric must be cache-line aligned");
_Static_assert(offsetof(HSGpu, participants_ready) % 64 == 0 &&
               offsetof(HSGpu, participants) - offsetof(HSGpu, participants_ready) >= 64,
               "Producer readiness must have a separate immutable cache line");

void hs_gpu_init(HSGpu* gpu) {
    memset(gpu, 0, sizeof(*gpu));
    atomic_init(&gpu->participant_count, 0);
    atomic_init(&gpu->participant_limit, 0);
    gpu->participants_ready = hs_fabric_init(&gpu->participants);
    if (!gpu->participants_ready) fprintf(stderr, "Participant fabric initialization failed\n");
    hs_init(&gpu->system, gpu->log_buffer, HS_MAX_MSG_LOG, gpu->payload_buffer);
    gpu->system.render_list = &gpu->render;

    hs_memory_init(&gpu->memory);

    gpu->shader_node.id = NODE_SHADER;
    gpu->buffer_node.id = NODE_BUFFER;
    gpu->texture_node.id = NODE_TEXTURE;
    gpu->output_node.id = NODE_OUTPUT;
    gpu->sound_node.id = NODE_SOUND;
    gpu->system_node.id = NODE_SYSTEM;

    gpu->shader_node.sys = &gpu->system;
    gpu->buffer_node.sys = &gpu->system;
    gpu->texture_node.sys = &gpu->system;
    gpu->output_node.sys = &gpu->system;
    gpu->sound_node.sys = &gpu->system;
    gpu->system_node.sys = &gpu->system;

    gpu->shader_node.state = &gpu->shader_state;
    gpu->buffer_node.state = &gpu->buffer_state;
    gpu->texture_node.state = &gpu->texture_state;
    gpu->output_node.state = &gpu->output_state;
    gpu->sound_node.state = &gpu->sound_state;
    gpu->system_node.state = &gpu->system_state_data;

    shader_node_init(&gpu->shader_node);
    buffer_node_init(&gpu->buffer_node);
    texture_node_init(&gpu->texture_node);
    output_node_init(&gpu->output_node);
    sound_node_init(&gpu->sound_node);
    system_node_init(&gpu->system_node);
    
    hs_register(&gpu->system, &gpu->shader_node);
    hs_register(&gpu->system, &gpu->buffer_node);
    hs_register(&gpu->system, &gpu->texture_node);
    hs_register(&gpu->system, &gpu->output_node);
    hs_register(&gpu->system, &gpu->sound_node);
    hs_register(&gpu->system, &gpu->system_node);
}

void hs_gpu_shutdown(HSGpu* gpu) {
    if (!gpu) return;
    if (gpu->backend && gpu->backend->ops && gpu->backend->ops->shutdown)
        gpu->backend->ops->shutdown(gpu->backend->ctx, gpu);
    gpu->backend = NULL;
    if (gpu->participants_ready) hs_fabric_destroy(&gpu->participants);
    gpu->participants_ready = false;
    atomic_store_explicit(&gpu->participant_count, 0, memory_order_relaxed);
    atomic_store_explicit(&gpu->participant_limit, 0, memory_order_relaxed);
    hs_memory_free(&gpu->memory);
    hs_destroy(&gpu->system);
}

bool hs_gpu_participants_ready(const HSGpu* gpu) {
    return gpu && gpu->participants_ready;
}

HSFabricResult hs_gpu_participant_join(HSGpu* gpu, HSParticipant* out) {
    if (!hs_gpu_participants_ready(gpu)) return HS_FABRIC_INVALID;
    HSFabricResult result = hs_fabric_join(&gpu->participants, out);
    if (result == HS_FABRIC_OK) {
        u32 limit = out->slot + 1;
        u32 previous = atomic_load_explicit(&gpu->participant_limit, memory_order_relaxed);
        /* Strong-CAS failures imply another monotonic advance (at most eight).
         * Never reset this bound while callers are live: reuse may race retire. */
        while (previous < limit && !atomic_compare_exchange_strong_explicit(
                   &gpu->participant_limit, &previous, limit, memory_order_release, memory_order_relaxed)) { }
        atomic_fetch_add_explicit(&gpu->participant_count, 1, memory_order_release);
    }
    return result;
}

HSFabricResult hs_gpu_participant_submit(HSGpu* gpu, HSParticipant owner, const HSParticipantFrame* frame) {
    if (!hs_gpu_participants_ready(gpu)) return HS_FABRIC_INVALID;
    return hs_fabric_submit(&gpu->participants, owner, frame);
}

HSFabricResult hs_gpu_participant_leave(HSGpu* gpu, HSParticipant owner) {
    if (!hs_gpu_participants_ready(gpu)) return HS_FABRIC_INVALID;
    return hs_fabric_leave(&gpu->participants, owner);
}

static u32 hs_gpu_service_participants(HSGpu* gpu) {
    u64 retired = gpu->participants.retired;
    u32 events = hs_fabric_step(&gpu->participants, HS_FABRIC_PARTICIPANTS);
    u32 reclaimed = (u32)(gpu->participants.retired - retired);
    if (reclaimed) atomic_fetch_sub_explicit(&gpu->participant_count, reclaimed, memory_order_release);
    return events;
}

HSBuffer* hs_gpu_get_buffer(HSGpu* gpu, u8 bank) {
    return hs_memory_get_buffer(&gpu->memory, bank);
}

void hs_gpu_attach_backend(HSGpu* gpu, HSBackend* backend) {
    gpu->backend = backend;
    if (gpu->backend && gpu->backend->ops && gpu->backend->ops->init) {
        if (!gpu->backend->ops->init(gpu->backend->ctx, gpu)) {
            fprintf(stderr, "Backend initialization failed\n");
            if (gpu->backend->ops->shutdown) gpu->backend->ops->shutdown(gpu->backend->ctx, gpu);
            gpu->backend = NULL;
        }
    }
}

static void hs_gpu_build_frame(const HSGpu* gpu, HSFrameContext* out) {
    memset(out, 0, sizeof(*out));
    out->gpu = gpu;
    out->tick = gpu->system.tick;
    out->sys = &gpu->system;
    out->shader_state = gpu->shader_node.state;
    out->buffer_state = gpu->buffer_node.state;
    out->texture_state = gpu->texture_node.state;
    out->output_state = gpu->output_node.state;
    out->sound_state = gpu->sound_node.state;
    out->system_state = gpu->system_node.state;
    out->render = &gpu->render;
}

static void hs_gpu_send_simple(HSGpu* gpu, u8 to, OpCode op, u32 payload_idx) {
    Message msg = {
        .to = to,
        .from = NODE_CPU,
        .op = op,
        .flags = 0,
        .cid = 0,
        .tick = 0,
        .payload_idx = (u16)payload_idx,
        .payload_len = 0
    };
    hs_send_trusted(&gpu->system, &msg, NULL, 0);
}

/* For the helpers below: fixed destination, opcode and packed length, so the
 * message conforms by construction (verified in HS_VERIFY_TRUSTED builds). The
 * public hs_gpu_send_with_payload() takes arbitrary arguments and is gated. */
static bool hs_gpu_send_built(HSGpu* gpu, u8 to, OpCode op, const void* data, u32 len) {
    if (!gpu) return false;
    Message msg = {.to = to, .from = NODE_CPU, .op = op};
    return hs_send_trusted(&gpu->system, &msg, data, len);
}

bool hs_gpu_send_with_payload(HSGpu* gpu, u8 to, OpCode op, const void* data, u32 len) {
    if (!gpu || len > HS_PAYLOAD_SIZE || (len && !data)) return false;
    /* Payload bytes travel inline with the message and are placed in the
     * payload ring by the step thread, so any producer thread may call this. */
    Message msg = {
        .to = to,
        .from = NODE_CPU,
        .op = op,
        .flags = 0,
        .cid = 0,
        .tick = 0,
        .payload_idx = 0,
        .payload_len = 0
    };
    return hs_send_with_payload(&gpu->system, &msg, data, len);
}

void hs_gpu_set_shader(HSGpu* gpu, u8 shader) {
    hs_gpu_send_simple(gpu, NODE_SHADER, OP_SET_SHADER, shader);
}

void hs_gpu_set_param(HSGpu* gpu, u8 idx, vec4 value) {
    u8 data[20];
    hs_pack_set_param(data, (u32)idx, value);
    hs_gpu_send_built(gpu, NODE_SHADER, OP_SET_PARAM, data, sizeof(data));
}

void hs_gpu_set_global(HSGpu* gpu, u8 idx, mat4 value) {
    /* The global index rides in Message.flags; the 64-byte matrix is the payload. */
    float arr[16];
    m4_to_array(value, arr);

    Message msg = {
        .to = NODE_SHADER,
        .from = NODE_CPU,
        .op = OP_SET_GLOBAL,
        .flags = idx,
        .cid = 0,
        .tick = 0,
        .payload_idx = 0,
        .payload_len = 0
    };
    (void)hs_send_trusted(&gpu->system, &msg, arr, sizeof(arr));
}

void hs_gpu_set_camera(HSGpu* gpu, mat4 view) {
    float arr[16];
    m4_to_array(view, arr);
    hs_gpu_send_built(gpu, NODE_SHADER, OP_SET_CAMERA, arr, sizeof(arr));
}

void hs_gpu_cull(HSGpu* gpu, u8 mode) {
    hs_gpu_send_simple(gpu, NODE_SHADER, OP_CULL, mode);
}

void hs_gpu_blend(HSGpu* gpu, u8 src, u8 dst) {
    u8 data[2];
    hs_pack_u8x2(data, src, dst);
    hs_gpu_send_built(gpu, NODE_SHADER, OP_BLEND, data, sizeof(data));
}

void hs_gpu_alpha(HSGpu* gpu, bool enable) {
    hs_gpu_send_simple(gpu, NODE_SHADER, OP_ALPHA, enable ? 1 : 0);
}

void hs_gpu_depth(HSGpu* gpu, bool enable) {
    hs_gpu_send_simple(gpu, NODE_SHADER, OP_DEPTH, enable ? 1 : 0);
}

void hs_gpu_color_mask(HSGpu* gpu, u8 mask) {
    hs_gpu_send_simple(gpu, NODE_SHADER, OP_COLOR_MASK, mask);
}

void hs_gpu_clip(HSGpu* gpu, u16 x, u16 y, u16 w, u16 h) {
    u8 data[8];
    hs_pack_u16x4(data, x, y, w, h);
    hs_gpu_send_built(gpu, NODE_SHADER, OP_CLIP, data, sizeof(data));
}

void hs_gpu_stencil(HSGpu* gpu, u8 op, u8 fail, u8 pass, u8 front) {
    u8 data[4]; /* exactly the 4 packed bytes the schema requires */
    hs_pack_stencil(data, op, fail, pass, front);
    hs_gpu_send_built(gpu, NODE_SHADER, OP_STENCIL, data, sizeof(data));
}

void hs_gpu_stencil_func(HSGpu* gpu, u8 compare, u8 ref, u8 read_mask, u8 write_mask) {
    u8 data[4]; /* exactly the 4 packed bytes the schema requires */
    hs_pack_stencil_func(data, compare, ref, read_mask, write_mask);
    hs_gpu_send_built(gpu, NODE_SHADER, OP_STENCIL_FUNC, data, sizeof(data));
}

void hs_gpu_depth_compare(HSGpu* gpu, u8 compare, bool write) {
    u8 data[2];
    hs_pack_depth_compare(data, compare, write);
    hs_gpu_send_built(gpu, NODE_SHADER, OP_DEPTH_COMPARE, data, sizeof(data));
}

void hs_gpu_load_buffer(HSGpu* gpu, u8 index) {
    hs_gpu_send_simple(gpu, NODE_BUFFER, OP_LOAD_BUFFER, index);
}

void hs_gpu_draw(HSGpu* gpu, u8 buffer) {
    hs_gpu_send_simple(gpu, NODE_BUFFER, OP_DRAW, buffer);
}

void hs_gpu_draw_instance(HSGpu* gpu, u8 buffer, u8 instance_buffer, u32 count) {
    u8 data[8];
    hs_pack_draw_instance(data, buffer, instance_buffer, count);
    hs_gpu_send_built(gpu, NODE_BUFFER, OP_DRAW_INSTANCE, data, sizeof(data));
}

void hs_gpu_draw_text(HSGpu* gpu, const char* text) {
    u32 len = 0;
    while (text[len] && len < HS_PAYLOAD_SIZE - 1) len++;
    u8 data[HS_PAYLOAD_SIZE];
    memcpy(data, text, len);
    data[len] = '\0';
    hs_gpu_send_built(gpu, NODE_BUFFER, OP_DRAW_TEXT, data, len + 1);
}

void hs_gpu_load_texture(HSGpu* gpu, u8 index) {
    hs_gpu_send_simple(gpu, NODE_TEXTURE, OP_LOAD_TEXTURE, index);
}

void hs_gpu_set_target(HSGpu* gpu, u8 texture, u8 depth_buffer) {
    u8 data[2];
    hs_pack_u8x2(data, texture, depth_buffer);
    hs_gpu_send_built(gpu, NODE_TEXTURE, OP_SET_TARGET, data, sizeof(data));
}

void hs_gpu_show_texture(HSGpu* gpu, u8 texture) {
    hs_gpu_send_simple(gpu, NODE_TEXTURE, OP_SHOW_TEXTURE, texture);
}

void hs_gpu_texture_filter(HSGpu* gpu, u8 texture, bool linear) {
    u8 data = texture | (linear ? 0x10 : 0);
    hs_gpu_send_simple(gpu, NODE_TEXTURE, OP_TEXTURE_FILTER, data);
}

void hs_gpu_texture_wrap(HSGpu* gpu, u8 texture, bool repeat) {
    u8 data = texture | (repeat ? 0x10 : 0);
    hs_gpu_send_simple(gpu, NODE_TEXTURE, OP_TEXTURE_WRAP, data);
}

void hs_gpu_clear(HSGpu* gpu, vec4 color) {
    hs_gpu_send_built(gpu, NODE_OUTPUT, OP_CLEAR, &color, sizeof(vec4));
}

void hs_gpu_clear_ds(HSGpu* gpu, f32 depth, u8 stencil) {
    u8 data[8];
    hs_pack_clear_ds(data, depth, stencil);
    hs_gpu_send_built(gpu, NODE_OUTPUT, OP_CLEAR_DS, data, sizeof(data));
}

void hs_gpu_set_channel(HSGpu* gpu, u8 channel, u8 shader) {
    u8 data[2];
    hs_pack_u8x2(data, channel, shader);
    hs_gpu_send_built(gpu, NODE_SOUND, OP_SET_CHANNEL, data, sizeof(data));
}

u32 hs_gpu_process(HSGpu* gpu) {
    u32 total = 0;
    for (;;) {
        u32 progress = hs_step(&gpu->system);
        if (atomic_load_explicit(&gpu->participant_count, memory_order_acquire))
            progress += hs_gpu_service_participants(gpu);
        if (!progress) break;
        total++;
        if (total > 1000) break;
        /* Skip the confirming empty step when nothing this step saw is left. */
        if (!gpu->system.step_pending &&
            !atomic_load_explicit(&gpu->participant_count, memory_order_acquire)) break;
    }
    return total;
}

u32 hs_gpu_step(HSGpu* gpu) {
    u32 processed = hs_step(&gpu->system);
    if (atomic_load_explicit(&gpu->participant_count, memory_order_acquire))
        processed += hs_gpu_service_participants(gpu);
    return processed;
}

void hs_gpu_start_recording(HSGpu* gpu) {
    hs_start_recording(&gpu->system);
}

u32 hs_gpu_stop_recording(HSGpu* gpu) {
    return hs_stop_recording(&gpu->system);
}

bool hs_gpu_replay(HSGpu* gpu, Message* msgs, u32 count) {
    return hs_replay(&gpu->system, msgs, count);
}

void hs_gpu_clear_log(HSGpu* gpu) {
    hs_clear(&gpu->system);
}

bool hs_gpu_has_overflow(HSGpu* gpu) {
    return hs_has_overflow(&gpu->system);
}

void hs_gpu_error(HSGpu* gpu, const char* msg) {
    hs_gpu_send_built(gpu, NODE_SYSTEM, OP_ERROR, (const u8*)msg, strlen(msg) + 1);
}

void hs_gpu_trace(HSGpu* gpu, const char* msg) {
    hs_gpu_send_built(gpu, NODE_SYSTEM, OP_TRACE, (const u8*)msg, strlen(msg) + 1);
}

void hs_gpu_stop(HSGpu* gpu) {
    hs_gpu_send_simple(gpu, NODE_SYSTEM, OP_STOP, 0);
}

void hs_gpu_frame_begin(HSGpu* gpu) {
    Message msg = {
        .to = NODE_SYSTEM,
        .from = NODE_CPU,
        .op = OP_FRAME_BEGIN,
        .flags = 0,
        .cid = 0,
        .tick = 0,
        .payload_idx = 0,
        .payload_len = 0,
        .channel = CHAN_RENDER,
    };
    (void)hs_send_trusted(&gpu->system, &msg, NULL, 0);
}

void hs_gpu_frame_end(HSGpu* gpu) {
    Message msg = {
        .to = NODE_SYSTEM,
        .from = NODE_CPU,
        .op = OP_FRAME_END,
        .flags = 0,
        .cid = 0,
        .tick = 0,
        .payload_idx = 0,
        .payload_len = 0,
        .channel = CHAN_RENDER,
    };
    (void)hs_send_trusted(&gpu->system, &msg, NULL, 0);
}

void hs_gpu_present(HSGpu* gpu) {
    Message msg = {
        .to = NODE_SYSTEM,
        .from = NODE_CPU,
        .op = OP_PRESENT,
        .flags = 0,
        .cid = 0,
        .tick = 0,
        .payload_idx = 0,
        .payload_len = 0,
        .channel = CHAN_RENDER,
    };
    (void)hs_send_trusted(&gpu->system, &msg, NULL, 0);
}

bool hs_gpu_fence(HSGpu* gpu, HSChannel target, u32 cid) {
    Message msg = {
        .to = NODE_SYSTEM,
        .from = NODE_CPU,
        .op = OP_FENCE,
        .flags = (u8)target,
        .cid = cid,
        .tick = 0,
        .payload_idx = 0,
        .payload_len = 0,
        .channel = CHAN_RT,
    };
    return hs_send_trusted(&gpu->system, &msg, NULL, 0);
}

void hs_gpu_begin_frame(HSGpu* gpu) {
    hs_render_reset(&gpu->render);
    if (gpu->backend && gpu->backend->ops && gpu->backend->ops->begin_frame) {
        HSFrameContext frame;
        hs_gpu_build_frame(gpu, &frame);
        gpu->backend->ops->begin_frame(gpu->backend->ctx, &frame);
    }
}

static void hs_gpu_execute_frame(HSGpu* gpu) {
    if (gpu->backend && gpu->backend->ops) {
        HSFrameContext frame;
        hs_gpu_build_frame(gpu, &frame);
        if (gpu->backend->ops->execute) gpu->backend->ops->execute(gpu->backend->ctx, &frame);
        if (gpu->backend->ops->end_frame) gpu->backend->ops->end_frame(gpu->backend->ctx, &frame);
    }
}

/* Keep composition's larger live register set off the idle legacy hot path. */
static __attribute__((noinline)) bool hs_gpu_end_participant_frame(HSGpu* gpu) {
    hs_gpu_service_participants(gpu);
    /* Read after service: acquiring a published frame also observes the bound
     * established before that producer's join returned. Never shrinks live. */
    u32 limit = atomic_load_explicit(&gpu->participant_limit, memory_order_relaxed);
    HSRenderList* render = &gpu->render;
    u32 added = 0;
    for (u32 i = 0; i < limit; i++) {
        const HSParticipantView* view = &gpu->participants.views[i];
        if (!view->active) continue;
        if (view->frame.count > HS_FABRIC_DRAWS) { render->overflow = true; return false; }
        added += view->frame.count;
    }
    if (render->overflow || render->count > HS_RENDER_MAX_CMDS || added > HS_RENDER_MAX_CMDS - render->count) {
        render->overflow = true;
        return false;
    }
    u32 original = render->count, insertion = original;
    u32 suffix = 0;
    if (added) {
        while (insertion && (render->cmds[insertion - 1].op == HS_RC_FRAME_END ||
                             render->cmds[insertion - 1].op == HS_RC_PRESENT)) insertion--;
        suffix = original - insertion;
        if (suffix) memmove(&render->cmds[insertion + added], &render->cmds[insertion], suffix * sizeof(HSRenderCmd));
        u32 next = insertion;
        for (u32 i = 0; i < limit; i++) {
            const HSParticipantView* view = &gpu->participants.views[i];
            if (!view->active || !view->frame.count) continue;
            memcpy(&render->cmds[next], view->frame.draws, view->frame.count * sizeof(HSRenderCmd));
            next += view->frame.count;
        }
        render->count += added;
    }
    gpu->frame_executing = true;
    hs_gpu_execute_frame(gpu);
    if (added) {
        if (suffix) memmove(&render->cmds[insertion], &render->cmds[insertion + added], suffix * sizeof(HSRenderCmd));
        render->count = original;
    }
    gpu->frame_executing = false;
    return true;
}

bool hs_gpu_end_frame_checked(HSGpu* gpu) {
    if (!gpu || gpu->frame_executing) return false;
    if (atomic_load_explicit(&gpu->participant_count, memory_order_acquire))
        return hs_gpu_end_participant_frame(gpu);
    if (gpu->render.overflow || gpu->render.count > HS_RENDER_MAX_CMDS) {
        gpu->render.overflow = true;
        return false;
    }
    gpu->frame_executing = true;
    hs_gpu_execute_frame(gpu);
    gpu->frame_executing = false;
    return true;
}

void hs_gpu_end_frame(HSGpu* gpu) {
    (void)hs_gpu_end_frame_checked(gpu);
}
