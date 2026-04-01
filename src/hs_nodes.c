#include "hs_nodes.h"
#include "hs_msg.h"
#include <stdio.h>
#include <string.h>

#ifndef HS_DEBUG
#define HS_DEBUG 0
#endif

#if HS_DEBUG
#define DBG_PRINT(...) printf(__VA_ARGS__)
#else
#define DBG_PRINT(...) ((void)0)
#endif

static HSSystem* hs_node_system(const Node* node) {
    return node ? node->sys : NULL;
}

static ShaderState* hs_shader_state(Node* node) {
    return node ? (ShaderState*)node->state : NULL;
}

static BufferState* hs_buffer_state(Node* node) {
    return node ? (BufferState*)node->state : NULL;
}

static TextureState* hs_texture_state(Node* node) {
    return node ? (TextureState*)node->state : NULL;
}

static OutputState* hs_output_state(Node* node) {
    return node ? (OutputState*)node->state : NULL;
}

static SoundState* hs_sound_state(Node* node) {
    return node ? (SoundState*)node->state : NULL;
}

static SystemState* hs_system_state(Node* node) {
    return node ? (SystemState*)node->state : NULL;
}

static void emit_ack(Node* node, const Message* req, u8 status) {
    HSSystem* sys = hs_node_system(node);
    if (!sys || !node || !req) return;
    if ((req->flags & HS_MSGF_ACK) == 0) return;

    Message ack = {
        .to = NODE_SYSTEM,
        .from = node->id,
        .op = OP_ACK,
        .flags = status,
        .cid = req->cid,
        .tick = 0,
        .payload_idx = (u16)req->op,
        .payload_len = 0,
    };
    (void)hs_send(sys, &ack);
}

static void emit_result(Node* node, const Message* req, u8 result_op, const void* payload, u32 payload_len) {
    HSSystem* sys = hs_node_system(node);
    if (!node || !sys || !req) return;

    u16 idx = 0;
    u32 copy_len = 0;
    if (payload_len) {
        if (!hs_payload_alloc_and_copy(sys, payload, payload_len, &idx, &copy_len)) {
            sys->dropped_result++;
            return;
        }
    }

    Message res = {
        .to = NODE_SYSTEM,
        .from = NODE_SYSTEM,
        .op = OP_RESULT,
        .flags = result_op,
        .cid = req->cid,
        .tick = 0,
        .payload_idx = idx,
        .payload_len = copy_len,
        .channel = CHAN_RT,
    };

    if (!mq_push(&node->outbox, &res)) {
        sys->dropped_result++;
    }
}

static void* get_payload(Node* node, u32 idx) {
    HSSystem* sys = hs_node_system(node);
    if (!sys || !sys->payloads || idx >= sys->payload_capacity) return NULL;
    return sys->payloads[idx].data;
}

const char* node_name(u8 id) {
    switch (id) {
        case NODE_CPU:    return "CPU";
        case NODE_SHADER: return "Shader";
        case NODE_BUFFER: return "Buffer";
        case NODE_TEXTURE:return "Texture";
        case NODE_OUTPUT: return "Output";
        case NODE_SOUND:  return "Sound";
        case NODE_SYSTEM: return "System";
        default:          return "Unknown";
    }
}

void shader_node_init(Node* node) {
    ShaderState* state = hs_shader_state(node);
    if (state) memset(state, 0, sizeof(*state));
    node->process_fn = shader_node_process;
    node->reset_fn = shader_node_reset;
}

void shader_node_reset(Node* node) {
    ShaderState* state = hs_shader_state(node);
    if (state) memset(state, 0, sizeof(*state));
}

int shader_node_process(Node* node) {
    ShaderState* state = hs_shader_state(node);
    int processed = 0;
    Message msg;

    if (!state) return 0;

    while (mq_pop(&node->inbox, &msg)) {
        u8 handled = 1;
        switch (msg.op) {
            case OP_SET_SHADER:
                state->current_shader = msg.payload_idx;
                DBG_PRINT("[%s] Set shader #%d\n", node_name(node->id), state->current_shader);
                break;

            case OP_SET_PARAM: {
                void* data = get_payload(node, msg.payload_idx);
                if (data && msg.payload_len >= 20) {
                    u32 param_idx = 0;
                    f32 v[4];
                    if (!hs_unpack_set_param(data, msg.payload_len, &param_idx, v)) break;
                    if (param_idx < 16) {
                        for (int i = 0; i < 4; i++) state->params[param_idx][i] = v[i];
                        state->params_set |= (1u << param_idx);
                        DBG_PRINT("[%s] Set param #%d = (%.2f, %.2f, %.2f, %.2f)\n",
                                  node_name(node->id), param_idx,
                                  state->params[param_idx][0], state->params[param_idx][1],
                                  state->params[param_idx][2], state->params[param_idx][3]);
                    }
                }
                break;
            }

            case OP_SET_GLOBAL: {
                void* data = get_payload(node, msg.payload_idx);
                if (data && msg.payload_len >= 16 * sizeof(f32)) {
                    f32* global_data = (f32*)data;
                    u8 global_idx = msg.flags & 0x7F;
                    if (global_idx == 0) {
                        for (int i = 0; i < 16; i++) state->camera_view_proj[i] = global_data[i];
                    }
                    DBG_PRINT("[%s] Set global #%d\n", node_name(node->id), global_idx);
                }
                break;
            }

            case OP_SET_CAMERA: {
                void* data = get_payload(node, msg.payload_idx);
                if (data && msg.payload_len >= 16 * sizeof(f32)) {
                    mat4 view = m4_from_array((f32*)data);
                    m4_to_array(view, state->camera_view_proj);
                }
                break;
            }

            case OP_CULL:
                state->cull_mode = (u8)msg.payload_idx;
                DBG_PRINT("[%s] Cull mode: %d\n", node_name(node->id), (int)msg.payload_idx);
                break;

            case OP_BLEND: {
                void* data = get_payload(node, msg.payload_idx);
                if (data) {
                    u8 src = 0, dst = 0;
                    if (!hs_unpack_u8x2(data, msg.payload_len, &src, &dst)) break;
                    state->blend_src = src;
                    state->blend_dst = dst;
                    DBG_PRINT("[%s] Blend: src=%d, dst=%d\n", node_name(node->id), (int)state->blend_src, (int)state->blend_dst);
                }
                break;
            }

            case OP_ALPHA:
                if (msg.payload_idx) {
                    state->blend_src = 5;
                    state->blend_dst = 6;
                } else {
                    state->blend_src = 1;
                    state->blend_dst = 0;
                }
                DBG_PRINT("[%s] Alpha blend: %s\n", node_name(node->id), msg.payload_idx ? "enabled" : "disabled");
                break;

            case OP_DEPTH:
                state->depth_enabled = (msg.payload_idx != 0);
                DBG_PRINT("[%s] Depth: %s\n", node_name(node->id), msg.payload_idx ? "enabled" : "disabled");
                break;

            case OP_COLOR_MASK:
                state->color_mask = (u8)msg.payload_idx;
                DBG_PRINT("[%s] Color mask: 0x%02X\n", node_name(node->id), (int)msg.payload_idx);
                break;

            case OP_CLIP: {
                void* data = get_payload(node, msg.payload_idx);
                if (data) {
                    u16 vals[4];
                    if (!hs_unpack_u16x4(data, msg.payload_len, vals)) break;
                    state->clip_x = vals[0];
                    state->clip_y = vals[1];
                    state->clip_w = vals[2];
                    state->clip_h = vals[3];
                    DBG_PRINT("[%s] Clip: x=%d, y=%d, w=%d, h=%d\n", node_name(node->id), vals[0], vals[1], vals[2], vals[3]);
                }
                break;
            }

            case OP_STENCIL: {
                void* data = get_payload(node, msg.payload_idx);
                if (data) {
                    u8 vals[4];
                    if (!hs_unpack_u8x4(data, msg.payload_len, vals)) break;
                    state->stencil_op = vals[0];
                    state->stencil_fail = vals[1];
                    state->stencil_pass = vals[2];
                    state->stencil_front = vals[3];
                    DBG_PRINT("[%s] Stencil: op=%d, fail=%d, pass=%d, front=%d\n", node_name(node->id), vals[0], vals[1], vals[2], vals[3]);
                }
                break;
            }

            case OP_STENCIL_FUNC: {
                void* data = get_payload(node, msg.payload_idx);
                if (data) {
                    u8 vals[4];
                    if (!hs_unpack_u8x4(data, msg.payload_len, vals)) break;
                    state->stencil_compare = vals[0];
                    state->stencil_ref = vals[1];
                    state->stencil_read_mask = vals[2];
                    state->stencil_write_mask = vals[3];
                    DBG_PRINT("[%s] Stencil func: compare=%d, ref=%d, readMask=0x%02X, writeMask=0x%02X\n", node_name(node->id), vals[0], vals[1], vals[2], vals[3]);
                }
                break;
            }

            case OP_DEPTH_COMPARE: {
                void* data = get_payload(node, msg.payload_idx);
                if (data) {
                    u8 cmp = 0, wr = 0;
                    if (!hs_unpack_u8x2(data, msg.payload_len, &cmp, &wr)) break;
                    state->depth_compare = cmp;
                    state->depth_write = wr;
                    DBG_PRINT("[%s] Depth compare: comp=%d, write=%d\n", node_name(node->id), (int)cmp, (int)wr);
                }
                break;
            }

            default:
                handled = 0;
                break;
        }
        emit_ack(node, &msg, handled ? 0 : 1);
        processed++;
    }

    return processed;
}

void buffer_node_init(Node* node) {
    BufferState* state = hs_buffer_state(node);
    if (state) memset(state, 0, sizeof(*state));
    node->process_fn = buffer_node_process;
    node->reset_fn = buffer_node_reset;
}

void buffer_node_reset(Node* node) {
    BufferState* state = hs_buffer_state(node);
    if (state) memset(state, 0, sizeof(*state));
}

int buffer_node_process(Node* node) {
    BufferState* state = hs_buffer_state(node);
    int processed = 0;
    Message msg;

    if (!state) return 0;

    while (mq_pop(&node->inbox, &msg)) {
        u8 handled = 1;
        switch (msg.op) {
            case OP_LOAD_BUFFER:
                state->buffer_handles[msg.payload_idx & 0xF] = msg.payload_idx;
                state->buffers_loaded |= (1 << (msg.payload_idx & 0xF));
                DBG_PRINT("[%s] Loaded buffer #%d\n", node_name(node->id), (int)(msg.payload_idx & 0xF));
                break;

            case OP_DRAW:
                DBG_PRINT("[%s] Draw buffer #%d\n", node_name(node->id), (int)(msg.payload_idx & 0xF));
                break;

            case OP_DRAW_INSTANCE: {
                void* data = get_payload(node, msg.payload_idx);
                if (data) {
                    u8 buffer_idx = 0;
                    u8 inst_buf = 0;
                    u32 count = 0;
                    if (!hs_unpack_draw_instance(data, msg.payload_len, &buffer_idx, &inst_buf, &count)) break;
                    DBG_PRINT("[%s] Draw instance: buffer=%d, inst_buffer=%d, count=%d\n", node_name(node->id), buffer_idx, inst_buf, count);
                }
                break;
            }

            case OP_DRAW_TEXT: {
                void* data = get_payload(node, msg.payload_idx);
                if (data) {
                    DBG_PRINT("[%s] Draw text: \"%.*s\"\n", node_name(node->id), (int)msg.payload_len, (char*)data);
                }
                break;
            }

            default:
                handled = 0;
                break;
        }
        emit_ack(node, &msg, handled ? 0 : 1);
        processed++;
    }

    return processed;
}

void texture_node_init(Node* node) {
    TextureState* state = hs_texture_state(node);
    if (state) memset(state, 0, sizeof(*state));
    node->process_fn = texture_node_process;
    node->reset_fn = texture_node_reset;
}

void texture_node_reset(Node* node) {
    TextureState* state = hs_texture_state(node);
    if (state) memset(state, 0, sizeof(*state));
}

int texture_node_process(Node* node) {
    TextureState* state = hs_texture_state(node);
    int processed = 0;
    Message msg;

    if (!state) return 0;

    while (mq_pop(&node->inbox, &msg)) {
        u8 handled = 1;
        switch (msg.op) {
            case OP_LOAD_TEXTURE:
                state->texture_handles[msg.payload_idx & 0xF] = msg.payload_idx;
                state->textures_loaded |= (1 << (msg.payload_idx & 0xF));
                DBG_PRINT("[%s] Loaded texture #%d\n", node_name(node->id), (int)(msg.payload_idx & 0xF));
                break;

            case OP_SET_TARGET: {
                void* data = get_payload(node, msg.payload_idx);
                if (data) {
                    u8 tex = 0;
                    u8 depth = 0;
                    if (!hs_unpack_u8x2(data, msg.payload_len, &tex, &depth)) break;
                    state->current_target = tex;
                    state->current_depth_target = depth;
                } else {
                    state->current_target = (msg.payload_idx & 0xF);
                    state->current_depth_target = 0xF;
                }
                DBG_PRINT("[%s] Set target: #%d, depth=#%d\n", node_name(node->id), state->current_target, state->current_depth_target);
                break;
            }

            case OP_SHOW_TEXTURE:
                state->show_texture = (msg.payload_idx & 0xF);
                DBG_PRINT("[%s] Show texture: #%d\n", node_name(node->id), state->show_texture);
                break;

            case OP_TEXTURE_FILTER: {
                u8 tex_idx = msg.payload_idx & 0xF;
                u8 filter = (msg.payload_idx >> 4) & 1;
                state->texture_filter[tex_idx] = filter;
                DBG_PRINT("[%s] Texture #%d filter: %s\n", node_name(node->id), tex_idx, filter ? "linear" : "nearest");
                break;
            }

            case OP_TEXTURE_WRAP: {
                u8 tex_idx = msg.payload_idx & 0xF;
                u8 wrap = (msg.payload_idx >> 4) & 1;
                state->texture_wrap[tex_idx] = wrap;
                DBG_PRINT("[%s] Texture #%d wrap: %s\n", node_name(node->id), tex_idx, wrap ? "repeat" : "clamp");
                break;
            }

            default:
                handled = 0;
                break;
        }
        emit_ack(node, &msg, handled ? 0 : 1);
        processed++;
    }

    return processed;
}

void output_node_init(Node* node) {
    OutputState* state = hs_output_state(node);
    if (state) {
        memset(state, 0, sizeof(*state));
        state->clear_color[3] = 1.0f;
        state->clear_depth = 1.0f;
    }
    node->process_fn = output_node_process;
    node->reset_fn = output_node_reset;
}

void output_node_reset(Node* node) {
    OutputState* state = hs_output_state(node);
    if (!state) return;
    memset(state, 0, sizeof(*state));
    state->clear_color[3] = 1.0f;
    state->clear_depth = 1.0f;
}

int output_node_process(Node* node) {
    OutputState* state = hs_output_state(node);
    int processed = 0;
    Message msg;

    if (!state) return 0;

    while (mq_pop(&node->inbox, &msg)) {
        u8 handled = 1;
        switch (msg.op) {
            case OP_CLEAR: {
                void* data = get_payload(node, msg.payload_idx);
                if (data && msg.payload_len >= 4 * sizeof(f32)) {
                    f32 color[4];
                    if (!hs_unpack_clear_color(data, msg.payload_len, color)) break;
                    state->clear_color[0] = color[0];
                    state->clear_color[1] = color[1];
                    state->clear_color[2] = color[2];
                    state->clear_color[3] = color[3];
                    DBG_PRINT("[%s] Clear color: (%.2f, %.2f, %.2f, %.2f)\n", node_name(node->id), color[0], color[1], color[2], color[3]);
                }
                break;
            }

            case OP_CLEAR_DS: {
                void* data = get_payload(node, msg.payload_idx);
                if (data) {
                    f32 depth = 0.0f;
                    u8 stencil = 0;
                    if (!hs_unpack_clear_ds(data, msg.payload_len, &depth, &stencil)) break;
                    state->clear_depth = depth;
                    state->clear_stencil = stencil;
                    DBG_PRINT("[%s] Clear DS: depth=%.2f, stencil=%d\n", node_name(node->id), state->clear_depth, state->clear_stencil);
                }
                break;
            }

            default:
                handled = 0;
                break;
        }
        emit_ack(node, &msg, handled ? 0 : 1);
        processed++;
    }

    return processed;
}

void sound_node_init(Node* node) {
    SoundState* state = hs_sound_state(node);
    if (state) memset(state, 0, sizeof(*state));
    node->process_fn = sound_node_process;
    node->reset_fn = sound_node_reset;
}

void sound_node_reset(Node* node) {
    SoundState* state = hs_sound_state(node);
    if (state) memset(state, 0, sizeof(*state));
}

int sound_node_process(Node* node) {
    SoundState* state = hs_sound_state(node);
    int processed = 0;
    Message msg;

    if (!state) return 0;

    while (mq_pop(&node->inbox, &msg)) {
        u8 handled = 1;
        switch (msg.op) {
            case OP_SET_CHANNEL: {
                void* data = get_payload(node, msg.payload_idx);
                if (data) {
                    u8 ch = 0;
                    u8 shader = 0;
                    if (!hs_unpack_u8x2(data, msg.payload_len, &ch, &shader)) break;
                    if (ch < 4) {
                        state->channel_shaders[ch] = shader;
                        DBG_PRINT("[%s] Channel %d -> shader #%d\n", node_name(node->id), ch, shader);
                    }
                }
                break;
            }

            default:
                handled = 0;
                break;
        }
        emit_ack(node, &msg, handled ? 0 : 1);
        processed++;
    }

    return processed;
}

void system_node_init(Node* node) {
    SystemState* state = hs_system_state(node);
    if (state) memset(state, 0, sizeof(*state));
    node->process_fn = system_node_process;
    node->reset_fn = system_node_reset;
}

void system_node_reset(Node* node) {
    SystemState* state = hs_system_state(node);
    if (state) memset(state, 0, sizeof(*state));
}

int system_node_process(Node* node) {
    HSSystem* sys = hs_node_system(node);
    SystemState* state = hs_system_state(node);
    int processed = 0;
    Message msg;

    if (!state) return 0;

    while (mq_pop(&node->inbox, &msg)) {
        switch (msg.op) {
            case OP_ACK:
                state->ack_count++;
                state->last_ack_cid = msg.cid;
                state->last_ack_from = msg.from;
                state->last_ack_op = (u8)msg.payload_idx;
                state->last_ack_status = msg.flags;
                break;

            case OP_RESULT:
                state->result_count++;
                state->last_result_cid = msg.cid;
                state->last_result_from = msg.from;
                state->last_result_op = msg.flags;
                state->last_result_len = (u16)msg.payload_len;
                state->last_result_payload_idx = (u16)msg.payload_idx;
                hs_toolbus_record_result(sys, &msg);
                break;

            case OP_QUEUE_FULL:
                state->queue_full_count++;
                state->last_queue_full_to = msg.flags;
                state->last_queue_full_op = (u8)msg.payload_idx;
                break;

            case OP_ASYNC_DONE: {
                void* data = get_payload(node, msg.payload_idx);
                if (data) {
                    u32 task_id = 0;
                    u8 type = 0, slot = 0, success = 0;
                    if (hs_unpack_async_done(data, msg.payload_len, &task_id, &type, &slot, &success)) {
                        state->async_done_count++;
                        state->last_async_task_id = task_id;
                        state->last_async_type = type;
                        state->last_async_slot = slot;
                        state->last_async_success = success;
                    }
                }
                break;
            }

            case OP_ERROR_EX: {
                void* data = get_payload(node, msg.payload_idx);
                if (data) {
                    u32 code = 0;
                    u8 op = 0, to = 0, from = 0, stage = 0;
                    u32 cid = 0;
                    if (hs_unpack_error_ex(data, msg.payload_len, &code, &op, &to, &from, &stage, &cid)) {
                        state->error_count++;
                        state->last_error_code = code;
                        state->last_error_stage = stage;
                        state->last_error_op = op;
                        state->last_error_to = to;
                        state->last_error_from = from;
#if HS_DEBUG
                        fprintf(stderr, "[ERROR_EX] code=%u stage=%u op=%u to=%u from=%u cid=%u\n",
                                (unsigned)code, (unsigned)stage, (unsigned)op, (unsigned)to, (unsigned)from, (unsigned)cid);
#endif
                    }
                }
                break;
            }

            case OP_ERROR: {
                void* data = get_payload(node, msg.payload_idx);
                if (data) fprintf(stderr, "[ERROR] %s\n", (char*)data);
                break;
            }

            case OP_TRACE: {
                void* data = get_payload(node, msg.payload_idx);
                if (data) printf("[TRACE] %s\n", (char*)data);
                break;
            }

            case OP_STOP:
                state->stopped = 1;
                DBG_PRINT("[%s] STOP command received\n", node_name(node->id));
                break;

            case OP_QUERY_STATS: {
                u8 out[64];
                u32 flags = 0;
                if (sys) {
                    if (sys->recording) flags |= 1u << 0;
                    if (sys->validate_on_send) flags |= 1u << 1;
                    if (sys->block_on_full_chan[CHAN_RT]) flags |= 1u << 2;
                    if (sys->block_on_full_chan[CHAN_RENDER]) flags |= 1u << 3;
                    if (sys->block_on_full_chan[CHAN_TELEM]) flags |= 1u << 4;
                }
                hs_pack_result_system_stats(out,
                                           sys ? sys->tick : 0,
                                           sys ? sys->log_head : 0,
                                           sys ? sys->record_mask : 0,
                                           sys ? sys->chan_budget[CHAN_RT] : 0,
                                           sys ? sys->chan_budget[CHAN_RENDER] : 0,
                                           sys ? sys->chan_budget[CHAN_TELEM] : 0,
                                           sys ? sys->dropped_error_ex : 0,
                                           sys ? sys->dropped_queue_full : 0,
                                           sys ? sys->dropped_system_nonrt : 0,
                                           sys ? sys->dropped_result : 0,
                                           sys ? atomic_load_explicit(&sys->producer_count, memory_order_relaxed) : 0,
                                           flags);
                emit_result(node, &msg, OP_QUERY_STATS, out, sizeof(out));
                break;
            }

            case OP_QUERY_FABRIC: {
                u8 out[96];
                u32 spsc_ok3[3] = {0}, spsc_full3[3] = {0}, mpsc_ok3[3] = {0}, submit_full3[3] = {0}, submit_hw3[3] = {0}, telem_dropped3[3] = {0};
                if (sys) {
                    u32 prod_count = atomic_load_explicit(&sys->producer_count, memory_order_relaxed);
                    if (prod_count > HS_MAX_PRODUCERS) prod_count = HS_MAX_PRODUCERS;
                    for (u32 p = 0; p < prod_count; p++) {
                        spsc_ok3[0] += atomic_load_explicit(&sys->spsc_ok_by_prod[CHAN_RT][p], memory_order_relaxed);
                        spsc_ok3[1] += atomic_load_explicit(&sys->spsc_ok_by_prod[CHAN_RENDER][p], memory_order_relaxed);
                        spsc_ok3[2] += atomic_load_explicit(&sys->spsc_ok_by_prod[CHAN_TELEM][p], memory_order_relaxed);
                        spsc_full3[0] += atomic_load_explicit(&sys->spsc_full_by_prod[CHAN_RT][p], memory_order_relaxed);
                        spsc_full3[1] += atomic_load_explicit(&sys->spsc_full_by_prod[CHAN_RENDER][p], memory_order_relaxed);
                        spsc_full3[2] += atomic_load_explicit(&sys->spsc_full_by_prod[CHAN_TELEM][p], memory_order_relaxed);
                    }
                    mpsc_ok3[0] = atomic_load_explicit(&sys->mpsc_ok[CHAN_RT], memory_order_relaxed);
                    mpsc_ok3[1] = atomic_load_explicit(&sys->mpsc_ok[CHAN_RENDER], memory_order_relaxed);
                    mpsc_ok3[2] = atomic_load_explicit(&sys->mpsc_ok[CHAN_TELEM], memory_order_relaxed);
                    submit_full3[0] = atomic_load_explicit(&sys->submit_full[CHAN_RT], memory_order_relaxed);
                    submit_full3[1] = atomic_load_explicit(&sys->submit_full[CHAN_RENDER], memory_order_relaxed);
                    submit_full3[2] = atomic_load_explicit(&sys->submit_full[CHAN_TELEM], memory_order_relaxed);
                    submit_hw3[0] = atomic_load_explicit(&sys->submit_hw[CHAN_RT], memory_order_relaxed);
                    submit_hw3[1] = atomic_load_explicit(&sys->submit_hw[CHAN_RENDER], memory_order_relaxed);
                    submit_hw3[2] = atomic_load_explicit(&sys->submit_hw[CHAN_TELEM], memory_order_relaxed);
                    telem_dropped3[0] = atomic_load_explicit(&sys->telem_dropped[CHAN_RT], memory_order_relaxed);
                    telem_dropped3[1] = atomic_load_explicit(&sys->telem_dropped[CHAN_RENDER], memory_order_relaxed);
                    telem_dropped3[2] = atomic_load_explicit(&sys->telem_dropped[CHAN_TELEM], memory_order_relaxed);
                }
                hs_pack_result_fabric(out, spsc_ok3, spsc_full3, mpsc_ok3, submit_full3, submit_hw3, telem_dropped3,
                                      sys ? atomic_load_explicit(&sys->producer_count, memory_order_relaxed) : 0,
                                      sys ? atomic_load_explicit(&sys->bp_waiters, memory_order_relaxed) : 0);
                emit_result(node, &msg, OP_QUERY_FABRIC, out, sizeof(out));
                break;
            }

            case OP_SET_RECORD_MASK: {
                void* data = get_payload(node, msg.payload_idx);
                u32 new_mask = 0;
                u32 old_mask = sys ? sys->record_mask : 0;
                if (data && sys && hs_unpack_set_record_mask(data, msg.payload_len, &new_mask)) sys->record_mask = new_mask;
                u8 out[8];
                hs_pack_u32x2(out, old_mask, sys ? sys->record_mask : old_mask);
                emit_result(node, &msg, OP_SET_RECORD_MASK, out, sizeof(out));
                break;
            }

            case OP_SET_CHAN_BUDGET: {
                void* data = get_payload(node, msg.payload_idx);
                u8 ch = 0;
                u32 budget = 0;
                u32 old = 0;
                if (sys) {
                    if (data && hs_unpack_set_chan_budget(data, msg.payload_len, &ch, &budget) && ch < CHAN_COUNT && ch != CHAN_DEFAULT) {
                        old = sys->chan_budget[ch];
                        sys->chan_budget[ch] = budget;
                    } else {
                        old = 0xFFFFFFFFu;
                        budget = 0xFFFFFFFFu;
                    }
                }
                u8 out[8];
                hs_pack_u32x2(out, old, budget);
                emit_result(node, &msg, OP_SET_CHAN_BUDGET, out, sizeof(out));
                break;
            }

            case OP_SET_BLOCK_POLICY: {
                void* data = get_payload(node, msg.payload_idx);
                u8 ch = 0, block = 0;
                u32 old = 0xFFFFFFFFu;
                u32 now = 0xFFFFFFFFu;
                if (sys && data && hs_unpack_set_block_policy(data, msg.payload_len, &ch, &block) && ch < CHAN_COUNT && ch != CHAN_DEFAULT) {
                    old = sys->block_on_full_chan[ch] ? 1u : 0u;
                    sys->block_on_full_chan[ch] = (block != 0);
                    now = sys->block_on_full_chan[ch] ? 1u : 0u;
                }
                u8 out[8];
                hs_pack_u32x2(out, old, now);
                emit_result(node, &msg, OP_SET_BLOCK_POLICY, out, sizeof(out));
                break;
            }

            default:
                break;
        }
        processed++;
    }

    return processed;
}

void input_node_init(Node* node) {
    InputState* state = (InputState*)node->state;
    if (state) memset(state, 0, sizeof(*state));
    node->process_fn = input_node_process;
    node->reset_fn = input_node_reset;
}

void input_node_reset(Node* node) {
    InputState* state = (InputState*)node->state;
    if (!state) return;
    memset(&state->base, 0, sizeof(state->base));
}

static InputState* hs_input_state(Node* node) {
    return node ? (InputState*)node->state : NULL;
}

int input_node_process(Node* node) {
    InputState* state = hs_input_state(node);
    int processed = 0;
    Message msg;

    if (!state) return 0;

    while (mq_pop(&node->inbox, &msg)) {
        u8 handled = 1;
        switch (msg.op) {
            case OP_INPUT_KEY: {
                void* data = get_payload(node, msg.payload_idx);
                if (data && msg.payload_len >= 3) {
                    u16 code = ((u8*)data)[0] | (((u8*)data)[1] << 8);
                    u8 pressed = ((u8*)data)[2];
                    switch (code) {
                        case 105: state->base.key_left = pressed; break;
                        case 106: state->base.key_right = pressed; break;
                        case 103: state->base.key_up = pressed; break;
                        case 108: state->base.key_down = pressed; break;
                        case 28: case 57: state->base.button1 = pressed; break;
                        case 14: case 111: state->base.button2 = pressed; break;
                        case 272: state->base.mouse_left = pressed; break;
                        case 273: state->base.mouse_right = pressed; break;
                        case 288: state->base.pad_a = pressed; break;
                        case 289: state->base.pad_b = pressed; break;
                    }
                }
                break;
            }

            case OP_INPUT_MOUSE: {
                void* data = get_payload(node, msg.payload_idx);
                if (data && msg.payload_len >= 8) {
                    s32 x = *(s32*)data;
                    s32 y = *(s32*)(data + 4);
                    state->base.mouse_x = x < 0 ? 0 : (x > HS_WIDTH ? HS_WIDTH : x);
                    state->base.mouse_y = y < 0 ? 0 : (y > HS_HEIGHT ? HS_HEIGHT : y);
                }
                break;
            }

            case OP_INPUT_GAMEPAD: {
                void* data = get_payload(node, msg.payload_idx);
                if (data && msg.payload_len >= 8) {
                    f32 x = *(f32*)data;
                    f32 y = *(f32*)(data + 4);
                    state->base.pad_x = x < -1.0f ? -1.0f : (x > 1.0f ? 1.0f : x);
                    state->base.pad_y = y < -1.0f ? -1.0f : (y > 1.0f ? 1.0f : y);
                }
                break;
            }

            default:
                handled = 0;
                break;
        }
        emit_ack(node, &msg, handled ? 0 : 1);
        processed++;
    }

    if (processed > 0) {
        f32 dx = 0.0f, dy = 0.0f;
        if (state->base.key_left)  dx -= 1.0f;
        if (state->base.key_right) dx += 1.0f;
        if (state->base.key_up)    dy -= 1.0f;
        if (state->base.key_down)  dy += 1.0f;
        dx += state->base.pad_x;
        dy += state->base.pad_y;
        if (dx < -1.0f) dx = -1.0f;
        if (dx >  1.0f) dx =  1.0f;
        if (dy < -1.0f) dy = -1.0f;
        if (dy >  1.0f) dy =  1.0f;
        state->base.dir_x = dx;
        state->base.dir_y = dy;
    }

    return processed;
}
