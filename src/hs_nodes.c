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

static ShaderState shader_state;
static BufferState buffer_state;
static TextureState texture_state;
static OutputState output_state;
static SoundState sound_state;
static SystemState system_state;

static HSSystem* g_sys = NULL;

static void emit_ack(u8 from, const Message* req, u8 status) {
    if (!g_sys || !req) return;
    if ((req->flags & HS_MSGF_ACK) == 0) return;

    Message ack = {
        .to = NODE_SYSTEM,
        .from = from,
        .op = OP_ACK,
        .flags = status,
        .cid = req->cid,
        .tick = 0,
        .payload_idx = (u16)req->op,
        .payload_len = 0,
    };
    (void)hs_send(g_sys, &ack);
}

void hs_nodes_set_system(HSSystem* sys) {
    g_sys = sys;
}

static void* get_payload(u32 idx) {
    if (!g_sys || !g_sys->payloads || idx >= g_sys->payload_capacity) return NULL;
    return g_sys->payloads[idx].data;
}

const char* node_name(u8 id) {
    switch (id) {
        case NODE_CPU:     return "CPU";
        case NODE_SHADER:  return "Shader";
        case NODE_BUFFER: return "Buffer";
        case NODE_TEXTURE: return "Texture";
        case NODE_OUTPUT: return "Output";
        case NODE_SOUND:  return "Sound";
        case NODE_SYSTEM: return "System";
        default:          return "Unknown";
    }
}

void shader_node_init(Node* node) {
    memset(&shader_state, 0, sizeof(ShaderState));
    node->state = &shader_state;
    node->process_fn = shader_node_process;
    node->reset_fn = shader_node_reset;
}

void shader_node_reset(Node* node) {
    (void)node;
    memset(&shader_state, 0, sizeof(ShaderState));
}

int shader_node_process(Node* node) {
    int processed = 0;
    Message msg;
    
    while (mq_pop(&node->inbox, &msg)) {
        u8 handled = 1;
        switch (msg.op) {
            case OP_SET_SHADER:
                shader_state.current_shader = msg.payload_idx;
                DBG_PRINT("[%s] Set shader #%d\n", node_name(node->id), shader_state.current_shader);
                break;
                
            case OP_SET_PARAM: {
                void* data = get_payload(msg.payload_idx);
                if (data && msg.payload_len >= 20) {
                    /* Layout: [u32 param_idx][f32 x][f32 y][f32 z][f32 w] */
                    u32 param_idx = 0;
                    f32 v[4];
                    if (!hs_unpack_set_param(data, msg.payload_len, &param_idx, v)) break;
                    if (param_idx < 16) {
                        for (int i = 0; i < 4; i++) {
                            shader_state.params[param_idx][i] = v[i];
                        }
                        shader_state.params_set |= (1u << param_idx);
                        DBG_PRINT("[%s] Set param #%d = (%.2f, %.2f, %.2f, %.2f)\n", 
                               node_name(node->id), param_idx,
                               shader_state.params[param_idx][0],
                               shader_state.params[param_idx][1],
                               shader_state.params[param_idx][2],
                               shader_state.params[param_idx][3]);
                    }
                }
                break;
            }
                
            case OP_SET_GLOBAL: {
                void* data = get_payload(msg.payload_idx);
                if (data && msg.payload_len >= 16 * sizeof(f32)) {
                    f32* global_data = (f32*)data;
                    u8 global_idx = msg.flags & 0x7F;
                    if (global_idx == 0) {
                        /* Global 0 = camera view-projection matrix */
                        for (int i = 0; i < 16; i++) {
                            shader_state.camera_view_proj[i] = global_data[i];
                        }
                    }
                    DBG_PRINT("[%s] Set global #%d\n", node_name(node->id), global_idx);
                }
                break;
            }
                
            case OP_SET_CAMERA: {
                void* data = get_payload(msg.payload_idx);
                if (data && msg.payload_len >= 16 * sizeof(f32)) {
                    mat4 view = m4_from_array((f32*)data);
                    m4_to_array(view, shader_state.camera_view_proj);
                }
                break;
            }
                
            case OP_CULL:
                shader_state.cull_mode = (u8)msg.payload_idx;
                DBG_PRINT("[%s] Cull mode: %d\n", node_name(node->id), (int)msg.payload_idx);
                break;
                
            case OP_BLEND: {
                void* data = get_payload(msg.payload_idx);
                if (data) {
                    u8 src = 0, dst = 0;
                    if (!hs_unpack_u8x2(data, msg.payload_len, &src, &dst)) break;
                    shader_state.blend_src = src;
                    shader_state.blend_dst = dst;
                    DBG_PRINT("[%s] Blend: src=%d, dst=%d\n", 
                           node_name(node->id), (int)shader_state.blend_src, (int)shader_state.blend_dst);
                }
                break;
            }
                
            case OP_ALPHA: {
                if (msg.payload_idx) {
                    shader_state.blend_src = 5;  /* SrcAlpha */
                    shader_state.blend_dst = 6;  /* OneMinusSrcAlpha */
                } else {
                    shader_state.blend_src = 1;  /* One */
                    shader_state.blend_dst = 0;  /* Zero */
                }
                DBG_PRINT("[%s] Alpha blend: %s\n", node_name(node->id), msg.payload_idx ? "enabled" : "disabled");
                break;
            }
                
            case OP_DEPTH:
                shader_state.depth_enabled = (msg.payload_idx != 0);
                DBG_PRINT("[%s] Depth: %s\n", node_name(node->id), msg.payload_idx ? "enabled" : "disabled");
                break;
                
            case OP_COLOR_MASK:
                shader_state.color_mask = (u8)msg.payload_idx;
                DBG_PRINT("[%s] Color mask: 0x%02X\n", node_name(node->id), (int)msg.payload_idx);
                break;
                
            case OP_CLIP: {
                void* data = get_payload(msg.payload_idx);
                if (data) {
                    u16 vals[4];
                    if (!hs_unpack_u16x4(data, msg.payload_len, vals)) break;
                    shader_state.clip_x = vals[0];
                    shader_state.clip_y = vals[1];
                    shader_state.clip_w = vals[2];
                    shader_state.clip_h = vals[3];
                    DBG_PRINT("[%s] Clip: x=%d, y=%d, w=%d, h=%d\n",
                           node_name(node->id), vals[0], vals[1], vals[2], vals[3]);
                }
                break;
            }
                
            case OP_STENCIL: {
                void* data = get_payload(msg.payload_idx);
                if (data) {
                    u8 vals[4];
                    if (!hs_unpack_u8x4(data, msg.payload_len, vals)) break;
                    shader_state.stencil_op = vals[0];
                    shader_state.stencil_fail = vals[1];
                    shader_state.stencil_pass = vals[2];
                    shader_state.stencil_front = vals[3];
                    DBG_PRINT("[%s] Stencil: op=%d, fail=%d, pass=%d, front=%d\n",
                           node_name(node->id), vals[0], vals[1], vals[2], vals[3]);
                }
                break;
            }
                
            case OP_STENCIL_FUNC: {
                void* data = get_payload(msg.payload_idx);
                if (data) {
                    u8 vals[4];
                    if (!hs_unpack_u8x4(data, msg.payload_len, vals)) break;
                    shader_state.stencil_compare = vals[0];
                    shader_state.stencil_ref = vals[1];
                    shader_state.stencil_read_mask = vals[2];
                    shader_state.stencil_write_mask = vals[3];
                    DBG_PRINT("[%s] Stencil func: compare=%d, ref=%d, readMask=0x%02X, writeMask=0x%02X\n",
                           node_name(node->id), vals[0], vals[1], vals[2], vals[3]);
                }
                break;
            }
                
            case OP_DEPTH_COMPARE: {
                void* data = get_payload(msg.payload_idx);
                if (data) {
                    u8 cmp = 0, wr = 0;
                    if (!hs_unpack_u8x2(data, msg.payload_len, &cmp, &wr)) break;
                    shader_state.depth_compare = cmp;
                    shader_state.depth_write = wr;
                    DBG_PRINT("[%s] Depth compare: comp=%d, write=%d\n",
                            node_name(node->id), (int)cmp, (int)wr);
                }
                break;
            }
                
            default:
                handled = 0;
                break;
        }
        emit_ack(node->id, &msg, handled ? 0 : 1);
        processed++;
    }
    
    return processed;
}

void buffer_node_init(Node* node) {
    memset(&buffer_state, 0, sizeof(BufferState));
    node->state = &buffer_state;
    node->process_fn = buffer_node_process;
    node->reset_fn = buffer_node_reset;
}

void buffer_node_reset(Node* node) {
    (void)node;
    memset(&buffer_state, 0, sizeof(BufferState));
}

int buffer_node_process(Node* node) {
    int processed = 0;
    Message msg;
    
    while (mq_pop(&node->inbox, &msg)) {
        u8 handled = 1;
        switch (msg.op) {
            case OP_LOAD_BUFFER:
                buffer_state.buffer_handles[msg.payload_idx & 0xF] = msg.payload_idx;
                buffer_state.buffers_loaded |= (1 << (msg.payload_idx & 0xF));
                DBG_PRINT("[%s] Loaded buffer #%d\n", node_name(node->id), (int)(msg.payload_idx & 0xF));
                break;
                
            case OP_DRAW:
                DBG_PRINT("[%s] Draw buffer #%d\n", node_name(node->id), (int)(msg.payload_idx & 0xF));
                break;
                
            case OP_DRAW_INSTANCE: {
                void* data = get_payload(msg.payload_idx);
                if (data) {
                    u8 buffer_idx = 0;
                    u8 inst_buf = 0;
                    u32 count = 0;
                    if (!hs_unpack_draw_instance(data, msg.payload_len, &buffer_idx, &inst_buf, &count)) break;
                    (void)buffer_idx;
                    (void)inst_buf;
                    (void)count;
                    DBG_PRINT("[%s] Draw instance: buffer=%d, inst_buffer=%d, count=%d\n", 
                           node_name(node->id), buffer_idx, inst_buf, count);
                }
                break;
            }
                
            case OP_DRAW_TEXT: {
                void* data = get_payload(msg.payload_idx);
                if (data) {
                    char* text = (char*)data;
                    (void)text;
                    /* Payload is null-terminated, max HS_PAYLOAD_SIZE chars */
                    DBG_PRINT("[%s] Draw text: \"%.*s\"\n", node_name(node->id), 
                           (int)msg.payload_len, text);
                }
                break;
            }
                
            default:
                handled = 0;
                break;
        }
        emit_ack(node->id, &msg, handled ? 0 : 1);
        processed++;
    }
    
    return processed;
}

void texture_node_init(Node* node) {
    memset(&texture_state, 0, sizeof(TextureState));
    node->state = &texture_state;
    node->process_fn = texture_node_process;
    node->reset_fn = texture_node_reset;
}

void texture_node_reset(Node* node) {
    (void)node;
    memset(&texture_state, 0, sizeof(TextureState));
}

int texture_node_process(Node* node) {
    int processed = 0;
    Message msg;
    
    while (mq_pop(&node->inbox, &msg)) {
        u8 handled = 1;
        switch (msg.op) {
            case OP_LOAD_TEXTURE:
                texture_state.texture_handles[msg.payload_idx & 0xF] = msg.payload_idx;
                texture_state.textures_loaded |= (1 << (msg.payload_idx & 0xF));
                DBG_PRINT("[%s] Loaded texture #%d\n", node_name(node->id), (int)(msg.payload_idx & 0xF));
                break;
                
            case OP_SET_TARGET: {
                void* data = get_payload(msg.payload_idx);
                if (data) {
                    u8 tex = 0;
                    u8 depth = 0;
                    if (!hs_unpack_u8x2(data, msg.payload_len, &tex, &depth)) break;
                    texture_state.current_target = tex;
                    texture_state.current_depth_target = depth;
                } else {
                    texture_state.current_target = (msg.payload_idx & 0xF);
                    texture_state.current_depth_target = 0xF;  /* No depth */
                }
                DBG_PRINT("[%s] Set target: #%d, depth=#%d\n", node_name(node->id), 
                          texture_state.current_target, texture_state.current_depth_target);
                break;
            }
                
            case OP_SHOW_TEXTURE:
                texture_state.show_texture = (msg.payload_idx & 0xF);
                DBG_PRINT("[%s] Show texture: #%d\n", node_name(node->id), texture_state.show_texture);
                break;
                
            case OP_TEXTURE_FILTER: {
                u8 tex_idx = msg.payload_idx & 0xF;
                u8 filter = (msg.payload_idx >> 4) & 1;
                texture_state.texture_filter[tex_idx] = filter;
                DBG_PRINT("[%s] Texture #%d filter: %s\n", node_name(node->id), tex_idx, filter ? "linear" : "nearest");
                break;
            }
                
            case OP_TEXTURE_WRAP: {
                u8 tex_idx = msg.payload_idx & 0xF;
                u8 wrap = (msg.payload_idx >> 4) & 1;
                texture_state.texture_wrap[tex_idx] = wrap;
                DBG_PRINT("[%s] Texture #%d wrap: %s\n", node_name(node->id), tex_idx, wrap ? "repeat" : "clamp");
                break;
            }
                 
            default:
                handled = 0;
                break;
        }
        emit_ack(node->id, &msg, handled ? 0 : 1);
        processed++;
    }
    
    return processed;
}

void output_node_init(Node* node) {
    memset(&output_state, 0, sizeof(OutputState));
    output_state.clear_color[0] = 0.0f;
    output_state.clear_color[1] = 0.0f;
    output_state.clear_color[2] = 0.0f;
    output_state.clear_color[3] = 1.0f;
    output_state.clear_depth = 1.0f;
    node->state = &output_state;
    node->process_fn = output_node_process;
    node->reset_fn = output_node_reset;
}

void output_node_reset(Node* node) {
    (void)node;
    output_state.clear_color[0] = 0.0f;
    output_state.clear_color[1] = 0.0f;
    output_state.clear_color[2] = 0.0f;
    output_state.clear_color[3] = 1.0f;
    output_state.clear_depth = 1.0f;
    output_state.clear_stencil = 0;
}

int output_node_process(Node* node) {
    int processed = 0;
    Message msg;
    
    while (mq_pop(&node->inbox, &msg)) {
        u8 handled = 1;
        switch (msg.op) {
            case OP_CLEAR: {
                void* data = get_payload(msg.payload_idx);
                if (data && msg.payload_len >= 4 * sizeof(f32)) {
                    f32 color[4];
                    if (!hs_unpack_clear_color(data, msg.payload_len, color)) break;
                    output_state.clear_color[0] = color[0];
                    output_state.clear_color[1] = color[1];
                    output_state.clear_color[2] = color[2];
                    output_state.clear_color[3] = color[3];
                    DBG_PRINT("[%s] Clear color: (%.2f, %.2f, %.2f, %.2f)\n",
                           node_name(node->id), color[0], color[1], color[2], color[3]);
                }
                break;
            }
                
            case OP_CLEAR_DS: {
                void* data = get_payload(msg.payload_idx);
                if (data) {
                    f32 depth = 0.0f;
                    u8 stencil = 0;
                    if (!hs_unpack_clear_ds(data, msg.payload_len, &depth, &stencil)) break;
                    output_state.clear_depth = depth;
                    output_state.clear_stencil = stencil;
                    DBG_PRINT("[%s] Clear DS: depth=%.2f, stencil=%d\n",
                           node_name(node->id), output_state.clear_depth, output_state.clear_stencil);
                }
                break;
            }
                
            default:
                handled = 0;
                break;
        }
        emit_ack(node->id, &msg, handled ? 0 : 1);
        processed++;
    }
    
    return processed;
}

void sound_node_init(Node* node) {
    memset(&sound_state, 0, sizeof(SoundState));
    node->state = &sound_state;
    node->process_fn = sound_node_process;
    node->reset_fn = sound_node_reset;
}

void sound_node_reset(Node* node) {
    (void)node;
    memset(&sound_state, 0, sizeof(SoundState));
}

int sound_node_process(Node* node) {
    int processed = 0;
    Message msg;
    
    while (mq_pop(&node->inbox, &msg)) {
        u8 handled = 1;
        switch (msg.op) {
            case OP_SET_CHANNEL: {
                void* data = get_payload(msg.payload_idx);
                if (data) {
                    u8 ch = 0;
                    u8 shader = 0;
                    if (!hs_unpack_u8x2(data, msg.payload_len, &ch, &shader)) break;
                    if (ch < 4) {
                        sound_state.channel_shaders[ch] = shader;
                        DBG_PRINT("[%s] Channel %d -> shader #%d\n", node_name(node->id), ch, shader);
                    }
                }
                break;
            }
                
            default:
                handled = 0;
                break;
        }
        emit_ack(node->id, &msg, handled ? 0 : 1);
        processed++;
    }
    
    return processed;
}

void system_node_init(Node* node) {
    memset(&system_state, 0, sizeof(SystemState));
    node->state = &system_state;
    node->process_fn = system_node_process;
    node->reset_fn = system_node_reset;
}

void system_node_reset(Node* node) {
    (void)node;
    memset(&system_state, 0, sizeof(SystemState));
}

int system_node_process(Node* node) {
    int processed = 0;
    Message msg;

    while (mq_pop(&node->inbox, &msg)) {
        switch (msg.op) {
            case OP_ACK:
                system_state.ack_count++;
                system_state.last_ack_cid = msg.cid;
                system_state.last_ack_from = msg.from;
                system_state.last_ack_op = (u8)msg.payload_idx;
                system_state.last_ack_status = msg.flags;
                break;

            case OP_RESULT:
                system_state.result_count++;
                system_state.last_result_cid = msg.cid;
                system_state.last_result_from = msg.from;
                system_state.last_result_op = msg.flags;
                system_state.last_result_len = (u16)msg.payload_len;
                break;

            case OP_QUEUE_FULL:
                system_state.queue_full_count++;
                system_state.last_queue_full_to = msg.flags;
                system_state.last_queue_full_op = (u8)msg.payload_idx;
                break;

            case OP_ASYNC_DONE: {
                void* data = get_payload(msg.payload_idx);
                if (data) {
                    u32 task_id = 0;
                    u8 type = 0, slot = 0, success = 0;
                    if (hs_unpack_async_done(data, msg.payload_len, &task_id, &type, &slot, &success)) {
                        system_state.async_done_count++;
                        system_state.last_async_task_id = task_id;
                        system_state.last_async_type = type;
                        system_state.last_async_slot = slot;
                        system_state.last_async_success = success;
                    }
                }
                break;
            }

            case OP_ERROR_EX: {
                void* data = get_payload(msg.payload_idx);
                if (data) {
                    u32 code = 0;
                    u8 op = 0, to = 0, from = 0, stage = 0;
                    u32 cid = 0;
                    if (hs_unpack_error_ex(data, msg.payload_len, &code, &op, &to, &from, &stage, &cid)) {
                        system_state.error_count++;
                        system_state.last_error_code = code;
                        system_state.last_error_stage = stage;
                        system_state.last_error_op = op;
                        system_state.last_error_to = to;
                        system_state.last_error_from = from;
#if HS_DEBUG
                        fprintf(stderr, "[ERROR_EX] code=%u stage=%u op=%u to=%u from=%u cid=%u\n",
                                (unsigned)code, (unsigned)stage, (unsigned)op, (unsigned)to, (unsigned)from, (unsigned)cid);
#endif
                    }
                }
                break;
            }

            case OP_ERROR: {
                void* data = get_payload(msg.payload_idx);
                if (data) {
                    fprintf(stderr, "[ERROR] %s\n", (char*)data);
                }
                break;
            }

            case OP_TRACE: {
                void* data = get_payload(msg.payload_idx);
                if (data) {
                    printf("[TRACE] %s\n", (char*)data);
                }
                break;
            }

            case OP_STOP:
                system_state.stopped = 1;
                DBG_PRINT("[%s] STOP command received\n", node_name(node->id));
                break;

            default:
                break;
        }
        processed++;
    }

    return processed;
}
