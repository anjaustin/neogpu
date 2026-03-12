#include "hs_gpu.h"
#include "hs_msg.h"
#include <stdio.h>
#include <string.h>

void hs_gpu_init(HSGpu* gpu) {
    hs_init(&gpu->system, gpu->log_buffer, HS_MAX_MSG_LOG, gpu->payload_buffer);
    hs_nodes_set_system(&gpu->system);
    
    gpu->shader_node.id = NODE_SHADER;
    gpu->buffer_node.id = NODE_BUFFER;
    gpu->texture_node.id = NODE_TEXTURE;
    gpu->output_node.id = NODE_OUTPUT;
    gpu->sound_node.id = NODE_SOUND;
    gpu->system_node.id = NODE_SYSTEM;
    
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

static void hs_gpu_send_simple(HSGpu* gpu, u8 to, OpCode op, u32 payload_idx, u32 payload_len) {
    Message msg = {
        .to = to,
        .from = NODE_CPU,
        .op = op,
        .flags = 0,
        .tick = 0,
        .payload_idx = (u16)payload_idx,
        .payload_len = payload_len
    };
    hs_send(&gpu->system, &msg);
}

bool hs_gpu_send_with_payload(HSGpu* gpu, u8 to, OpCode op, const void* data, u32 len) {
    u32 idx = gpu->system.payload_head;
    u32 cap = gpu->system.payload_capacity ? gpu->system.payload_capacity : (u32)HS_MAX_PAYLOADS;
    gpu->system.payload_head = (gpu->system.payload_head + 1) % cap;
    
    u32 copy_len = len < HS_PAYLOAD_SIZE ? len : HS_PAYLOAD_SIZE;
    memcpy(gpu->system.payloads[idx].data, data, copy_len);
    
    Message msg = {
        .to = to,
        .from = NODE_CPU,
        .op = op,
        .flags = 0,
        .tick = 0,
        .payload_idx = (u16)idx,
        .payload_len = copy_len
    };
    return hs_send(&gpu->system, &msg);
}

void hs_gpu_set_shader(HSGpu* gpu, u8 shader) {
    hs_gpu_send_simple(gpu, NODE_SHADER, OP_SET_SHADER, shader, 0);
}

void hs_gpu_set_param(HSGpu* gpu, u8 idx, vec4 value) {
    u8 data[20];
    hs_pack_set_param(data, (u32)idx, value);
    hs_gpu_send_with_payload(gpu, NODE_SHADER, OP_SET_PARAM, data, sizeof(data));
}

void hs_gpu_set_global(HSGpu* gpu, u8 idx, mat4 value) {
    /* Layout: [u8 global_idx][padding x3][f32 x 16] = 68 bytes, but payload max is 64.
     * For now, global idx 0 = camera. Store idx at bytes 0-3, matrix at 4-67.
     * Since HS_PAYLOAD_SIZE is 64, we can fit idx(4) + 15 floats(60) = 64.
     * Actually 4 + 64 = 68 > 64. So pack idx as u8 prefix and 15.75 floats won't fit.
     * Solution: use the simple message field for the index. */
    float arr[16];
    m4_to_array(value, arr);
    
    /* Allocate payload manually so we can set payload_idx for the index. */
    u32 pidx = gpu->system.payload_head;
    u32 cap = gpu->system.payload_capacity ? gpu->system.payload_capacity : (u32)HS_MAX_PAYLOADS;
    gpu->system.payload_head = (gpu->system.payload_head + 1) % cap;
    u32 copy_len = sizeof(arr) < HS_PAYLOAD_SIZE ? sizeof(arr) : HS_PAYLOAD_SIZE;
    memcpy(gpu->system.payloads[pidx].data, arr, copy_len);
    
    Message msg = {
        .to = NODE_SHADER,
        .from = NODE_CPU,
        .op = OP_SET_GLOBAL,
        .flags = idx,          /* Use flags field for the global index */
        .tick = 0,
        .payload_idx = (u16)pidx,
        .payload_len = copy_len
    };
    hs_send(&gpu->system, &msg);
}

void hs_gpu_set_camera(HSGpu* gpu, mat4 view) {
    float arr[16];
    m4_to_array(view, arr);
    hs_gpu_send_with_payload(gpu, NODE_SHADER, OP_SET_CAMERA, arr, sizeof(arr));
}

void hs_gpu_cull(HSGpu* gpu, u8 mode) {
    hs_gpu_send_simple(gpu, NODE_SHADER, OP_CULL, mode, 0);
}

void hs_gpu_blend(HSGpu* gpu, u8 src, u8 dst) {
    u8 data[2];
    hs_pack_u8x2(data, src, dst);
    hs_gpu_send_with_payload(gpu, NODE_SHADER, OP_BLEND, data, sizeof(data));
}

void hs_gpu_alpha(HSGpu* gpu, bool enable) {
    hs_gpu_send_simple(gpu, NODE_SHADER, OP_ALPHA, enable ? 1 : 0, 0);
}

void hs_gpu_depth(HSGpu* gpu, bool enable) {
    hs_gpu_send_simple(gpu, NODE_SHADER, OP_DEPTH, enable ? 1 : 0, 0);
}

void hs_gpu_color_mask(HSGpu* gpu, u8 mask) {
    hs_gpu_send_simple(gpu, NODE_SHADER, OP_COLOR_MASK, mask, 0);
}

void hs_gpu_clip(HSGpu* gpu, u16 x, u16 y, u16 w, u16 h) {
    u8 data[8];
    hs_pack_u16x4(data, x, y, w, h);
    hs_gpu_send_with_payload(gpu, NODE_SHADER, OP_CLIP, data, sizeof(data));
}

void hs_gpu_stencil(HSGpu* gpu, u8 op, u8 fail, u8 pass, u8 front) {
    u8 data[4];
    hs_pack_stencil(data, op, fail, pass, front);
    hs_gpu_send_with_payload(gpu, NODE_SHADER, OP_STENCIL, data, sizeof(data));
}

void hs_gpu_stencil_func(HSGpu* gpu, u8 compare, u8 ref, u8 read_mask, u8 write_mask) {
    u8 data[4];
    hs_pack_stencil_func(data, compare, ref, read_mask, write_mask);
    hs_gpu_send_with_payload(gpu, NODE_SHADER, OP_STENCIL_FUNC, data, sizeof(data));
}

void hs_gpu_depth_compare(HSGpu* gpu, u8 compare, bool write) {
    u8 data[2];
    hs_pack_depth_compare(data, compare, write);
    hs_gpu_send_with_payload(gpu, NODE_SHADER, OP_DEPTH_COMPARE, data, sizeof(data));
}

void hs_gpu_load_buffer(HSGpu* gpu, u8 index) {
    hs_gpu_send_simple(gpu, NODE_BUFFER, OP_LOAD_BUFFER, index, 0);
}

void hs_gpu_draw(HSGpu* gpu, u8 buffer) {
    hs_gpu_send_simple(gpu, NODE_BUFFER, OP_DRAW, buffer, 0);
}

void hs_gpu_draw_instance(HSGpu* gpu, u8 buffer, u8 instance_buffer, u32 count) {
    u8 data[4];
    hs_pack_draw_instance(data, buffer, instance_buffer, count);
    hs_gpu_send_with_payload(gpu, NODE_BUFFER, OP_DRAW_INSTANCE, data, sizeof(data));
}

void hs_gpu_draw_text(HSGpu* gpu, const char* text) {
    u32 len = 0;
    while (text[len] && len < HS_PAYLOAD_SIZE - 1) len++;
    u8 data[HS_PAYLOAD_SIZE];
    memcpy(data, text, len);
    data[len] = '\0';
    hs_gpu_send_with_payload(gpu, NODE_BUFFER, OP_DRAW_TEXT, data, len + 1);
}

void hs_gpu_load_texture(HSGpu* gpu, u8 index) {
    hs_gpu_send_simple(gpu, NODE_TEXTURE, OP_LOAD_TEXTURE, index, 0);
}

void hs_gpu_set_target(HSGpu* gpu, u8 texture, u8 depth_buffer) {
    u8 data[2];
    hs_pack_u8x2(data, texture, depth_buffer);
    hs_gpu_send_with_payload(gpu, NODE_TEXTURE, OP_SET_TARGET, data, sizeof(data));
}

void hs_gpu_show_texture(HSGpu* gpu, u8 texture) {
    hs_gpu_send_simple(gpu, NODE_TEXTURE, OP_SHOW_TEXTURE, texture, 0);
}

void hs_gpu_texture_filter(HSGpu* gpu, u8 texture, bool linear) {
    u8 data = texture | (linear ? 0x10 : 0);
    hs_gpu_send_simple(gpu, NODE_TEXTURE, OP_TEXTURE_FILTER, data, 0);
}

void hs_gpu_texture_wrap(HSGpu* gpu, u8 texture, bool repeat) {
    u8 data = texture | (repeat ? 0x10 : 0);
    hs_gpu_send_simple(gpu, NODE_TEXTURE, OP_TEXTURE_WRAP, data, 0);
}

void hs_gpu_clear(HSGpu* gpu, vec4 color) {
    hs_gpu_send_with_payload(gpu, NODE_OUTPUT, OP_CLEAR, &color, sizeof(vec4));
}

void hs_gpu_clear_ds(HSGpu* gpu, f32 depth, u8 stencil) {
    u8 data[8];
    hs_pack_clear_ds(data, depth, stencil);
    hs_gpu_send_with_payload(gpu, NODE_OUTPUT, OP_CLEAR_DS, data, sizeof(data));
}

void hs_gpu_set_channel(HSGpu* gpu, u8 channel, u8 shader) {
    u8 data[2];
    hs_pack_u8x2(data, channel, shader);
    hs_gpu_send_with_payload(gpu, NODE_SOUND, OP_SET_CHANNEL, data, sizeof(data));
}

u32 hs_gpu_process(HSGpu* gpu) {
    u32 total = 0;
    while (hs_step(&gpu->system) > 0) {
        total++;
        if (total > 1000) break;
    }
    return total;
}

u32 hs_gpu_step(HSGpu* gpu) {
    return hs_step(&gpu->system);
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
    hs_gpu_send_with_payload(gpu, NODE_SYSTEM, OP_ERROR, (const u8*)msg, strlen(msg) + 1);
}

void hs_gpu_trace(HSGpu* gpu, const char* msg) {
    hs_gpu_send_with_payload(gpu, NODE_SYSTEM, OP_TRACE, (const u8*)msg, strlen(msg) + 1);
}

void hs_gpu_stop(HSGpu* gpu) {
    hs_gpu_send_simple(gpu, NODE_SYSTEM, OP_STOP, 0, 0);
}

void hs_gpu_begin_frame(HSGpu* gpu) {
    (void)gpu;
}

void hs_gpu_end_frame(HSGpu* gpu) {
    (void)gpu;
}
