#ifndef HS_GPU_H
#define HS_GPU_H

#include "hs_core.h"
#include "hs_nodes.h"
#include "hs_math_neon.h"

typedef struct {
    HSSystem    system;
    Message     log_buffer[HS_MAX_MSG_LOG];
    Payload     payload_buffer[HS_MAX_PAYLOADS];
    Node        shader_node;
    Node        buffer_node;
    Node        texture_node;
    Node        output_node;
    Node        sound_node;
    Node        system_node;
} HSGpu;

void hs_gpu_init(HSGpu* gpu);

void hs_gpu_set_shader(HSGpu* gpu, u8 shader);
void hs_gpu_set_param(HSGpu* gpu, u8 idx, vec4 value);
void hs_gpu_set_global(HSGpu* gpu, u8 idx, mat4 value);
void hs_gpu_set_camera(HSGpu* gpu, mat4 view);
void hs_gpu_cull(HSGpu* gpu, u8 mode);
void hs_gpu_blend(HSGpu* gpu, u8 src, u8 dst);
void hs_gpu_alpha(HSGpu* gpu, bool enable);
void hs_gpu_depth(HSGpu* gpu, bool enable);
void hs_gpu_color_mask(HSGpu* gpu, u8 mask);
void hs_gpu_clip(HSGpu* gpu, u16 x, u16 y, u16 w, u16 h);

void hs_gpu_stencil(HSGpu* gpu, u8 op, u8 fail, u8 pass, u8 front);
void hs_gpu_stencil_func(HSGpu* gpu, u8 compare, u8 ref, u8 read_mask, u8 write_mask);
void hs_gpu_depth_compare(HSGpu* gpu, u8 compare, bool write);

void hs_gpu_load_buffer(HSGpu* gpu, u8 index);
void hs_gpu_draw(HSGpu* gpu, u8 buffer);
void hs_gpu_draw_instance(HSGpu* gpu, u8 buffer, u8 instance_buffer, u32 count);
void hs_gpu_draw_text(HSGpu* gpu, const char* text);

void hs_gpu_load_texture(HSGpu* gpu, u8 index);
void hs_gpu_set_target(HSGpu* gpu, u8 texture, u8 depth_buffer);
void hs_gpu_show_texture(HSGpu* gpu, u8 texture);
void hs_gpu_texture_filter(HSGpu* gpu, u8 texture, bool linear);
void hs_gpu_texture_wrap(HSGpu* gpu, u8 texture, bool repeat);

void hs_gpu_clear(HSGpu* gpu, vec4 color);
void hs_gpu_clear_ds(HSGpu* gpu, f32 depth, u8 stencil);

void hs_gpu_set_channel(HSGpu* gpu, u8 channel, u8 shader);

u32 hs_gpu_process(HSGpu* gpu);
u32 hs_gpu_step(HSGpu* gpu);
void hs_gpu_start_recording(HSGpu* gpu);
u32 hs_gpu_stop_recording(HSGpu* gpu);
bool hs_gpu_replay(HSGpu* gpu, Message* msgs, u32 count);
void hs_gpu_clear_log(HSGpu* gpu);
bool hs_gpu_has_overflow(HSGpu* gpu);

void hs_gpu_error(HSGpu* gpu, const char* msg);
void hs_gpu_trace(HSGpu* gpu, const char* msg);
void hs_gpu_stop(HSGpu* gpu);

void hs_gpu_begin_frame(HSGpu* gpu);
void hs_gpu_end_frame(HSGpu* gpu);

#endif
