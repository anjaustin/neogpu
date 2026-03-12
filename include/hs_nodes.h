#ifndef HS_NODES_H
#define HS_NODES_H

#include "hs_core.h"
#include "hs_math_neon.h"

#define NODE_CPU      0
#define NODE_SHADER  1
#define NODE_BUFFER  2
#define NODE_TEXTURE 3
#define NODE_OUTPUT  4
#define NODE_SOUND   5
#define NODE_SYSTEM  6

typedef struct {
    u8   current_shader;
    f32  params[16][4];
    u16  params_set;
    f32  camera_view_proj[16];
    /* Render state */
    u8   cull_mode;
    u8   blend_src;
    u8   blend_dst;
    bool depth_enabled;
    u8   depth_compare;
    bool depth_write;
    u8   color_mask;
    u16  clip_x, clip_y, clip_w, clip_h;
    /* Stencil state */
    u8   stencil_op;
    u8   stencil_fail;
    u8   stencil_pass;
    u8   stencil_front;
    u8   stencil_compare;
    u8   stencil_ref;
    u8   stencil_read_mask;
    u8   stencil_write_mask;
} ShaderState;

typedef struct {
    u32  buffer_handles[16];
    u8   buffers_loaded;
    u8   last_draw_buffer;
    u32  last_draw_tri_start;
    u32  last_draw_tri_count;
} BufferState;

typedef struct {
    u32  texture_handles[16];
    u32  depth_buffer_handles[16];
    u8   textures_loaded;
    u8   current_target;
    u8   current_depth_target;
    u8   show_texture;
    /* Texture sampler state */
    u8   texture_filter[16];   /* 0=nearest, 1=linear */
    u8   texture_wrap[16];     /* 0=clamp, 1=repeat */
} TextureState;

typedef struct {
    f32  clear_color[4];
    f32  clear_depth;
    u8   clear_stencil;
} OutputState;

typedef struct {
    u8 channel_shaders[4];
} SoundState;

typedef struct {
    u8 stopped;
    u32 ack_count;
    u32 last_ack_cid;
    u8  last_ack_from;
    u8  last_ack_op;
    u8  last_ack_status;
    u8  pad0;
    u32 result_count;
    u32 last_result_cid;
    u8  last_result_from;
    u8  last_result_op;
    u16 last_result_len;
} SystemState;

int shader_node_process(Node* node);
int buffer_node_process(Node* node);
int texture_node_process(Node* node);
int output_node_process(Node* node);
int sound_node_process(Node* node);
int system_node_process(Node* node);

void shader_node_init(Node* node);
void buffer_node_init(Node* node);
void texture_node_init(Node* node);
void output_node_init(Node* node);
void sound_node_init(Node* node);
void system_node_init(Node* node);

void shader_node_reset(Node* node);
void buffer_node_reset(Node* node);
void texture_node_reset(Node* node);
void output_node_reset(Node* node);
void sound_node_reset(Node* node);
void system_node_reset(Node* node);

void hs_nodes_set_system(HSSystem* sys);
const char* node_name(u8 id);

#endif
