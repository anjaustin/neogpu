/*
 * NeoGPU - Message Packing Helpers
 *
 * Single-source helpers for packing opcode payloads.
 * These helpers match the current message ABI documented in docs/MESSAGE_ABI.md.
 */

#ifndef HS_MSG_H
#define HS_MSG_H

#include "hs_core.h"
#include "hs_math_neon.h"

static inline void hs_pack_set_param(u8 out[20], u32 param_idx, vec4 value) {
    /* Layout: [u32 param_idx][f32 x][f32 y][f32 z][f32 w] */
    memcpy(&out[0], &param_idx, 4);
    f32 arr[4];
    arr[0] = vgetq_lane_f32(value, 0);
    arr[1] = vgetq_lane_f32(value, 1);
    arr[2] = vgetq_lane_f32(value, 2);
    arr[3] = vgetq_lane_f32(value, 3);
    memcpy(&out[4], arr, 16);
}

static inline void hs_pack_u8x2(u8 out[2], u8 a, u8 b) {
    out[0] = a;
    out[1] = b;
}

static inline void hs_pack_u16x4(u8 out[8], u16 x, u16 y, u16 w, u16 h) {
    u16 vals[4] = {x, y, w, h};
    memcpy(out, vals, 8);
}

static inline void hs_pack_stencil(u8 out[4], u8 op, u8 fail, u8 pass, u8 front) {
    out[0] = op;
    out[1] = fail;
    out[2] = pass;
    out[3] = front;
}

static inline void hs_pack_stencil_func(u8 out[4], u8 compare, u8 ref, u8 read_mask, u8 write_mask) {
    out[0] = compare;
    out[1] = ref;
    out[2] = read_mask;
    out[3] = write_mask;
}

static inline void hs_pack_depth_compare(u8 out[2], u8 compare, bool write) {
    out[0] = compare;
    out[1] = write ? 1 : 0;
}

static inline void hs_pack_draw_instance(u8 out[4], u8 buffer, u8 instance_buffer, u32 count) {
    out[0] = buffer;
    out[1] = instance_buffer;
    out[2] = (u8)((count >> 0) & 0xFF);
    out[3] = (u8)((count >> 8) & 0xFF);
}

static inline void hs_pack_clear_ds(u8 out[8], f32 depth, u8 stencil) {
    memcpy(&out[0], &depth, 4);
    out[4] = stencil;
    out[5] = 0;
    out[6] = 0;
    out[7] = 0;
}

#endif
