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

static inline bool hs_unpack_set_param(const void* data, u32 len, u32* out_param_idx, f32 out_xyzw[4]) {
    if (!data || len < 20) return false;
    memcpy(out_param_idx, data, 4);
    memcpy(out_xyzw, (const u8*)data + 4, 16);
    return true;
}

static inline bool hs_unpack_u8x2(const void* data, u32 len, u8* out_a, u8* out_b) {
    if (!data || len < 2) return false;
    const u8* b = (const u8*)data;
    if (out_a) *out_a = b[0];
    if (out_b) *out_b = b[1];
    return true;
}

static inline bool hs_unpack_u16x4(const void* data, u32 len, u16 out_xywh[4]) {
    if (!data || len < 8) return false;
    memcpy(out_xywh, data, 8);
    return true;
}

static inline bool hs_unpack_u8x4(const void* data, u32 len, u8 out_vals[4]) {
    if (!data || len < 4) return false;
    memcpy(out_vals, data, 4);
    return true;
}

static inline bool hs_unpack_draw_instance(const void* data, u32 len, u8* out_buffer, u8* out_instance_buffer, u32* out_count) {
    if (!data || len < 4) return false;
    const u8* v = (const u8*)data;
    if (out_buffer) *out_buffer = v[0];
    if (out_instance_buffer) *out_instance_buffer = v[1];
    if (out_count) *out_count = (u32)v[2] | ((u32)v[3] << 8);
    return true;
}

static inline bool hs_unpack_clear_color(const void* data, u32 len, f32 out_rgba[4]) {
    if (!data || len < 16) return false;
    memcpy(out_rgba, data, 16);
    return true;
}

static inline bool hs_unpack_clear_ds(const void* data, u32 len, f32* out_depth, u8* out_stencil) {
    if (!data || len < 8) return false;
    if (out_depth) memcpy(out_depth, data, 4);
    if (out_stencil) *out_stencil = ((const u8*)data)[4];
    return true;
}

/*
 * Structured error payload (OP_ERROR_EX):
 * [u32 code][u8 op][u8 to][u8 from][u8 stage][u32 cid][u32 arg0][u32 arg1][char msg[32]]
 */
static inline void hs_pack_error_ex(u8 out[52], u32 code, u8 op, u8 to, u8 from, u8 stage, u32 cid, u32 arg0, u32 arg1, const char* msg) {
    memset(out, 0, 52);
    memcpy(&out[0], &code, 4);
    out[4] = op;
    out[5] = to;
    out[6] = from;
    out[7] = stage;
    memcpy(&out[8], &cid, 4);
    memcpy(&out[12], &arg0, 4);
    memcpy(&out[16], &arg1, 4);
    if (msg) {
        /* msg[32] at offset 20 */
        strncpy((char*)&out[20], msg, 31);
        out[20 + 31] = 0;
    }
}

static inline bool hs_unpack_error_ex(const void* data, u32 len, u32* out_code, u8* out_op, u8* out_to, u8* out_from, u8* out_stage, u32* out_cid) {
    if (!data || len < 52) return false;
    const u8* b = (const u8*)data;
    if (out_code) memcpy(out_code, &b[0], 4);
    if (out_op) *out_op = b[4];
    if (out_to) *out_to = b[5];
    if (out_from) *out_from = b[6];
    if (out_stage) *out_stage = b[7];
    if (out_cid) memcpy(out_cid, &b[8], 4);
    return true;
}

/* Async completion payload (OP_ASYNC_DONE): [u32 task_id][u8 type][u8 slot][u8 success][u8 reserved] */
static inline void hs_pack_async_done(u8 out[8], u32 task_id, u8 type, u8 slot, u8 success) {
    memcpy(&out[0], &task_id, 4);
    out[4] = type;
    out[5] = slot;
    out[6] = success;
    out[7] = 0;
}

static inline bool hs_unpack_async_done(const void* data, u32 len, u32* out_task_id, u8* out_type, u8* out_slot, u8* out_success) {
    if (!data || len < 8) return false;
    const u8* b = (const u8*)data;
    if (out_task_id) memcpy(out_task_id, &b[0], 4);
    if (out_type) *out_type = b[4];
    if (out_slot) *out_slot = b[5];
    if (out_success) *out_success = b[6];
    return true;
}

#endif
