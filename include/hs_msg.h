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

/* Fence result payload (OP_RESULT for OP_FENCE): [u32 tick][u8 channel][u8 reserved][u16 reserved] */
static inline void hs_pack_result_fence(u8 out[8], u32 tick, u8 channel) {
    memcpy(&out[0], &tick, 4);
    out[4] = channel;
    out[5] = 0;
    out[6] = 0;
    out[7] = 0;
}

static inline bool hs_unpack_result_fence(const void* data, u32 len, u32* out_tick, u8* out_channel) {
    if (!data || len < 8) return false;
    const u8* b = (const u8*)data;
    if (out_tick) memcpy(out_tick, &b[0], 4);
    if (out_channel) *out_channel = b[4];
    return true;
}

/* Generic u32x2 result payload: [u32 a][u32 b] */
static inline void hs_pack_u32x2(u8 out[8], u32 a, u32 b) {
    memcpy(&out[0], &a, 4);
    memcpy(&out[4], &b, 4);
}

static inline bool hs_unpack_u32x2(const void* data, u32 len, u32* out_a, u32* out_b) {
    if (!data || len < 8) return false;
    const u8* b = (const u8*)data;
    if (out_a) memcpy(out_a, &b[0], 4);
    if (out_b) memcpy(out_b, &b[4], 4);
    return true;
}

/* OP_SET_RECORD_MASK payload: [u32 mask] */
static inline void hs_pack_set_record_mask(u8 out[4], u32 mask) {
    memcpy(&out[0], &mask, 4);
}

static inline bool hs_unpack_set_record_mask(const void* data, u32 len, u32* out_mask) {
    if (!data || len < 4) return false;
    if (out_mask) memcpy(out_mask, data, 4);
    return true;
}

/* OP_SET_CHAN_BUDGET payload: [u8 channel][u8 rsv0][u8 rsv1][u8 rsv2][u32 budget] */
static inline void hs_pack_set_chan_budget(u8 out[8], u8 channel, u32 budget) {
    out[0] = channel;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    memcpy(&out[4], &budget, 4);
}

static inline bool hs_unpack_set_chan_budget(const void* data, u32 len, u8* out_channel, u32* out_budget) {
    if (!data || len < 8) return false;
    const u8* b = (const u8*)data;
    if (out_channel) *out_channel = b[0];
    if (out_budget) memcpy(out_budget, &b[4], 4);
    return true;
}

/* OP_SET_BLOCK_POLICY payload: [u8 channel][u8 block] */
static inline void hs_pack_set_block_policy(u8 out[2], u8 channel, u8 block) {
    out[0] = channel;
    out[1] = block;
}

static inline bool hs_unpack_set_block_policy(const void* data, u32 len, u8* out_channel, u8* out_block) {
    if (!data || len < 2) return false;
    const u8* b = (const u8*)data;
    if (out_channel) *out_channel = b[0];
    if (out_block) *out_block = b[1];
    return true;
}

/* OP_QUERY_STATS result payload (64B)
 * Offsets:
 *  0 tick(u32)
 *  4 log_head(u32)
 *  8 record_mask(u32)
 * 12 budget_rt(u32)
 * 16 budget_render(u32)
 * 20 budget_telem(u32)
 * 24 dropped_error_ex(u32)
 * 28 dropped_queue_full(u32)
 * 32 dropped_system_nonrt(u32)
 * 36 dropped_result(u32)
 * 40 producer_count(u32)
 * 44 flags(u32): bit0 recording, bit1 validate_on_send, bit2 block_rt, bit3 block_render, bit4 block_telem
 */
static inline void hs_pack_result_system_stats(u8 out[64], u32 tick, u32 log_head, u32 record_mask,
                                              u32 bud_rt, u32 bud_render, u32 bud_telem,
                                              u32 dropped_error_ex, u32 dropped_queue_full, u32 dropped_system_nonrt, u32 dropped_result,
                                              u32 producer_count, u32 flags) {
    memset(out, 0, 64);
    memcpy(&out[0], &tick, 4);
    memcpy(&out[4], &log_head, 4);
    memcpy(&out[8], &record_mask, 4);
    memcpy(&out[12], &bud_rt, 4);
    memcpy(&out[16], &bud_render, 4);
    memcpy(&out[20], &bud_telem, 4);
    memcpy(&out[24], &dropped_error_ex, 4);
    memcpy(&out[28], &dropped_queue_full, 4);
    memcpy(&out[32], &dropped_system_nonrt, 4);
    memcpy(&out[36], &dropped_result, 4);
    memcpy(&out[40], &producer_count, 4);
    memcpy(&out[44], &flags, 4);
}

static inline bool hs_unpack_result_system_stats(const void* data, u32 len,
                                                u32* out_tick, u32* out_log_head, u32* out_record_mask,
                                                u32 out_budgets3[3], u32 out_dropped4[4], u32* out_producer_count, u32* out_flags) {
    if (!data || len < 64) return false;
    const u8* b = (const u8*)data;
    if (out_tick) memcpy(out_tick, &b[0], 4);
    if (out_log_head) memcpy(out_log_head, &b[4], 4);
    if (out_record_mask) memcpy(out_record_mask, &b[8], 4);
    if (out_budgets3) {
        memcpy(&out_budgets3[0], &b[12], 4);
        memcpy(&out_budgets3[1], &b[16], 4);
        memcpy(&out_budgets3[2], &b[20], 4);
    }
    if (out_dropped4) {
        memcpy(&out_dropped4[0], &b[24], 4);
        memcpy(&out_dropped4[1], &b[28], 4);
        memcpy(&out_dropped4[2], &b[32], 4);
        memcpy(&out_dropped4[3], &b[36], 4);
    }
    if (out_producer_count) memcpy(out_producer_count, &b[40], 4);
    if (out_flags) memcpy(out_flags, &b[44], 4);
    return true;
}

/* OP_QUERY_FABRIC result payload (64B)
 * Layout: u32 arrays per channel (RT, RENDER, TELEM):
 *  0..11  spsc_ok[3], spsc_full[3], mpsc_ok[3], submit_full[3]
 * 48 producer_count(u32)
 * 52 bp_waiters(u32)
 */
static inline void hs_pack_result_fabric(u8 out[64], const u32 spsc_ok3[3], const u32 spsc_full3[3],
                                        const u32 mpsc_ok3[3], const u32 submit_full3[3], u32 producer_count, u32 bp_waiters) {
    memset(out, 0, 64);
    memcpy(&out[0], spsc_ok3, 12);
    memcpy(&out[12], spsc_full3, 12);
    memcpy(&out[24], mpsc_ok3, 12);
    memcpy(&out[36], submit_full3, 12);
    memcpy(&out[48], &producer_count, 4);
    memcpy(&out[52], &bp_waiters, 4);
}

static inline bool hs_unpack_result_fabric(const void* data, u32 len, u32 spsc_ok3[3], u32 spsc_full3[3],
                                          u32 mpsc_ok3[3], u32 submit_full3[3], u32* out_producer_count, u32* out_bp_waiters) {
    if (!data || len < 64) return false;
    const u8* b = (const u8*)data;
    if (spsc_ok3) memcpy(spsc_ok3, &b[0], 12);
    if (spsc_full3) memcpy(spsc_full3, &b[12], 12);
    if (mpsc_ok3) memcpy(mpsc_ok3, &b[24], 12);
    if (submit_full3) memcpy(submit_full3, &b[36], 12);
    if (out_producer_count) memcpy(out_producer_count, &b[48], 4);
    if (out_bp_waiters) memcpy(out_bp_waiters, &b[52], 4);
    return true;
}

#endif
