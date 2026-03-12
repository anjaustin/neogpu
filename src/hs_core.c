#include "hs_core.h"
#include "hs_nodes.h"
#include "hs_msg.h"
#include "hs_render.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    u8  magic[8];      /* "HSCAP1\0" */
    u32 endian;        /* 0x01020304 */
    u32 version;       /* 1 */
    u32 msg_size;
    u32 payload_size;
    u32 count;
} HSCaptureHeader;

static void hs_capture_header_init(HSCaptureHeader* h, u32 count) {
    memset(h, 0, sizeof(*h));
    memcpy(h->magic, "HSCAP1\0", 7);
    h->endian = 0x01020304u;
    h->version = 2u;
    h->msg_size = (u32)sizeof(Message);
    h->payload_size = (u32)sizeof(Payload);
    h->count = count;
}

static bool hs_capture_header_valid(const HSCaptureHeader* h) {
    if (!h) return false;
    if (memcmp(h->magic, "HSCAP1\0", 7) != 0) return false;
    if (h->endian != 0x01020304u) return false;
    if (h->version != 2u) return false;
    if (h->msg_size != (u32)sizeof(Message)) return false;
    if (h->payload_size != (u32)sizeof(Payload)) return false;
    return true;
}

typedef enum {
    HS_ERR_VALIDATE = 1,
    HS_ERR_ROUTE = 2,
    HS_ERR_QUEUE_FULL = 3,
} HSErrorCode;

typedef enum {
    HS_ERR_STAGE_SEND = 1,
    HS_ERR_STAGE_REPLAY = 2,
} HSErrorStage;

static HSChannel hs_default_channel_for_op(OpCode op);

static inline Node* hs_node_by_id(HSSystem* sys, u8 id) {
    return sys ? sys->node_map[id] : NULL;
}

static inline bool hs_system_has_node(const HSSystem* sys, u8 id) {
    return sys ? (sys->node_map[id] != NULL) : false;
}

static void hs_report_error_ex(HSSystem* sys, const Message* bad_msg, u32 code, u8 stage, const char* detail) {
    if (!sys) return;
    Node* sys_node = hs_node_by_id(sys, NODE_SYSTEM);
    if (!sys_node) {
        if (detail) fprintf(stderr, "[HS] ERROR_EX: %s\n", detail);
        return;
    }

    u8 payload[52];
    hs_pack_error_ex(
        payload,
        code,
        bad_msg ? bad_msg->op : 0,
        bad_msg ? bad_msg->to : 0,
        bad_msg ? bad_msg->from : 0,
        stage,
        bad_msg ? bad_msg->cid : 0,
        bad_msg ? bad_msg->payload_len : 0,
        bad_msg ? bad_msg->payload_idx : 0,
        detail
    );

    u16 idx = 0;
    u32 copy_len = 0;
    if (!hs_payload_alloc_and_copy(sys, payload, sizeof(payload), &idx, &copy_len)) return;

    Message emsg = {
        .to = NODE_SYSTEM,
        .from = NODE_CPU,
        .op = OP_ERROR_EX,
        .flags = 0,
        .cid = bad_msg ? bad_msg->cid : 0,
        .tick = (u16)sys->tick,
        .payload_idx = idx,
        .payload_len = copy_len,
        .channel = CHAN_TELEM,
    };

    if (!mq_push(&sys_node->inbox, &emsg)) {
        sys->dropped_error_ex++;
    }
}

static void hs_report_queue_full(HSSystem* sys, const Message* msg, u8 dest_node) {
    if (!sys) return;
    Node* sys_node = hs_node_by_id(sys, NODE_SYSTEM);
    if (!sys_node) return;

    Message q = {
        .to = NODE_SYSTEM,
        .from = NODE_CPU,
        .op = OP_QUEUE_FULL,
        .flags = dest_node,
        .cid = msg ? msg->cid : 0,
        .tick = (u16)sys->tick,
        .payload_idx = (u16)(msg ? msg->op : 0),
        .payload_len = 0,
        .channel = CHAN_TELEM,
    };
    if (!mq_push(&sys_node->inbox, &q)) {
        sys->dropped_queue_full++;
    }
}

static void hs_render_record(HSSystem* sys, const Message* msg) {
    if (!sys || !sys->render_list || !msg) return;

    HSRenderCmd cmd;
    memset(&cmd, 0, sizeof(cmd));

    switch ((OpCode)msg->op) {
        case OP_FRAME_BEGIN:
            hs_render_reset(sys->render_list);
            cmd.op = HS_RC_FRAME_BEGIN;
            break;

        case OP_FRAME_END:
            cmd.op = HS_RC_FRAME_END;
            break;

        case OP_PRESENT:
            cmd.op = HS_RC_PRESENT;
            break;

        case OP_CULL:
            cmd.op = HS_RC_SET_CULL;
            cmd.a = (u8)msg->payload_idx;
            break;

        case OP_BLEND: {
            if (msg->payload_len != 2 || msg->payload_idx >= sys->payload_capacity) return;
            u8 src = 0, dst = 0;
            if (!hs_unpack_u8x2(sys->payloads[msg->payload_idx].data, msg->payload_len, &src, &dst)) return;
            cmd.op = HS_RC_SET_BLEND;
            cmd.a = src;
            cmd.b = dst;
            break;
        }

        case OP_ALPHA:
            cmd.op = HS_RC_SET_ALPHA;
            cmd.a = (u8)(msg->payload_idx ? 1 : 0);
            break;

        case OP_DEPTH:
            cmd.op = HS_RC_SET_DEPTH;
            cmd.a = (u8)(msg->payload_idx ? 1 : 0);
            break;

        case OP_DEPTH_COMPARE: {
            if (msg->payload_len != 2 || msg->payload_idx >= sys->payload_capacity) return;
            u8 cmp = 0, wr = 0;
            if (!hs_unpack_u8x2(sys->payloads[msg->payload_idx].data, msg->payload_len, &cmp, &wr)) return;
            cmd.op = HS_RC_SET_DEPTH_COMPARE;
            cmd.a = cmp;
            cmd.b = wr;
            break;
        }

        case OP_COLOR_MASK:
            cmd.op = HS_RC_SET_COLOR_MASK;
            cmd.a = (u8)msg->payload_idx;
            break;

        case OP_CLIP: {
            if (msg->payload_len != 8 || msg->payload_idx >= sys->payload_capacity) return;
            u16 v[4];
            if (!hs_unpack_u16x4(sys->payloads[msg->payload_idx].data, msg->payload_len, v)) return;
            cmd.op = HS_RC_SET_CLIP;
            cmd.x = v[0];
            cmd.y = v[1];
            cmd.payload_idx = v[2];
            cmd.payload_len = v[3];
            break;
        }

        case OP_TEXTURE_FILTER:
            cmd.op = HS_RC_SET_TEX_FILTER;
            cmd.a = (u8)(msg->payload_idx & 0x0F);
            cmd.b = (u8)((msg->payload_idx >> 4) & 1);
            break;

        case OP_TEXTURE_WRAP:
            cmd.op = HS_RC_SET_TEX_WRAP;
            cmd.a = (u8)(msg->payload_idx & 0x0F);
            cmd.b = (u8)((msg->payload_idx >> 4) & 1);
            break;

        case OP_CLEAR: {
            if (msg->payload_len != 16 || msg->payload_idx >= sys->payload_capacity) return;
            f32 rgba[4];
            if (!hs_unpack_clear_color(sys->payloads[msg->payload_idx].data, msg->payload_len, rgba)) return;
            cmd.op = HS_RC_CLEAR;
            cmd.f0 = rgba[0];
            cmd.f1 = rgba[1];
            cmd.f2 = rgba[2];
            cmd.f3 = rgba[3];
            break;
        }

        case OP_CLEAR_DS: {
            if (msg->payload_len != 8 || msg->payload_idx >= sys->payload_capacity) return;
            f32 depth = 0.0f;
            u8 stencil = 0;
            if (!hs_unpack_clear_ds(sys->payloads[msg->payload_idx].data, msg->payload_len, &depth, &stencil)) return;
            cmd.op = HS_RC_CLEAR_DS;
            cmd.f0 = depth;
            cmd.a = stencil;
            break;
        }

        case OP_DRAW:
            cmd.op = HS_RC_DRAW;
            cmd.a = (u8)(msg->payload_idx & 0xF);
            break;

        case OP_DRAW_INSTANCE: {
            if (msg->payload_len != 4 || msg->payload_idx >= sys->payload_capacity) return;
            u8 b = 0, ib = 0;
            u32 count = 0;
            if (!hs_unpack_draw_instance(sys->payloads[msg->payload_idx].data, msg->payload_len, &b, &ib, &count)) return;
            cmd.op = HS_RC_DRAW_INSTANCE;
            cmd.a = b;
            cmd.b = ib;
            cmd.x = count;
            break;
        }

        case OP_DRAW_TEXT:
            cmd.op = HS_RC_DRAW_TEXT;
            cmd.payload_idx = msg->payload_idx;
            cmd.payload_len = (u16)msg->payload_len;
            break;

        case OP_SHOW_TEXTURE:
            cmd.op = HS_RC_SHOW_TEXTURE;
            cmd.a = (u8)(msg->payload_idx & 0xF);
            break;

        default:
            return;
    }

    (void)hs_render_push(sys->render_list, &cmd);
}

static bool hs_submit_enqueue(HSSystem* sys, HSChannel ch, const Message* msg, const void* payload, u32 payload_len) {
    if (!sys || !msg) return false;

    /* fast reject invalid destination/op */
    if (msg->op >= OP_COUNT) return false;
    if (!hs_system_has_node(sys, msg->to)) return false;

    u32 len = payload_len;
    if (len > HS_PAYLOAD_SIZE) len = HS_PAYLOAD_SIZE;

    HSSubmitQueue* q = &sys->submit[(u32)ch];
    u32 pos = atomic_load_explicit(&q->enqueue_pos, memory_order_relaxed);
    for (;;) {
        HSSubmitSlot* slot = &q->slots[pos % HS_SUBMIT_SIZE];
        u32 seq = atomic_load_explicit(&slot->seq, memory_order_acquire);
        intptr_t dif = (intptr_t)seq - (intptr_t)pos;
        if (dif == 0) {
            if (atomic_compare_exchange_weak_explicit(
                    &q->enqueue_pos, &pos, pos + 1,
                    memory_order_relaxed, memory_order_relaxed)) {
                /* claimed */
                slot->msg = *msg;
                slot->payload_len = len;
                if (len && payload) memcpy(slot->payload, payload, len);
                atomic_store_explicit(&slot->seq, pos + 1, memory_order_release);
                atomic_fetch_add_explicit(&sys->mpsc_ok[(u32)ch], 1, memory_order_relaxed);
                return true;
            }
            continue;
        }
        if (dif < 0) {
            atomic_fetch_add_explicit(&sys->submit_full[(u32)ch], 1, memory_order_relaxed);
            return false;
        }
        pos = atomic_load_explicit(&q->enqueue_pos, memory_order_relaxed);
    }
}

static bool hs_submit_dequeue(HSSystem* sys, HSChannel ch, Message* out_msg, u8 out_payload[HS_PAYLOAD_SIZE], u32* out_payload_len) {
    if (!sys || !out_msg || !out_payload_len) return false;

    HSSubmitQueue* q = &sys->submit[(u32)ch];
    u32 pos = atomic_load_explicit(&q->dequeue_pos, memory_order_relaxed);
    for (;;) {
        HSSubmitSlot* slot = &q->slots[pos % HS_SUBMIT_SIZE];
        u32 seq = atomic_load_explicit(&slot->seq, memory_order_acquire);
        intptr_t dif = (intptr_t)seq - (intptr_t)(pos + 1);
        if (dif == 0) {
            if (atomic_compare_exchange_weak_explicit(
                    &q->dequeue_pos, &pos, pos + 1,
                    memory_order_relaxed, memory_order_relaxed)) {
                *out_msg = slot->msg;
                *out_payload_len = slot->payload_len;
                if (out_payload && slot->payload_len) {
                    memcpy(out_payload, slot->payload, slot->payload_len);
                }
                atomic_store_explicit(&slot->seq, pos + HS_SUBMIT_SIZE, memory_order_release);
                return true;
            }
            continue;
        }
        if (dif < 0) {
            return false; /* empty */
        }
        pos = atomic_load_explicit(&q->dequeue_pos, memory_order_relaxed);
    }
}

static bool hs_spsc_push(HSSpscQueue* q, const Message* msg, const void* payload, u32 payload_len) {
    if (!q || !msg) return false;
    u32 len = payload_len;
    if (len > HS_PAYLOAD_SIZE) len = HS_PAYLOAD_SIZE;

    u32 tail = atomic_load_explicit(&q->tail.v, memory_order_relaxed);
    u32 head = atomic_load_explicit(&q->head.v, memory_order_acquire);
    if ((tail - head) >= HS_SPSC_SIZE) return false;

    HSSpscSlot* s = &q->slots[tail % HS_SPSC_SIZE];
    s->msg = *msg;
    s->payload_len = len;
    if (len && payload) memcpy(s->payload, payload, len);
    atomic_store_explicit(&q->tail.v, tail + 1, memory_order_release);
    return true;
}

static bool hs_route_immediate(HSSystem* sys, Message* msg, const void* payload, u32 payload_len) {
    /* Called by the step thread only. */
    if (!sys || !msg) return false;

    if (msg->channel == CHAN_DEFAULT) {
        msg->channel = (u8)hs_default_channel_for_op((OpCode)msg->op);
    }

    msg->tick = sys->tick;

    /* If message carries a payload in-band, allocate it into system payload ring now. */
    if (payload_len) {
        u16 idx = 0;
        u32 copy_len = 0;
        if (!hs_payload_alloc_and_copy(sys, payload, payload_len, &idx, &copy_len)) return false;
        msg->payload_idx = idx;
        msg->payload_len = copy_len;
    }

    if (sys->validate_on_send) {
        const char* err = NULL;
        if (!hs_validate_message(sys, msg, &err)) {
            hs_report_error_ex(sys, msg, HS_ERR_VALIDATE, HS_ERR_STAGE_SEND, err ? err : "validate failed");
            return false;
        }
    }

    bool record_this = false;
    {
        HSChannel ch = (HSChannel)msg->channel;
        if (sys->recording && ch < CHAN_COUNT) {
            record_this = (sys->record_mask & (1u << (u32)ch)) != 0;
        }
    }

    if (record_this && sys->log_head >= sys->log_capacity) {
        sys->log_overflow = true;
        sys->recording = false;
        record_this = false;
    }

    /* Frame/QoS control ops are handled by the system itself. */
    if ((OpCode)msg->op == OP_FRAME_BEGIN || (OpCode)msg->op == OP_FRAME_END || (OpCode)msg->op == OP_PRESENT) {
        hs_render_record(sys, msg);
        if (record_this) {
            sys->log[sys->log_head++] = *msg;
        }
        return true;
    }

    if ((OpCode)msg->op == OP_FENCE) {
        Node* sys_node = hs_node_by_id(sys, NODE_SYSTEM);
        HSChannel target = (HSChannel)msg->flags;
        if (target == CHAN_DEFAULT) target = CHAN_RENDER;
        if (target >= CHAN_COUNT) target = CHAN_RENDER;

        u8 res_payload[8];
        hs_pack_result_fence(res_payload, (u32)sys->tick, (u8)target);

        u16 ridx = 0;
        u32 rlen = 0;
        if (sys_node && hs_payload_alloc_and_copy(sys, res_payload, sizeof(res_payload), &ridx, &rlen)) {
            Message res = {
                .to = NODE_SYSTEM,
                .from = NODE_SYSTEM,
                .op = OP_RESULT,
                .flags = OP_FENCE,
                .cid = msg->cid,
                .tick = (u16)sys->tick,
                .payload_idx = ridx,
                .payload_len = rlen,
                .channel = CHAN_RT,
            };

            if (!mq_push(&sys_node->inbox, &res)) {
                sys->dropped_result++;
            } else {
                if (sys->recording && (sys->record_mask & (1u << (u32)CHAN_RT))) {
                    if (sys->log_head < sys->log_capacity) {
                        sys->log[sys->log_head++] = res;
                    }
                }
            }
        }

        if (record_this) {
            sys->log[sys->log_head++] = *msg;
        }
        return true;
    }

    Node* dest = hs_node_by_id(sys, msg->to);
    if (!dest) {
        hs_report_error_ex(sys, msg, HS_ERR_ROUTE, HS_ERR_STAGE_SEND, "invalid destination node");
        return false;
    }

    /* Keep RT pristine even if system inbox is flooded by non-RT. */
    if (msg->to == NODE_SYSTEM) {
        HSChannel ch = (HSChannel)msg->channel;
        enum { HS_SYSTEM_RT_RESERVE = 32 };
        if (ch != CHAN_RT) {
            if (dest->inbox.count >= (HS_QUEUE_SIZE - HS_SYSTEM_RT_RESERVE)) {
                sys->dropped_system_nonrt++;
                return false;
            }
        }
    }

    if (!mq_push(&dest->inbox, msg)) {
        hs_report_queue_full(sys, msg, msg->to);
        return false;
    }

    if (record_this) {
        sys->log[sys->log_head++] = *msg;
    }

    hs_render_record(sys, msg);
    return true;
}

typedef struct {
    HSSystem* sys;
    int id;
    u32 epoch;
} HSProducerTLS;

static _Thread_local HSProducerTLS g_tls_prod = {0};
static _Thread_local HSSystem* g_tls_in_step = NULL;

static int hs_get_producer_id(HSSystem* sys) {
    u32 epoch = atomic_load_explicit(&sys->producer_epoch, memory_order_relaxed);
    if (g_tls_prod.sys != sys || g_tls_prod.epoch != epoch) {
        g_tls_prod.sys = sys;
        g_tls_prod.epoch = epoch;
        g_tls_prod.id = -1;
    }
    /* Cache both real producer ids (>=0) and fallback (-2). */
    if (g_tls_prod.id != -1) return g_tls_prod.id;

    u32 id = atomic_fetch_add_explicit(&sys->producer_count, 1, memory_order_relaxed);
    if (id >= HS_MAX_PRODUCERS) {
        g_tls_prod.id = -2; /* fallback */
        return -2;
    }
    g_tls_prod.id = (int)id;
    return g_tls_prod.id;
}

static void hs_backpressure_wait(HSSystem* sys) {
    if (!sys) return;
    atomic_fetch_add_explicit(&sys->bp_waiters, 1, memory_order_relaxed);
    pthread_mutex_lock(&sys->bp_lock);
    /* Timed wait to avoid missed wakeups causing long stalls. */
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    long nsec = ts.tv_nsec + 1000000L; /* +1ms */
    ts.tv_sec += nsec / 1000000000L;
    ts.tv_nsec = nsec % 1000000000L;
    (void)pthread_cond_timedwait(&sys->bp_cv, &sys->bp_lock, &ts);
    pthread_mutex_unlock(&sys->bp_lock);
    atomic_fetch_sub_explicit(&sys->bp_waiters, 1, memory_order_relaxed);
}

static void hs_backpressure_wake(HSSystem* sys) {
    if (!sys) return;
    if (atomic_load_explicit(&sys->bp_waiters, memory_order_relaxed) == 0) return;
    pthread_mutex_lock(&sys->bp_lock);
    pthread_cond_broadcast(&sys->bp_cv);
    pthread_mutex_unlock(&sys->bp_lock);
}

void hs_wake_senders(HSSystem* sys) {
    hs_backpressure_wake(sys);
}

static bool hs_send_enqueue(HSSystem* sys, Message* msg, const void* payload, u32 len) {
    if (!sys || !msg) return false;

    /* Enforce channel semantics for critical system ops (producer mistakes should not downgrade QoS). */
    switch ((OpCode)msg->op) {
        case OP_FENCE:
        case OP_QUERY_STATS:
        case OP_QUERY_FABRIC:
        case OP_SET_RECORD_MASK:
        case OP_SET_CHAN_BUDGET:
        case OP_SET_BLOCK_POLICY:
            msg->channel = CHAN_RT;
            break;

        case OP_FRAME_BEGIN:
        case OP_FRAME_END:
        case OP_PRESENT:
            msg->channel = CHAN_RENDER;
            break;

        default:
            break;
    }

    if (msg->channel == CHAN_DEFAULT) {
        msg->channel = (u8)hs_default_channel_for_op((OpCode)msg->op);
    }

    HSChannel ch = (HSChannel)msg->channel;
    if (ch <= CHAN_DEFAULT || ch >= CHAN_COUNT) {
        ch = hs_default_channel_for_op((OpCode)msg->op);
        msg->channel = (u8)ch;
    }

    if (g_tls_in_step == sys) {
        return hs_route_immediate(sys, msg, payload, len);
    }

    int pid = hs_get_producer_id(sys);
    if (pid >= 0) {
        if (hs_spsc_push(&sys->producers[(u32)ch][pid], msg, payload, len)) {
            atomic_fetch_add_explicit(&sys->spsc_ok[(u32)ch], 1, memory_order_relaxed);
            atomic_fetch_add_explicit(&sys->spsc_ok_by_prod[(u32)ch][pid], 1, memory_order_relaxed);
            return true;
        }
        atomic_fetch_add_explicit(&sys->spsc_full[(u32)ch], 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&sys->spsc_full_by_prod[(u32)ch][pid], 1, memory_order_relaxed);
        /* fall through to MPSC */
    }

    return hs_submit_enqueue(sys, ch, msg, payload, len);
}

typedef enum {
    HS_PAYLOAD_NONE = 0,
    HS_PAYLOAD_FIXED,
    HS_PAYLOAD_RANGE,
    HS_PAYLOAD_STRING
} HSPayloadKind;

typedef struct {
    u8           expected_to; /* 0xFF means any registered node */
    HSPayloadKind kind;
    u16          min_len;
    u16          max_len;
} HSOpSpec;

static HSChannel hs_default_channel_for_op(OpCode op) {
    switch (op) {
        /* telemetry */
        case OP_ERROR_EX:
        case OP_ERROR:
        case OP_TRACE:
        case OP_QUEUE_FULL:
            return CHAN_TELEM;

        /* rt/control */
        case OP_ACK:
        case OP_RESULT:
        case OP_ASYNC_DONE:
        case OP_STOP:
        case OP_FENCE:
        case OP_QUERY_STATS:
        case OP_QUERY_FABRIC:
        case OP_SET_RECORD_MASK:
        case OP_SET_CHAN_BUDGET:
        case OP_SET_BLOCK_POLICY:
            return CHAN_RT;

        /* render/frame */
        case OP_FRAME_BEGIN:
        case OP_FRAME_END:
        case OP_PRESENT:
            return CHAN_RENDER;

        /* default render */
        default:
            return CHAN_RENDER;
    }
}

static const HSOpSpec hs_op_spec_table[OP_COUNT] = {
    [OP_NOOP]          = {0xFF,        HS_PAYLOAD_NONE,   0, 0},

    [OP_SET_SHADER]    = {NODE_SHADER, HS_PAYLOAD_NONE,   0, 0},
    [OP_SET_PARAM]     = {NODE_SHADER, HS_PAYLOAD_FIXED, 20, 20},
    [OP_SET_GLOBAL]    = {NODE_SHADER, HS_PAYLOAD_FIXED, 64, 64},
    [OP_SET_CAMERA]    = {NODE_SHADER, HS_PAYLOAD_FIXED, 64, 64},
    [OP_CULL]          = {NODE_SHADER, HS_PAYLOAD_NONE,   0, 0},
    [OP_BLEND]         = {NODE_SHADER, HS_PAYLOAD_FIXED,  2,  2},
    [OP_ALPHA]         = {NODE_SHADER, HS_PAYLOAD_NONE,   0, 0},
    [OP_DEPTH]         = {NODE_SHADER, HS_PAYLOAD_NONE,   0, 0},
    [OP_COLOR_MASK]    = {NODE_SHADER, HS_PAYLOAD_NONE,   0, 0},
    [OP_CLIP]          = {NODE_SHADER, HS_PAYLOAD_FIXED,  8,  8},
    [OP_STENCIL]       = {NODE_SHADER, HS_PAYLOAD_FIXED,  4,  4},
    [OP_STENCIL_FUNC]  = {NODE_SHADER, HS_PAYLOAD_FIXED,  4,  4},
    [OP_DEPTH_COMPARE] = {NODE_SHADER, HS_PAYLOAD_FIXED,  2,  2},

    [OP_LOAD_BUFFER]   = {NODE_BUFFER, HS_PAYLOAD_NONE,   0, 0},
    [OP_DRAW]          = {NODE_BUFFER, HS_PAYLOAD_NONE,   0, 0},
    [OP_DRAW_INSTANCE] = {NODE_BUFFER, HS_PAYLOAD_FIXED,  4,  4},
    [OP_DRAW_TEXT]     = {NODE_BUFFER, HS_PAYLOAD_STRING, 1, HS_PAYLOAD_SIZE},

    [OP_LOAD_TEXTURE]  = {NODE_TEXTURE, HS_PAYLOAD_NONE,  0, 0},
    [OP_SET_TARGET]    = {NODE_TEXTURE, HS_PAYLOAD_FIXED, 2, 2},
    [OP_SHOW_TEXTURE]  = {NODE_TEXTURE, HS_PAYLOAD_NONE,  0, 0},
    [OP_TEXTURE_FILTER]= {NODE_TEXTURE, HS_PAYLOAD_NONE,  0, 0},
    [OP_TEXTURE_WRAP]  = {NODE_TEXTURE, HS_PAYLOAD_NONE,  0, 0},

    [OP_ACK]           = {NODE_SYSTEM, HS_PAYLOAD_NONE,   0, 0},
    [OP_RESULT]        = {NODE_SYSTEM, HS_PAYLOAD_RANGE,  0, HS_PAYLOAD_SIZE},

    [OP_ERROR_EX]      = {NODE_SYSTEM, HS_PAYLOAD_FIXED, 52, 52},
    [OP_QUEUE_FULL]    = {NODE_SYSTEM, HS_PAYLOAD_NONE,   0, 0},
    [OP_ASYNC_DONE]    = {NODE_SYSTEM, HS_PAYLOAD_FIXED,  8, 8},

    [OP_CLEAR]         = {NODE_OUTPUT, HS_PAYLOAD_FIXED, 16, 16},
    [OP_CLEAR_DS]      = {NODE_OUTPUT, HS_PAYLOAD_FIXED,  8,  8},

    [OP_SET_CHANNEL]   = {NODE_SOUND,  HS_PAYLOAD_FIXED,  2,  2},

    [OP_ERROR]         = {NODE_SYSTEM, HS_PAYLOAD_STRING, 1, HS_PAYLOAD_SIZE},
    [OP_TRACE]         = {NODE_SYSTEM, HS_PAYLOAD_STRING, 1, HS_PAYLOAD_SIZE},
    [OP_STOP]          = {NODE_SYSTEM, HS_PAYLOAD_NONE,   0, 0},

    [OP_FRAME_BEGIN]   = {NODE_SYSTEM, HS_PAYLOAD_NONE,   0, 0},
    [OP_FRAME_END]     = {NODE_SYSTEM, HS_PAYLOAD_NONE,   0, 0},
    [OP_PRESENT]       = {NODE_SYSTEM, HS_PAYLOAD_NONE,   0, 0},
    [OP_FENCE]         = {NODE_SYSTEM, HS_PAYLOAD_NONE,   0, 0},

    [OP_QUERY_STATS]     = {NODE_SYSTEM, HS_PAYLOAD_NONE,  0, 0},
    [OP_QUERY_FABRIC]    = {NODE_SYSTEM, HS_PAYLOAD_NONE,  0, 0},
    [OP_SET_RECORD_MASK] = {NODE_SYSTEM, HS_PAYLOAD_FIXED, 4, 4},
    [OP_SET_CHAN_BUDGET] = {NODE_SYSTEM, HS_PAYLOAD_FIXED, 8, 8},
    [OP_SET_BLOCK_POLICY]= {NODE_SYSTEM, HS_PAYLOAD_FIXED, 2, 2},
};

bool hs_payload_alloc_and_copy(HSSystem* sys, const void* data, u32 len, u16* out_idx, u32* out_len) {
    if (!sys) return false;
    if (!sys->payloads || sys->payload_capacity == 0) {
        return false;
    }

    u32 copy_len = len;
    if (copy_len > HS_PAYLOAD_SIZE) copy_len = HS_PAYLOAD_SIZE;

    u32 idx = sys->payload_head;
    sys->payload_head = (sys->payload_head + 1) % sys->payload_capacity;

    if (copy_len > 0) {
        if (data) memcpy(sys->payloads[idx].data, data, copy_len);
        else memset(sys->payloads[idx].data, 0, copy_len);
    }

    if (out_idx) *out_idx = (u16)idx;
    if (out_len) *out_len = copy_len;
    return true;
}

bool hs_validate_message(const HSSystem* sys, const Message* msg, const char** out_err) {
    const char* err = NULL;
    if (!sys || !msg) {
        err = "null sys/msg";
        if (out_err) *out_err = err;
        return false;
    }

    if (msg->op >= OP_COUNT) {
        err = "invalid opcode";
        if (out_err) *out_err = err;
        return false;
    }

    if (!hs_system_has_node(sys, msg->to)) {
        err = "invalid destination node";
        if (out_err) *out_err = err;
        return false;
    }

    const HSOpSpec* spec = &hs_op_spec_table[(u32)msg->op];
    if (spec->expected_to != 0xFF && msg->to != spec->expected_to) {
        err = "bad destination";
        if (out_err) *out_err = err;
        return false;
    }

    switch (spec->kind) {
        case HS_PAYLOAD_NONE:
            if (msg->payload_len != 0) err = "unexpected payload";
            break;
        case HS_PAYLOAD_FIXED:
            if (msg->payload_len != spec->min_len) err = "bad payload len";
            break;
        case HS_PAYLOAD_RANGE:
        case HS_PAYLOAD_STRING:
            if (msg->payload_len < spec->min_len || msg->payload_len > spec->max_len) err = "bad payload len";
            break;
    }

    if (err) {
        if (out_err) *out_err = err;
        return false;
    }

    /* For payload-bearing ops, validate payload_idx range. */
    if (msg->payload_len != 0) {
        if (!sys->payloads) {
            if (out_err) *out_err = "null payload buffer";
            return false;
        }
        if (msg->payload_idx >= sys->payload_capacity) {
            if (out_err) *out_err = "invalid payload index";
            return false;
        }

        if (spec->kind == HS_PAYLOAD_STRING) {
            if (sys->payloads[msg->payload_idx].data[msg->payload_len - 1] != 0) {
                if (out_err) *out_err = "string not terminated";
                return false;
            }
        }
    }

    if (out_err) *out_err = NULL;
    return true;
}

void mq_init(MessageQueue* q) {
    q->head = 0;
    q->tail = 0;
    q->count = 0;
    q->padding = 0;
}

void hs_lock(HSSystem* sys) {
    if (!sys || !sys->lock_inited) return;
    pthread_mutex_lock(&sys->lock);
}

void hs_unlock(HSSystem* sys) {
    if (!sys || !sys->lock_inited) return;
    pthread_mutex_unlock(&sys->lock);
}

void hs_set_record_mask(HSSystem* sys, u32 mask) {
    if (!sys) return;
    hs_lock(sys);
    sys->record_mask = mask;
    hs_unlock(sys);
}

void hs_set_channel_budget(HSSystem* sys, HSChannel ch, u32 budget) {
    if (!sys) return;
    if (ch <= CHAN_DEFAULT || ch >= CHAN_COUNT) return;
    hs_lock(sys);
    sys->chan_budget[(u32)ch] = budget;
    hs_unlock(sys);
}

bool mq_push(MessageQueue* q, Message* msg) {
    if (q->count >= HS_QUEUE_SIZE) return false;
    q->msgs[q->tail] = *msg;
    q->tail = (q->tail + 1) % HS_QUEUE_SIZE;
    q->count++;
    return true;
}

bool mq_pop(MessageQueue* q, Message* msg) {
    if (q->count == 0) return false;
    *msg = q->msgs[q->head];
    q->head = (q->head + 1) % HS_QUEUE_SIZE;
    q->count--;
    return true;
}

bool mq_empty(MessageQueue* q) {
    return q->count == 0;
}

u16 mq_count(MessageQueue* q) {
    return q->count;
}

void hs_init(HSSystem* sys, Message* log_buffer, u32 log_capacity, Payload* payload_buffer) {
    memset(sys, 0, sizeof(HSSystem));
    sys->log = log_buffer;
    sys->log_capacity = log_capacity;
    sys->payloads = payload_buffer;
    sys->payload_capacity = HS_MAX_PAYLOADS;
    sys->payload_head = 0;
    sys->recording = true;
    sys->validate_on_send = false;
    sys->block_on_full = false;
    sys->render_list = NULL;
    sys->log_overflow = false;
    sys->dropped_error_ex = 0;
    sys->dropped_queue_full = 0;
    sys->dropped_system_nonrt = 0;
    sys->dropped_result = 0;

    /* Capture defaults: record render channel only. */
    sys->record_mask = hs_channel_bit(CHAN_RENDER);

    /* Default per-channel drain budgets (messages per hs_step). */
    for (u32 c = 0; c < CHAN_COUNT; c++) sys->chan_budget[c] = 0;
    sys->chan_budget[CHAN_RT] = 4096;
    sys->chan_budget[CHAN_RENDER] = 16384;
    sys->chan_budget[CHAN_TELEM] = 1024;
    for (u32 c = 0; c < CHAN_COUNT; c++) {
        atomic_init(&sys->submit_full[c], 0);
        atomic_init(&sys->spsc_full[c], 0);
        atomic_init(&sys->spsc_ok[c], 0);
        atomic_init(&sys->mpsc_ok[c], 0);

        for (u32 i = 0; i < HS_MAX_PRODUCERS; i++) {
            atomic_init(&sys->spsc_full_by_prod[c][i], 0);
            atomic_init(&sys->spsc_ok_by_prod[c][i], 0);
        }

        atomic_init(&sys->submit[c].enqueue_pos, 0);
        atomic_init(&sys->submit[c].dequeue_pos, 0);
        for (u32 i = 0; i < HS_SUBMIT_SIZE; i++) {
            atomic_init(&sys->submit[c].slots[i].seq, i);
            memset(&sys->submit[c].slots[i].msg, 0, sizeof(Message));
            sys->submit[c].slots[i].payload_len = 0;
            memset(sys->submit[c].slots[i].payload, 0, HS_PAYLOAD_SIZE);
        }
    }

    atomic_init(&sys->producer_count, 0);
    atomic_init(&sys->producer_epoch, 1);
    for (u32 c = 0; c < CHAN_COUNT; c++) {
        for (u32 i = 0; i < HS_MAX_PRODUCERS; i++) {
            atomic_init(&sys->producers[c][i].head.v, 0);
            atomic_init(&sys->producers[c][i].tail.v, 0);
            for (u32 j = 0; j < HS_SPSC_SIZE; j++) {
                memset(&sys->producers[c][i].slots[j].msg, 0, sizeof(Message));
                sys->producers[c][i].slots[j].payload_len = 0;
                memset(sys->producers[c][i].slots[j].payload, 0, HS_PAYLOAD_SIZE);
            }
        }
    }

    /* default channel policies */
    for (u32 c = 0; c < CHAN_COUNT; c++) {
        sys->block_on_full_chan[c] = false;
    }
    sys->block_on_full_chan[CHAN_RT] = true;
    sys->block_on_full_chan[CHAN_RENDER] = true;
    sys->block_on_full_chan[CHAN_TELEM] = false;

    {
        pthread_mutexattr_t attr;
        pthread_mutexattr_init(&attr);
        pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
        pthread_mutex_init(&sys->lock, &attr);
        pthread_mutexattr_destroy(&attr);
        sys->lock_inited = true;
    }

    pthread_mutex_init(&sys->bp_lock, NULL);
    pthread_cond_init(&sys->bp_cv, NULL);
    atomic_init(&sys->bp_waiters, 0);
}

void hs_register(HSSystem* sys, Node* node) {
    if (sys->node_count >= HS_MAX_NODES) return;
    node->next = NULL;
    sys->nodes[sys->node_count++] = node;
    sys->node_map[node->id] = node;
}

void hs_capture_init(HSCapture* cap, Message* msg_buf, Payload* payload_buf, u32 capacity) {
    if (!cap) return;
    cap->msgs = msg_buf;
    cap->payloads = payload_buf;
    cap->capacity = capacity;
    cap->count = 0;
}

bool hs_capture_from_log(const HSSystem* sys, const Message* msgs, u32 count, HSCapture* out) {
    /* Capturing is read-only but depends on stable payload storage; caller should serialize appropriately. */
    if (!sys || !msgs || !out || !out->msgs || !out->payloads) return false;
    if (count > out->capacity) return false;
    if (!sys->payloads || sys->payload_capacity == 0) return false;

    for (u32 i = 0; i < count; i++) {
        Message m = msgs[i];
        if (m.payload_len != 0) {
            if (m.payload_idx >= sys->payload_capacity) return false;
            memcpy(out->payloads[i].data, sys->payloads[m.payload_idx].data, HS_PAYLOAD_SIZE);
            m.payload_idx = (u16)i;
        }
        out->msgs[i] = m;
    }

    out->count = count;
    return true;
}

bool hs_capture_replay(HSSystem* sys, const HSCapture* cap) {
    if (!sys || !cap || !cap->msgs || !cap->payloads) return false;
    if (cap->count > cap->capacity) return false;

    Payload* saved_payloads = sys->payloads;
    u32 saved_cap = sys->payload_capacity;

    sys->payloads = cap->payloads;
    sys->payload_capacity = cap->capacity;

    bool ok = hs_replay(sys, cap->msgs, cap->count);

    sys->payloads = saved_payloads;
    sys->payload_capacity = saved_cap;

    return ok;
}

bool hs_capture_write_file(const HSCapture* cap, const char* path) {
    if (!cap || !path || !cap->msgs || !cap->payloads) return false;
    FILE* f = fopen(path, "wb");
    if (!f) return false;

    HSCaptureHeader h;
    hs_capture_header_init(&h, cap->count);

    bool ok = true;
    ok = ok && (fwrite(&h, sizeof(h), 1, f) == 1);
    ok = ok && (fwrite(cap->msgs, sizeof(Message), cap->count, f) == cap->count);
    ok = ok && (fwrite(cap->payloads, sizeof(Payload), cap->count, f) == cap->count);

    fclose(f);
    return ok;
}

bool hs_capture_read_file(HSCapture* cap, const char* path, Message* msg_buf, Payload* payload_buf, u32 capacity) {
    if (!cap || !path || !msg_buf || !payload_buf || capacity == 0) return false;
    FILE* f = fopen(path, "rb");
    if (!f) return false;

    HSCaptureHeader h;
    bool ok = (fread(&h, sizeof(h), 1, f) == 1);
    ok = ok && hs_capture_header_valid(&h);
    ok = ok && (h.count <= capacity);

    if (!ok) {
        fclose(f);
        return false;
    }

    ok = ok && (fread(msg_buf, sizeof(Message), h.count, f) == h.count);
    ok = ok && (fread(payload_buf, sizeof(Payload), h.count, f) == h.count);
    fclose(f);

    if (!ok) return false;

    hs_capture_init(cap, msg_buf, payload_buf, capacity);
    cap->count = h.count;
    return true;
}

bool hs_send(HSSystem* sys, Message* msg) {
    if (!sys || !msg) return false;
    for (;;) {
        if (hs_send_enqueue(sys, msg, NULL, 0)) return true;
        HSChannel ch = (HSChannel)msg->channel;
        bool block = (ch != CHAN_TELEM) && (sys->block_on_full || (ch < CHAN_COUNT && sys->block_on_full_chan[(u32)ch]));
        if (!block) return false;
        hs_backpressure_wait(sys);
    }
}

bool hs_send_with_payload(HSSystem* sys, Message* msg, const void* data, u32 len) {
    if (!sys || !msg) return false;
    for (;;) {
        if (hs_send_enqueue(sys, msg, data, len)) return true;
        HSChannel ch = (HSChannel)msg->channel;
        bool block = (ch != CHAN_TELEM) && (sys->block_on_full || (ch < CHAN_COUNT && sys->block_on_full_chan[(u32)ch]));
        if (!block) return false;
        hs_backpressure_wait(sys);
    }
}

u32 hs_step(HSSystem* sys) {
    hs_lock(sys);
    u32 processed = 0;

    g_tls_in_step = sys;

    u32 prod_count = atomic_load_explicit(&sys->producer_count, memory_order_acquire);
    if (prod_count > HS_MAX_PRODUCERS) prod_count = HS_MAX_PRODUCERS;

    for (HSChannel ch = CHAN_RT; ch <= CHAN_TELEM; ch++) {
        u32 budget = sys->chan_budget[(u32)ch];
        u32 used = 0;

        /* Drain per-producer SPSC lanes */
        for (u32 p = 0; p < prod_count && used < budget; p++) {
            HSSpscQueue* q = &sys->producers[(u32)ch][p];
            for (;;) {
                if (used >= budget) break;
                u32 head = atomic_load_explicit(&q->head.v, memory_order_relaxed);
                u32 tail = atomic_load_explicit(&q->tail.v, memory_order_acquire);
                u32 avail = tail - head;
                if (avail == 0) break;

                u32 n = avail;
                if (n > 32) n = 32;
                if (n > (budget - used)) n = (budget - used);

                for (u32 i = 0; i < n; i++) {
                    HSSpscSlot* s = &q->slots[(head + i) % HS_SPSC_SIZE];
                    (void)hs_route_immediate(sys, &s->msg, s->payload_len ? s->payload : NULL, s->payload_len);
                }

                atomic_store_explicit(&q->head.v, head + n, memory_order_release);
                used += n;
                processed += n;
            }
        }

        /* Drain fallback MPSC submit queue */
        while (used < budget) {
            Message m;
            u8 payload[HS_PAYLOAD_SIZE];
            u32 payload_len = 0;
            if (!hs_submit_dequeue(sys, ch, &m, payload, &payload_len)) break;
            (void)hs_route_immediate(sys, &m, payload_len ? payload : NULL, payload_len);
            used++;
            processed++;
        }
    }
    
    for (u8 i = 0; i < sys->node_count; i++) {
        Node* node = sys->nodes[i];
        if (node->process_fn) {
            processed += node->process_fn(node);
        }
    }
    
    for (u8 i = 0; i < sys->node_count; i++) {
        Node* node = sys->nodes[i];
        while (!mq_empty(&node->outbox)) {
            Message msg;
            mq_pop(&node->outbox, &msg);
            (void)hs_route_immediate(sys, &msg, NULL, 0);
        }
    }
    
    sys->tick++;
    g_tls_in_step = NULL;
    hs_backpressure_wake(sys);
    hs_unlock(sys);
    return processed;
}

void hs_start_recording(HSSystem* sys) {
    hs_lock(sys);
    sys->recording = true;
    hs_unlock(sys);
}

u32 hs_stop_recording(HSSystem* sys) {
    hs_lock(sys);
    sys->recording = false;
    u32 n = sys->log_head;
    hs_unlock(sys);
    return n;
}

bool hs_has_overflow(HSSystem* sys) {
    hs_lock(sys);
    bool of = sys->log_overflow;
    hs_unlock(sys);
    return of;
}

bool hs_replay(HSSystem* sys, Message* msgs, u32 count) {
    hs_lock(sys);
    for (u8 i = 0; i < sys->node_count; i++) {
        Node* node = sys->nodes[i];
        if (node->reset_fn) {
            node->reset_fn(node);
        }
    }
    
    sys->tick = 0;
    sys->payload_head = 0;
    
    for (u32 i = 0; i < count; i++) {
        Message msg = msgs[i];
        msg.tick = sys->tick;

        if (sys->validate_on_send) {
            const char* err = NULL;
            if (!hs_validate_message(sys, &msg, &err)) {
                hs_report_error_ex(sys, &msg, HS_ERR_VALIDATE, HS_ERR_STAGE_REPLAY, err ? err : "validate failed");
                hs_unlock(sys);
                return false;
            }
        }
        
        bool found = false;
        for (u8 j = 0; j < sys->node_count; j++) {
            if (sys->nodes[j]->id == msg.to) {
                hs_render_record(sys, &msg);
                mq_push(&sys->nodes[j]->inbox, &msg);
                sys->nodes[j]->process_fn(sys->nodes[j]);
                found = true;
                break;
            }
        }
        if (!found) {
            hs_report_error_ex(sys, &msg, HS_ERR_ROUTE, HS_ERR_STAGE_REPLAY, "invalid destination node");
            hs_unlock(sys);
            return false;
        }
        sys->tick++;
    }
    hs_unlock(sys);
    return true;
}

void hs_clear(HSSystem* sys) {
    hs_lock(sys);
    sys->log_head = 0;
    sys->tick = 0;
    sys->payload_head = 0;
    sys->log_overflow = false;
    sys->dropped_error_ex = 0;
    sys->dropped_queue_full = 0;
    sys->dropped_system_nonrt = 0;
    sys->dropped_result = 0;

    /* Reset producer lanes and submit queues (no concurrent producers allowed). */
    atomic_store_explicit(&sys->producer_count, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&sys->producer_epoch, 1, memory_order_relaxed);
    for (u32 c = 0; c < CHAN_COUNT; c++) {
        atomic_store_explicit(&sys->spsc_full[c], 0, memory_order_relaxed);
        atomic_store_explicit(&sys->spsc_ok[c], 0, memory_order_relaxed);
        atomic_store_explicit(&sys->mpsc_ok[c], 0, memory_order_relaxed);
        atomic_store_explicit(&sys->submit_full[c], 0, memory_order_relaxed);

        for (u32 i = 0; i < HS_MAX_PRODUCERS; i++) {
            atomic_store_explicit(&sys->producers[c][i].head.v, 0, memory_order_relaxed);
            atomic_store_explicit(&sys->producers[c][i].tail.v, 0, memory_order_relaxed);
            atomic_store_explicit(&sys->spsc_full_by_prod[c][i], 0, memory_order_relaxed);
            atomic_store_explicit(&sys->spsc_ok_by_prod[c][i], 0, memory_order_relaxed);
        }

        atomic_store_explicit(&sys->submit[c].enqueue_pos, 0, memory_order_relaxed);
        atomic_store_explicit(&sys->submit[c].dequeue_pos, 0, memory_order_relaxed);
        for (u32 i = 0; i < HS_SUBMIT_SIZE; i++) {
            atomic_store_explicit(&sys->submit[c].slots[i].seq, i, memory_order_relaxed);
            sys->submit[c].slots[i].payload_len = 0;
        }
    }
    for (u8 i = 0; i < sys->node_count; i++) {
        mq_init(&sys->nodes[i]->inbox);
        mq_init(&sys->nodes[i]->outbox);
        if (sys->nodes[i]->reset_fn) {
            sys->nodes[i]->reset_fn(sys->nodes[i]);
        }
    }

    hs_unlock(sys);
}

const char* hs_op_name(OpCode op) {
    switch (op) {
        case OP_NOOP:          return "NOOP";
        case OP_SET_SHADER:    return "SET_SHADER";
        case OP_SET_PARAM:     return "SET_PARAM";
        case OP_SET_GLOBAL:    return "SET_GLOBAL";
        case OP_SET_CAMERA:    return "SET_CAMERA";
        case OP_SET_TARGET:    return "SET_TARGET";
        case OP_LOAD_BUFFER:   return "LOAD_BUFFER";
        case OP_LOAD_TEXTURE:  return "LOAD_TEXTURE";
        case OP_DRAW:          return "DRAW";
        case OP_DRAW_INSTANCE: return "DRAW_INSTANCE";
        case OP_CLEAR:         return "CLEAR";
        case OP_CLEAR_DS:      return "CLEAR_DS";
        case OP_CULL:          return "CULL";
        case OP_BLEND:         return "BLEND";
        case OP_ALPHA:         return "ALPHA";
        case OP_DEPTH:         return "DEPTH";
        case OP_COLOR_MASK:    return "COLOR_MASK";
        case OP_CLIP:          return "CLIP";
        case OP_SET_CHANNEL:   return "SET_CHANNEL";
        case OP_STENCIL:       return "STENCIL";
        case OP_STENCIL_FUNC:  return "STENCIL_FUNC";
        case OP_DEPTH_COMPARE: return "DEPTH_COMPARE";
        case OP_DRAW_TEXT:     return "DRAW_TEXT";
        case OP_SHOW_TEXTURE:  return "SHOW_TEXTURE";
        case OP_TEXTURE_FILTER: return "TEXTURE_FILTER";
        case OP_TEXTURE_WRAP:  return "TEXTURE_WRAP";
        case OP_ACK:           return "ACK";
        case OP_RESULT:        return "RESULT";
        case OP_ERROR_EX:      return "ERROR_EX";
        case OP_QUEUE_FULL:    return "QUEUE_FULL";
        case OP_ASYNC_DONE:    return "ASYNC_DONE";
        case OP_ERROR:         return "ERROR";
        case OP_TRACE:         return "TRACE";
        case OP_STOP:          return "STOP";
        case OP_FRAME_BEGIN:   return "FRAME_BEGIN";
        case OP_FRAME_END:     return "FRAME_END";
        case OP_PRESENT:       return "PRESENT";
        case OP_FENCE:         return "FENCE";
        case OP_QUERY_STATS:   return "QUERY_STATS";
        case OP_QUERY_FABRIC:  return "QUERY_FABRIC";
        case OP_SET_RECORD_MASK: return "SET_RECORD_MASK";
        case OP_SET_CHAN_BUDGET: return "SET_CHAN_BUDGET";
        case OP_SET_BLOCK_POLICY: return "SET_BLOCK_POLICY";
        case OP_COUNT:         return "COUNT";
    }
    return "UNKNOWN";
}
