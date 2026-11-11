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
    h->version = 1u;
    h->msg_size = (u32)sizeof(Message);
    h->payload_size = (u32)sizeof(Payload);
    h->count = count;
}

static bool hs_capture_header_valid(const HSCaptureHeader* h) {
    if (!h) return false;
    if (memcmp(h->magic, "HSCAP1\0", 7) != 0) return false;
    if (h->endian != 0x01020304u) return false;
    if (h->version != 1u) return false;
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
static bool hs_send_enqueue(HSSystem* sys, Message* msg, const void* payload, u32 len, bool trusted);

#define HS_RING_MASK(size) ((size) - 1u)

_Static_assert((HS_QUEUE_SIZE & (HS_QUEUE_SIZE - 1)) == 0, "HS_QUEUE_SIZE must be power of two");
_Static_assert((HS_SUBMIT_SIZE & (HS_SUBMIT_SIZE - 1)) == 0, "HS_SUBMIT_SIZE must be power of two");
_Static_assert((HS_SPSC_SIZE & (HS_SPSC_SIZE - 1)) == 0, "HS_SPSC_SIZE must be power of two");
_Static_assert((HS_TOOLBUS_SIZE & (HS_TOOLBUS_SIZE - 1)) == 0, "HS_TOOLBUS_SIZE must be power of two");

static inline u32 hs_ring_index(u32 pos, u32 size) {
    return pos & HS_RING_MASK(size);
}

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

static inline HSRenderCmd* hs_render_claim(HSRenderList* rl) {
    if (rl->count >= HS_RENDER_MAX_CMDS) {
        rl->overflow = true;
        return NULL;
    }
    HSRenderCmd* c = &rl->cmds[rl->count++];
    *c = (HSRenderCmd){0};
    return c;
}

static void hs_render_record(HSSystem* sys, const Message* msg) {
    if (!sys || !sys->render_list || !msg) return;

    /* Validate first, then write the command straight into the list. */
    HSRenderList* rl = sys->render_list;
    HSRenderCmd* c;

    switch ((OpCode)msg->op) {
        case OP_FRAME_BEGIN:
            hs_render_reset(rl);
            if (!(c = hs_render_claim(rl))) return;
            c->op = HS_RC_FRAME_BEGIN;
            break;

        case OP_FRAME_END:
            if (!(c = hs_render_claim(rl))) return;
            c->op = HS_RC_FRAME_END;
            break;

        case OP_PRESENT:
            if (!(c = hs_render_claim(rl))) return;
            c->op = HS_RC_PRESENT;
            break;

        case OP_CULL:
            if (!(c = hs_render_claim(rl))) return;
            c->op = HS_RC_SET_CULL;
            c->a = (u8)msg->payload_idx;
            break;

        case OP_BLEND: {
            if (msg->payload_len != 2 || msg->payload_idx >= sys->payload_capacity) return;
            u8 src = 0, dst = 0;
            if (!hs_unpack_u8x2(sys->payloads[msg->payload_idx].data, msg->payload_len, &src, &dst)) return;
            if (!(c = hs_render_claim(rl))) return;
            c->op = HS_RC_SET_BLEND;
            c->a = src;
            c->b = dst;
            break;
        }

        case OP_ALPHA:
            if (!(c = hs_render_claim(rl))) return;
            c->op = HS_RC_SET_ALPHA;
            c->a = (u8)(msg->payload_idx ? 1 : 0);
            break;

        case OP_DEPTH:
            if (!(c = hs_render_claim(rl))) return;
            c->op = HS_RC_SET_DEPTH;
            c->a = (u8)(msg->payload_idx ? 1 : 0);
            break;

        case OP_DEPTH_COMPARE: {
            if (msg->payload_len != 2 || msg->payload_idx >= sys->payload_capacity) return;
            u8 cmp = 0, wr = 0;
            if (!hs_unpack_u8x2(sys->payloads[msg->payload_idx].data, msg->payload_len, &cmp, &wr)) return;
            if (!(c = hs_render_claim(rl))) return;
            c->op = HS_RC_SET_DEPTH_COMPARE;
            c->a = cmp;
            c->b = wr;
            break;
        }

        case OP_COLOR_MASK:
            if (!(c = hs_render_claim(rl))) return;
            c->op = HS_RC_SET_COLOR_MASK;
            c->a = (u8)msg->payload_idx;
            break;

        case OP_CLIP: {
            if (msg->payload_len != 8 || msg->payload_idx >= sys->payload_capacity) return;
            u16 v[4];
            if (!hs_unpack_u16x4(sys->payloads[msg->payload_idx].data, msg->payload_len, v)) return;
            if (!(c = hs_render_claim(rl))) return;
            c->op = HS_RC_SET_CLIP;
            c->x = v[0];
            c->y = v[1];
            c->payload_idx = v[2];
            c->payload_len = v[3];
            break;
        }

        case OP_TEXTURE_FILTER:
            if (!(c = hs_render_claim(rl))) return;
            c->op = HS_RC_SET_TEX_FILTER;
            c->a = (u8)(msg->payload_idx & 0x0F);
            c->b = (u8)((msg->payload_idx >> 4) & 1);
            break;

        case OP_TEXTURE_WRAP:
            if (!(c = hs_render_claim(rl))) return;
            c->op = HS_RC_SET_TEX_WRAP;
            c->a = (u8)(msg->payload_idx & 0x0F);
            c->b = (u8)((msg->payload_idx >> 4) & 1);
            break;

        case OP_CLEAR: {
            if (msg->payload_len != 16 || msg->payload_idx >= sys->payload_capacity) return;
            f32 rgba[4];
            if (!hs_unpack_clear_color(sys->payloads[msg->payload_idx].data, msg->payload_len, rgba)) return;
            if (!(c = hs_render_claim(rl))) return;
            c->op = HS_RC_CLEAR;
            c->f0 = rgba[0];
            c->f1 = rgba[1];
            c->f2 = rgba[2];
            c->f3 = rgba[3];
            break;
        }

        case OP_CLEAR_DS: {
            if (msg->payload_len != 8 || msg->payload_idx >= sys->payload_capacity) return;
            f32 depth = 0.0f;
            u8 stencil = 0;
            if (!hs_unpack_clear_ds(sys->payloads[msg->payload_idx].data, msg->payload_len, &depth, &stencil)) return;
            if (!(c = hs_render_claim(rl))) return;
            c->op = HS_RC_CLEAR_DS;
            c->f0 = depth;
            c->a = stencil;
            break;
        }

        case OP_DRAW:
            if (!(c = hs_render_claim(rl))) return;
            c->op = HS_RC_DRAW;
            c->a = (u8)(msg->payload_idx & 0xF);
            break;

        case OP_DRAW_INSTANCE: {
            if (msg->payload_len != 8 || msg->payload_idx >= sys->payload_capacity) return;
            u8 b = 0, ib = 0;
            u32 count = 0;
            if (!hs_unpack_draw_instance(sys->payloads[msg->payload_idx].data, msg->payload_len, &b, &ib, &count)) return;
            if (!(c = hs_render_claim(rl))) return;
            c->op = HS_RC_DRAW_INSTANCE;
            c->a = b;
            c->b = ib;
            c->x = count;
            break;
        }

        case OP_DRAW_TEXT:
            if (!(c = hs_render_claim(rl))) return;
            c->op = HS_RC_DRAW_TEXT;
            c->payload_idx = msg->payload_idx;
            c->payload_len = (u16)msg->payload_len;
            break;

        case OP_SHOW_TEXTURE:
            if (!(c = hs_render_claim(rl))) return;
            c->op = HS_RC_SHOW_TEXTURE;
            c->a = (u8)(msg->payload_idx & 0xF);
            break;

        default:
            return;
    }

}

static bool hs_submit_enqueue(HSSystem* sys, HSChannel ch, const Message* msg, const void* payload, u32 payload_len, bool trusted) {
    if (!sys || !msg) return false;

    /* fast reject invalid destination/op */
    if (msg->op >= OP_COUNT) return false;
    if (!hs_system_has_node(sys, msg->to)) return false;

    u32 len = payload_len;
    if (len > HS_PAYLOAD_SIZE) len = HS_PAYLOAD_SIZE;

    HSSubmitQueue* q = &sys->submit[(u32)ch];
    u32 pos = atomic_load_explicit(&q->enqueue_pos, memory_order_relaxed);
    for (;;) {
        HSSubmitSlot* slot = &q->slots[hs_ring_index(pos, HS_SUBMIT_SIZE)];
        u32 seq = atomic_load_explicit(&slot->seq, memory_order_acquire);
        int32_t dif = (int32_t)(seq - pos);
        if (dif == 0) {
            if (atomic_compare_exchange_weak_explicit(
                    &q->enqueue_pos, &pos, pos + 1,
                    memory_order_relaxed, memory_order_relaxed)) {
                /* claimed */
                slot->msg = *msg;
                slot->payload_len = len;
                slot->trusted = trusted;
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

static bool hs_submit_dequeue(HSSystem* sys, HSChannel ch, Message* out_msg, u8 out_payload[HS_PAYLOAD_SIZE], u32* out_payload_len, bool* out_trusted) {
    if (!sys || !out_msg || !out_payload_len || !out_trusted) return false;

    HSSubmitQueue* q = &sys->submit[(u32)ch];
    u32 pos = atomic_load_explicit(&q->dequeue_pos, memory_order_relaxed);
    for (;;) {
        HSSubmitSlot* slot = &q->slots[hs_ring_index(pos, HS_SUBMIT_SIZE)];
        u32 seq = atomic_load_explicit(&slot->seq, memory_order_acquire);
        int32_t dif = (int32_t)(seq - (pos + 1));
        if (dif == 0) {
            if (atomic_compare_exchange_weak_explicit(
                    &q->dequeue_pos, &pos, pos + 1,
                    memory_order_relaxed, memory_order_relaxed)) {
                *out_msg = slot->msg;
                *out_payload_len = slot->payload_len;
                *out_trusted = slot->trusted;
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

static bool hs_spsc_push(HSSpscQueue* q, const Message* msg, const void* payload, u32 payload_len, bool trusted) {
    if (!q || !msg) return false;
    u32 len = payload_len;
    if (len > HS_PAYLOAD_SIZE) len = HS_PAYLOAD_SIZE;

    u32 tail = atomic_load_explicit(&q->tail.v, memory_order_relaxed);
    if ((tail - q->tail.head_cache) >= HS_SPSC_SIZE) {
        /* Acquire: slots the consumer released are free to overwrite. */
        q->tail.head_cache = atomic_load_explicit(&q->head.v, memory_order_acquire);
        if ((tail - q->tail.head_cache) >= HS_SPSC_SIZE) return false;
    }

    HSSpscSlot* s = &q->slots[hs_ring_index(tail, HS_SPSC_SIZE)];
    s->msg = *msg;
    s->payload_len = len;
    s->trusted = trusted;
    if (len && payload) memcpy(s->payload, payload, len);
    atomic_store_explicit(&q->tail.v, tail + 1, memory_order_release);
    return true;
}

static int hs_process_node(Node* node) {
    if (!node || !node->process_fn || node->processing) return 0;
    node->processing = true;
    int processed = node->process_fn(node);
    node->processing = false;
    return processed;
}

static void hs_record_message(HSSystem* sys, const Message* msg) {
    if (!sys->log || sys->log_head >= sys->log_capacity || sys->log_head > UINT16_MAX ||
        (msg->payload_len && (!sys->payloads ||
         msg->payload_idx >= sys->payload_capacity || msg->payload_len > HS_PAYLOAD_SIZE))) {
        sys->log_overflow = true;
        sys->recording = false;
        return;
    }
    u32 slot = sys->log_head;
    if (msg->payload_len) {
        if (!sys->log_payloads) {
            if (sys->log_capacity > UINT16_MAX + 1u) {
                sys->log_overflow = true; sys->recording = false; return;
            }
            sys->log_payloads = aligned_alloc(64, (size_t)sys->log_capacity * sizeof(Payload));
            if (!sys->log_payloads) {
                sys->log_overflow = true; sys->recording = false; return;
            }
            sys->owns_log_payloads = true;
        }
        memset(&sys->log_payloads[slot], 0, sizeof(Payload));
        memcpy(sys->log_payloads[slot].data, sys->payloads[msg->payload_idx].data, msg->payload_len);
    }
    sys->log[sys->log_head++] = *msg;
}

/* A message refused by the schema gate. With validate_on_send the sender asked
 * for diagnostics (HS_ERR_VALIDATE with the reason). Otherwise a registered
 * destination answers HS_ACK_INVALID if an ack was requested, exactly as the
 * node would have; an unregistered one is a routing error. */
__attribute__((noinline, cold))
static void hs_reject_message(HSSystem* sys, const Message* msg, u8 stage) {
    if (sys->validate_on_send) {
        const char* err = NULL;
        (void)hs_validate_message(sys, msg, &err);
        hs_report_error_ex(sys, msg, HS_ERR_VALIDATE, stage, err ? err : "validate failed");
        return;
    }
    if (!sys->node_map[msg->to]) {
        hs_report_error_ex(sys, msg, HS_ERR_ROUTE, stage, "invalid destination node");
        return;
    }
    sys->rejected_invalid++;
    if ((msg->flags & HS_MSGF_ACK) == 0) return;
    Message ack = {
        .to = NODE_SYSTEM,
        .from = msg->to,
        .op = OP_ACK,
        .flags = HS_ACK_INVALID,
        .cid = msg->cid,
        .payload_idx = (u16)msg->op,
        .channel = CHAN_RT,
    };
    (void)hs_send_enqueue(sys, &ack, NULL, 0, true);
}

static bool hs_route_immediate(HSSystem* sys, Message* msg, const void* payload, u32 payload_len, bool trusted) {
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

    if (!trusted || sys->validate_on_send) {
        if (!hs_message_conforms(sys, msg)) {
            hs_reject_message(sys, msg, HS_ERR_STAGE_SEND);
            return false;
        }
    }
#ifdef HS_VERIFY_TRUSTED
    else if (!hs_message_conforms(sys, msg)) {
        fprintf(stderr, "hs_send_trusted: op %u to %u len %u does not conform\n",
                (unsigned)msg->op, (unsigned)msg->to, (unsigned)msg->payload_len);
        abort();
    }
#endif

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
            hs_record_message(sys, msg);
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
                        hs_record_message(sys, &res);
                    }
                }
            }
        }

        if (record_this) {
            hs_record_message(sys, msg);
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

    /* Transport admission must not become loss at the smaller actor inbox.
     * Apply the full inbox on the sole step thread, then retry the handoff. */
    if (dest->inbox.count == HS_QUEUE_SIZE && !dest->processing) {
        sys->inline_processed += (u32)hs_process_node(dest);
    }
    if (!mq_push(&dest->inbox, msg)) {
        hs_report_queue_full(sys, msg, msg->to);
        return false;
    }

    if (record_this) {
        hs_record_message(sys, msg);
    }

    hs_render_record(sys, msg);
    return true;
}

typedef struct {
    HSSystem* sys;
    int id;
    u32 epoch;
    u32 spsc_ok_local[CHAN_COUNT];
    u32 spsc_full_local[CHAN_COUNT];
} HSProducerTLS;

static _Thread_local HSProducerTLS g_tls_prod = {0};
static _Thread_local HSSystem* g_tls_in_step = NULL;

static int hs_get_producer_id(HSSystem* sys) {
    u32 epoch = atomic_load_explicit(&sys->producer_epoch, memory_order_relaxed);
    if (g_tls_prod.sys != sys || g_tls_prod.epoch != epoch) {
        g_tls_prod.sys = sys;
        g_tls_prod.epoch = epoch;
        g_tls_prod.id = -1;
        memset(g_tls_prod.spsc_ok_local, 0, sizeof(g_tls_prod.spsc_ok_local));
        memset(g_tls_prod.spsc_full_local, 0, sizeof(g_tls_prod.spsc_full_local));
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

static inline void hs_record_spsc_ok(HSSystem* sys, HSChannel ch, int pid) {
    if (!sys || pid < 0 || pid >= HS_MAX_PRODUCERS || ch <= CHAN_DEFAULT || ch >= CHAN_COUNT) return;
    u32 v = ++g_tls_prod.spsc_ok_local[(u32)ch];
    atomic_store_explicit(&sys->spsc_ok_by_prod[(u32)ch][pid], v, memory_order_relaxed);
}

static inline void hs_record_spsc_full(HSSystem* sys, HSChannel ch, int pid) {
    if (!sys || pid < 0 || pid >= HS_MAX_PRODUCERS || ch <= CHAN_DEFAULT || ch >= CHAN_COUNT) return;
    u32 v = ++g_tls_prod.spsc_full_local[(u32)ch];
    atomic_store_explicit(&sys->spsc_full_by_prod[(u32)ch][pid], v, memory_order_relaxed);
}

static void hs_backpressure_wait(HSSystem* sys) {
    if (!sys) return;
    atomic_fetch_add_explicit(&sys->bp_waiters, 1, memory_order_relaxed);
    pthread_mutex_lock(&sys->bp_lock);
    /* Timed wait to avoid missed wakeups causing long stalls. */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
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

u32 hs_toolbus_seq(HSSystem* sys) {
    if (!sys) return 0;
    if (!sys->toolbus_inited) return 0;
    pthread_mutex_lock(&sys->toolbus_lock);
    u32 s = sys->toolbus_seq;
    pthread_mutex_unlock(&sys->toolbus_lock);
    return s;
}

void hs_toolbus_record_result(HSSystem* sys, const Message* result) {
    if (!sys || !result) return;
    if (!sys->toolbus_inited) return;
    if ((OpCode)result->op != OP_RESULT) return;

    pthread_mutex_lock(&sys->toolbus_lock);

    if (sys->toolbus_count >= HS_TOOLBUS_SIZE) {
        sys->toolbus_dropped++;
    } else {
        sys->toolbus_count++;
    }

    u32 idx = sys->toolbus_head;
    sys->toolbus_head = hs_ring_index(sys->toolbus_head + 1, HS_TOOLBUS_SIZE);
    sys->toolbus_seq++;

    HSToolbusEntry* e = &sys->toolbus[idx];
    e->seq = sys->toolbus_seq;
    e->cid = result->cid;
    e->result_op = result->flags;
    u32 n = result->payload_len;
    if (n > HS_PAYLOAD_SIZE) n = HS_PAYLOAD_SIZE;
    e->payload_len = (u8)n;

    if (n > 0 && sys->payloads && result->payload_idx < sys->payload_capacity) {
        memcpy(e->payload, sys->payloads[result->payload_idx].data, n);
    }

    pthread_cond_broadcast(&sys->toolbus_cv);
    pthread_mutex_unlock(&sys->toolbus_lock);
}

bool hs_toolbus_wait(HSSystem* sys, u32 after_seq, u32 cid, u8 result_op, u8* out_payload, u32* out_len, u32 timeout_ms) {
    if (!sys) return false;
    if (!sys->toolbus_inited) return false;

    pthread_mutex_lock(&sys->toolbus_lock);

    const u64 deadline_ns = (u64)timeout_ms * 1000000ull;
    struct timespec start_ts;
    clock_gettime(CLOCK_MONOTONIC, &start_ts);

    for (;;) {
        for (u32 i = 0; i < HS_TOOLBUS_SIZE; i++) {
            u32 pos = hs_ring_index(sys->toolbus_head + HS_TOOLBUS_SIZE - 1 - i, HS_TOOLBUS_SIZE);
            const HSToolbusEntry* e = &sys->toolbus[pos];
            if (e->seq == 0 || e->seq <= after_seq) break;
            if (e->cid == cid && e->result_op == result_op) {
                u32 n = e->payload_len;
                if (out_payload && n) memcpy(out_payload, e->payload, n);
                if (out_len) *out_len = n;
                pthread_mutex_unlock(&sys->toolbus_lock);
                return true;
            }
        }

        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        u64 elapsed_ns = (u64)(now.tv_sec - start_ts.tv_sec) * 1000000000ull;
        long dn = now.tv_nsec - start_ts.tv_nsec;
        if (dn < 0) { elapsed_ns -= 1000000000ull; dn += 1000000000L; }
        elapsed_ns += (u64)dn;
        if (elapsed_ns >= deadline_ns) {
            pthread_mutex_unlock(&sys->toolbus_lock);
            return false;
        }

        u64 remain_ns = deadline_ns - elapsed_ns;
        u64 wait_ns = remain_ns > 10000000ull ? 10000000ull : remain_ns;
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        u64 add = (u64)ts.tv_nsec + wait_ns;
        ts.tv_sec += (time_t)(add / 1000000000ull);
        ts.tv_nsec = (long)(add % 1000000000ull);
        (void)pthread_cond_timedwait(&sys->toolbus_cv, &sys->toolbus_lock, &ts);
    }
}

static bool hs_send_enqueue(HSSystem* sys, Message* msg, const void* payload, u32 len, bool trusted) {
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
        return hs_route_immediate(sys, msg, payload, len, trusted);
    }

    int pid = hs_get_producer_id(sys);
    if (pid >= 0) {
        if (hs_spsc_push(&sys->producers[(u32)ch][pid], msg, payload, len, trusted)) {
            hs_record_spsc_ok(sys, ch, pid);
            return true;
        }
        hs_record_spsc_full(sys, ch, pid);
        /* TELEM: auto-drop instead of falling to MPSC */
        if (ch == CHAN_TELEM) {
            atomic_fetch_add_explicit(&sys->telem_dropped[(u32)ch], 1, memory_order_relaxed);
            return true;  /* "delivered" (as drop) */
        }
        /* fall through to MPSC */
    }

    return hs_submit_enqueue(sys, ch, msg, payload, len, trusted);
}

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

/* Every entry obeys the invariants hs_message_conforms() relies on:
 * HS_PAYLOAD_NONE has min_len == max_len == 0, HS_PAYLOAD_FIXED has
 * min_len == max_len (checked by test_p0). Missing entries are zero. */
const HSOpSpec hs_op_spec_table[OP_COUNT] = {
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
    [OP_DRAW_INSTANCE] = {NODE_BUFFER, HS_PAYLOAD_FIXED,  8,  8},
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

    /* Input ops previously had no entry, so validation compared them against
     * a zeroed spec (destination NODE_CPU) and rejected every one. Minimum
     * lengths match what input_node_process decodes. */
    [OP_INPUT_KEY]          = {NODE_INPUT, HS_PAYLOAD_RANGE, 3, HS_PAYLOAD_SIZE},
    [OP_INPUT_MOUSE]        = {NODE_INPUT, HS_PAYLOAD_RANGE, 8, HS_PAYLOAD_SIZE},
    [OP_INPUT_GAMEPAD]      = {NODE_INPUT, HS_PAYLOAD_RANGE, 8, HS_PAYLOAD_SIZE},
    [OP_INPUT_DEVICE_ADD]   = {NODE_INPUT, HS_PAYLOAD_RANGE, 0, HS_PAYLOAD_SIZE},
    [OP_INPUT_DEVICE_REMOVE]= {NODE_INPUT, HS_PAYLOAD_RANGE, 0, HS_PAYLOAD_SIZE},

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

bool hs_set_channel_budget(HSSystem* sys, HSChannel ch, u32 budget) {
    if (!sys) return false;
    if (ch <= CHAN_DEFAULT || ch >= CHAN_COUNT) return false;
    if (budget < HS_CHAN_BUDGET_MIN) return false;
    hs_lock(sys);
    sys->chan_budget[(u32)ch] = budget;
    hs_unlock(sys);
    return true;
}

bool mq_push(MessageQueue* q, Message* msg) {
    if (q->count >= HS_QUEUE_SIZE) return false;
    q->msgs[q->tail] = *msg;
    q->tail = hs_ring_index(q->tail + 1, HS_QUEUE_SIZE);
    q->count++;
    return true;
}

bool mq_pop(MessageQueue* q, Message* msg) {
    if (q->count == 0) return false;
    *msg = q->msgs[q->head];
    q->head = hs_ring_index(q->head + 1, HS_QUEUE_SIZE);
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
    atomic_init(&sys->block_on_full, false);
    sys->render_list = NULL;
    sys->log_overflow = false;
    sys->dropped_error_ex = 0;
    sys->dropped_queue_full = 0;
    sys->dropped_system_nonrt = 0;
    sys->dropped_result = 0;
    sys->rejected_invalid = 0;
    sys->step_pending = false;

    /* Capture defaults: record render channel only. */
    sys->record_mask = hs_channel_bit(CHAN_RENDER);

    /* Default per-channel drain budgets (messages per hs_step). */
    for (u32 c = 0; c < CHAN_COUNT; c++) sys->chan_budget[c] = 0;
    sys->chan_budget[CHAN_RT] = 4096;
    sys->chan_budget[CHAN_RENDER] = 16384;
    sys->chan_budget[CHAN_TELEM] = 1024;
    for (u32 c = 0; c < CHAN_COUNT; c++) {
        atomic_init(&sys->submit_full[c], 0);
        atomic_init(&sys->submit_hw[c], 0);
        atomic_init(&sys->telem_dropped[c], 0);
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
            sys->producers[c][i].tail.head_cache = 0;
            for (u32 j = 0; j < HS_SPSC_SIZE; j++) {
                memset(&sys->producers[c][i].slots[j].msg, 0, sizeof(Message));
                sys->producers[c][i].slots[j].payload_len = 0;
                memset(sys->producers[c][i].slots[j].payload, 0, HS_PAYLOAD_SIZE);
            }
        }
    }

    /* default channel policies */
    for (u32 c = 0; c < CHAN_COUNT; c++) {
        atomic_init(&sys->block_on_full_chan[c], false);
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

    /* Timed waits use CLOCK_MONOTONIC: the Pi has no RTC, so wall-clock
     * steps from NTP after boot are routine and must not stretch or cut waits. */
    pthread_condattr_t cattr;
    pthread_condattr_init(&cattr);
    pthread_condattr_setclock(&cattr, CLOCK_MONOTONIC);
    pthread_mutex_init(&sys->bp_lock, NULL);
    pthread_cond_init(&sys->bp_cv, &cattr);
    atomic_init(&sys->bp_waiters, 0);

    pthread_mutex_init(&sys->toolbus_lock, NULL);
    pthread_cond_init(&sys->toolbus_cv, &cattr);
    pthread_condattr_destroy(&cattr);
    sys->toolbus_inited = true;
    sys->toolbus_seq = 0;
    sys->toolbus_head = 0;
    sys->toolbus_count = 0;
    sys->toolbus_dropped = 0;
    memset(sys->toolbus, 0, sizeof(sys->toolbus));
}

void hs_register(HSSystem* sys, Node* node) {
    if (sys->node_count >= HS_MAX_NODES) return;
    node->next = NULL;
    sys->nodes[sys->node_count++] = node;
    sys->node_map[node->id] = node;
    node->processing = false;
}

void hs_destroy(HSSystem* sys) {
    if (!sys) return;
    if (sys->owns_log_payloads) free(sys->log_payloads);
    sys->log_payloads = NULL;
    sys->owns_log_payloads = false;
    if (sys->toolbus_inited) {
        pthread_cond_destroy(&sys->toolbus_cv);
        pthread_mutex_destroy(&sys->toolbus_lock);
        sys->toolbus_inited = false;
    }
    if (sys->lock_inited) {
        pthread_cond_destroy(&sys->bp_cv);
        pthread_mutex_destroy(&sys->bp_lock);
        pthread_mutex_destroy(&sys->lock);
        sys->lock_inited = false;
    }
    if (g_tls_prod.sys == sys) memset(&g_tls_prod, 0, sizeof(g_tls_prod));
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
    if (out) out->count = 0;
    if (!sys || !msgs || !out || !out->msgs || !out->payloads) return false;
    if (count == 0) return false;
    if (count > out->capacity) return false;
    if (!sys->payloads || sys->payload_capacity == 0) return false;

    if (count > UINT16_MAX + 1u || msgs != sys->log || count > sys->log_head ||
        sys->log_overflow) return false;
    for (u32 i = 0; i < count; i++) {
        Message m = msgs[i];
        memset(&out->payloads[i], 0, sizeof(Payload));
        if (m.payload_len != 0) {
            if (!sys->log_payloads || m.payload_len > HS_PAYLOAD_SIZE) return false;
            memcpy(out->payloads[i].data, sys->log_payloads[i].data, m.payload_len);
            m.payload_idx = (u16)i;
        }
        out->msgs[i] = m;
    }

    out->count = count;
    return true;
}

static bool hs_replay_source(HSSystem* sys, Message* msgs, u32 count,
                             const Payload* source, u32 source_capacity, bool journal);

bool hs_capture_replay(HSSystem* sys, const HSCapture* cap) {
    if (!sys || !cap || !cap->msgs || !cap->payloads) return false;
    if (cap->count == 0 || cap->count > cap->capacity) return false;

    fprintf(stderr, "Warning: replay assumes deterministic execution (rand/time will replay identically)\n");
    fflush(stderr);

    return hs_replay_source(sys, cap->msgs, cap->count, cap->payloads, cap->capacity, false);
}

bool hs_capture_write_file(const HSCapture* cap, const char* path) {
    if (!cap || !path || !cap->msgs || !cap->payloads) return false;
    if (cap->count > cap->capacity || cap->count > UINT16_MAX + 1u) return false;
    FILE* f = fopen(path, "wb");
    if (!f) return false;

    HSCaptureHeader h;
    hs_capture_header_init(&h, cap->count);

    bool ok = true;
    ok = ok && (fwrite(&h, sizeof(h), 1, f) == 1);
    ok = ok && (fwrite(cap->msgs, sizeof(Message), cap->count, f) == cap->count);
    ok = ok && (fwrite(cap->payloads, sizeof(Payload), cap->count, f) == cap->count);

    if (fclose(f) != 0) ok = false;
    return ok;
}

bool hs_capture_read_file(HSCapture* cap, const char* path, Message* msg_buf, Payload* payload_buf, u32 capacity) {
    if (!cap || !path || !msg_buf || !payload_buf || capacity == 0) return false;
    FILE* f = fopen(path, "rb");
    if (!f) return false;

    HSCaptureHeader h;
    size_t nread = fread(&h, sizeof(h), 1, f);
    if (nread != 1) {
        fprintf(stderr, "Capture read: failed to read header (got %zu)\n", nread);
        fclose(f);
        return false;
    }
    
    bool ok = hs_capture_header_valid(&h);
    
    if (!ok) {
        fprintf(stderr, "Capture read: invalid header magic=%.7s version=%u endian=0x%x\n", 
                h.magic, h.version, h.endian);
        fclose(f);
        return false;
    }
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

bool hs_try_send(HSSystem* sys, Message* msg, const void* payload, u32 len) {
    if (!sys || !msg || len > HS_PAYLOAD_SIZE || (len && !payload)) return false;
    return hs_send_enqueue(sys, msg, payload, len, false);
}

/* Queue full: wait for space if this channel blocks, then retry. Kept out of
 * line so the common case (first enqueue succeeds) needs no loop set-up. */
__attribute__((noinline, cold))
static bool hs_send_wait(HSSystem* sys, Message* msg, const void* data, u32 len, bool trusted) {
    for (;;) {
        if (g_tls_in_step == sys) return false; /* consumer cannot wait on itself */
        HSChannel ch = (HSChannel)msg->channel;
        bool block = (ch != CHAN_TELEM) && (sys->block_on_full || (ch < CHAN_COUNT && sys->block_on_full_chan[(u32)ch]));
        if (!block) return false;
        hs_backpressure_wait(sys);
        if (hs_send_enqueue(sys, msg, data, len, trusted)) return true;
    }
}

static inline __attribute__((always_inline))
bool hs_send_blocking(HSSystem* sys, Message* msg, const void* data, u32 len, bool trusted) {
    if (!sys || !msg) return false;
    if (__builtin_expect(hs_send_enqueue(sys, msg, data, len, trusted), 1)) return true;
    return hs_send_wait(sys, msg, data, len, trusted);
}

bool hs_send(HSSystem* sys, Message* msg) {
    return hs_send_blocking(sys, msg, NULL, 0, false);
}

bool hs_send_with_payload(HSSystem* sys, Message* msg, const void* data, u32 len) {
    return hs_send_blocking(sys, msg, data, len, false);
}

bool hs_send_trusted(HSSystem* sys, Message* msg, const void* data, u32 len) {
    if (len > HS_PAYLOAD_SIZE || (len && !data)) return false;
    return hs_send_blocking(sys, msg, data, len, true);
}

static inline void hs_submit_note_depth_step(HSSystem* sys, HSChannel ch) {
    if (!sys || ch <= CHAN_DEFAULT || ch >= CHAN_COUNT) return;
    HSSubmitQueue* q = &sys->submit[(u32)ch];
    u32 enq = atomic_load_explicit(&q->enqueue_pos, memory_order_relaxed);
    u32 deq = atomic_load_explicit(&q->dequeue_pos, memory_order_relaxed);
    u32 depth = enq - deq;
    u32 old = atomic_load_explicit(&sys->submit_hw[(u32)ch], memory_order_relaxed);
    if (depth > old) {
        atomic_store_explicit(&sys->submit_hw[(u32)ch], depth, memory_order_relaxed);
    }
}

u32 hs_step(HSSystem* sys) {
    hs_lock(sys);
    u32 processed = 0;
    bool pending = false;

    g_tls_in_step = sys;
    sys->inline_processed = 0;

    u32 prod_count = atomic_load_explicit(&sys->producer_count, memory_order_acquire);
    if (prod_count > HS_MAX_PRODUCERS) prod_count = HS_MAX_PRODUCERS;

    for (HSChannel ch = CHAN_RT; ch <= CHAN_TELEM; ch++) {
        u32 budget = sys->chan_budget[(u32)ch];
        u32 used = 0;

        hs_submit_note_depth_step(sys, ch);

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
                    HSSpscSlot* s = &q->slots[hs_ring_index(head + i, HS_SPSC_SIZE)];
                    (void)hs_route_immediate(sys, &s->msg, s->payload_len ? s->payload : NULL, s->payload_len, s->trusted);
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
            bool trusted = false;
            if (!hs_submit_dequeue(sys, ch, &m, payload, &payload_len, &trusted)) break;
            (void)hs_route_immediate(sys, &m, payload_len ? payload : NULL, payload_len, trusted);
            used++;
            processed++;
        }
        pending |= used >= budget;
    }
    
    for (u8 i = 0; i < sys->node_count; i++) {
        Node* node = sys->nodes[i];
        if (node->process_fn && node->inbox.count) {
            processed += (u32)hs_process_node(node);
        }
    }
    
    for (u8 i = 0; i < sys->node_count; i++) {
        Node* node = sys->nodes[i];
        while (!mq_empty(&node->outbox)) {
            Message msg;
            mq_pop(&node->outbox, &msg);
            (void)hs_route_immediate(sys, &msg, NULL, 0, true); /* node-built results */
        }
    }
    /* Messages sent while nodes ran (acks, results) wait in inboxes. */
    for (u8 i = 0; i < sys->node_count && !pending; i++)
        pending = sys->nodes[i]->process_fn && sys->nodes[i]->inbox.count;
    sys->step_pending = pending;
    
    sys->tick++;
    processed += sys->inline_processed;
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

static bool hs_replay_source(HSSystem* sys, Message* msgs, u32 count,
                             const Payload* source, u32 source_capacity, bool journal) {
    if (!sys || !msgs || count > UINT16_MAX + 1u) return false;
    hs_lock(sys);
    if (journal) {
        if (count > sys->log_head || sys->log_overflow) {
            hs_unlock(sys);
            return false;
        }
    }
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
        if (source && msg.payload_len) {
            u32 idx = journal ? i : msg.payload_idx;
            if (idx >= source_capacity || msg.payload_len > HS_PAYLOAD_SIZE ||
                !hs_payload_alloc_and_copy(sys, source[idx].data, msg.payload_len,
                                           &msg.payload_idx, &msg.payload_len)) {
                hs_unlock(sys);
                return false;
            }
        }
        msg.tick = sys->tick;

        /* Captures are external input: the routing schema gate applies. */
        if (!hs_message_conforms(sys, &msg)) {
            if (sys->validate_on_send || !sys->node_map[msg.to]) {
                hs_reject_message(sys, &msg, HS_ERR_STAGE_REPLAY);
                hs_unlock(sys);
                return false;
            }
            hs_reject_message(sys, &msg, HS_ERR_STAGE_REPLAY);
            continue;
        }
        
        bool found = false;
        for (u8 j = 0; j < sys->node_count; j++) {
            if (sys->nodes[j]->id == msg.to) {
                hs_render_record(sys, &msg);
                mq_push(&sys->nodes[j]->inbox, &msg);
                hs_process_node(sys->nodes[j]);
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

bool hs_replay(HSSystem* sys, Message* msgs, u32 count) {
    if (!sys) return false;
    bool journal = msgs == sys->log;
    return hs_replay_source(sys, msgs, count, journal ? sys->log_payloads : NULL,
                            journal ? sys->log_capacity : 0, journal);
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
    sys->rejected_invalid = 0;
    sys->step_pending = false;

    /* Reset producer lanes and submit queues (no concurrent producers allowed). */
    atomic_store_explicit(&sys->producer_count, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&sys->producer_epoch, 1, memory_order_relaxed);
    for (u32 c = 0; c < CHAN_COUNT; c++) {
        atomic_store_explicit(&sys->spsc_full[c], 0, memory_order_relaxed);
        atomic_store_explicit(&sys->spsc_ok[c], 0, memory_order_relaxed);
        atomic_store_explicit(&sys->mpsc_ok[c], 0, memory_order_relaxed);
        atomic_store_explicit(&sys->submit_full[c], 0, memory_order_relaxed);
        atomic_store_explicit(&sys->submit_hw[c], 0, memory_order_relaxed);
        atomic_store_explicit(&sys->telem_dropped[c], 0, memory_order_relaxed);

        for (u32 i = 0; i < HS_MAX_PRODUCERS; i++) {
            atomic_store_explicit(&sys->producers[c][i].head.v, 0, memory_order_relaxed);
            atomic_store_explicit(&sys->producers[c][i].tail.v, 0, memory_order_relaxed);
            sys->producers[c][i].tail.head_cache = 0;
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

    if (sys->toolbus_inited) {
        pthread_mutex_lock(&sys->toolbus_lock);
        sys->toolbus_seq++;
        sys->toolbus_head = 0;
        sys->toolbus_count = 0;
        memset(sys->toolbus, 0, sizeof(sys->toolbus));
        pthread_cond_broadcast(&sys->toolbus_cv);
        pthread_mutex_unlock(&sys->toolbus_lock);
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
        case OP_INPUT_KEY:     return "INPUT_KEY";
        case OP_INPUT_MOUSE:   return "INPUT_MOUSE";
        case OP_INPUT_GAMEPAD: return "INPUT_GAMEPAD";
        case OP_INPUT_DEVICE_ADD: return "INPUT_DEVICE_ADD";
        case OP_INPUT_DEVICE_REMOVE: return "INPUT_DEVICE_REMOVE";
        case OP_COUNT:         return "COUNT";
    }
    return "UNKNOWN";
}
