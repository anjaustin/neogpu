#include "hs_core.h"
#include "hs_nodes.h"
#include <stdio.h>
#include <stdlib.h>

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

    [OP_CLEAR]         = {NODE_OUTPUT, HS_PAYLOAD_FIXED, 16, 16},
    [OP_CLEAR_DS]      = {NODE_OUTPUT, HS_PAYLOAD_FIXED,  8,  8},

    [OP_SET_CHANNEL]   = {NODE_SOUND,  HS_PAYLOAD_FIXED,  2,  2},

    [OP_ERROR]         = {NODE_SYSTEM, HS_PAYLOAD_STRING, 1, HS_PAYLOAD_SIZE},
    [OP_TRACE]         = {NODE_SYSTEM, HS_PAYLOAD_STRING, 1, HS_PAYLOAD_SIZE},
    [OP_STOP]          = {NODE_SYSTEM, HS_PAYLOAD_NONE,   0, 0},
};

static bool hs_system_has_node(const HSSystem* sys, u8 id) {
    if (!sys) return false;
    for (u8 i = 0; i < sys->node_count; i++) {
        if (sys->nodes[i] && sys->nodes[i]->id == id) return true;
    }
    return false;
}

bool hs_payload_alloc_and_copy(HSSystem* sys, const void* data, u32 len, u16* out_idx, u32* out_len) {
    if (!sys || !sys->payloads || sys->payload_capacity == 0) return false;

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
    sys->log_overflow = false;
}

void hs_register(HSSystem* sys, Node* node) {
    if (sys->node_count >= HS_MAX_NODES) return;
    node->next = NULL;
    sys->nodes[sys->node_count++] = node;
}

void hs_capture_init(HSCapture* cap, Message* msg_buf, Payload* payload_buf, u32 capacity) {
    if (!cap) return;
    cap->msgs = msg_buf;
    cap->payloads = payload_buf;
    cap->capacity = capacity;
    cap->count = 0;
}

bool hs_capture_from_log(const HSSystem* sys, const Message* msgs, u32 count, HSCapture* out) {
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

bool hs_send(HSSystem* sys, Message* msg) {
    msg->tick = sys->tick;

#ifndef NDEBUG
    {
        const char* err = NULL;
        if (!hs_validate_message(sys, msg, &err)) {
            fprintf(stderr, "[HS] SEND REJECTED: op=%s to=%u err=%s\n",
                    hs_op_name((OpCode)msg->op), (unsigned)msg->to, err ? err : "unknown");
            return false;
        }
    }
#endif
    
    if (sys->recording) {
        if (sys->log_head >= sys->log_capacity) {
            sys->log_overflow = true;
            return false;
        }
        sys->log[sys->log_head++] = *msg;
    }
    
    for (u8 i = 0; i < sys->node_count; i++) {
        if (sys->nodes[i]->id == msg->to) {
            return mq_push(&sys->nodes[i]->inbox, msg);
        }
    }
    fprintf(stderr, "[HS] ERROR: Invalid destination node %d\n", msg->to);
    return false;
}

bool hs_send_with_payload(HSSystem* sys, Message* msg, const void* data, u32 len) {
    u16 idx = 0;
    u32 copy_len = 0;
    if (!hs_payload_alloc_and_copy(sys, data, len, &idx, &copy_len)) return false;
    msg->payload_idx = idx;
    msg->payload_len = copy_len;
    return hs_send(sys, msg);
}

u32 hs_step(HSSystem* sys) {
    u32 processed = 0;
    
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
            hs_send(sys, &msg);
        }
    }
    
    sys->tick++;
    return processed;
}

void hs_start_recording(HSSystem* sys) {
    sys->recording = true;
}

u32 hs_stop_recording(HSSystem* sys) {
    sys->recording = false;
    return sys->log_head;
}

bool hs_has_overflow(HSSystem* sys) {
    return sys->log_overflow;
}

bool hs_replay(HSSystem* sys, Message* msgs, u32 count) {
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

#ifndef NDEBUG
        {
            const char* err = NULL;
            if (!hs_validate_message(sys, &msg, &err)) {
                fprintf(stderr, "[HS] REPLAY REJECTED: i=%u op=%s to=%u err=%s\n",
                        (unsigned)i, hs_op_name((OpCode)msg.op), (unsigned)msg.to, err ? err : "unknown");
                return false;
            }
        }
#endif
        
        bool found = false;
        for (u8 j = 0; j < sys->node_count; j++) {
            if (sys->nodes[j]->id == msg.to) {
                mq_push(&sys->nodes[j]->inbox, &msg);
                sys->nodes[j]->process_fn(sys->nodes[j]);
                found = true;
                break;
            }
        }
        if (!found) {
            fprintf(stderr, "[HS] REPLAY ERROR: Invalid destination node %d at tick %u\n", msg.to, sys->tick);
            return false;
        }
        sys->tick++;
    }
    return true;
}

void hs_clear(HSSystem* sys) {
    sys->log_head = 0;
    sys->tick = 0;
    sys->payload_head = 0;
    sys->log_overflow = false;
    for (u8 i = 0; i < sys->node_count; i++) {
        mq_init(&sys->nodes[i]->inbox);
        mq_init(&sys->nodes[i]->outbox);
        if (sys->nodes[i]->reset_fn) {
            sys->nodes[i]->reset_fn(sys->nodes[i]);
        }
    }
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
        case OP_ERROR:         return "ERROR";
        case OP_TRACE:         return "TRACE";
        case OP_STOP:          return "STOP";
        case OP_COUNT:         return "COUNT";
    }
    return "UNKNOWN";
}
