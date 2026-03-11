#include "hs_core.h"
#include <stdio.h>
#include <stdlib.h>

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
    sys->payload_head = 0;
    sys->recording = true;
    sys->log_overflow = false;
}

void hs_register(HSSystem* sys, Node* node) {
    if (sys->node_count >= HS_MAX_NODES) return;
    node->next = NULL;
    sys->nodes[sys->node_count++] = node;
}

static u32 allocate_payload(HSSystem* sys, const void* data, u32 len) {
    if (len > HS_PAYLOAD_SIZE) {
        len = HS_PAYLOAD_SIZE;
    }
    u32 idx = sys->payload_head;
    sys->payload_head = (sys->payload_head + 1) % HS_MAX_PAYLOADS;
    memcpy(sys->payloads[idx].data, data, len);
    return idx;
}

bool hs_send(HSSystem* sys, Message* msg) {
    msg->tick = sys->tick;
    
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
    msg->payload_idx = allocate_payload(sys, data, len);
    msg->payload_len = len;
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
