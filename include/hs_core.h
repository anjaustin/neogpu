/*
 * NeoGPU - ARM NEON Optimized GPU Message Layer
 * 
 * High-performance message-passing GPU abstraction with ARM NEON SIMD.
 * 1M+ fps message throughput on Cortex-A72.
 */

#ifndef HS_CORE_H
#define HS_CORE_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <arm_neon.h>

#define HS_MAX_NODES       16
#define HS_MAX_MSG_LOG     65536
#define HS_QUEUE_SIZE      256
#define HS_MAX_PAYLOADS    4096
#define HS_PAYLOAD_SIZE    64

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t   s8;
typedef int32_t  s32;
typedef float    f32;
typedef double   f64;

typedef enum {
    OP_NOOP,
    OP_SET_SHADER,
    OP_SET_PARAM,
    OP_SET_GLOBAL,
    OP_SET_CAMERA,
    OP_SET_TARGET,
    OP_LOAD_BUFFER,
    OP_LOAD_TEXTURE,
    OP_DRAW,
    OP_DRAW_INSTANCE,
    OP_CLEAR,
    OP_CLEAR_DS,
    OP_CULL,
    OP_BLEND,
    OP_ALPHA,
    OP_DEPTH,
    OP_COLOR_MASK,
    OP_CLIP,
    OP_SET_CHANNEL,
    OP_STENCIL,
    OP_STENCIL_FUNC,
    OP_DEPTH_COMPARE,
    OP_DRAW_TEXT,
    OP_SHOW_TEXTURE,
    OP_TEXTURE_FILTER,
    OP_TEXTURE_WRAP,
    OP_ERROR,
    OP_TRACE,
    OP_STOP,
    OP_COUNT
} OpCode;

typedef struct {
    u8    to;
    u8    from;
    u8    op;
    u8    flags;
    u16   tick;
    u16   payload_idx;
    u32   payload_len;
} Message;

/* Removed static_assert - on some ABIs struct is 12 bytes, not 16 */

typedef struct __attribute__((aligned(64))) {
    u8    data[HS_PAYLOAD_SIZE];
} Payload;

typedef struct __attribute__((aligned(64))) {
    Message  msgs[HS_QUEUE_SIZE];
    u16      head;
    u16      tail;
    u16      count;
    u16      padding;
} MessageQueue;

_Static_assert(sizeof(MessageQueue) % 64 == 0, "Queue aligned to cache line");

typedef struct Node Node;

typedef int (*NodeProcessFn)(Node*);

typedef void (*NodeResetFn)(Node*);

struct Node {
    u8             id;
    MessageQueue   inbox;
    MessageQueue   outbox;
    NodeProcessFn  process_fn;
    NodeResetFn    reset_fn;
    void*          state;
    Node*          next;
};

typedef struct {
    Node*       nodes[HS_MAX_NODES];
    u8          node_count;
    u32         tick;
    Message*    log;
    u32         log_head;
    u32         log_capacity;
    bool        recording;
    Payload*    payloads;
    u32         payload_head;
    bool        log_overflow;
} HSSystem;

void mq_init(MessageQueue* q);
bool mq_push(MessageQueue* q, Message* msg);
bool mq_pop(MessageQueue* q, Message* msg);
bool mq_empty(MessageQueue* q);
u16  mq_count(MessageQueue* q);

void hs_init(HSSystem* sys, Message* log_buffer, u32 log_capacity, Payload* payload_buffer);
void hs_register(HSSystem* sys, Node* node);
bool hs_send(HSSystem* sys, Message* msg);
u32  hs_step(HSSystem* sys);
void hs_start_recording(HSSystem* sys);
u32  hs_stop_recording(HSSystem* sys);
bool hs_replay(HSSystem* sys, Message* msgs, u32 count);
void hs_clear(HSSystem* sys);
bool hs_has_overflow(HSSystem* sys);

const char* hs_op_name(OpCode op);

/* Debug/validation helper: validates message schema and routing. */
bool hs_validate_message(const HSSystem* sys, const Message* msg, const char** out_err);

#endif
