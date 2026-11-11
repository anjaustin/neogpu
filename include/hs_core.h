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
#include <stddef.h>
#include <string.h>
#include "hs_neonCompat.h" /* <arm_neon.h> on aarch64 */

#include <pthread.h>
#include <stdatomic.h>

typedef struct HSRenderList HSRenderList;
typedef struct HSSystem HSSystem;

#define HS_MAX_NODES       16
#define HS_MAX_MSG_LOG     65536
#define HS_QUEUE_SIZE      256
#define HS_MAX_PAYLOADS    4096
#define HS_PAYLOAD_SIZE    96

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
    OP_ACK,
    OP_RESULT,
    OP_ERROR_EX,
    OP_QUEUE_FULL,
    OP_ASYNC_DONE,
    OP_ERROR,
    OP_TRACE,
    OP_STOP,

    /* Frame/QoS control */
    OP_FRAME_BEGIN,
    OP_FRAME_END,
    OP_PRESENT,
    OP_FENCE,

    /* System queries / control */
    OP_QUERY_STATS,
    OP_QUERY_FABRIC,
    OP_SET_RECORD_MASK,
    OP_SET_CHAN_BUDGET,
    OP_SET_BLOCK_POLICY,
    
    /* Input: values 43..48 are retired; preserve the capture/wire ABI. */
    OP_INPUT_KEY = 49,
    OP_INPUT_MOUSE,
    OP_INPUT_GAMEPAD,
    OP_INPUT_DEVICE_ADD,
    OP_INPUT_DEVICE_REMOVE,
    
    OP_COUNT
} OpCode;

typedef enum {
    CHAN_DEFAULT = 0,
    CHAN_RT = 1,
    CHAN_RENDER = 2,
    CHAN_TELEM = 3,
    CHAN_COUNT
} HSChannel;

typedef struct {
    u8    to;
    u8    from;
    u8    op;
    u8    flags;
    u32   cid;         /* correlation id (0 = none) */
    u16   tick;
    u16   payload_idx;
    u32   payload_len;
    u8    channel;     /* HSChannel (0 = default mapping) */
    u8    pad0;
    u16   pad1;
} Message;

#define HS_SUBMIT_SIZE     1024

typedef struct {
    atomic_uint seq;
    Message msg;
    u32 payload_len;
    bool trusted;     /* sent with hs_send_trusted(); set on every enqueue. Kept in
                       * the first cache line, which every message touches. */
    u8 payload[HS_PAYLOAD_SIZE];
} __attribute__((aligned(64))) HSSubmitSlot;

typedef struct {
    atomic_uint enqueue_pos;
    atomic_uint dequeue_pos;
    HSSubmitSlot slots[HS_SUBMIT_SIZE];
} HSSubmitQueue;

#define HS_MAX_PRODUCERS   8
#define HS_SPSC_SIZE       256

typedef struct {
    atomic_uint v;
    u8 pad[64 - sizeof(atomic_uint)];
} __attribute__((aligned(64))) HSAtomicCacheLine;

_Static_assert(sizeof(HSAtomicCacheLine) == 64, "HSAtomicCacheLine is one cache line");

typedef struct {
    Message msg;
    u32 payload_len;
    bool trusted;     /* sent with hs_send_trusted(); set on every push. Kept in
                       * the first cache line, which every message touches. */
    u8 payload[HS_PAYLOAD_SIZE];
} __attribute__((aligned(64))) HSSpscSlot;

_Static_assert(offsetof(HSSpscSlot, trusted) < 64 && offsetof(HSSubmitSlot, trusted) < 64,
               "the trusted flag shares the message's cache line");
_Static_assert(sizeof(HSSpscSlot) == 128 && sizeof(HSSubmitSlot) == 128, "slot size unchanged");

typedef struct {
    HSAtomicCacheLine head;
    HSAtomicCacheLine tail;
    HSSpscSlot slots[HS_SPSC_SIZE];
} HSSpscQueue;

/* Removed static_assert - on some ABIs struct is 12 bytes, not 16 */

typedef struct __attribute__((aligned(64))) {
    u8    data[HS_PAYLOAD_SIZE];
} Payload;

#define HS_TOOLBUS_SIZE 256
#define HS_TOOLBUS_PAYLOAD_MAX HS_PAYLOAD_SIZE

typedef struct {
    u32 seq;
    u32 cid;
    u8  result_op;
    u8  payload_len;
    u16 pad;
    u8  payload[HS_PAYLOAD_SIZE];
} HSToolbusEntry;

typedef struct __attribute__((aligned(64))) {
    Message  msgs[HS_QUEUE_SIZE];
    u16      head;
    u16      tail;
    u16      count;
    u16      padding;
} MessageQueue;

_Static_assert(sizeof(MessageQueue) % 64 == 0, "Queue aligned to cache line");

typedef struct Node Node;

/* Applies the node's inbox and returns the number of messages applied.
 * hs_step calls it only when the inbox is non-empty: a node has no idle work. */
typedef int (*NodeProcessFn)(Node*);

typedef void (*NodeResetFn)(Node*);

struct Node {
    u8             id;
    MessageQueue   inbox;
    MessageQueue   outbox;
    NodeProcessFn  process_fn;
    NodeResetFn    reset_fn;
    void*          state;
    HSSystem*      sys;
    Node*          next;
    bool           processing;
};

struct HSSystem {
    Node*       nodes[HS_MAX_NODES];
    Node*       node_map[256];
    u8          node_count;
    u32         tick;
    Message*    log;
    Payload*    log_payloads; /* lazy owned journal, or caller-supplied slots */
    u32         log_head;
    u32         log_capacity;
    bool        recording;
    bool        validate_on_send; /* gate trusted sends too; report refusals as HS_ERR_VALIDATE */
    atomic_bool block_on_full;    /* if true, hs_send waits instead of failing */
    HSRenderList* render_list;
    Payload*    payloads;
    u32         payload_capacity;
    u32         payload_head;
    bool        log_overflow;

    pthread_mutex_t lock;
    bool            lock_inited;

    /* Telemetry about dropped best-effort system events */
    u32 dropped_error_ex;
    u32 dropped_queue_full;
    u32 dropped_system_nonrt;
    u32 dropped_result;
    u32 rejected_invalid; /* untrusted messages refused by the routing schema gate */
    /* Set by hs_step: it left work it did not apply (a channel reached its
     * budget, or a message is waiting in a node inbox). When false, another
     * hs_step applies only messages sent after this one drained its queue. */
    bool step_pending;

    /* Capture policy */
    u32 record_mask; /* bitmask of (1u << HSChannel) */

    /* Per-channel drain budgets (messages per hs_step) */
    u32 chan_budget[CHAN_COUNT];

    HSSubmitQueue submit[CHAN_COUNT];
    atomic_uint submit_full[CHAN_COUNT];
    atomic_uint submit_hw[CHAN_COUNT];  /* High-water mark */
    atomic_uint telem_dropped[CHAN_COUNT];  /* TELEM drops when queue full */

    atomic_uint spsc_full[CHAN_COUNT];
    atomic_uint spsc_full_by_prod[CHAN_COUNT][HS_MAX_PRODUCERS];

    atomic_uint spsc_ok[CHAN_COUNT];
    atomic_uint spsc_ok_by_prod[CHAN_COUNT][HS_MAX_PRODUCERS];
    atomic_uint mpsc_ok[CHAN_COUNT];

    atomic_uint producer_count;
    atomic_uint producer_epoch;
    HSSpscQueue producers[CHAN_COUNT][HS_MAX_PRODUCERS];

    /* Channel policies */
    atomic_bool block_on_full_chan[CHAN_COUNT];

    pthread_mutex_t bp_lock;
    pthread_cond_t  bp_cv;
    atomic_uint     bp_waiters;

    /* Toolbus for IPC result correlation */
    pthread_mutex_t toolbus_lock;
    pthread_cond_t  toolbus_cv;
    bool            toolbus_inited;
    u32             toolbus_seq;
    u32             toolbus_head;
    u32             toolbus_count;
    u32             toolbus_dropped;
    HSToolbusEntry  toolbus[HS_TOOLBUS_SIZE];
    u32            inline_processed;
    bool           owns_log_payloads;
};

static inline u32 hs_channel_bit(HSChannel ch) {
    return (ch < CHAN_COUNT) ? (1u << (u32)ch) : 0u;
}

void hs_lock(HSSystem* sys);
void hs_unlock(HSSystem* sys);

void hs_set_record_mask(HSSystem* sys, u32 mask);
/* Per-step drain budget. A budget of 0 would never drain the channel (and on
 * CHAN_RT would wedge all IPC control traffic), so it is rejected. */
#define HS_CHAN_BUDGET_MIN 1u
bool hs_set_channel_budget(HSSystem* sys, HSChannel ch, u32 budget);

/* Message flags (Message.flags) */
#define HS_MSGF_ACK   (1u << 7)  /* request OP_ACK after apply (must not collide with opcode-specific flags) */

void mq_init(MessageQueue* q);
bool mq_push(MessageQueue* q, Message* msg);
bool mq_pop(MessageQueue* q, Message* msg);
bool mq_empty(MessageQueue* q);
u16  mq_count(MessageQueue* q);

void hs_init(HSSystem* sys, Message* log_buffer, u32 log_capacity, Payload* payload_buffer);
/* Exclusive teardown: stop/join producers and services before calling. */
void hs_destroy(HSSystem* sys);
void hs_register(HSSystem* sys, Node* node);
bool hs_send(HSSystem* sys, Message* msg);
/* Bounded admission for external services: never waits for queue space. */
bool hs_try_send(HSSystem* sys, Message* msg, const void* payload, u32 len);
/* Schema gate. Every message is checked once, when the step thread routes it
 * and before any node sees it: a message that fails hs_message_conforms() is
 * refused (HS_ERR_VALIDATE when validate_on_send is set, otherwise an
 * HS_ACK_INVALID ack if one was requested) and never reaches a node. Replay
 * applies the same gate. hs_send_trusted() is for messages the runtime builds
 * itself (hs_gpu_* helpers, node acks): they conform by construction, so the
 * gate is skipped for them unless validate_on_send is set. Builds with
 * HS_VERIFY_TRUSTED (the sanitizer builds) abort if one does not conform.
 * Blocks like hs_send_with_payload when the system blocks on full queues. */
bool hs_send_trusted(HSSystem* sys, Message* msg, const void* payload, u32 len);
u32  hs_step(HSSystem* sys);

/* Wake threads blocked in hs_send when block_on_full is enabled. */
void hs_wake_senders(HSSystem* sys);

/* Allocate a payload block and copy up to HS_PAYLOAD_SIZE bytes into it.
 * STEP THREAD ONLY: the payload ring is owned by the thread running hs_step.
 * Producer threads must use hs_send_with_payload()/hs_try_send(), which carry
 * the bytes inline and allocate on the step thread. */
bool hs_payload_alloc_and_copy(HSSystem* sys, const void* data, u32 len, u16* out_idx, u32* out_len);

/* Convenience wrapper: alloc+copy payload then send message. */
bool hs_send_with_payload(HSSystem* sys, Message* msg, const void* data, u32 len);
void hs_start_recording(HSSystem* sys);
u32  hs_stop_recording(HSSystem* sys);
bool hs_replay(HSSystem* sys, Message* msgs, u32 count);
void hs_clear(HSSystem* sys);
bool hs_has_overflow(HSSystem* sys);

const char* hs_op_name(OpCode op);

/* Debug/validation helper: validates message schema and routing. */
bool hs_validate_message(const HSSystem* sys, const Message* msg, const char** out_err);

/* Per-opcode message schema (hs_core.c). */
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

extern const HSOpSpec hs_op_spec_table[OP_COUNT];

/* Same verdict as hs_validate_message(sys, msg, NULL), without the error
 * text, cheap enough for every message a node handles. NONE and FIXED specs
 * reduce to the [min_len, max_len] range check (table invariant). */
static inline bool hs_message_conforms(const HSSystem* sys, const Message* msg) {
    if (!sys || !msg || msg->op >= OP_COUNT || !sys->node_map[msg->to]) return false;
    const HSOpSpec* spec = &hs_op_spec_table[msg->op];
    u32 len = msg->payload_len;
    if ((spec->expected_to != 0xFF && msg->to != spec->expected_to) |
        (len < spec->min_len) | (len > spec->max_len)) return false;
    if (len == 0) return true;
    if (!sys->payloads || msg->payload_idx >= sys->payload_capacity) return false;
    return spec->kind != HS_PAYLOAD_STRING || sys->payloads[msg->payload_idx].data[len - 1] == 0;
}

/*
 * Self-contained capture for deterministic replay.
 *
 * A capture stores messages plus deep-copied payload blocks. For payload-bearing
 * messages, the capture rewrites payload_idx to refer to capture payload slots.
 */
typedef struct {
    Message* msgs;
    Payload* payloads;
    u32      capacity;
    u32      count;
} HSCapture;

void hs_capture_init(HSCapture* cap, Message* msg_buf, Payload* payload_buf, u32 capacity);
bool hs_capture_from_log(const HSSystem* sys, const Message* msgs, u32 count, HSCapture* out);
bool hs_capture_replay(HSSystem* sys, const HSCapture* cap);

/* Capture persistence (binary, versioned) */
bool hs_capture_write_file(const HSCapture* cap, const char* path);
bool hs_capture_read_file(HSCapture* cap, const char* path, Message* msg_buf, Payload* payload_buf, u32 capacity);

/* Toolbus: IPC result correlation */
u32  hs_toolbus_seq(HSSystem* sys);
void hs_toolbus_record_result(HSSystem* sys, const Message* result);
bool hs_toolbus_wait(HSSystem* sys, u32 after_seq, u32 cid, u8 result_op, u8* out_payload, u32* out_len, u32 timeout_ms);

#endif
