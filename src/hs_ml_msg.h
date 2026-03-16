/*
 * NeoGPU ML - Message-Passing Inference Layer
 * 
 * Applies lessons from NeoGPU's graphics messaging fabric to ML inference:
 * - Channel-based QoS (prefill vs decode vs telemetry)
 * - SPSC queues for low-latency routing
 * - Semantic message classes
 * - Deterministic scheduling with budgets
 * - Capture/replay for debugging
 * 
 * Core insight: Weights are routing tables, activations are messages.
 * Inference is message-passing through a graph of routing nodes.
 */

#ifndef HS_ML_MSG_H
#define HS_ML_MSG_H

#include "hs_core.h"
#include "hs_ml_routing.h"

/*============================================================================
 * Message Types
 *============================================================================*/

/* Layer types - the "destination nodes" in the message graph */
typedef enum {
    ML_LAYER_EMBED = 0,     /* Token embedding lookup */
    ML_LAYER_NORM,          /* RMSNorm / LayerNorm */
    ML_LAYER_ATTN_QKV,      /* Attention Q/K/V projection */
    ML_LAYER_ATTN_SCORE,    /* Attention score computation */
    ML_LAYER_ATTN_OUT,      /* Attention output projection */
    ML_LAYER_FFN_GATE,      /* FFN gate projection */
    ML_LAYER_FFN_UP,        /* FFN up projection */
    ML_LAYER_FFN_DOWN,      /* FFN down projection */
    ML_LAYER_HEAD,          /* LM head (logits) */
    ML_LAYER_SAMPLE,        /* Token sampling */
    ML_LAYER_COUNT
} MLLayerType;

/* Operation codes - what to do with the message */
typedef enum {
    ML_OP_GEMM = 0,         /* Matrix multiply (the core operation) */
    ML_OP_NORM,             /* Normalization */
    ML_OP_ROPE,             /* Rotary position embedding */
    ML_OP_SOFTMAX,          /* Softmax attention or sampling */
    ML_OP_ACTIVATE,         /* Activation function (SiLU, GELU, etc) */
    ML_OP_ADD,              /* Residual add */
    ML_OP_COPY,             /* Memory copy / gather */
    ML_OP_FENCE,            /* Synchronization point */
    ML_OP_COUNT
} MLOpCode;

/* QoS channels - different guarantees for different phases */
typedef enum {
    ML_CHAN_PREFILL = 0,    /* Bulk context processing (high throughput) */
    ML_CHAN_DECODE,         /* Token-by-token generation (low latency) */
    ML_CHAN_TELEM,          /* Telemetry/profiling (droppable) */
    ML_CHAN_COUNT
} MLChannel;

/* Message flags */
#define ML_MSGF_ACK       (1u << 0)  /* Request acknowledgment */
#define ML_MSGF_FENCE     (1u << 1)  /* Wait for completion before next */
#define ML_MSGF_LAST_TOK  (1u << 2)  /* Last token in sequence (for KV cache) */
#define ML_MSGF_FIRST_TOK (1u << 3)  /* First token (reset state) */

/*
 * ML Message - the unit of work in the inference pipeline
 * 
 * 128 bytes for cache-line pair alignment on 64-bit systems.
 * Pointers are 8 bytes each on aarch64.
 */
typedef struct __attribute__((aligned(64))) {
    /* Routing (8 bytes) */
    u8  layer_type;         /* MLLayerType: destination */
    u8  layer_idx;          /* Which layer (0..num_layers-1) */
    u8  op;                 /* MLOpCode: what operation */
    u8  channel;            /* MLChannel: QoS class */
    u8  flags;              /* ML_MSGF_* */
    u8  format;             /* HSRouteFormat: weight encoding */
    u16 seq_id;             /* Sequence/batch ID for ordering */
    
    /* Dimensions (12 bytes + 4 padding) */
    u32 M;                  /* Batch/sequence dimension */
    u32 N;                  /* Output dimension */
    u32 K;                  /* Input/reduction dimension */
    u32 pad0;               /* Alignment padding */
    
    /* Pointers (24 bytes on 64-bit) */
    void* input;            /* Input activation tensor */
    void* weights;          /* Weight tensor (routing table) */
    void* output;           /* Output buffer */
    
    /* Metadata (16 bytes) */
    u32 cid;                /* Correlation ID for request/response */
    u32 tick;               /* Assigned when processed */
    u32 position;           /* Token position (for RoPE, KV cache) */
    u32 head_idx;           /* Attention head index (for multi-head) */
    
    /* Extended metadata (to fill 128 bytes) */
    u64 timestamp;          /* Submission timestamp (ns) */
    u32 batch_idx;          /* Index within batch */
    u32 reserved[7];        /* Future use */
} MLMsg;

_Static_assert(sizeof(MLMsg) == 128, "MLMsg must be exactly 128 bytes");

/*============================================================================
 * Message Queue (SPSC Ring Buffer)
 *============================================================================*/

#define ML_QUEUE_SIZE 64   /* Must be power of 2 */
#define ML_QUEUE_MASK (ML_QUEUE_SIZE - 1)

/*
 * Cache-line padded atomic for avoiding false sharing
 */
typedef struct {
    atomic_uint v;
    u8 pad[64 - sizeof(atomic_uint)];
} __attribute__((aligned(64))) MLAtomicU32;

/*
 * Single-Producer Single-Consumer queue
 * 
 * Producer writes to tail, consumer reads from head.
 * Lock-free, wait-free for both sides.
 */
typedef struct {
    MLAtomicU32 head;               /* Consumer position */
    MLAtomicU32 tail;               /* Producer position */
    MLMsg slots[ML_QUEUE_SIZE];     /* Message slots */
} MLSpscQueue;

/* Initialize queue */
static inline void ml_spsc_init(MLSpscQueue* q) {
    atomic_store_explicit(&q->head.v, 0, memory_order_relaxed);
    atomic_store_explicit(&q->tail.v, 0, memory_order_relaxed);
}

/* Push message (producer side). Returns false if full. */
static inline bool ml_spsc_push(MLSpscQueue* q, const MLMsg* msg) {
    u32 tail = atomic_load_explicit(&q->tail.v, memory_order_relaxed);
    u32 head = atomic_load_explicit(&q->head.v, memory_order_acquire);
    
    if (tail - head >= ML_QUEUE_SIZE) {
        return false;  /* Full */
    }
    
    q->slots[tail & ML_QUEUE_MASK] = *msg;
    atomic_store_explicit(&q->tail.v, tail + 1, memory_order_release);
    return true;
}

/* Pop message (consumer side). Returns false if empty. */
static inline bool ml_spsc_pop(MLSpscQueue* q, MLMsg* msg) {
    u32 head = atomic_load_explicit(&q->head.v, memory_order_relaxed);
    u32 tail = atomic_load_explicit(&q->tail.v, memory_order_acquire);
    
    if (head == tail) {
        return false;  /* Empty */
    }
    
    *msg = q->slots[head & ML_QUEUE_MASK];
    atomic_store_explicit(&q->head.v, head + 1, memory_order_release);
    return true;
}

/* Check available space */
static inline u32 ml_spsc_available(MLSpscQueue* q) {
    u32 tail = atomic_load_explicit(&q->tail.v, memory_order_relaxed);
    u32 head = atomic_load_explicit(&q->head.v, memory_order_acquire);
    return tail - head;
}

static inline bool ml_spsc_empty(MLSpscQueue* q) {
    return ml_spsc_available(q) == 0;
}

static inline bool ml_spsc_full(MLSpscQueue* q) {
    return ml_spsc_available(q) >= ML_QUEUE_SIZE;
}

/*============================================================================
 * ML Inference System
 *============================================================================*/

#define ML_MAX_LAYERS 64
#define ML_MAX_PRODUCERS 2

/* Per-layer statistics */
typedef struct {
    u64 messages_processed;
    u64 total_ops;              /* M*N*K accumulated */
    u64 total_ns;               /* Nanoseconds spent */
    u32 queue_full_count;
} MLLayerStats;

/* Channel budget and policy */
typedef struct {
    u32 budget;                 /* Max messages per step */
    bool blocking;              /* Block on full (vs drop) */
    u32 dropped;                /* Count of dropped messages */
} MLChannelPolicy;

/*
 * ML Inference System - the message router
 */
typedef struct {
    /* Per-channel, per-producer queues (fast path) */
    MLSpscQueue producers[ML_CHAN_COUNT][ML_MAX_PRODUCERS];
    atomic_uint producer_count;
    
    /* Channel policies */
    MLChannelPolicy channels[ML_CHAN_COUNT];
    
    /* Layer statistics */
    MLLayerStats layer_stats[ML_LAYER_COUNT];
    
    /* Global state */
    atomic_uint tick;
    atomic_bool running;
    
    /* Capture buffer for replay */
    MLMsg* capture_buf;
    u32 capture_capacity;
    u32 capture_count;
    bool capturing;
    
    /* Backpressure */
    pthread_mutex_t bp_lock;
    pthread_cond_t bp_cv;
    atomic_uint bp_waiters;
    
    /* Kernel dispatch table */
    void (*dispatch[ML_OP_COUNT])(const MLMsg* msg);
    
} MLSystem;

/*============================================================================
 * System Lifecycle
 *============================================================================*/

/* Initialize the ML inference system */
void ml_sys_init(MLSystem* sys);

/* Shutdown and free resources */
void ml_sys_free(MLSystem* sys);

/* Register as a producer (returns producer ID, or -1 on failure) */
int ml_sys_register_producer(MLSystem* sys);

/*
 * Register a GPU node for concurrent dispatch of GPU-eligible ops
 * (ML_OP_ROPE, ML_OP_SOFTMAX, ML_OP_ACTIVATE, ML_OP_ADD).
 *
 * submit_fn: called from ml_sys_dispatch for GPU-eligible messages.
 *            Must be non-blocking (enqueue to GPU thread inbox).
 * sync_fn:   called at ML_OP_FENCE — waits for GPU to drain its inbox.
 * ctx:       opaque pointer passed to both functions.
 *
 * Pass NULL for submit_fn to disable GPU dispatch (CPU handles all ops).
 */
void ml_sys_set_gpu_node(void (*submit_fn)(void*, const MLMsg*),
                          void (*sync_fn)(void*), void *ctx);

/*============================================================================
 * Message Submission
 *============================================================================*/

/* 
 * Submit a message to the system.
 * 
 * producer_id: From ml_sys_register_producer()
 * msg: The message to submit (copied into queue)
 * 
 * Returns: true if enqueued, false if queue full
 * 
 * If channel policy is blocking and queue is full, this will wait.
 */
bool ml_sys_submit(MLSystem* sys, int producer_id, const MLMsg* msg);

/*
 * Submit a GEMM operation (convenience wrapper)
 */
bool ml_sys_submit_gemm(MLSystem* sys, int producer_id,
                        MLLayerType layer, u8 layer_idx,
                        MLChannel channel, HSRouteFormat format,
                        void* input, void* weights, void* output,
                        u32 M, u32 N, u32 K);

/*============================================================================
 * Processing (Consumer Side)
 *============================================================================*/

/*
 * Process pending messages.
 * 
 * Called by the inference executor (single consumer thread).
 * Drains channels according to budget, in priority order:
 *   1. PREFILL (bulk processing)
 *   2. DECODE (low-latency tokens)
 *   3. TELEM (droppable)
 * 
 * Returns: Number of messages processed
 */
u32 ml_sys_step(MLSystem* sys);

/*
 * Process a single message (dispatch to kernel).
 * Called internally by ml_sys_step().
 */
void ml_sys_dispatch(MLSystem* sys, MLMsg* msg);

/*============================================================================
 * Capture/Replay
 *============================================================================*/

/* Start capturing messages */
void ml_sys_capture_start(MLSystem* sys, MLMsg* buf, u32 capacity);

/* Stop capturing and return count */
u32 ml_sys_capture_stop(MLSystem* sys);

/* Replay captured messages */
bool ml_sys_replay(MLSystem* sys, const MLMsg* msgs, u32 count);

/*============================================================================
 * Statistics and Debugging
 *============================================================================*/

/* Get layer statistics */
const MLLayerStats* ml_sys_layer_stats(MLSystem* sys, MLLayerType layer);

/* Reset all statistics */
void ml_sys_reset_stats(MLSystem* sys);

/* Set channel budget */
void ml_sys_set_budget(MLSystem* sys, MLChannel ch, u32 budget);

/* Set channel blocking policy */
void ml_sys_set_blocking(MLSystem* sys, MLChannel ch, bool blocking);

/*============================================================================
 * Helper: Build messages for common operations
 *============================================================================*/

static inline MLMsg ml_msg_gemm(MLLayerType layer, u8 layer_idx, 
                                 HSRouteFormat format, MLChannel channel,
                                 void* input, void* weights, void* output,
                                 u32 M, u32 N, u32 K) {
    MLMsg msg;
    memset(&msg, 0, sizeof(msg));
    msg.layer_type = layer;
    msg.layer_idx = layer_idx;
    msg.op = ML_OP_GEMM;
    msg.channel = channel;
    msg.format = format;
    msg.input = input;
    msg.weights = weights;
    msg.output = output;
    msg.M = M;
    msg.N = N;
    msg.K = K;
    return msg;
}

static inline MLMsg ml_msg_norm(u8 layer_idx, MLChannel channel,
                                 void* input, void* output, u32 N) {
    MLMsg msg;
    memset(&msg, 0, sizeof(msg));
    msg.layer_type = ML_LAYER_NORM;
    msg.layer_idx = layer_idx;
    msg.op = ML_OP_NORM;
    msg.channel = channel;
    msg.input = input;
    msg.output = output;
    msg.N = N;
    return msg;
}

static inline MLMsg ml_msg_fence(MLChannel channel, u32 cid) {
    MLMsg msg;
    memset(&msg, 0, sizeof(msg));
    msg.op = ML_OP_FENCE;
    msg.channel = channel;
    msg.flags = ML_MSGF_FENCE;
    msg.cid = cid;
    return msg;
}

/*============================================================================
 * Layer Name Helpers
 *============================================================================*/

static inline const char* ml_layer_name(MLLayerType t) {
    static const char* names[] = {
        "EMBED", "NORM", "ATTN_QKV", "ATTN_SCORE", "ATTN_OUT",
        "FFN_GATE", "FFN_UP", "FFN_DOWN", "HEAD", "SAMPLE"
    };
    return (t < ML_LAYER_COUNT) ? names[t] : "UNKNOWN";
}

static inline const char* ml_op_name(MLOpCode op) {
    static const char* names[] = {
        "GEMM", "NORM", "ROPE", "SOFTMAX", "ACTIVATE", "ADD", "COPY", "FENCE"
    };
    return (op < ML_OP_COUNT) ? names[op] : "UNKNOWN";
}

static inline const char* ml_channel_name(MLChannel ch) {
    static const char* names[] = { "PREFILL", "DECODE", "TELEM" };
    return (ch < ML_CHAN_COUNT) ? names[ch] : "UNKNOWN";
}

#endif /* HS_ML_MSG_H */
