/*
 * NeoGPU ML - Message-Passing Inference Implementation
 *
 * Implements the message routing layer for ML inference.
 * Single consumer thread processes messages in priority order.
 *
 * DISPATCH TABLE:
 *   ML_OP_GEMM    → hs_ml_ternary_f32_proj (LUT kernel, 4-thread NEON)
 *                   for HS_ROUTE_TERNARY_2BIT (I2_S packed, float32 acts)
 *   ML_OP_NORM    → rmsnorm (float32, in-place)
 *   ML_OP_ROPE    → rope_apply (float32, NEON) — GPU-eligible
 *   ML_OP_SOFTMAX → softmax (float32)          — GPU-eligible
 *   ML_OP_ACTIVATE→ relu2_mul (float32, NEON)  — GPU-eligible
 *   ML_OP_ADD     → vector add (float32, NEON) — GPU-eligible
 *   ML_OP_FENCE   → tick increment
 *
 * GPU concurrency: when ml_sys_set_gpu_node() is called, GPU-eligible ops
 * are submitted to the GPU node concurrently while the CPU handles GEMM.
 * The GPU runs a dedicated thread draining its own inbox.
 */

#include "hs_ml_msg.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* LUT-based ternary float32 projection kernel (hs_ml_ternary_neon.c) */
extern void hs_ml_ternary_f32_proj(float *out, const float *in,
                                    const uint8_t *W, uint32_t N, uint32_t K);

/*============================================================================
 * Timing
 *============================================================================*/

static inline u64 get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (u64)ts.tv_sec * 1000000000ULL + (u64)ts.tv_nsec;
}

/*============================================================================
 * GPU node (optional — fabric works without GPU)
 *
 * When registered, GPU-eligible ops are submitted to the GPU's inbox
 * concurrently with CPU GEMM. GPU node runs on a dedicated thread.
 *============================================================================*/

typedef struct {
    void (*submit)(void *ctx, const MLMsg *msg);  /* enqueue to GPU */
    void (*sync)(void *ctx);                       /* wait for GPU idle */
    void *ctx;
    bool  active;
} MLGpuNode;

static MLGpuNode g_gpu_node = { NULL, NULL, NULL, false };

void ml_sys_set_gpu_node(void (*submit_fn)(void*, const MLMsg*),
                          void (*sync_fn)(void*), void *ctx) {
    g_gpu_node.submit = submit_fn;
    g_gpu_node.sync   = sync_fn;
    g_gpu_node.ctx    = ctx;
    g_gpu_node.active = (submit_fn != NULL);
}

/* GPU-eligible: small ops that fit in V3D shared memory / shader registers */
static inline bool op_is_gpu_eligible(MLOpCode op) {
    return op == ML_OP_ROPE     ||
           op == ML_OP_SOFTMAX  ||
           op == ML_OP_ACTIVATE ||
           op == ML_OP_ADD;
}

/*============================================================================
 * System Lifecycle
 *============================================================================*/

void ml_sys_init(MLSystem* sys) {
    memset(sys, 0, sizeof(*sys));
    
    /* Initialize queues */
    for (int ch = 0; ch < ML_CHAN_COUNT; ch++) {
        for (int p = 0; p < ML_MAX_PRODUCERS; p++) {
            ml_spsc_init(&sys->producers[ch][p]);
        }
    }
    
    /* Default channel policies */
    sys->channels[ML_CHAN_PREFILL].budget = 4096;   /* High throughput for prefill */
    sys->channels[ML_CHAN_PREFILL].blocking = true;
    
    sys->channels[ML_CHAN_DECODE].budget = 256;     /* Lower budget, low latency */
    sys->channels[ML_CHAN_DECODE].blocking = true;
    
    sys->channels[ML_CHAN_TELEM].budget = 64;       /* Best effort */
    sys->channels[ML_CHAN_TELEM].blocking = false;
    
    /* Initialize backpressure primitives */
    pthread_mutex_init(&sys->bp_lock, NULL);
    pthread_cond_init(&sys->bp_cv, NULL);
    
    atomic_store(&sys->running, true);
    atomic_store(&sys->tick, 0);
    atomic_store(&sys->producer_count, 0);
}

void ml_sys_free(MLSystem* sys) {
    atomic_store(&sys->running, false);
    
    /* Wake any blocked producers */
    pthread_mutex_lock(&sys->bp_lock);
    pthread_cond_broadcast(&sys->bp_cv);
    pthread_mutex_unlock(&sys->bp_lock);
    
    pthread_mutex_destroy(&sys->bp_lock);
    pthread_cond_destroy(&sys->bp_cv);
}

int ml_sys_register_producer(MLSystem* sys) {
    u32 id = atomic_fetch_add(&sys->producer_count, 1);
    if (id >= ML_MAX_PRODUCERS) {
        atomic_fetch_sub(&sys->producer_count, 1);
        return -1;
    }
    return (int)id;
}

/*============================================================================
 * Message Submission
 *============================================================================*/

bool ml_sys_submit(MLSystem* sys, int producer_id, const MLMsg* msg) {
    if (!sys || producer_id < 0 || producer_id >= ML_MAX_PRODUCERS) {
        return false;
    }
    
    MLChannel ch = (MLChannel)msg->channel;
    if (ch >= ML_CHAN_COUNT) ch = ML_CHAN_PREFILL;
    
    MLSpscQueue* q = &sys->producers[ch][producer_id];
    MLChannelPolicy* policy = &sys->channels[ch];
    
    /* Try to push */
    if (ml_spsc_push(q, msg)) {
        return true;
    }
    
    /* Queue full */
    if (!policy->blocking) {
        /* Non-blocking channel: drop the message */
        policy->dropped++;
        return false;
    }
    
    /* Blocking channel: wait for space */
    atomic_fetch_add(&sys->bp_waiters, 1);
    pthread_mutex_lock(&sys->bp_lock);
    
    while (atomic_load(&sys->running) && ml_spsc_full(q)) {
        pthread_cond_wait(&sys->bp_cv, &sys->bp_lock);
    }
    
    pthread_mutex_unlock(&sys->bp_lock);
    atomic_fetch_sub(&sys->bp_waiters, 1);
    
    /* Try again after wakeup */
    if (!atomic_load(&sys->running)) {
        return false;
    }
    
    return ml_spsc_push(q, msg);
}

bool ml_sys_submit_gemm(MLSystem* sys, int producer_id,
                        MLLayerType layer, u8 layer_idx,
                        MLChannel channel, HSRouteFormat format,
                        void* input, void* weights, void* output,
                        u32 M, u32 N, u32 K) {
    MLMsg msg = ml_msg_gemm(layer, layer_idx, format, channel,
                            input, weights, output, M, N, K);
    return ml_sys_submit(sys, producer_id, &msg);
}

/*============================================================================
 * Kernel Dispatch
 *============================================================================*/

static void dispatch_gemm(MLSystem* sys, MLMsg* msg) {
    u64 t0 = get_time_ns();

    /* Dispatch to the LUT-based ternary float32 kernel.
     * Weights are I2_S packed (HS_ROUTE_TERNARY_2BIT): N rows x K/4 bytes.
     * Input and output are float32 — no int8 quantization of activations.
     * The 4KB LUT is built once (static, cached in ARM L1) and reused.
     *
     * For M=1 (decode step): N rows x K cols, one output vector of length N.
     * The kernel is already 4-threaded internally via pthreads.
     */
    if ((HSRouteFormat)msg->format == HS_ROUTE_TERNARY_2BIT) {
        hs_ml_ternary_f32_proj((float*)msg->output,
                               (const float*)msg->input,
                               (const uint8_t*)msg->weights,
                               msg->N, msg->K);
    } else {
        /* Fallback to legacy routing kernel for other formats */
        HSRouteDesc route;
        route.format = (HSRouteFormat)msg->format;
        route.K      = msg->K;
        route.N      = msg->N;
        route.routes = msg->weights;
        int threads = hs_ml_route_optimal_threads(&route, msg->M);
        hs_ml_route_mt((s32*)msg->output, msg->input, &route, msg->M, threads);
    }

    u64 elapsed = get_time_ns() - t0;

    MLLayerStats* stats = &sys->layer_stats[msg->layer_type];
    stats->messages_processed++;
    stats->total_ops += (u64)msg->M * msg->N * msg->K;
    stats->total_ns  += elapsed;
}

static void dispatch_norm(MLSystem* sys, MLMsg* msg) {
    (void)sys;
    float *input  = (float*)msg->input;
    float *output = (float*)msg->output;
    u32 N = msg->N;
    float ss = 0.0f;
#ifdef __ARM_NEON
    float32x4_t acc = vdupq_n_f32(0.0f);
    u32 n4 = N & ~3u;
    for (u32 i = 0; i < n4; i += 4) {
        float32x4_t v = vld1q_f32(input + i);
        acc = vfmaq_f32(acc, v, v);
    }
    ss = vaddvq_f32(acc);
    for (u32 i = n4; i < N; i++) ss += input[i] * input[i];
#else
    for (u32 i = 0; i < N; i++) ss += input[i] * input[i];
#endif
    float scale = 1.0f / sqrtf(ss / (float)N + 1e-5f);
    /* weights pointer carries the norm weight vector (optional) */
    const float *w = (const float*)msg->weights;
#ifdef __ARM_NEON
    float32x4_t vsc = vdupq_n_f32(scale);
    for (u32 i = 0; i < n4; i += 4) {
        float32x4_t v = vmulq_f32(vld1q_f32(input + i), vsc);
        if (w) v = vmulq_f32(v, vld1q_f32(w + i));
        vst1q_f32(output + i, v);
    }
    for (u32 i = n4; i < N; i++)
        output[i] = input[i] * scale * (w ? w[i] : 1.0f);
#else
    for (u32 i = 0; i < N; i++)
        output[i] = input[i] * scale * (w ? w[i] : 1.0f);
#endif
}

static void dispatch_rope(MLSystem* sys, MLMsg* msg) {
    (void)sys;
    /* RoPE: rotate pairs of dimensions in Q or K heads.
     * input/output: float[num_heads * head_dim]
     * M = num_heads, N = head_dim, position in msg->position
     * weights = float[head_dim/2] precomputed cos/sin (optional)
     * If weights==NULL, compute on the fly using rope_theta from msg->batch_idx
     */
    float *x = (float*)msg->input;   /* in-place if output==input */
    float *out = msg->output ? (float*)msg->output : x;
    u32 num_heads = msg->M;
    u32 head_dim  = msg->N;
    u32 pos       = msg->position;
    float theta   = (float)msg->batch_idx; /* rope_theta packed here */
    if (theta < 1.0f) theta = 500000.0f;  /* default BitNet rope_theta */

    for (u32 h = 0; h < num_heads; h++) {
        float *xh    = x   + h * head_dim;
        float *outh  = out + h * head_dim;
        u32 half = head_dim / 2;
        for (u32 i = 0; i < half; i++) {
            float freq  = 1.0f / powf(theta, (float)(2*i) / (float)head_dim);
            float angle = (float)pos * freq;
            float cs = cosf(angle), sn = sinf(angle);
            float x0 = xh[i], x1 = xh[i + half];
            outh[i]      = x0 * cs - x1 * sn;
            outh[i+half] = x0 * sn + x1 * cs;
        }
    }
}

static void dispatch_softmax(MLSystem* sys, MLMsg* msg) {
    (void)sys;
    float *x = (float*)msg->input;
    float *out = msg->output ? (float*)msg->output : x;
    u32 N = msg->N;
    float mx = x[0];
    for (u32 i = 1; i < N; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (u32 i = 0; i < N; i++) { out[i] = expf(x[i] - mx); sum += out[i]; }
    float inv = 1.0f / sum;
    for (u32 i = 0; i < N; i++) out[i] *= inv;
}

static void dispatch_activate(MLSystem* sys, MLMsg* msg) {
    (void)sys;
    /* ReLU^2(gate) * up:
     * input  = gate_proj output [N]
     * weights = up_proj output [N]  (packed in weights pointer)
     * output  = ffn activation [N]
     */
    float *gate = (float*)msg->input;
    float *up   = (float*)msg->weights;
    float *out  = (float*)msg->output;
    u32 N = msg->N;
#ifdef __ARM_NEON
    float32x4_t zero = vdupq_n_f32(0.0f);
    u32 n4 = N & ~3u;
    for (u32 i = 0; i < n4; i += 4) {
        float32x4_t g = vmaxq_f32(vld1q_f32(gate+i), zero);
        vst1q_f32(out+i, vmulq_f32(vmulq_f32(g,g), vld1q_f32(up+i)));
    }
    for (u32 i = n4; i < N; i++) {
        float g = gate[i] > 0.0f ? gate[i] : 0.0f;
        out[i] = g * g * up[i];
    }
#else
    for (u32 i = 0; i < N; i++) {
        float g = gate[i] > 0.0f ? gate[i] : 0.0f;
        out[i] = g * g * up[i];
    }
#endif
}

static void dispatch_add(MLSystem* sys, MLMsg* msg) {
    (void)sys;
    float *a   = (float*)msg->input;
    float *b   = (float*)msg->weights;  /* second operand */
    float *out = (float*)msg->output;
    u32 N = msg->N;
#ifdef __ARM_NEON
    u32 n4 = N & ~3u;
    for (u32 i = 0; i < n4; i += 4)
        vst1q_f32(out+i, vaddq_f32(vld1q_f32(a+i), vld1q_f32(b+i)));
    for (u32 i = n4; i < N; i++) out[i] = a[i] + b[i];
#else
    for (u32 i = 0; i < N; i++) out[i] = a[i] + b[i];
#endif
}

static void dispatch_fence(MLSystem* sys, MLMsg* msg) {
    (void)msg;
    /* Sync point: if GPU node active, wait for it to drain */
    if (g_gpu_node.active && g_gpu_node.sync)
        g_gpu_node.sync(g_gpu_node.ctx);
    atomic_fetch_add(&sys->tick, 1);
}

void ml_sys_dispatch(MLSystem* sys, MLMsg* msg) {
    MLOpCode op = (MLOpCode)msg->op;

    /* GPU-eligible ops: submit to GPU node concurrently, skip CPU dispatch */
    if (g_gpu_node.active && op_is_gpu_eligible(op)) {
        g_gpu_node.submit(g_gpu_node.ctx, msg);
        /* GPU runs asynchronously — CPU continues to next message immediately */
        if (sys->capturing && sys->capture_count < sys->capture_capacity)
            sys->capture_buf[sys->capture_count++] = *msg;
        return;
    }

    switch (op) {
        case ML_OP_GEMM:     dispatch_gemm(sys, msg);     break;
        case ML_OP_NORM:     dispatch_norm(sys, msg);     break;
        case ML_OP_ROPE:     dispatch_rope(sys, msg);     break;
        case ML_OP_SOFTMAX:  dispatch_softmax(sys, msg);  break;
        case ML_OP_ACTIVATE: dispatch_activate(sys, msg); break;
        case ML_OP_ADD:      dispatch_add(sys, msg);      break;
        case ML_OP_FENCE:    dispatch_fence(sys, msg);    break;
        case ML_OP_COPY:
            /* Memory copy: input -> output, N bytes */
            if (msg->input && msg->output && msg->N)
                memcpy(msg->output, msg->input, msg->N * sizeof(float));
            break;
        default: break;
    }

    if (sys->capturing && sys->capture_count < sys->capture_capacity)
        sys->capture_buf[sys->capture_count++] = *msg;
}

/*============================================================================
 * Processing (Consumer Side)
 *============================================================================*/

u32 ml_sys_step(MLSystem* sys) {
    u32 processed = 0;
    u32 prod_count = atomic_load(&sys->producer_count);
    if (prod_count > ML_MAX_PRODUCERS) prod_count = ML_MAX_PRODUCERS;
    
    /* Process channels in priority order: PREFILL, DECODE, TELEM */
    for (int ch = 0; ch < ML_CHAN_COUNT; ch++) {
        u32 budget = sys->channels[ch].budget;
        u32 used = 0;
        
        /* Drain all producer queues for this channel */
        for (u32 p = 0; p < prod_count && used < budget; p++) {
            MLSpscQueue* q = &sys->producers[ch][p];
            
            /* Drain up to remaining budget from this producer */
            while (used < budget) {
                MLMsg msg;
                if (!ml_spsc_pop(q, &msg)) {
                    break;  /* Queue empty */
                }
                
                /* Assign tick */
                msg.tick = atomic_load(&sys->tick);
                
                /* Dispatch to kernel */
                ml_sys_dispatch(sys, &msg);
                
                used++;
                processed++;
            }
        }
    }
    
    /* Increment tick */
    atomic_fetch_add(&sys->tick, 1);
    
    /* Wake blocked producers if any */
    if (atomic_load(&sys->bp_waiters) > 0) {
        pthread_mutex_lock(&sys->bp_lock);
        pthread_cond_broadcast(&sys->bp_cv);
        pthread_mutex_unlock(&sys->bp_lock);
    }
    
    return processed;
}

/*============================================================================
 * Capture/Replay
 *============================================================================*/

void ml_sys_capture_start(MLSystem* sys, MLMsg* buf, u32 capacity) {
    sys->capture_buf = buf;
    sys->capture_capacity = capacity;
    sys->capture_count = 0;
    sys->capturing = true;
}

u32 ml_sys_capture_stop(MLSystem* sys) {
    sys->capturing = false;
    return sys->capture_count;
}

bool ml_sys_replay(MLSystem* sys, const MLMsg* msgs, u32 count) {
    /* Reset stats */
    ml_sys_reset_stats(sys);
    atomic_store(&sys->tick, 0);
    
    /* Process each message directly (single-threaded replay) */
    for (u32 i = 0; i < count; i++) {
        MLMsg msg = msgs[i];
        msg.tick = atomic_load(&sys->tick);
        ml_sys_dispatch(sys, &msg);
        atomic_fetch_add(&sys->tick, 1);
    }
    
    return true;
}

/*============================================================================
 * Statistics
 *============================================================================*/

const MLLayerStats* ml_sys_layer_stats(MLSystem* sys, MLLayerType layer) {
    if (layer >= ML_LAYER_COUNT) return NULL;
    return &sys->layer_stats[layer];
}

void ml_sys_reset_stats(MLSystem* sys) {
    memset(sys->layer_stats, 0, sizeof(sys->layer_stats));
    for (int ch = 0; ch < ML_CHAN_COUNT; ch++) {
        sys->channels[ch].dropped = 0;
    }
}

void ml_sys_set_budget(MLSystem* sys, MLChannel ch, u32 budget) {
    if (ch < ML_CHAN_COUNT) {
        sys->channels[ch].budget = budget;
    }
}

void ml_sys_set_blocking(MLSystem* sys, MLChannel ch, bool blocking) {
    if (ch < ML_CHAN_COUNT) {
        sys->channels[ch].blocking = blocking;
    }
}
