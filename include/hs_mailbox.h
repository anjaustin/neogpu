/* hs_mailbox: a resident QPU program that takes work from the CPU through
 * shared memory, without a kernel job per request (P0-5,
 * docs/audit/HARDWARE_HANDOFF_P0.md). Built on hs_qpu, hs_v3d and pthreads.
 *
 * How it works (each point measured; see the P0-5 section):
 * - The program runs as a chain of "leases": CSD jobs that each end after a
 *   poll budget. The CPU keeps `qdepth` leases queued while it uses the
 *   mailbox.
 * - The GPU polls a sequence word with TMU op 2 (cache publish + reload) and
 *   reads each request from a ring of 512-byte slots with plain loads. The GPU
 *   fetches 256-byte blocks, so a slot is only read fresh if the GPU has not
 *   read it in the current job; a lease ends where the ring wraps, so it
 *   takes at most `slots` requests.
 * - Data the CPU changes between requests (indices, inputs, a changing gather
 *   table) is reloaded with op 2 per 16-word load; results are published with
 *   op 2 per 64-byte line before the GPU acknowledges.
 * - With batches = 8 the request runs on all 8 batches (every QPU); each
 *   acknowledges separately.
 * - A batch's published acknowledgement is its only durable state. A lease
 *   the kernel kills is resumed by the next one at the first unacknowledged
 *   request, so a request can run more than once; requests whose output
 *   overlaps their input are refused for that reason.
 *
 * How long a lease can hold the GPU:
 * - A poll count is not a time: beside a renderer one poll takes several times
 *   longer, and the kernel resets a job that makes no progress for too long.
 *   A watchdog thread (started by hs_mailbox_start) checks every 1 ms:
 *   - if no lease job has ended for `max_lease_us`, every batch ends its
 *     lease at its next poll, with requests still pending left to the next
 *     lease (a request a batch is running is finished first);
 *   - once the caller has not sent or waited for `idle_us`, leases end as
 *     soon as no request is pending.
 *   Without the thread (watchdog_off, or a stopped process) only the poll
 *   budget, the ring wrap and a stop end a lease.
 * - Other compute jobs queue behind the leases. To run one promptly, call
 *   hs_mailbox_stop first (measured: 0.15 ms for stop + job, against 3.1 ms
 *   behind the leases, or 20-27 ms with the watchdog off).
 * - One request is not interruptible; send bounds its size (see below).
 * - While in use the mailbox shares the QPUs with any renderer. Measured
 *   beside a GPU-bound Mesa renderer at 110 FPS, one request every 100 us:
 *   107-108 FPS on 1 batch, 67-73 FPS on 8. Use 8 batches when the GPU is
 *   otherwise free, or in bursts. A caller quiet for idle_us costs nothing.
 * - The first request after the GPU has been idle for more than ~100 ms
 *   takes about 10 ms (the device's runtime resume), as any job does.
 *
 * Not thread-safe: one thread per hs_mailbox, and the struct must not move
 * after hs_mailbox_start. Every BO a request touches must be listed in
 * hs_mailbox_cfg.bos and stay allocated until hs_mailbox_stop returns 0.
 * Gather indices are not checked: an index outside the table reads whatever
 * the GPU address holds, as with any gather.
 */
#ifndef HS_MAILBOX_H
#define HS_MAILBOX_H

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>

#include "hs_v3d.h"

enum {
    HS_MB_OP_ECHO = 0,   /* acknowledge only */
    HS_MB_OP_GATHER = 1, /* out[e] = table[idx[e]], 4-byte elements */
    HS_MB_OP_CHAIN = 2,  /* y = x; k times: y = y * c1 + c0 (float), out = y */
    HS_MB_OP_STOP = 3,   /* internal: use hs_mailbox_stop */
};
enum {
    HS_MB_MAX_BOS = 13,
    HS_MB_MAX_BUDGET = 250000, /* an idle lease alone: 0.32 us (1 batch) to 0.44 us (8) per poll */
    HS_MB_MAX_UNITS = 65536,   /* budget units one chain request may take per batch */
    HS_MB_MAX_GATHER_CHUNKS = 512, /* 16-element chunks per batch in one gather request */
};

typedef struct {
    uint32_t batches;      /* 1 or 8 */
    uint32_t budget;       /* polls per lease, 2..HS_MB_MAX_BUDGET; 0: 30000 */
    uint32_t qdepth;       /* leases kept queued, at most 8; 0: 2 */
    uint32_t slots;        /* request slots, a power of two 2..65536; 0: 256 */
    const hs_v3d_bo* const* bos; /* the caller's BOs that requests touch */
    uint32_t nbos;         /* at most HS_MB_MAX_BOS */
    uint32_t idle_us;      /* leases end this long after the last send/wait; 0: 2000 */
    uint32_t max_lease_us; /* a lease older than this is ended; 0: 50000 */
    int watchdog_off;      /* 1: no watchdog thread (tests; only the budget ends a lease) */
} hs_mailbox_cfg;

typedef struct {
    hs_v3d* dev;
    hs_v3d_bo code, unif, ctl;
    uint32_t handles[16], nhandles;
    uint32_t bo_lo[HS_MB_MAX_BOS], nbos;
    uint64_t bo_hi[HS_MB_MAX_BOS];
    uint32_t prog;
    uint32_t batches, budget, qdepth, slots;
    uint32_t seq;        /* last request sent */
    uint32_t done_seq;   /* last request every batch is known to have acknowledged */
    uint32_t submitted;  /* leases submitted */
    int running, stop_sent, stop_acked, yielding;
    int test_stop_fail_once; /* tests: the next stop returns -ETIME after sending its request */
    uint64_t waits_over_20us, handovers_seen;
    uint64_t stall_yields; /* waits that asked the leases to yield (batches serialized) */
    uint64_t lost_leases;  /* times queued leases turned out to be gone (killed) */
    /* watchdog */
    pthread_t wd_thread;
    pthread_mutex_t wd_mu;
    pthread_cond_t wd_cv;
    int wd_started;
    uint64_t idle_ns, max_lease_ns;
    _Atomic int wd_state, wd_quit;
    _Atomic uint32_t wd_act; /* bumped by every send and wait */
    _Atomic uint64_t wd_idle_yields, wd_age_yields;
} hs_mailbox;

/* Allocates the buffers, builds the program, runs a ~1 ms GPU warm-up and
 * starts the watchdog thread. Leases are submitted on the first request.
 * 0 or -errno (-EINVAL for a config outside the ranges above). */
int hs_mailbox_start(hs_mailbox* mb, hs_v3d* dev, const hs_mailbox_cfg* cfg);

/* Sends one request (op, then 7 argument words; NULL: zeros) and returns its
 * sequence number (0..2^32-1; it wraps), or -errno. Does not wait.
 *   GATHER  args: chunks per batch (16 elements each), idx, table, out,
 *           table_changes (0/1), table elements, 0
 *   CHAIN   args: pairs of chunks per batch (32 elements each), in, 0, out,
 *           k/4 (1..16384), c1 bits, c0 bits
 * -EINVAL: an op other than ECHO/GATHER/CHAIN; an address that is not
 * word-aligned; a range (idx, table, in, out) not inside one listed BO; an
 * output that overlaps an input or the table; a request over the size limit (gather: HS_MB_MAX_GATHER_CHUNKS
 * chunks per batch; chain: pairs * (k/4 + 4) + 4 over HS_MB_MAX_UNITS). The
 * limits keep one request, which cannot be interrupted, to a few ms alone on
 * the GPU; split larger work into several requests. -EBUSY: `slots` requests are
 * unacknowledged, or a stop is under way. Another -errno: the lease could
 * not be submitted; nothing was sent. */
int64_t hs_mailbox_send(hs_mailbox* mb, uint32_t op, const uint32_t args[7]);
/* Waits (spinning) until every batch acknowledged request `seq` and all
 * before it. 0; -EINVAL if `seq` was never sent; -ETIME after timeout_ns, in
 * which case the request is still queued and will run: wait again or call
 * hs_mailbox_stop before reusing or freeing its buffers. */
int hs_mailbox_wait(hs_mailbox* mb, uint32_t seq, uint64_t timeout_ns);

/* Synchronous helpers (send + wait with a 1 s timeout). 0 or -errno as above. */
int hs_mailbox_echo(hs_mailbox* mb);
/* elements: a multiple of 16 * batches, at most 16 * batches *
 * HS_MB_MAX_GATHER_CHUNKS (8192 on 1 batch, 65536 on 8). table_elements: the
 * table's length, for the range and overlap checks (indices themselves are
 * not checked). table_changes: 1 if the CPU may have written the table since
 * the mailbox started (each table load is then reloaded with op 2); 0 if only
 * the GPU writes it or nobody does. */
int hs_mailbox_gather(hs_mailbox* mb, uint32_t idx_gpu, uint32_t table_gpu, uint32_t table_elements, uint32_t out_gpu,
                      uint32_t elements, int table_changes);
/* elements: a multiple of 32 * batches. k: a multiple of 4, 4..65536. */
int hs_mailbox_chain(hs_mailbox* mb, uint32_t in_gpu, uint32_t out_gpu, uint32_t elements, uint32_t k, float c1,
                     float c0);

/* Waits for every request sent, ends the leases and waits for the GPU to
 * finish them. 0, or -errno with the stop still under way: call again to
 * finish it (sends are refused with -EBUSY until it has). The mailbox can be
 * used again after a stop: the next request restarts the leases. */
int hs_mailbox_stop(hs_mailbox* mb);
/* Stops (best effort), ends the watchdog and frees everything. */
void hs_mailbox_free(hs_mailbox* mb);

/* For tests: leases finished by batch b; on a stopped mailbox, continue from
 * sequence number `seq` (to reach the 32-bit wrap); and count one more lease
 * for batch b, as a killed job that only batch b finished would leave it. */
uint32_t hs_mailbox_leases_done(const hs_mailbox* mb, uint32_t b);
int hs_mailbox_test_seek(hs_mailbox* mb, uint32_t seq);
void hs_mailbox_test_count_lease(hs_mailbox* mb, uint32_t b);

#endif
