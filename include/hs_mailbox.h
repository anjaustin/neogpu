/* hs_mailbox: a resident QPU program that takes work from the CPU through
 * shared memory, without a kernel job per request (P0-5,
 * docs/audit/HARDWARE_HANDOFF_P0.md). Built on hs_qpu and hs_v3d only.
 *
 * How it works (each point measured; see the P0-5 section):
 * - The program runs as a chain of "leases": CSD jobs that each end after a
 *   poll budget, well under the kernel's ~1 s job reset. The CPU keeps
 *   `qdepth` leases queued while it uses the mailbox; when it stops calling,
 *   the queue drains and the GPU is free again (no background thread).
 * - The GPU polls a sequence word with TMU op 2 (cache publish + reload) and
 *   reads each request from a ring of 512-byte slots with plain loads. The GPU
 *   fetches 256-byte blocks, so a slot is only read fresh if the GPU has not
 *   read it in the current lease; a lease takes at most `slots` requests.
 * - Data the CPU changes between requests (indices, inputs, a changing gather
 *   table) is reloaded with op 2 per 16-word load; results are published with
 *   op 2 per 64-byte line before the GPU acknowledges.
 * - With batches = 8 the request runs on all 8 batches (every QPU); each
 *   acknowledges separately and has its own lease state.
 *
 * Costs that bind (measured): a request that meets a lease hand-over waits
 * ~50-180 us; a resident mailbox costs a GPU-bound renderer ~3.5% while its
 * leases run; while other GPU work runs, the mailbox's latency rises.
 *
 * Not thread-safe: one thread per hs_mailbox. Every BO a request touches must
 * be listed in hs_mailbox_cfg.bos (the kernel tracks BOs per job) and must
 * stay allocated until hs_mailbox_stop returns.
 */
#ifndef HS_MAILBOX_H
#define HS_MAILBOX_H

#include <stdint.h>

#include "hs_v3d.h"

enum {
    HS_MB_OP_ECHO = 0,   /* acknowledge only */
    HS_MB_OP_GATHER = 1, /* out[e] = table[idx[e]], 4-byte elements */
    HS_MB_OP_CHAIN = 2,  /* y = x; k times: y = y * c1 + c0 (float), out = y */
    HS_MB_OP_STOP = 3,   /* end the leases; queued leases quit at once */
};

typedef struct {
    uint32_t batches;    /* 1 or 8 */
    uint32_t budget;     /* polls per lease; 0: 100000 (~32 ms alone on the GPU) */
    uint32_t qdepth;     /* leases kept queued; 0: 2 */
    uint32_t slots;      /* request slots; 0: 256 (requests per lease at most) */
    const uint32_t* bos; /* the caller's BO handles that requests touch */
    uint32_t nbos;       /* at most 13 */
} hs_mailbox_cfg;

typedef struct {
    hs_v3d* dev;
    hs_v3d_bo code, unif, ctl;
    uint32_t handles[16], nhandles;
    uint32_t prog;
    uint32_t batches, budget, qdepth, slots;
    uint32_t seq;        /* last request sent */
    uint32_t done_seq;   /* last request every batch acknowledged */
    uint32_t submitted;  /* leases submitted */
    int running;
    uint64_t waits_over_20us, handovers_seen;
} hs_mailbox;

/* Allocates the buffers, builds the program and runs a ~1 ms GPU warm-up.
 * Leases are submitted on the first request. 0 or -errno. */
int hs_mailbox_start(hs_mailbox* mb, hs_v3d* dev, const hs_mailbox_cfg* cfg);

/* Sends one request (op, then up to 7 argument words; NULL: zeros) and returns its sequence number (> 0), or -errno:
 * -EINVAL for arguments the program cannot run (see each op), -EBUSY when
 * `slots` requests are already unacknowledged. Does not wait. */
int64_t hs_mailbox_send(hs_mailbox* mb, uint32_t op, const uint32_t args[7]);
/* Waits until every batch acknowledged request `seq` (and all before it).
 * 0, or -ETIME after timeout_ns. Keeps the lease queue topped up. */
int hs_mailbox_wait(hs_mailbox* mb, uint32_t seq, uint64_t timeout_ns);

/* Synchronous helpers (send + wait, 1 s timeout). 0 or -errno. */
int hs_mailbox_echo(hs_mailbox* mb);
/* elements: a multiple of 16 * batches. table_changes: 1 if the CPU may have
 * changed the table since the mailbox started (each table load is then
 * reloaded with op 2); 0 if the table is only read. */
int hs_mailbox_gather(hs_mailbox* mb, uint32_t idx_gpu, uint32_t table_gpu, uint32_t out_gpu, uint32_t elements,
                      int table_changes);
/* elements: a multiple of 32 * batches. k: a multiple of 4, 4..65536. */
int hs_mailbox_chain(hs_mailbox* mb, uint32_t in_gpu, uint32_t out_gpu, uint32_t elements, uint32_t k, float c1,
                     float c0);

/* Ends the leases and waits for the GPU to finish them. The mailbox can be
 * used again afterwards (the next request restarts the leases). */
int hs_mailbox_stop(hs_mailbox* mb);
void hs_mailbox_free(hs_mailbox* mb);

/* GPU-side lease state, for tests: leases finished by batch b. */
uint32_t hs_mailbox_leases_done(const hs_mailbox* mb, uint32_t b);

#endif
