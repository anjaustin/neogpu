/* rt_mailbox: adversarial tests for hs_mailbox (P0-5 red team,
 * docs/audit/HARDWARE_HANDOFF_P0.md). Each case prints one JSON line with
 * "ok": true/false; exit 1 if any case is not ok.
 *   rt_mailbox cases [batches]  every functional case below (default: 1, then 8)
 *   rt_mailbox lease <batches> <budget> [n=20]
 *       how long one idle lease runs (ms): the bound that must stay far
 *       below the kernel's job reset
 *   rt_mailbox soak <batches> <seconds>
 *       random checked requests, pauses and stops; latency and lease lengths
 *   rt_mailbox busy <batches> <seconds> <gap_us> <0 echo | 1 chain>
 *       a steady caller: latency and the longest lease (env RT_WD_OFF, RT_BUDGET)
 *   rt_mailbox pipe <batches> <seconds> [chunks per batch per request]
 *       requests always in flight: the lease bound under continuous load
 *   rt_mailbox chainload <batches> <seconds> <pairs per batch> <k> [gap_us]
 *       chain requests back to back: what a renderer beside them gets
 *   rt_mailbox reset <batches>      a REAL kernel reset of a lease mid-request
 *   rt_mailbox foreign <batches>    a REAL reset caused by another client's hung job
 *       (both add to the GPU reset count: run them alone and on purpose)
 *   rt_mailbox jobcheck <batches> <seconds>
 *       the shipped gather kernel as plain jobs, every word checked
 *   rt_mailbox idlegap <batches>
 *       first request after the GPU sat idle for 1..1000 ms: plain job vs mailbox
 *   rt_mailbox bowait           cost of a zero-timeout BO wait while a lease runs
 *   rt_mailbox jobwait <batches> [n=30]
 *       how long an ordinary compute job waits behind the mailbox's leases
 * Run through tests/redteam-p0-5/wrap.sh (GPU lock, governor, reset count).
 */
#define _GNU_SOURCE
#include <errno.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "hs_mailbox.h"
#include "hs_qpu.h"
#include "hs_qpu_kernels.h"
#include "hs_v3d.h"

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
static void die(const char* w, int e) {
    fprintf(stderr, "%s: %s\n", w, strerror(e < 0 ? -e : e));
    exit(2);
}
static uint32_t rng = 987654321;
static uint32_t rnd(void) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}
static int cmpd(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}

#define TABLE_N (1u << 24) /* 64 MiB: random indices miss every cache */
#define MAXE (1u << 18)
static hs_v3d dev;
static hs_v3d_bo bt, bi, bo, bx, by;
static const hs_v3d_bo* bos[5] = {&bt, &bi, &bo, &bx, &by};
static double last_call_ms;
static int failures;

static void report(const char* name, uint32_t batches, int ok, const char* fmt, ...) __attribute__((format(printf, 4, 5)));
#include <stdarg.h>
static void report(const char* name, uint32_t batches, int ok, const char* fmt, ...) {
    char d[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(d, sizeof d, fmt, ap);
    va_end(ap);
    printf("{\"case\":\"%s\",\"batches\":%u,\"ok\":%s,%s}\n", name, batches, ok ? "true" : "false", d);
    if (!ok) failures++;
}

static float chain_ref(float x, uint32_t k, float c1, float c0) {
    volatile float y = x;
    for (uint32_t i = 0; i < k; i++) {
        float m = y * c1;
        y = m + c0;
    }
    return y;
}
static int wd_off;
static void start(hs_mailbox* mb, uint32_t batches, uint32_t budget, uint32_t slots) {
    hs_mailbox_cfg cfg = {.batches = batches, .budget = budget, .slots = slots, .bos = bos, .nbos = 5, .watchdog_off = wd_off};
    if (getenv("RT_QDEPTH")) cfg.qdepth = (uint32_t)atoi(getenv("RT_QDEPTH"));
    if (getenv("RT_STRETCH")) cfg.chain_stretch = (uint32_t)strtoul(getenv("RT_STRETCH"), 0, 0); /* 0xffffffff: no switches */
    int e = hs_mailbox_start(mb, &dev, &cfg);
    if (e) die("hs_mailbox_start", e);
}
/* one checked gather of n elements at byte offsets io/oo into bi/bo */
static int gather_checked(hs_mailbox* mb, uint32_t n, uint32_t io, uint32_t oo, int* rc) {
    uint32_t* T = (uint32_t*)bt.map;
    uint32_t* ix = (uint32_t*)(bi.map + io);
    uint32_t* out = (uint32_t*)(bo.map + oo);
    for (uint32_t i = 0; i < n; i++) {
        ix[i] = rnd() % TABLE_N;
        out[i] = 0xDEADBEEF;
    }
    uint64_t t = now_ns();
    *rc = hs_mailbox_gather(mb, bi.gpu + io, bt.gpu, TABLE_N, bo.gpu + oo, n, 0);
    last_call_ms = (now_ns() - t) / 1e6;
    int bad = 0;
    for (uint32_t i = 0; i < n; i++) bad += out[i] != T[ix[i]];
    return bad;
}
static int chain_checked(hs_mailbox* mb, uint32_t n, uint32_t k, uint32_t io, uint32_t oo, int* rc) {
    float* x = (float*)(bx.map + io);
    float* y = (float*)(by.map + oo);
    for (uint32_t i = 0; i < n; i++) {
        x[i] = (float)(rnd() % 1000000) / 1e6f;
        y[i] = -1.0f;
    }
    float c1 = 0.9995f, c0 = 0.0007f;
    uint64_t t = now_ns();
    *rc = hs_mailbox_chain(mb, bx.gpu + io, by.gpu + oo, n, k, c1, c0);
    last_call_ms = (now_ns() - t) / 1e6;
    int bad = 0;
    for (uint32_t i = 0; i < n; i++) {
        float r = chain_ref(x[i], k, c1, c0);
        bad += memcmp(&r, &y[i], 4) != 0;
    }
    return bad;
}

static void cases(uint32_t batches) {
    hs_mailbox mb;
    int rc, bad;
    uint32_t per = 32 * batches;

    /* raw send must refuse everything the program cannot run safely */
    start(&mb, batches, 0, 0);
    {
        uint32_t per16 = 16 * batches;
        uint32_t bad_args[][8] = {
            {HS_MB_OP_GATHER, 1, bi.gpu + 2, bt.gpu, bo.gpu, 0, TABLE_N, 0},             /* 0 misaligned */
            {HS_MB_OP_GATHER, 0x7FFFFFFF, bi.gpu, bt.gpu, bo.gpu, 0, TABLE_N, 0},        /* 1 huge count */
            {HS_MB_OP_GATHER, HS_MB_MAX_GATHER_CHUNKS + 1, bi.gpu, bt.gpu, bo.gpu, 0, TABLE_N, 0}, /* 2 over the size limit */
            {HS_MB_OP_GATHER, 1, bi.gpu, bt.gpu, bo.gpu, 2, TABLE_N, 0},                 /* 3 flag not 0/1 */
            {HS_MB_OP_GATHER, 1, bi.gpu, bt.gpu, bi.gpu, 0, TABLE_N, 0},                 /* 4 out == idx */
            {HS_MB_OP_GATHER, 4, bi.gpu, bt.gpu, bi.gpu + 64 * batches, 0, TABLE_N, 0},  /* 5 out overlaps idx */
            {HS_MB_OP_GATHER, 4, bi.gpu, bo.gpu + 8, bo.gpu, 0, 16, 0},                  /* 6 table starts inside out */
            {HS_MB_OP_GATHER, 1, bi.gpu, bt.gpu, (uint32_t)(bo.gpu + bo.size - 4 * per16 + 4), 0, TABLE_N, 0}, /* 7 past the BO end */
            {HS_MB_OP_GATHER, 1, bi.gpu, bt.gpu, 0x10000, 0, TABLE_N, 0},                /* 8 not in a listed BO */
            {HS_MB_OP_GATHER, 1, 0xFFFFFFC0u, bt.gpu, bo.gpu, 0, TABLE_N, 0},            /* 9 wraps the address space */
            {HS_MB_OP_CHAIN, 1, bx.gpu, 0, by.gpu, 0, 0, 0},                             /* 10 k = 0 */
            {HS_MB_OP_CHAIN, 1, bx.gpu, 0, by.gpu, 16385, 0, 0},                         /* 11 k too large */
            {HS_MB_OP_CHAIN, 8, bx.gpu, 0, by.gpu, 16384, 0, 0},                         /* 12 over the unit limit */
            {HS_MB_OP_CHAIN, 1, bx.gpu, 0, bx.gpu, 4, 0, 0},                             /* 13 in place */
            {HS_MB_OP_CHAIN, 1, bx.gpu, 0, by.gpu + 1, 4, 0, 0},                         /* 14 misaligned */
            {HS_MB_OP_STOP, 0, 0, 0, 0, 0, 0, 0},                                        /* 15 stop is not a request */
            {77, 0, 0, 0, 0, 0, 0, 0},                                                   /* 16 unknown op */
            {HS_MB_OP_GATHER, 4, bi.gpu, bo.gpu, bo.gpu + 4096, 0, 2048, 0},             /* 17 out inside the table */
            {HS_MB_OP_GATHER, 1, bi.gpu, bt.gpu, bo.gpu, 0, 0, 0},                       /* 18 no table length */
            {HS_MB_OP_GATHER, 1, bi.gpu, bt.gpu, bo.gpu, 0, TABLE_N + 1, 0},             /* 19 table past its BO */
        };
        int n = sizeof bad_args / sizeof bad_args[0], accepted = 0;
        uint32_t seq0 = mb.seq;
        for (int i = 0; i < n; i++) {
            int64_t r = hs_mailbox_send(&mb, bad_args[i][0], &bad_args[i][1]);
            if (r != -EINVAL) accepted |= 1 << i;
        }
        bad = gather_checked(&mb, 1024, 0, 0, &rc);
        report("raw send refuses bad arguments", batches, !accepted && mb.seq == seq0 + 1 && !rc && !bad,
               "\"tried\":%d,\"accepted_mask\":%d,\"sent\":%u,\"rc_after\":%d,\"wrong\":%d", n, accepted, mb.seq - seq0, rc, bad);
        /* a table and an output side by side in one BO are fine */
        uint32_t* T = (uint32_t*)bt.map;
        uint32_t* ix = (uint32_t*)bi.map;
        uint32_t* out = (uint32_t*)bo.map;
        uint32_t m = 16 * batches * 4;
        for (uint32_t i = 0; i < 1024; i++) out[i] = T[i];
        for (uint32_t i = 0; i < m; i++) ix[i] = rnd() % 1024, out[1024 + i] = 0xDEADBEEF;
        rc = hs_mailbox_gather(&mb, bi.gpu, bo.gpu, 1024, bo.gpu + 4096, m, 1);
        bad = 0;
        for (uint32_t i = 0; i < m; i++) bad += out[1024 + i] != T[ix[i]];
        report("table and output adjacent in one BO", batches, !rc && !bad, "\"rc\":%d,\"wrong\":%d", rc, bad);
    }
    /* wait for a sequence that was never sent */
    {
        uint64_t t = now_ns();
        rc = hs_mailbox_wait(&mb, mb.seq + 5, 50000000ull);
        report("wait on unsent sequence", batches, rc == -EINVAL, "\"rc\":%d,\"ms\":%.2f", rc, (now_ns() - t) / 1e6);
    }
    /* addresses that are word-aligned but not 64- or 256-byte aligned */
    {
        static const uint32_t offs[][2] = {{4, 60}, {252, 4}, {68, 196}, {1020, 508}};
        bad = 0, rc = 0;
        int rc2 = 0;
        for (int i = 0; i < 4 && !rc && !rc2; i++)
            for (int rep = 0; rep < 10; rep++) {
                bad += gather_checked(&mb, 1024, offs[i][0], offs[i][1], &rc);
                bad += chain_checked(&mb, 1024, 16, offs[i][0], offs[i][1], &rc2);
            }
        report("odd word offsets", batches, !bad && !rc && !rc2, "\"wrong\":%d,\"rc\":%d,\"rc_chain\":%d", bad, rc, rc2);
    }
    /* zero-length requests */
    {
        rc = hs_mailbox_gather(&mb, bi.gpu, bt.gpu, TABLE_N, bo.gpu, 0, 0);
        int rc2 = hs_mailbox_chain(&mb, bx.gpu, by.gpu, 0, 4, 1.f, 0.f);
        report("zero elements", batches, !rc && !rc2, "\"rc\":%d,\"rc_chain\":%d", rc, rc2);
        /* the chain runs its steps in stretches of HS_MB_CHAIN_STRETCH_DEFAULT * 4
         * with a thread switch between: step counts around the boundaries */
        static const uint32_t ks[] = {4, 8, 508, 512, 516, 1020, 1024, 1028, 2044, 4000, 4096, 4100};
        int badk = 0, rck = 0;
        for (unsigned i = 0; i < sizeof ks / sizeof ks[0] && !rck; i++) badk += chain_checked(&mb, per * 3, ks[i], 0, 0, &rck);
        report("chain step counts around the stretch boundaries", batches, !badk && !rck, "\"wrong\":%d,\"rc\":%d", badk, rck);
    }
    /* a table the mailbox itself wrote, read with table_changes = 0 and 1 */
    for (int tc = 0; tc <= 1; tc++) {
        uint32_t n = 4096, *T = (uint32_t*)bt.map, *mid = (uint32_t*)bo.map, *fin = (uint32_t*)by.map;
        uint32_t* ix = (uint32_t*)bi.map;
        static uint32_t ix1[4096];
        bad = 0, rc = 0;
        for (int rep = 0; rep < 30 && !rc; rep++) {
            for (uint32_t i = 0; i < n; i++) ix[i] = ix1[i] = rnd() % TABLE_N;
            rc = hs_mailbox_gather(&mb, bi.gpu, bt.gpu, TABLE_N, bo.gpu, n, 0); /* mid = T[ix1] */
            for (uint32_t i = 0; i < n; i++) ix[i] = (i * 7 + (uint32_t)rep) % n;
            if (!rc) rc = hs_mailbox_gather(&mb, bi.gpu, bo.gpu, n, by.gpu, n, tc); /* fin = mid[ix] */
            for (uint32_t i = 0; i < n; i++) bad += fin[i] != T[ix1[ix[i]]] || fin[i] != mid[ix[i]];
        }
        report(tc ? "table written by an earlier request, table_changes=1"
                  : "table written by an earlier request, table_changes=0",
               batches, !bad && !rc, "\"wrong\":%d,\"rc\":%d", bad, rc);
    }
    /* in place (out == in) must either be right or be refused */
    {
        uint32_t n = 1024;
        float* x = (float*)bx.map;
        static float ref[1024];
        for (uint32_t i = 0; i < n; i++) {
            x[i] = (float)(rnd() % 1000) / 1e3f;
            ref[i] = chain_ref(x[i], 16, 0.999f, 0.001f);
        }
        rc = hs_mailbox_chain(&mb, bx.gpu, bx.gpu, n, 16, 0.999f, 0.001f);
        bad = 0;
        for (uint32_t i = 0; i < n; i++) bad += memcmp(&ref[i], &x[i], 4) != 0;
        report("chain in place is refused (a killed lease would run it twice)", batches, rc == -EINVAL, "\"rc\":%d", rc);
    }
    /* send without waiting until the ring refuses: nothing sent may be lost */
    {
        uint32_t n = 16 * batches, *T = (uint32_t*)bt.map, *ix = (uint32_t*)bi.map, *out = (uint32_t*)bo.map;
        uint32_t max = MAXE / n;
        for (uint32_t i = 0; i < MAXE; i++) ix[i] = rnd() % TABLE_N, out[i] = 0xDEADBEEF;
        uint32_t sent = 0, refused = 0;
        int64_t last = 0, r = 0;
        for (int round = 0; round < 3; round++) {
            while (sent < max) {
                uint32_t a[7] = {1, bi.gpu + 4 * n * sent, bt.gpu, bo.gpu + 4 * n * sent, 0, TABLE_N, 0};
                r = hs_mailbox_send(&mb, HS_MB_OP_GATHER, a);
                if (r < 0) break;
                last = r, sent++;
            }
            if (r == -EBUSY) refused++;
            rc = hs_mailbox_wait(&mb, (uint32_t)last, 1000000000ull);
            if (rc || (r < 0 && r != -EBUSY)) break;
        }
        bad = 0;
        for (uint32_t i = 0; i < sent * n; i++) bad += out[i] != T[ix[i]];
        int rc2;
        bad += gather_checked(&mb, 1024, 0, 0, &rc2);
        report("pipelined sends until the ring refuses", batches, sent >= mb.slots && (r >= 0 || r == -EBUSY) && !rc && !rc2 && !bad,
               "\"sent\":%u,\"slots\":%u,\"refusals\":%u,\"last\":%lld,\"rc\":%d,\"wrong\":%d", sent, mb.slots, refused,
               (long long)r, rc, bad);
    }
    /* stop with requests still in flight: every one must have run */
    {
        uint32_t n = 1024, *T = (uint32_t*)bt.map, *ix = (uint32_t*)bi.map, *out = (uint32_t*)bo.map;
        for (uint32_t i = 0; i < 16 * n; i++) ix[i] = rnd() % TABLE_N, out[i] = 0xDEADBEEF;
        int64_t r = 1;
        for (uint32_t j = 0; j < 16 && r >= 0; j++) {
            uint32_t a[7] = {n / (16 * batches), bi.gpu + 4 * n * j, bt.gpu, bo.gpu + 4 * n * j, 0, TABLE_N, 0};
            r = hs_mailbox_send(&mb, HS_MB_OP_GATHER, a);
        }
        rc = hs_mailbox_stop(&mb);
        bad = 0;
        for (uint32_t i = 0; i < 16 * n; i++) bad += out[i] != T[ix[i]];
        int rc2;
        bad += gather_checked(&mb, 1024, 0, 0, &rc2);
        report("stop with 16 requests in flight", batches, r > 0 && !rc && !rc2 && !bad,
               "\"send\":%lld,\"rc_stop\":%d,\"rc_after\":%d,\"wrong\":%d", (long long)r, rc, rc2, bad);
    }
    /* 200 stop/restart cycles */
    {
        bad = 0, rc = 0;
        int rc2 = 0;
        for (int i = 0; i < 200 && !rc && !rc2; i++) {
            bad += gather_checked(&mb, per * 4, 0, 0, &rc);
            rc2 = hs_mailbox_stop(&mb);
        }
        report("200 stop/restart cycles", batches, !bad && !rc && !rc2, "\"wrong\":%d,\"rc\":%d,\"rc_stop\":%d", bad, rc, rc2);
    }
    /* sequence numbers across the 32-bit wrap */
    {
        hs_mailbox_stop(&mb);
        rc = hs_mailbox_test_seek(&mb, 0xFFFFFF00u);
        bad = 0;
        int n = 0, rc2 = 0;
        for (; n < 600 && !rc && !rc2; n++) {
            bad += gather_checked(&mb, per * 2, 0, 0, &rc);
            if (n % 97 == 0) rc2 = hs_mailbox_stop(&mb);
        }
        report("600 requests across the sequence wrap", batches, !bad && !rc && !rc2 && n == 600 && mb.seq < 1000,
               "\"done\":%d,\"wrong\":%d,\"rc\":%d,\"rc_stop\":%d,\"seq\":%u", n, bad, rc, rc2, mb.seq);
    }
    hs_mailbox_free(&mb);

    /* the aftermath of leases that died without storing their state: the
     * library believes `qdepth` leases are still queued */
    start(&mb, batches, 0, 0);
    {
        rc = hs_mailbox_echo(&mb);
        hs_v3d_bo_wait(&dev, &mb.ctl, 2000000000ull); /* leases drained */
        mb.submitted += mb.qdepth;
        uint64_t t = now_ns();
        int rc2;
        bad = gather_checked(&mb, 1024, 0, 0, &rc2);
        double ms = (now_ns() - t) / 1e6;
        int rc3;
        bad += gather_checked(&mb, 1024, 0, 0, &rc3);
        report("lost leases are replaced", batches, !rc && !rc2 && !rc3 && !bad, "\"rc\":%d,\"rc_after\":%d,\"rc_next\":%d,\"wrong\":%d,\"first_ms\":%.2f,\"lost_leases_counted\":%llu", rc,
               rc2, rc3, bad, ms, (unsigned long long)mb.lost_leases);
    }
    /* a killed job that batch 0 finished and the others did not: batch 0's
     * lease count runs ahead of the jobs that finished everywhere */
    {
        hs_v3d_bo_wait(&dev, &mb.ctl, 2000000000ull);
        hs_mailbox_test_count_lease(&mb, 0);
        int rc2, rc3;
        bad = gather_checked(&mb, 1024, 0, 0, &rc2);
        bad += gather_checked(&mb, 1024, 0, 0, &rc3);
        hs_mailbox_stop(&mb);
        int rc4;
        bad += gather_checked(&mb, 1024, 0, 0, &rc4);
        report("lease counts that differ between batches", batches, !rc2 && !rc3 && !rc4 && !bad,
               "\"rc\":%d,\"rc_next\":%d,\"rc_after_stop\":%d,\"wrong\":%d", rc2, rc3, rc4, bad);
    }
    /* a stop whose wait times out, then is called again: it must finish, and
     * nothing may be sent in between */
    {
        int timed_out = 0, refused = 0, rc2 = 0, rc3 = 0;
        bad = 0;
        for (int i = 0; i < 50 && !rc2 && !rc3; i++) {
            bad += gather_checked(&mb, 16 * batches, 0, 0, &rc3);
            mb.test_stop_fail_once = 1;
            int e = hs_mailbox_stop(&mb);
            if (e == -ETIME) {
                timed_out++;
                refused += hs_mailbox_send(&mb, HS_MB_OP_ECHO, NULL) == -EBUSY;
                rc2 = hs_mailbox_stop(&mb);
            } else if (e)
                rc2 = e;
        }
        int rc4;
        bad += gather_checked(&mb, 1024, 0, 0, &rc4);
        report("stop that times out, then is finished by a second call", batches,
               timed_out > 0 && refused == timed_out && !rc2 && !rc3 && !rc4 && !bad,
               "\"stops_timed_out\":%d,\"sends_refused_meanwhile\":%d,\"rc_second_stop\":%d,\"rc\":%d,\"rc_after\":%d,\"wrong\":%d",
               timed_out, refused, rc2, rc3, rc4, bad);
    }
    hs_mailbox_free(&mb);

    /* The GPU's read cache is shared by the batches of a job
     * (tools/p0_5_batch_cache_probe). Batch 0 starts a lease one request
     * behind the others, as after a lease it ended early; with 2 slots the
     * others then meet, in the same job, a slot index batch 0 already read
     * for an older request. They must not run the cached older request. */
    if (batches > 1) {
        start(&mb, batches, 0, 2);
        int rc2 = 0, rc3 = 0, bad1 = 0, bad2 = 0, rounds = 0;
        rc = hs_mailbox_echo(&mb);
        for (; rounds < 50 && !rc && !rc2 && !rc3; rounds++) {
            rc = hs_mailbox_stop(&mb);
            uint32_t q = 1000 + 3 * (uint32_t)rounds; /* both slot parities */
            if (!rc) rc = hs_mailbox_test_seek(&mb, q);
            hs_mailbox_test_patch_request(&mb, q, 0, HS_MB_OP_ECHO); /* request q, which only batch 0 still owes */
            hs_mailbox_test_set_ack(&mb, 0, q - 1);
            bad1 += gather_checked(&mb, 16 * batches * 4, 0, 0, &rc2); /* q + 1 */
            bad2 += gather_checked(&mb, 16 * batches * 4, 0, 0, &rc3); /* q + 2: the slot index of q */
        }
        report("a batch one request behind the others, 2 slots", batches, !rc && !rc2 && !rc3 && !bad1 && !bad2,
               "\"rounds\":%d,\"rc\":%d,\"rc_first\":%d,\"rc_second\":%d,\"wrong_first\":%d,\"wrong_second\":%d", rounds, rc,
               rc2, rc3, bad1, bad2);
        hs_mailbox_free(&mb);
    }

    /* small rings: every slot is reused constantly */
    for (uint32_t slots = 2; slots <= 4; slots += 2) {
        start(&mb, batches, 0, slots);
        bad = 0, rc = 0;
        int rc2 = 0, n = 0;
        for (; n < 3000 && !rc && !rc2; n++) {
            if (n & 1)
                bad += gather_checked(&mb, per * (1 + rnd() % 8), 0, 0, &rc);
            else
                bad += chain_checked(&mb, per * (1 + rnd() % 8), 4 * (1 + rnd() % 8), 0, 0, &rc2);
        }
        char nm[48];
        snprintf(nm, sizeof nm, "3000 mixed requests, %u slots", slots);
        report(nm, batches, !bad && !rc && !rc2, "\"done\":%d,\"wrong\":%d,\"rc\":%d,\"rc_chain\":%d,\"leases\":%u", n, bad, rc, rc2,
               hs_mailbox_leases_done(&mb, 0));
        hs_mailbox_free(&mb);
    }
    /* small budget: a lease per few requests */
    {
        start(&mb, batches, 4, 0);
        bad = 0, rc = 0;
        int rc2 = 0, n = 0;
        for (; n < 300 && !rc && !rc2; n++) {
            bad += gather_checked(&mb, per * 4, 0, 0, &rc);
            bad += chain_checked(&mb, per * 4, 64, 0, 0, &rc2);
        }
        report("budget 4", batches, !bad && !rc && !rc2, "\"done\":%d,\"wrong\":%d,\"rc\":%d,\"rc_chain\":%d,\"leases\":%u", n, bad, rc, rc2,
               hs_mailbox_leases_done(&mb, 0));
        hs_mailbox_free(&mb);
    }
    /* the largest requests the library accepts, back to back */
    {
        start(&mb, batches, 0, 0);
        bad = 0, rc = 0;
        int rc2 = 0;
        uint32_t gmax = 16 * batches * HS_MB_MAX_GATHER_CHUNKS;
        double worst = 0, worstc = 0;
        for (int i = 0; i < 20 && !rc && !rc2; i++) {
            uint32_t* T = (uint32_t*)bt.map;
            uint32_t* ix = (uint32_t*)bi.map;
            uint32_t* out = (uint32_t*)bo.map;
            for (uint32_t j = 0; j < gmax; j++) ix[j] = rnd() % TABLE_N, out[j] = 0xDEADBEEF;
            hs_mailbox_echo(&mb); /* the CPU-side check before this left the GPU idle: wake it (~10 ms) first */
            uint64_t t = now_ns();
            rc = hs_mailbox_gather(&mb, bi.gpu, bt.gpu, TABLE_N, bo.gpu, gmax, 1);
            double ms = (now_ns() - t) / 1e6;
            if (ms > worst) worst = ms;
            for (uint32_t j = 0; j < gmax; j++) bad += out[j] != T[ix[j]];
            hs_mailbox_echo(&mb);
            bad += chain_checked(&mb, per * 3, 65536, 0, 0, &rc2);
            if (last_call_ms > worstc) worstc = last_call_ms;
        }
        report("largest requests x20", batches, !bad && !rc && !rc2, "\"gather_elements\":%u,\"wrong\":%d,\"rc\":%d,\"rc_chain\":%d,\"worst_gather_ms\":%.2f,\"worst_chain_ms\":%.2f",
               gmax, bad, rc, rc2, worst, worstc);
        hs_mailbox_free(&mb);
    }
    /* two mailboxes used alternately from one thread */
    {
        hs_mailbox m2;
        start(&mb, batches, 0, 0);
        start(&m2, batches, 0, 0);
        bad = 0, rc = 0;
        int rc2 = 0;
        double lat[60];
        int n = 0;
        for (; n < 30 && !rc && !rc2; n++) {
            uint64_t t = now_ns();
            bad += gather_checked(&mb, 1024, 0, 0, &rc);
            lat[2 * n] = (now_ns() - t) / 1e6;
            t = now_ns();
            bad += gather_checked(&m2, 1024, 0, 0, &rc2);
            lat[2 * n + 1] = (now_ns() - t) / 1e6;
        }
        qsort(lat, 2 * n, sizeof(double), cmpd);
        report("two mailboxes alternating", batches, !bad && !rc && !rc2, "\"wrong\":%d,\"rc\":%d,\"rc2\":%d,\"median_ms\":%.2f,\"max_ms\":%.2f", bad, rc,
               rc2, lat[n], lat[2 * n - 1]);
        hs_mailbox_free(&m2);
        hs_mailbox_free(&mb);
    }
    /* use after stop + free + start again on the same struct; double free */
    {
        start(&mb, batches, 0, 0);
        bad = gather_checked(&mb, 1024, 0, 0, &rc);
        hs_mailbox_free(&mb);
        hs_mailbox_free(&mb);
        start(&mb, batches, 0, 0);
        int rc2;
        bad += gather_checked(&mb, 1024, 0, 0, &rc2);
        hs_mailbox_free(&mb);
        hs_mailbox_cfg c[] = {{3, 0, 0, 0, bos, 5, 0, 0, 0, 0},       {1, 0, 0, 0, bos, 14, 0, 0, 0, 0},
                              {1, 1, 0, 0, bos, 5, 0, 0, 0, 0},       {1, 0, 0, 1, bos, 5, 0, 0, 0, 0},
                              {1, 0, 9, 0, bos, 5, 0, 0, 0, 0},       {1, 0, 0, 0, NULL, 5, 0, 0, 0, 0},
                              {1, 0, 0, 3, bos, 5, 0, 0, 0, 0},       {1, 250001, 0, 0, bos, 5, 0, 0, 0, 0},
                              {1, 4000000, 0, 0, bos, 5, 0, 0, 0, 0}, {1, 0, 0, 131072, bos, 5, 0, 0, 0, 0},
                              {1, 0, 0, 0, bos, 5, 0, 0, 0, 3},       {1, 0, 0, 0, bos, 5, 0, 0, 0, 65536}};
        int nc = sizeof c / sizeof c[0], okc = 0;
        for (int i = 0; i < nc; i++) okc += hs_mailbox_start(&mb, &dev, &c[i]) == -EINVAL;
        okc += hs_mailbox_start(&mb, &dev, NULL) == -EINVAL;
        hs_mailbox_free(&mb);
        report("restart on the same struct; bad configs refused", batches, !bad && !rc && !rc2 && okc == nc + 1,
               "\"wrong\":%d,\"rc\":%d,\"rc2\":%d,\"configs_refused\":%d,\"configs_tried\":%d", bad, rc, rc2, okc, nc + 1);
    }
    /* the watchdog: leases end soon after the caller goes quiet, and no lease
     * outlives max_lease_us while the caller keeps it busy */
    for (int off = 0; off <= 1; off++) {
        hs_mailbox_cfg cfg = {.batches = batches, .budget = 250000, .slots = 4096, .bos = bos, .nbos = 5, .max_lease_us = 20000, .watchdog_off = off};
        int e = hs_mailbox_start(&mb, &dev, &cfg);
        if (e) die("hs_mailbox_start", e);
        double drain[10], dmax = 0;
        rc = 0;
        for (int i = 0; i < 10 && !rc; i++) {
            rc = hs_mailbox_echo(&mb);
            uint64_t t = now_ns();
            while (hs_v3d_bo_wait(&dev, &mb.ctl, 0) && now_ns() - t < 3000000000ull) {
            }
            drain[i] = (now_ns() - t) / 1e6;
            if (drain[i] > dmax) dmax = drain[i];
            struct timespec ts = {0, 20000000};
            nanosleep(&ts, NULL);
        }
        qsort(drain, 10, sizeof(double), cmpd);
        /* 2 s of echoes every 100 us: the longest time between lease ends */
        uint32_t lc = hs_mailbox_leases_done(&mb, batches - 1), nle = 0;
        uint64_t t_end = now_ns() + 2000000000ull, t_last = 0, gap = 0;
        bad = 0;
        while (now_ns() < t_end && !rc) {
            rc = hs_mailbox_echo(&mb);
            uint64_t t = now_ns();
            uint32_t l = hs_mailbox_leases_done(&mb, batches - 1);
            if (l != lc) {
                if (t_last && t - t_last > gap) gap = t - t_last;
                t_last = t, lc = l, nle++;
            }
            while (now_ns() - t < 100000) {
            }
        }
        int ok = off ? 1 : (dmax < 10.0 && gap / 1e6 < 25.0);
        report(off ? "watchdog off (reference): drain after last call, longest lease while busy"
                   : "watchdog: drain after last call < 10 ms, longest lease while busy < 25 ms",
               batches, ok && !rc,
               "\"rc\":%d,\"drain_median_ms\":%.2f,\"drain_max_ms\":%.2f,\"longest_lease_ms\":%.2f,\"leases\":%u,"
               "\"idle_yields\":%llu,\"age_yields\":%llu",
               rc, drain[5], dmax, gap / 1e6, nle, (unsigned long long)atomic_load(&mb.wd_idle_yields),
               (unsigned long long)atomic_load(&mb.wd_age_yields));
        hs_mailbox_free(&mb);
    }
}

/* how long one idle lease runs */
static void lease(uint32_t batches, uint32_t budget, int n) {
    double* ms = malloc(sizeof(double) * (size_t)n);
    hs_mailbox mb;
    wd_off = 1; /* the budget alone */
    start(&mb, batches, budget, 0);
    for (int i = 0; i < n; i++) {
        hs_v3d_bo_wait(&dev, &mb.ctl, 5000000000ull);
        uint32_t l0 = hs_mailbox_leases_done(&mb, batches - 1);
        int rc = hs_mailbox_echo(&mb);
        if (rc) die("echo", rc);
        uint64_t t1 = 0, t2 = 0, lim = now_ns() + 10000000000ull;
        while (!t2 && now_ns() < lim) { /* the first lease served the echo; time the second, which only polls */
            uint32_t l = hs_mailbox_leases_done(&mb, batches - 1) - l0;
            if (l >= 1 && !t1) t1 = now_ns();
            if (l >= 2) t2 = now_ns();
        }
        if (!t2) die("lease did not end", ETIME);
        ms[i] = (t2 - t1) / 1e6;
    }
    hs_mailbox_free(&mb);
    qsort(ms, n, sizeof(double), cmpd);
    printf("{\"lease\":true,\"batches\":%u,\"budget\":%u,\"n\":%d,\"min_ms\":%.2f,\"median_ms\":%.2f,\"max_ms\":%.2f}\n", batches, budget, n,
           ms[0], ms[n / 2], ms[n - 1]);
}

/* Random validated requests with random pauses (the idle stop and restart
 * paths) and occasional stops. */
static void soak(uint32_t batches, double seconds) {
    hs_mailbox mb;
    wd_off = !!getenv("RT_WD_OFF");
    start(&mb, batches, 0, 0);
    uint32_t per = 32 * batches, nlat = 0, cap = 4000000;
    double* lat = malloc(sizeof(double) * cap);
    uint64_t t_end = now_ns() + (uint64_t)(seconds * 1e9), reqs = 0;
    int bad = 0, rc = 0, stops = 0, pauses = 0;
    double worst_g = 0, worst_c = 0;
    int prev = 0, slow_after[5] = {0}, nopause = !!getenv("SOAK_NOPAUSE"), nostop = !!getenv("SOAK_NOSTOP");
    while (now_ns() < t_end && !rc) {
        uint32_t r = rnd() % 1000;
        if (nopause && r >= 960 && r < 995) r = 0;
        if (nostop && r >= 995) r = 0;
        int kind = r < 600 ? 0 : r < 780 ? 1 : r < 960 ? 2 : r < 995 ? 3 : 4;
        if (r < 600) {
            uint64_t t = now_ns();
            rc = hs_mailbox_echo(&mb);
            if (nlat < cap) lat[nlat++] = (now_ns() - t) / 1e3;
            if (now_ns() - t > 1000000) slow_after[prev]++;
        } else if (r < 780) {
            uint32_t n = r < 610 ? 16 * batches * HS_MB_MAX_GATHER_CHUNKS : per * (1 + rnd() % 64);
            bad += gather_checked(&mb, n, 4 * (rnd() % 64), 4 * (rnd() % 64), &rc);
            if (last_call_ms > worst_g) worst_g = last_call_ms;
        } else if (r < 960) {
            uint32_t k = r < 790 ? 65536 : 4 * (1 + rnd() % 256), n = r < 790 ? per * 3 : per * (1 + rnd() % 16);
            bad += chain_checked(&mb, n, k, 4 * (rnd() % 64), 4 * (rnd() % 64), &rc);
            if (last_call_ms > worst_c) worst_c = last_call_ms;
        } else if (r < 995) {
            struct timespec ts = {0, (long)(rnd() % 6000000)};
            nanosleep(&ts, NULL);
            pauses++;
        } else {
            rc = hs_mailbox_stop(&mb);
            stops++;
        }
        reqs++;
        prev = kind;
    }
    qsort(lat, nlat, sizeof(double), cmpd);
    int ok = !bad && !rc && !mb.lost_leases;
    printf("{\"soak\":true,\"batches\":%u,\"seconds\":%.0f,\"ok\":%s,\"steps\":%llu,\"wrong\":%d,\"rc\":%d,\"stops\":%d,\"pauses\":%d,"
           "\"echo_us\":{\"p50\":%.2f,\"p99\":%.2f,\"p999\":%.2f,\"max\":%.1f},\"worst_gather_ms\":%.2f,\"worst_chain_ms\":%.2f,"
           "\"leases\":%u,\"stall_yields\":%llu,\"lost_leases\":%llu,\"idle_yields\":%llu,"
           "\"age_yields\":%llu,\"echo_over_1ms_after\":{\"echo\":%d,\"gather\":%d,\"chain\":%d,\"pause\":%d,\"stop\":%d}}\n",
           batches, seconds, ok ? "true" : "false", (unsigned long long)reqs, bad, rc, stops, pauses, nlat ? lat[nlat / 2] : 0,
           nlat ? lat[(size_t)(nlat * 0.99)] : 0, nlat ? lat[(size_t)(nlat * 0.999)] : 0, nlat ? lat[nlat - 1] : 0, worst_g, worst_c,
           hs_mailbox_leases_done(&mb, batches - 1), (unsigned long long)mb.stall_yields,
           (unsigned long long)mb.lost_leases, (unsigned long long)atomic_load(&mb.wd_idle_yields),
           (unsigned long long)atomic_load(&mb.wd_age_yields), slow_after[0], slow_after[1], slow_after[2], slow_after[3],
           slow_after[4]);
    if (!ok) failures++;
    hs_mailbox_free(&mb);
}

/* A steady caller: one request every `gap_us`. kind 0: echo; 1: chain of
 * 32*batches*8 elements x 256 steps, checked. Reports latency and the longest
 * time any batch went without ending a lease (the lease bound in practice). */
static void busy(uint32_t batches, double seconds, uint32_t gap_us, int kind) {
    hs_mailbox mb;
    wd_off = !!getenv("RT_WD_OFF");
    hs_mailbox_cfg cfg = {.batches = batches, .bos = bos, .nbos = 5, .watchdog_off = wd_off};
    if (getenv("RT_BUDGET")) cfg.budget = (uint32_t)atoi(getenv("RT_BUDGET"));
    int e = hs_mailbox_start(&mb, &dev, &cfg);
    if (e) die("hs_mailbox_start", e);
    uint32_t cap = 2000000, nlat = 0, lc[8], n = 32 * batches * 8;
    uint64_t tl[8], gap = 0, reqs = 0, t_end = now_ns() + (uint64_t)(seconds * 1e9);
    double* lat = malloc(sizeof(double) * cap);
    int rc = 0, bad = 0, timeouts = 0;
    float* x = (float*)bx.map;
    float* y = (float*)by.map;
    static float ref[32 * 8 * 8];
    for (uint32_t i = 0; i < n; i++) x[i] = (float)(rnd() % 1000000) / 1e6f, ref[i] = chain_ref(x[i], 256, 0.9995f, 0.0007f);
    for (uint32_t b = 0; b < batches; b++) lc[b] = hs_mailbox_leases_done(&mb, b), tl[b] = 0;
    while (now_ns() < t_end) {
        uint64_t t = now_ns();
        if (kind) {
            for (uint32_t i = 0; i < n; i += 37) y[i] = -1.f;
            rc = hs_mailbox_chain(&mb, bx.gpu, by.gpu, n, 256, 0.9995f, 0.0007f);
        } else
            rc = hs_mailbox_echo(&mb);
        uint64_t t1 = now_ns();
        if (rc == -ETIME) { /* count it and wait it out */
            timeouts++;
            rc = hs_mailbox_wait(&mb, mb.seq, 10000000000ull);
            t1 = now_ns();
        }
        if (rc) break;
        if (kind)
            for (uint32_t i = 0; i < n; i += 37) bad += memcmp(&ref[i], &y[i], 4) != 0;
        if (nlat < cap) lat[nlat++] = (t1 - t) / 1e3;
        reqs++;
        for (uint32_t b = 0; b < batches; b++) {
            uint32_t l = hs_mailbox_leases_done(&mb, b);
            if (l != lc[b]) {
                if (tl[b] && t1 - tl[b] > gap) gap = t1 - tl[b];
                tl[b] = t1, lc[b] = l;
            }
        }
        while (now_ns() - t1 < (uint64_t)gap_us * 1000) {
        }
    }
    qsort(lat, nlat, sizeof(double), cmpd);
    int ok = !rc && !bad && !mb.lost_leases;
    printf("{\"busy\":\"%s\",\"batches\":%u,\"watchdog\":%s,\"budget\":%u,\"ok\":%s,\"requests\":%llu,\"rc\":%d,\"wrong\":%d,"
           "\"timeouts_1s\":%d,\"us\":{\"p50\":%.2f,\"p99\":%.2f,\"p999\":%.2f,\"max\":%.1f},\"longest_lease_ms\":%.2f,\"leases\":%u,"
           "\"stall_yields\":%llu,\"lost_leases\":%llu,\"age_yields\":%llu}\n",
           kind ? "chain 256 steps" : "echo", batches, wd_off ? "false" : "true", mb.budget, ok ? "true" : "false",
           (unsigned long long)reqs, rc, bad, timeouts, nlat ? lat[nlat / 2] : 0, nlat ? lat[(size_t)(nlat * 0.99)] : 0,
           nlat ? lat[(size_t)(nlat * 0.999)] : 0, nlat ? lat[nlat - 1] : 0, gap / 1e6, hs_mailbox_leases_done(&mb, 0),
           (unsigned long long)mb.stall_yields, (unsigned long long)mb.lost_leases,
           (unsigned long long)atomic_load(&mb.wd_age_yields));
    if (!ok) failures++;
    hs_mailbox_free(&mb);
}

/* The shipped gather kernel as plain jobs, every output word checked: does
 * an 8-batch job cover every batch's share when the GPU is shared? */
static void jobcheck(uint32_t batches, double seconds) {
    hs_v3d_bo code, unif;
    static uint64_t buf[4096];
    if (hs_v3d_bo_new(&dev, &code, 64 << 10) || hs_v3d_bo_new(&dev, &unif, 4096)) die("bo", EIO);
    hs_qpu_prog p;
    hs_qpu_prog_init(&p, buf, 4096);
    int len = hs_qpu_kernel_gather(&p, (int)batches, 8);
    if (len < 0) die("gather build", EIO);
    uint32_t g = hs_v3d_load(&code, 0, buf, (uint32_t)len), n = 8192;
    uint32_t jh[5] = {code.handle, unif.handle, bt.handle, bi.handle, bo.handle};
    uint32_t* T = (uint32_t*)bt.map;
    uint32_t* ix = (uint32_t*)bi.map;
    uint32_t* out = (uint32_t*)bo.map;
    if (hs_qpu_gather_uniforms((uint32_t*)unif.map, n, (int)batches, 8, bi.gpu, bt.gpu, bo.gpu)) die("uniforms", EINVAL);
    uint64_t t_end = now_ns() + (uint64_t)(seconds * 1e9), jobs = 0, bad_jobs = 0, bad_words = 0;
    uint32_t first_bad[8], nfb = 0;
    while (now_ns() < t_end) {
        for (uint32_t i = 0; i < n; i++) ix[i] = rnd() % TABLE_N, out[i] = 0xDEADBEEF;
        hs_v3d_dispatch_desc d = {g, unif.gpu, batches, jh, 5, 0};
        int e = hs_v3d_dispatch(&dev, &d);
        if (!e) e = hs_v3d_bo_wait(&dev, &bo, 5000000000ull);
        if (e) die("job", e);
        uint32_t bad = 0;
        for (uint32_t i = 0; i < n; i++)
            if (out[i] != T[ix[i]]) {
                if (!bad && nfb < 8) first_bad[nfb++] = i;
                bad++;
            }
        jobs++, bad_jobs += bad != 0, bad_words += bad;
    }
    printf("{\"jobcheck\":\"shipped gather kernel, %u elements\",\"batches\":%u,\"ok\":%s,\"jobs\":%llu,\"jobs_with_wrong_words\":%llu,"
           "\"wrong_words\":%llu,\"first_wrong_index_of_first_bad_jobs\":[",
           n, batches, bad_jobs ? "false" : "true", (unsigned long long)jobs, (unsigned long long)bad_jobs,
           (unsigned long long)bad_words);
    for (uint32_t i = 0; i < nfb; i++) printf("%s%u", i ? "," : "", first_bad[i]);
    printf("]}\n");
    if (bad_jobs) failures++;
}

/* Requests kept in flight without a gap: 4 of the largest gathers at a time,
 * each to its own region, every output checked. No poll ever finds the ring
 * empty for long, so only the budget, the ring wrap and the watchdog's age
 * yield can end a lease. Reports the longest lease. */
static void pipe_mode(uint32_t batches, double seconds, uint32_t chunks) {
    hs_mailbox mb;
    wd_off = !!getenv("RT_WD_OFF");
    hs_mailbox_cfg cfg = {.batches = batches, .budget = 100000, .slots = 4096, .bos = bos, .nbos = 5, .max_lease_us = 20000,
                          .watchdog_off = wd_off};
    int e = hs_mailbox_start(&mb, &dev, &cfg);
    if (e) die("hs_mailbox_start", e);
    uint32_t n = 16 * batches * chunks, regions = MAXE / n < 4 ? MAXE / n : 4;
    uint32_t* T = (uint32_t*)bt.map;
    uint32_t* ix = (uint32_t*)bi.map;
    uint32_t* out = (uint32_t*)bo.map;
    for (uint32_t i = 0; i < regions * n; i++) ix[i] = rnd() % TABLE_N, out[i] = 0xDEADBEEF;
    int64_t seq[4];
    uint64_t reqs = 0, t_end = now_ns() + (uint64_t)(seconds * 1e9), tl = 0, gap = 0;
    uint32_t lc = hs_mailbox_leases_done(&mb, batches - 1);
    int rc = 0, bad = 0;
    for (uint32_t r = 0; r < regions && !rc; r++) {
        uint32_t a[7] = {chunks, bi.gpu + 4 * n * r, bt.gpu, bo.gpu + 4 * n * r, 0, TABLE_N, 0};
        seq[r] = hs_mailbox_send(&mb, HS_MB_OP_GATHER, a);
        if (seq[r] < 0) rc = (int)seq[r];
    }
    for (uint32_t r = 0; !rc && now_ns() < t_end; r = (r + 1) % regions) {
        rc = hs_mailbox_wait(&mb, (uint32_t)seq[r], 5000000000ull);
        if (rc) break;
        uint32_t step = n > 61 * 16 ? 61 : 1;
        for (uint32_t i = 0; i < n; i += step) bad += out[n * r + i] != T[ix[n * r + i]];
        for (uint32_t i = 0; i < n; i += step) out[n * r + i] = 0xDEADBEEF;
        uint32_t a[7] = {chunks, bi.gpu + 4 * n * r, bt.gpu, bo.gpu + 4 * n * r, 0, TABLE_N, 0};
        seq[r] = hs_mailbox_send(&mb, HS_MB_OP_GATHER, a);
        if (seq[r] < 0) rc = (int)seq[r];
        reqs++;
        uint64_t t = now_ns();
        uint32_t l = hs_mailbox_leases_done(&mb, batches - 1);
        if (l != lc) {
            if (tl && t - tl > gap) gap = t - tl;
            tl = t, lc = l;
        }
    }
    if (!rc) rc = hs_mailbox_stop(&mb);
    double limit = getenv("RT_PIPE_LIMIT_MS") ? atof(getenv("RT_PIPE_LIMIT_MS")) : 35.0; /* 20 + a tick + one request */
    /* The bound is on the job the watchdog measures. `gap`, one batch's lease
     * ends as this thread happens to see them, can span two jobs. */
    int ok = !rc && !bad && !mb.lost_leases && (wd_off || atomic_load(&mb.wd_job_gap_max_ns) / 1e6 < limit);
    printf("{\"pipe\":\"4 gathers always in flight\",\"chunks_per_batch\":%u,\"batches\":%u,\"watchdog\":%s,\"budget\":%u,"
           "\"max_lease_ms\":20,\"ok\":%s,\"requests\":%llu,\"rc\":%d,\"wrong\":%d,\"longest_gap_between_one_batchs_lease_ends_ms\":%.2f,\"leases\":%u,"
           "\"lost_leases\":%llu,\"age_yields\":%llu,\"longest_yield_to_job_end_ms\":%.2f,\"longest_job_ms\":%.2f}\n",
           chunks, batches, wd_off ? "false" : "true", mb.budget, ok ? "true" : "false", (unsigned long long)reqs, rc, bad,
           gap / 1e6, hs_mailbox_leases_done(&mb, 0), (unsigned long long)mb.lost_leases,
           (unsigned long long)atomic_load(&mb.wd_age_yields), atomic_load(&mb.wd_age_hold_max_ns) / 1e6,
           atomic_load(&mb.wd_job_gap_max_ns) / 1e6);
    if (!ok) failures++;
    hs_mailbox_free(&mb);
}

/* A REAL kernel reset of a lease. A chain request that cannot finish (sent
 * past the checks) makes the kernel reset the GPU while the lease runs it;
 * three ordinary requests are queued behind it. While it spins, its step
 * count in the slot is rewritten to 4, so whichever lease runs the request
 * again can finish it. Run alone, on purpose: it adds to the reset count. */
static void reset_mode(uint32_t batches) {
    hs_mailbox mb;
    start(&mb, batches, 0, 0);
    int rc, rc0, bad0 = gather_checked(&mb, 1024, 0, 0, &rc0);
    uint32_t n = 32 * batches, g = 1024, *T = (uint32_t*)bt.map, *ix = (uint32_t*)bi.map, *out = (uint32_t*)bo.map;
    float* x = (float*)bx.map;
    float* y = (float*)by.map;
    float c1 = 0.9995f, c0 = 0.0007f;
    uint32_t c1b, c0b;
    memcpy(&c1b, &c1, 4);
    memcpy(&c0b, &c0, 4);
    for (uint32_t i = 0; i < n; i++) x[i] = (float)(rnd() % 1000000) / 1e6f, y[i] = -1.f;
    for (uint32_t i = 0; i < 3 * g; i++) ix[i] = rnd() % TABLE_N, out[i] = 0xDEADBEEF;
    uint32_t leases0 = hs_mailbox_leases_done(&mb, 0);
    uint32_t a[7] = {1, bx.gpu, 0, by.gpu, 75000000, c1b, c0b}; /* ~4 s of steps: longer than the kernel allows */
    uint64_t t0 = now_ns();
    int64_t big = hs_mailbox_test_send_unchecked(&mb, HS_MB_OP_CHAIN, a), last = big;
    for (uint32_t j = 0; j < 3 && last >= 0; j++) {
        uint32_t ga[7] = {g / (16 * batches), bi.gpu + 4 * g * j, bt.gpu, bo.gpu + 4 * g * j, 0, TABLE_N, 0};
        last = hs_mailbox_send(&mb, HS_MB_OP_GATHER, ga);
    }
    struct timespec ts = {0, 100000000};
    nanosleep(&ts, NULL);
    uint32_t acked_at_100ms = 0;
    for (uint32_t b = 0; b < batches; b++) acked_at_100ms += hs_mailbox_wait(&mb, (uint32_t)big, 0) == 0;
    hs_mailbox_test_patch_request(&mb, (uint32_t)big, 5, 1); /* k/4 = 1: 4 steps */
    int waits = 0;
    do rc = last < 0 ? (int)last : hs_mailbox_wait(&mb, (uint32_t)last, 1000000000ull);
    while (rc == -ETIME && ++waits < 12);
    double ms = (now_ns() - t0) / 1e6;
    int bad_chain = 0, fixed_point = 0, bad_g = 0;
    for (uint32_t i = 0; i < n; i++) {
        float r = chain_ref(x[i], 4, c1, c0);
        bad_chain += memcmp(&r, &y[i], 4) != 0;
        fixed_point += y[i] > 1.39f && y[i] < 1.41f; /* where 300 million steps would end */
    }
    for (uint32_t i = 0; i < 3 * g; i++) bad_g += out[i] != T[ix[i]];
    uint64_t lost = mb.lost_leases;
    uint32_t leases = hs_mailbox_leases_done(&mb, 0) - leases0;
    /* afterwards: ordinary work, a stop and a restart */
    int rc2 = 0, rc3 = 0, bad_after = 0;
    uint64_t t1 = now_ns();
    for (int i = 0; i < 2000 && !rc2 && !rc3; i++) {
        bad_after += gather_checked(&mb, 32 * batches * 4, 0, 0, &rc2);
        bad_after += chain_checked(&mb, 32 * batches * 4, 16, 0, 0, &rc3);
    }
    double after_ms = (now_ns() - t1) / 1e6;
    int rc4 = hs_mailbox_stop(&mb), rc5;
    bad_after += gather_checked(&mb, 1024, 0, 0, &rc5);
    int ok = big > 0 && !rc && !rc0 && !bad0 && !bad_chain && !bad_g && !rc2 && !rc3 && !rc4 && !rc5 && !bad_after;
    printf("{\"reset\":\"lease killed while running a request\",\"batches\":%u,\"ok\":%s,\"acked_after_100ms\":%u,"
           "\"all_acknowledged_after_ms\":%.1f,\"waits_timed_out\":%d,\"rc\":%d,\"chain_wrong_vs_4_steps\":%d,"
           "\"chain_elements_at_fixed_point\":%d,\"queued_gathers_wrong\":%d,\"lost_leases_counted\":%llu,\"leases_ended\":%u,"
           "\"after\":{\"requests\":4000,\"ms\":%.1f,\"rc\":[%d,%d],\"rc_stop\":%d,\"rc_restart\":%d,\"wrong\":%d}}\n",
           batches, ok ? "true" : "false", acked_at_100ms, ms, waits, rc, bad_chain, fixed_point, bad_g, (unsigned long long)lost,
           leases, after_ms, rc2, rc3, rc4, rc5, bad_after);
    if (!ok) failures++;
    hs_mailbox_free(&mb);
}

/* A REAL kernel reset caused by someone else: another client (a second
 * device handle) submits a job that never ends while the mailbox is in use.
 * Checked gathers keep being sent throughout. Adds to the reset count. */
static void foreign_mode(uint32_t batches) {
    hs_v3d dev2;
    hs_v3d_bo code, unif;
    static uint64_t buf[64];
    int e = hs_v3d_open(&dev2, NULL);
    if (e || hs_v3d_bo_new(&dev2, &code, 4096) || hs_v3d_bo_new(&dev2, &unif, 4096)) die("second device", e ? e : EIO);
    hs_qpu_prog p;
    hs_qpu_prog_init(&p, buf, 64);
    hs_qpu_alu ld = hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDUNIFRF, HS_QPU_RF(0));
    hs_qpu_emit(&p, &ld);
    int l = hs_qpu_label_new(&p);
    hs_qpu_label_bind(&p, l);
    hs_qpu_alu dec = hs_qpu_a(HS_QPU_A_SUB, HS_QPU_RF(0), HS_QPU_RF(0), hs_qpu_imm_int(1));
    dec.af = HS_QPU_PF_PUSHZ;
    hs_qpu_emit(&p, &dec);
    hs_qpu_branch(&p, HS_QPU_B_NA0, l, 0, 0);
    hs_qpu_alu np = hs_qpu_alu_nop();
    for (int i = 0; i < 3; i++) hs_qpu_emit(&p, &np);
    hs_qpu_emit_end(&p);
    int len = hs_qpu_prog_finish(&p);
    uint32_t spin = len > 0 ? hs_v3d_load(&code, 0, buf, (uint32_t)len) : 0;
    if (!spin) die("spin build", EIO);
    ((uint32_t*)unif.map)[0] = 400000000; /* ~10 s at the 37,500 iterations per ms hs_mailbox's warm-up measured */
    uint32_t h[2] = {code.handle, unif.handle};
    hs_mailbox mb;
    start(&mb, batches, 0, 0);
    int rc = 0, bad = gather_checked(&mb, 1024, 0, 0, &rc);
    uint32_t* T = (uint32_t*)bt.map;
    uint32_t* ix = (uint32_t*)bi.map;
    uint32_t* out = (uint32_t*)bo.map;
    hs_v3d_dispatch_desc d = {spin, unif.gpu, 1, h, 2, 0};
    uint64_t t0 = now_ns();
    /* RT_FOREIGN_NOSPIN: no job of our own; something else (a renderer whose job hangs) causes the reset */
    if (!getenv("RT_FOREIGN_NOSPIN") && (e = hs_v3d_dispatch(&dev2, &d))) die("spin dispatch", e);
    uint64_t reqs = 0, slow = 0, slow100 = 0, lat_max = 0;
    double spin_gone_ms = 0;
    while (now_ns() - t0 < 8000000000ull && !rc) {
        uint32_t n = 16 * batches * 4;
        for (uint32_t i = 0; i < n; i++) ix[i] = rnd() % TABLE_N, out[i] = 0xDEADBEEF;
        uint32_t a[7] = {4, bi.gpu, bt.gpu, bo.gpu, 0, TABLE_N, 0};
        uint64_t t = now_ns();
        int64_t s = hs_mailbox_send(&mb, HS_MB_OP_GATHER, a);
        if (s < 0) {
            rc = (int)s;
            break;
        }
        int w = 0;
        while ((rc = hs_mailbox_wait(&mb, (uint32_t)s, 100000000ull)) == -ETIME && ++w < 150) {
        }
        uint64_t dt = now_ns() - t;
        if (dt > lat_max) lat_max = dt;
        slow += dt > 1000000000ull;
        slow100 += dt > 100000000ull;
        for (uint32_t i = 0; i < n; i++) bad += out[i] != T[ix[i]];
        reqs++;
        if (!spin_gone_ms && !hs_v3d_bo_wait(&dev2, &code, 0)) spin_gone_ms = (now_ns() - t0) / 1e6;
    }
    int rc2 = hs_mailbox_stop(&mb), rc3;
    bad += gather_checked(&mb, 1024, 0, 0, &rc3);
    int ok = !rc && !rc2 && !rc3 && !bad && spin_gone_ms > 0;
    printf("{\"reset\":\"another client's job hangs while the mailbox is in use\",\"batches\":%u,\"ok\":%s,\"requests\":%llu,"
           "\"rc\":%d,\"wrong\":%d,\"longest_request_ms\":%.1f,\"requests_over_1s\":%llu,\"hung_job_gone_after_ms\":%.0f,"
           "\"lost_leases_counted\":%llu,\"rc_stop\":%d,\"rc_restart\":%d,\"requests_over_100ms\":%llu}\n",
           batches, ok ? "true" : "false", (unsigned long long)reqs, rc, bad, lat_max / 1e6, (unsigned long long)slow, spin_gone_ms,
           (unsigned long long)mb.lost_leases, rc2, rc3, (unsigned long long)slow100);
    if (!ok) failures++;
    hs_mailbox_free(&mb);
    hs_v3d_bo_free(&dev2, &code);
    hs_v3d_bo_free(&dev2, &unif);
    hs_v3d_close(&dev2);
}

/* Chain requests back to back for `seconds`: `pairs` pairs of chunks per
 * batch, k steps each. A chain pair runs its k steps with no thread switch
 * (one switch per pair, at its loads), and a compute job gives the QPUs to a
 * renderer only at thread switches (tools/p0_5_spin_probe). Beside a renderer
 * under reset_trace.bt this measures what the renderer gets. gap_us: pause
 * between requests (0: none). One request in 64 is checked against the CPU
 * (checking takes the CPU longer than the request takes the GPU). */
static void chainload(uint32_t batches, double seconds, uint32_t pairs, uint32_t k, uint32_t gap_us) {
    hs_mailbox mb;
    start(&mb, batches, 0, 0);
    uint32_t n = 32 * batches * pairs;
    int rc = 0, bad = 0;
    int copy = !!getenv("RT_COPY");
    static float cx[MAXE], cy[MAXE];
    for (uint32_t i = 0; i < n && i < MAXE; i++) cx[i] = (float)(rnd() % 1000000) / 1e6f;
    uint64_t reqs = 0, checked = 0, t0 = now_ns(), busy = 0, mx = 0;
    while (now_ns() - t0 < (uint64_t)(seconds * 1e9) && !rc) {
        uint64_t t = now_ns(), d;
        if (reqs % 64 == 0) {
            bad += chain_checked(&mb, n, k, 0, 0, &rc);
            d = (uint64_t)(last_call_ms * 1e6);
            checked++;
        } else {
            /* RT_COPY: the caller also copies its operands in and the results out, as an inline message would */
            if (copy) memcpy(bx.map, cx, n * 4);
            rc = hs_mailbox_chain(&mb, bx.gpu, by.gpu, n, k, 0.9995f, 0.0007f);
            if (copy) memcpy(cy, by.map, n * 4);
            d = now_ns() - t;
        }
        busy += d;
        if (d > mx) mx = d;
        reqs++;
        if (gap_us) {
            struct timespec ts = {0, (long)gap_us * 1000};
            nanosleep(&ts, NULL);
        }
    }
    int ok = !rc && !bad && !mb.lost_leases;
    printf("{\"chainload\":true,\"batches\":%u,\"pairs_per_batch\":%u,\"k\":%u,\"gap_us\":%u,\"ok\":%s,\"requests\":%llu,\"checked\":%llu,\"rc\":%d,"
           "\"wrong\":%d,\"gpu_share_of_time\":%.3f,\"request_ms\":{\"mean\":%.3f,\"max\":%.3f},\"request_us_mean\":%.2f,\"caller_copies\":%s,\"leases\":%u,\"age_yields\":%llu,\"longest_job_ms\":%.2f}\n",
           batches, pairs, k, gap_us, ok ? "true" : "false", (unsigned long long)reqs, (unsigned long long)checked, rc, bad,
           (double)busy / (double)(now_ns() - t0), busy / 1e6 / (double)(reqs ? reqs : 1),
           mx / 1e6, busy / 1e3 / (double)(reqs ? reqs : 1), copy ? "true" : "false", hs_mailbox_leases_done(&mb, 0), (unsigned long long)atomic_load(&mb.wd_age_yields),
           atomic_load(&mb.wd_job_gap_max_ns) / 1e6);
    if (!ok) failures++;
    hs_mailbox_free(&mb);
}

/* first request after the GPU sat idle: a plain job against the mailbox */
static void idlegap(uint32_t batches) {
    hs_v3d_bo code, unif;
    static uint64_t buf[4096];
    if (hs_v3d_bo_new(&dev, &code, 64 << 10) || hs_v3d_bo_new(&dev, &unif, 4096)) die("bo", EIO);
    hs_qpu_prog p;
    hs_qpu_prog_init(&p, buf, 4096);
    int len = hs_qpu_kernel_gather(&p, 1, 8);
    if (len < 0) die("gather build", EIO);
    uint32_t g = hs_v3d_load(&code, 0, buf, (uint32_t)len);
    uint32_t jh[5] = {code.handle, unif.handle, bt.handle, bi.handle, by.handle};
    for (uint32_t i = 0; i < 128; i++) ((uint32_t*)bi.map)[i] = i;
    if (hs_qpu_gather_uniforms((uint32_t*)unif.map, 128, 1, 8, bi.gpu, bt.gpu, by.gpu)) die("uniforms", EINVAL);
    hs_mailbox mb;
    start(&mb, batches, 0, 0);
    static const long gaps_ms[] = {1, 5, 20, 50, 100, 200, 400, 1000};
    for (unsigned gi = 0; gi < sizeof gaps_ms / sizeof gaps_ms[0]; gi++) {
        double job[7], box[7];
        for (int i = 0; i < 7; i++) {
            for (int m = 0; m < 2; m++) {
                hs_mailbox_stop(&mb); /* GPU idle from here */
                struct timespec ts = {gaps_ms[gi] / 1000, (gaps_ms[gi] % 1000) * 1000000};
                nanosleep(&ts, NULL);
                uint64_t a = now_ns();
                int e;
                if (m) {
                    e = hs_mailbox_echo(&mb);
                } else {
                    hs_v3d_dispatch_desc d = {g, unif.gpu, 1, jh, 5, 0};
                    e = hs_v3d_dispatch(&dev, &d);
                    if (!e) e = hs_v3d_bo_wait(&dev, &by, 5000000000ull);
                }
                if (e) die("idlegap", e);
                (m ? box : job)[i] = (now_ns() - a) / 1e6;
            }
        }
        qsort(job, 7, sizeof(double), cmpd);
        qsort(box, 7, sizeof(double), cmpd);
        printf("{\"idle_gap_ms\":%ld,\"batches\":%u,\"plain_job_ms\":{\"median\":%.3f,\"max\":%.3f},"
               "\"mailbox_echo_ms\":{\"median\":%.3f,\"max\":%.3f}}\n",
               gaps_ms[gi], batches, job[3], job[6], box[3], box[6]);
    }
    hs_mailbox_free(&mb);
}

/* what a zero-timeout BO wait costs while a lease runs */
static void bowait(void) {
    hs_mailbox mb;
    wd_off = 1;
    start(&mb, 1, 250000, 0);
    double v[20];
    for (int i = 0; i < 20; i++) {
        hs_mailbox_echo(&mb);
        uint64_t t = now_ns();
        int r = hs_v3d_bo_wait(&dev, &mb.ctl, 0);
        v[i] = (now_ns() - t) / 1e6;
        if (!r) i--;
    }
    hs_mailbox_free(&mb);
    qsort(v, 20, sizeof(double), cmpd);
    printf("{\"bo_wait_zero_timeout_while_busy_ms\":{\"min\":%.3f,\"median\":%.3f,\"max\":%.3f}}\n", v[0], v[10], v[19]);
}

/* an ordinary compute job submitted while the mailbox's leases are queued */
static void jobwait(uint32_t batches, int n) {
    hs_v3d_bo code, unif;
    static uint64_t buf[4096];
    if (hs_v3d_bo_new(&dev, &code, 64 << 10) || hs_v3d_bo_new(&dev, &unif, 4096)) die("bo", EIO);
    hs_qpu_prog p;
    hs_qpu_prog_init(&p, buf, 4096);
    int len = hs_qpu_kernel_gather(&p, 1, 8);
    if (len < 0) die("gather build", EIO);
    uint32_t g = hs_v3d_load(&code, 0, buf, (uint32_t)len);
    uint32_t jh[5] = {code.handle, unif.handle, bt.handle, bi.handle, by.handle};
    if (hs_qpu_gather_uniforms((uint32_t*)unif.map, 128, 1, 8, bi.gpu, bt.gpu, by.gpu)) die("uniforms", EINVAL);
    hs_mailbox mb;
    wd_off = !getenv("RT_WD_ON"); /* default: leases stay queued, the case being measured */
    start(&mb, batches, 0, 0);
    double* w[3];
    for (int m = 0; m < 3; m++) w[m] = malloc(sizeof(double) * (size_t)n);
    for (int m = 0; m < 3; m++) /* 0: no mailbox leases; 1: leases queued; 2: hs_mailbox_stop first */
        for (int i = 0; i < n; i++) {
            hs_v3d_bo_wait(&dev, &mb.ctl, 5000000000ull);
            if (m) hs_mailbox_echo(&mb);
            uint64_t a = now_ns();
            if (m == 2) hs_mailbox_stop(&mb);
            hs_v3d_dispatch_desc d = {g, unif.gpu, 1, jh, 5, 0};
            int e = hs_v3d_dispatch(&dev, &d);
            if (!e) e = hs_v3d_bo_wait(&dev, &by, 5000000000ull);
            if (e) die("job", e);
            w[m][i] = (now_ns() - a) / 1e6;
        }
    hs_mailbox_free(&mb);
    for (int m = 0; m < 3; m++) qsort(w[m], n, sizeof(double), cmpd);
    printf("{\"jobwait\":true,\"batches\":%u,\"n\":%d,\"alone_ms\":[%.3f,%.3f],\"behind_leases_ms\":[%.3f,%.3f],"
           "\"stop_first_ms\":[%.3f,%.3f],\"note\":\"[median,max]\"}\n",
           batches, n, w[0][n / 2], w[0][n - 1], w[1][n / 2], w[1][n - 1], w[2][n / 2], w[2][n - 1]);
}

int main(int argc, char** argv) {
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(2, &cs);
    sched_setaffinity(0, sizeof cs, &cs);
    setvbuf(stdout, NULL, _IOLBF, 0);
    int e = hs_v3d_open(&dev, NULL);
    if (e) die("hs_v3d_open", e);
    if ((e = hs_v3d_bo_new(&dev, &bt, TABLE_N * 4)) || (e = hs_v3d_bo_new(&dev, &bi, MAXE * 4 + 4096)) ||
        (e = hs_v3d_bo_new(&dev, &bo, MAXE * 4 + 4096)) || (e = hs_v3d_bo_new(&dev, &bx, MAXE * 4 + 4096)) ||
        (e = hs_v3d_bo_new(&dev, &by, MAXE * 4 + 4096)))
        die("bo", e);
    uint32_t* T = (uint32_t*)bt.map;
    for (uint32_t i = 0; i < TABLE_N; i++) T[i] = i * 2654435761u ^ 0x5A5A5A5Au;
    const char* mode = argc > 1 ? argv[1] : "cases";
    if (!strcmp(mode, "lease") && argc > 3)
        lease((uint32_t)atoi(argv[2]), (uint32_t)strtoul(argv[3], 0, 0), argc > 4 ? atoi(argv[4]) : 20);
    else if (!strcmp(mode, "soak") && argc > 3)
        soak((uint32_t)atoi(argv[2]), atof(argv[3]));
    else if (!strcmp(mode, "busy") && argc > 5)
        busy((uint32_t)atoi(argv[2]), atof(argv[3]), (uint32_t)atoi(argv[4]), atoi(argv[5]));
    else if (!strcmp(mode, "reset") && argc > 2)
        reset_mode((uint32_t)atoi(argv[2]));
    else if (!strcmp(mode, "foreign") && argc > 2)
        foreign_mode((uint32_t)atoi(argv[2]));
    else if (!strcmp(mode, "chainload") && argc > 5)
        chainload((uint32_t)atoi(argv[2]), atof(argv[3]), (uint32_t)atoi(argv[4]), (uint32_t)atoi(argv[5]),
                  argc > 6 ? (uint32_t)atoi(argv[6]) : 0);
    else if (!strcmp(mode, "pipe") && argc > 3)
        pipe_mode((uint32_t)atoi(argv[2]), atof(argv[3]), argc > 4 ? (uint32_t)atoi(argv[4]) : HS_MB_MAX_GATHER_CHUNKS);
    else if (!strcmp(mode, "jobcheck") && argc > 3)
        jobcheck((uint32_t)atoi(argv[2]), atof(argv[3]));
    else if (!strcmp(mode, "idlegap") && argc > 2)
        idlegap((uint32_t)atoi(argv[2]));
    else if (!strcmp(mode, "bowait"))
        bowait();
    else if (!strcmp(mode, "jobwait") && argc > 2)
        jobwait((uint32_t)atoi(argv[2]), argc > 3 ? atoi(argv[3]) : 30);
    else if (argc > 2) {
        cases((uint32_t)atoi(argv[2]));
    } else {
        cases(1);
        cases(8);
    }
    printf("{\"rt_mailbox\":\"%s\",\"failures\":%d}\n", mode, failures);
    return failures ? 1 : 0;
}
