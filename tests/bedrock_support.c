#define _GNU_SOURCE
#include "bedrock_support.h"
#include "hs_fabric.h"
#include <fcntl.h>
#include <errno.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifdef HS_FABRIC_SPLIT_PUBLICATION
#define BR_HEAD read_index
#define BR_COUNT write_index
#define BR_HEAD_NAME "read_index"
#define BR_COUNT_NAME "write_index"
#else
#define BR_HEAD head
#define BR_COUNT count
#define BR_HEAD_NAME "head"
#define BR_COUNT_NAME "count"
#endif

static BedrockCounters counters[HS_FABRIC_PARTICIPANTS + 1];
_Thread_local BedrockCounters* br_current;
_Thread_local int br_collect;
unsigned br_participants;

void bedrock_thread(unsigned index) {
    if (index > HS_FABRIC_PARTICIPANTS) abort();
    br_current = &counters[index];
    br_current->tid = (int)syscall(SYS_gettid);
    char name[16];
    if (!index) snprintf(name, sizeof(name), "br-consumer");
    else snprintf(name, sizeof(name), "br-prod-%u", index - 1);
    if (pthread_setname_np(pthread_self(), name) != 0) abort();
}
void bedrock_start(void) { br_collect = 1; }

void bedrock_control(const char* command) {
    static _Thread_local bool previous_ack;
    const char* ctl = getenv("BEDROCK_CTL"), *ack = getenv("BEDROCK_ACK");
    if (!ctl && !ack) return;
    if (!ctl || !ack) { fprintf(stderr, "bedrock control requires both FIFO paths\n"); abort(); }
    int writer = open(ctl, O_RDWR | O_CLOEXEC), reader = open(ack, O_RDWR | O_CLOEXEC);
    if (writer < 0 || reader < 0) { perror("bedrock control open"); abort(); }
    char request[32];
    int length = snprintf(request, sizeof(request), "%s\n", command);
    if (length <= 0 || length >= (int)sizeof(request)) abort();
    ssize_t sent;
    do { sent = write(writer, request, (size_t)length); } while (sent < 0 && errno == EINTR);
    if (sent != length) { perror("bedrock control write"); abort(); }
    /* FIFOs are byte streams, not packet transports. Read through newline;
     * consume an optional trailing NUL from the previous ACK on the next call.
     * This accepts both perf encodings even when reads/writes are fragmented. */
    const char expected[] = "ack\n";
    size_t count = 0;
    bool skipped_nul = false;
    while (count < sizeof(expected) - 1) {
        char byte;
        ssize_t received;
        do { received = read(reader, &byte, 1); } while (received < 0 && errno == EINTR);
        if (received != 1) { perror("bedrock control read"); abort(); }
        if (!count && !byte && previous_ack && !skipped_nul) { skipped_nul = true; continue; }
        if (byte != expected[count++]) { fprintf(stderr, "bedrock %s malformed ACK\n", command); abort(); }
    }
    previous_ack = true;
    close(writer); close(reader);
}
#ifdef BEDROCK_COUNTS
extern int __real_sched_yield(void);
int __wrap_sched_yield(void) { BR_INC(yields); return __real_sched_yield(); }
#endif

void bedrock_emit(const char* implementation, void* context, unsigned participants) {
    for (unsigned i = 0; i <= participants; i++) {
        const BedrockCounters* c = &counters[i];
        printf("{\"record\":\"bedrock_counts\",\"thread\":%u,\"tid\":%d,\"attempts\":%" PRIu64
               ",\"busy\":%" PRIu64 ",\"full\":%" PRIu64 ",\"admitted\":%" PRIu64
               ",\"validated_draws\":%" PRIu64 ",\"canonical_bytes\":%" PRIu64
               ",\"visits\":%" PRIu64 ",\"active_visits\":%" PRIu64 ",\"inactive_visits\":%" PRIu64
               ",\"service_busy\":%" PRIu64 ",\"empty\":%" PRIu64 ",\"applied\":%" PRIu64
               ",\"copy_bytes\":%" PRIu64 ",\"yields\":%" PRIu64 "}\n",
               i, c->tid, c->attempts, c->busy, c->full, c->admitted, c->validated_draws, c->canonical_bytes,
               c->visits, c->active_visits, c->inactive_visits, c->service_busy, c->empty, c->applied, c->copy_bytes, c->yields);
    }
    if (!strcmp(implementation, "neogpu")) {
        /* comparison_native's Context begins with HSFabric (checked by source). */
        HSFabric* fabric = context;
        for (unsigned i = 0; i < HS_FABRIC_PARTICIPANTS; i++) {
            HSParticipantSlot* s = &fabric->slots[i];
            printf("{\"record\":\"layout\",\"slot\":%u,\"fabric_bytes\":%zu,\"slot_bytes\":%zu,"
                   "\"slot_address\":%" PRIuPTR ",\"mutex_offset\":%zu,\"active_offset\":%zu,"
                   "\"admitted_offset\":%zu,\"" BR_HEAD_NAME "_offset\":%zu,\"" BR_COUNT_NAME "_offset\":%zu,\"queue_offset\":%zu,"
                   "\"mutex_mod64\":%" PRIuPTR ",\"" BR_COUNT_NAME "_mod64\":%" PRIuPTR ",\"queue_mod64\":%" PRIuPTR "}\n",
                   i, sizeof(HSFabric), sizeof(*s), (uintptr_t)s, offsetof(HSParticipantSlot, mutex),
                   offsetof(HSParticipantSlot, active), offsetof(HSParticipantSlot, admitted), offsetof(HSParticipantSlot, BR_HEAD),
                   offsetof(HSParticipantSlot, BR_COUNT), offsetof(HSParticipantSlot, queue),
                   (uintptr_t)&s->mutex % 64, (uintptr_t)&s->BR_COUNT % 64, (uintptr_t)&s->queue % 64);
        }
    }
}
