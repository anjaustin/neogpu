#ifndef HS_FABRIC_H
#define HS_FABRIC_H

#include "hs_render.h"
#include <pthread.h>

#define HS_FABRIC_PARTICIPANTS 8
#define HS_FABRIC_QUEUE 4
#define HS_FABRIC_SPLIT_PUBLICATION 1
#define HS_FABRIC_DRAWS 64

typedef enum {
    HS_FABRIC_OK, HS_FABRIC_FULL, HS_FABRIC_BUSY, HS_FABRIC_STALE,
    HS_FABRIC_CLOSING, HS_FABRIC_SEQUENCE, HS_FABRIC_INVALID
} HSFabricResult;

typedef struct { u32 slot; u64 generation; } HSParticipant;
typedef struct {
    u64 sequence;
    u32 count;
    HSRenderCmd draws[HS_FABRIC_DRAWS];
} HSParticipantFrame;

/* Internals are public for fixed storage, not caller mutation. Never copy a
 * live fabric. All functions require successful init and a live instance. */
typedef struct {
    pthread_mutex_t mutex;
    bool active, closing;
    u64 generation, admitted;
    /* Monotonic modulo-2^32 ring cursors; never reset at join. */
    u32 read_index, write_index;
    HSParticipantFrame queue[HS_FABRIC_QUEUE];
} HSParticipantSlot;

typedef struct {
    HSParticipant owner;
    bool active;
    HSParticipantFrame frame;
} HSParticipantView;

typedef struct {
    HSParticipantSlot slots[HS_FABRIC_PARTICIPANTS];
    HSParticipantView views[HS_FABRIC_PARTICIPANTS]; /* consumer-owned */
    u32 cursor;
    u64 applied, retired; /* consumer-owned event counters */
} HSFabric;

bool hs_fabric_init(HSFabric* fabric);
void hs_fabric_destroy(HSFabric* fabric);
HSFabricResult hs_fabric_join(HSFabric* fabric, HSParticipant* out);
HSFabricResult hs_fabric_leave(HSFabric* fabric, HSParticipant owner);
/* Input must be stable for the call; accepted frames are copied. No blocking
 * retry, fallback, overwrite or silent drop; rejection consumes no sequence. */
HSFabricResult hs_fabric_submit(HSFabric* fabric, HSParticipant owner,
                                 const HSParticipantFrame* frame);
/* Single externally serialized consumer only; returns applied + retired events.
 * A call visits each slot at most once even if budget exceeds slot count.
 * Zero means no progress this pass, NOT proof of emptiness: slots may be busy. */
u32 hs_fabric_step(HSFabric* fabric, u32 budget);
/* Consumer-only: materialize all last-complete contributions into a single list.
 * Vertices are immutable, renderer-owned resources in bank owner.slot. */
bool hs_fabric_compose(const HSFabric* fabric, HSRenderList* out, const f32 clear[4]);

#endif
