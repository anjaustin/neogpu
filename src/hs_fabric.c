#include "hs_fabric.h"
#include "hs_buffer.h"
#include <string.h>

_Static_assert(HS_FABRIC_PARTICIPANTS * HS_FABRIC_DRAWS + 4 <= HS_RENDER_MAX_CMDS,
               "Every complete participant view must fit one render list");
_Static_assert(HS_FABRIC_PARTICIPANTS <= HS_MAX_BANKS,
               "Participants require distinct renderer banks");

bool hs_fabric_init(HSFabric* fabric) {
    if (!fabric) return false;
    memset(fabric, 0, sizeof(*fabric));
    for (u32 i = 0; i < HS_FABRIC_PARTICIPANTS; i++) {
        if (pthread_mutex_init(&fabric->slots[i].mutex, NULL) != 0) {
            while (i) pthread_mutex_destroy(&fabric->slots[--i].mutex);
            return false;
        }
    }
    return true;
}

void hs_fabric_destroy(HSFabric* fabric) {
    if (!fabric) return;
    for (u32 i = 0; i < HS_FABRIC_PARTICIPANTS; i++)
        pthread_mutex_destroy(&fabric->slots[i].mutex);
}

HSFabricResult hs_fabric_join(HSFabric* fabric, HSParticipant* out) {
    if (!fabric || !out) return HS_FABRIC_INVALID;
    bool busy = false;
    for (u32 i = 0; i < HS_FABRIC_PARTICIPANTS; i++) {
        HSParticipantSlot* slot = &fabric->slots[i];
        if (pthread_mutex_trylock(&slot->mutex) != 0) { busy = true; continue; }
        if (!slot->active && slot->generation != UINT64_MAX) {
            slot->active = true;
            __atomic_store_n(&slot->closing, false, __ATOMIC_RELEASE);
            slot->admitted = 0;
            *out = (HSParticipant){i, ++slot->generation};
            pthread_mutex_unlock(&slot->mutex);
            return HS_FABRIC_OK;
        }
        pthread_mutex_unlock(&slot->mutex);
    }
    return busy ? HS_FABRIC_BUSY : HS_FABRIC_FULL;
}

static HSFabricResult ownership(const HSParticipantSlot* slot, HSParticipant owner) {
    if (!slot->active || slot->generation != owner.generation) return HS_FABRIC_STALE;
    return __atomic_load_n(&slot->closing, __ATOMIC_RELAXED) ? HS_FABRIC_CLOSING : HS_FABRIC_OK;
}

HSFabricResult hs_fabric_leave(HSFabric* fabric, HSParticipant owner) {
    if (!fabric || owner.slot >= HS_FABRIC_PARTICIPANTS) return HS_FABRIC_INVALID;
    HSParticipantSlot* slot = &fabric->slots[owner.slot];
    if (pthread_mutex_trylock(&slot->mutex) != 0) return HS_FABRIC_BUSY;
    HSFabricResult result = ownership(slot, owner);
    if (result == HS_FABRIC_OK) __atomic_store_n(&slot->closing, true, __ATOMIC_RELEASE);
    pthread_mutex_unlock(&slot->mutex);
    return result;
}

HSFabricResult hs_fabric_submit(HSFabric* fabric, HSParticipant owner,
                                 const HSParticipantFrame* frame) {
    if (!fabric || !frame || owner.slot >= HS_FABRIC_PARTICIPANTS ||
        frame->count > HS_FABRIC_DRAWS) return HS_FABRIC_INVALID;
    for (u32 i = 0; i < frame->count; i++) {
        const HSRenderCmd* cmd = &frame->draws[i];
        if (cmd->op != HS_RC_DRAW || cmd->a != owner.slot || cmd->payload_len)
            return HS_FABRIC_INVALID;
    }
    HSParticipantSlot* slot = &fabric->slots[owner.slot];
    if (pthread_mutex_trylock(&slot->mutex) != 0) return HS_FABRIC_BUSY;
    HSFabricResult result = ownership(slot, owner);
    if (result == HS_FABRIC_OK) {
        u32 read = __atomic_load_n(&slot->read_index, __ATOMIC_ACQUIRE);
        u32 write = __atomic_load_n(&slot->write_index, __ATOMIC_RELAXED);
        if (slot->admitted == UINT64_MAX || frame->sequence != slot->admitted + 1)
            result = HS_FABRIC_SEQUENCE;
        else if ((u32)(write - read) >= HS_FABRIC_QUEUE) result = HS_FABRIC_FULL;
        else {
            HSParticipantFrame* dest = &slot->queue[write % HS_FABRIC_QUEUE];
            dest->sequence = frame->sequence;
            dest->count = frame->count;
            /* Canonicalize unused fields: no borrowed pointers/payload/state. */
            for (u32 i = 0; i < frame->count; i++)
                dest->draws[i] = (HSRenderCmd){.op = HS_RC_DRAW, .a = (u8)owner.slot};
            slot->admitted = frame->sequence;
            __atomic_store_n(&slot->write_index, write + 1, __ATOMIC_RELEASE);
        }
    }
    pthread_mutex_unlock(&slot->mutex);
    return result;
}

/* Cursor wrap preserves ring positions because capacity divides 2^32. */
_Static_assert(HS_FABRIC_QUEUE > 0 && HS_FABRIC_QUEUE <= UINT32_MAX / 2 &&
               (HS_FABRIC_QUEUE & (HS_FABRIC_QUEUE - 1)) == 0, "Ring capacity must be a bounded power of two");
_Static_assert(__atomic_always_lock_free(sizeof(u32), 0), "Ring cursors must be lock-free");
_Static_assert(__atomic_always_lock_free(sizeof(bool), 0), "Closing notification must be lock-free");

u32 hs_fabric_step(HSFabric* fabric, u32 budget) {
    if (!fabric || !budget) return 0;
    u32 events = 0;
    for (u32 visited = 0; visited < HS_FABRIC_PARTICIPANTS && events < budget; visited++) {
        u32 i = fabric->cursor;
        fabric->cursor = (i + 1) % HS_FABRIC_PARTICIPANTS;
        HSParticipantSlot* slot = &fabric->slots[i];
        u32 read = __atomic_load_n(&slot->read_index, __ATOMIC_RELAXED);
        u32 write = __atomic_load_n(&slot->write_index, __ATOMIC_ACQUIRE);
        if (read != write) {
            HSParticipantFrame* source = &slot->queue[read % HS_FABRIC_QUEUE];
            HSParticipantView* view = &fabric->views[i];
            /* Generation is published before write_index and cannot change
             * until this same consumer retires the drained generation. */
            view->owner = (HSParticipant){i, slot->generation};
            view->active = true;
            view->frame.sequence = source->sequence;
            view->frame.count = source->count;
            memcpy(view->frame.draws, source->draws, source->count * sizeof(HSRenderCmd));
            __atomic_store_n(&slot->read_index, read + 1, __ATOMIC_RELEASE);
            fabric->applied++;
            events++;
        } else if (__atomic_load_n(&slot->closing, __ATOMIC_ACQUIRE)) {
            if (pthread_mutex_trylock(&slot->mutex) != 0) continue;
            /* Recheck after excluding submit/leave/join. Closing rejects new
             * admissions, and retirement never overtakes accepted frames. */
            if (slot->active && __atomic_load_n(&slot->closing, __ATOMIC_RELAXED) &&
                __atomic_load_n(&slot->read_index, __ATOMIC_RELAXED) ==
                __atomic_load_n(&slot->write_index, __ATOMIC_ACQUIRE)) {
                fabric->views[i].active = false;
                slot->active = false;
                __atomic_store_n(&slot->closing, false, __ATOMIC_RELEASE);
                fabric->retired++;
                events++;
            }
            pthread_mutex_unlock(&slot->mutex);
        }
    }
    return events;
}

bool hs_fabric_compose(const HSFabric* fabric, HSRenderList* out, const f32 clear[4]) {
    if (!fabric || !out || !clear) return false;
    u32 total = 4;
    for (u32 i = 0; i < HS_FABRIC_PARTICIPANTS; i++) {
        const HSParticipantView* view = &fabric->views[i];
        if (!view->active) continue;
        if (view->frame.count > HS_FABRIC_DRAWS) return false;
        total += view->frame.count;
    }
    if (total > HS_RENDER_MAX_CMDS) return false; /* do not touch out on failure */
    hs_render_reset(out);
    out->cmds[out->count++] = (HSRenderCmd){.op = HS_RC_FRAME_BEGIN};
    out->cmds[out->count++] = (HSRenderCmd){.op = HS_RC_CLEAR,
        .f0 = clear[0], .f1 = clear[1], .f2 = clear[2], .f3 = clear[3]};
    for (u32 i = 0; i < HS_FABRIC_PARTICIPANTS; i++) {
        const HSParticipantView* view = &fabric->views[i];
        if (!view->active) continue;
        memcpy(&out->cmds[out->count], view->frame.draws, view->frame.count * sizeof(HSRenderCmd));
        out->count += view->frame.count;
    }
    out->cmds[out->count++] = (HSRenderCmd){.op = HS_RC_FRAME_END};
    out->cmds[out->count++] = (HSRenderCmd){.op = HS_RC_PRESENT};
    return true;
}
