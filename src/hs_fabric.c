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
            slot->closing = false;
            slot->admitted = 0;
            slot->head = slot->count = 0;
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
    return slot->closing ? HS_FABRIC_CLOSING : HS_FABRIC_OK;
}

HSFabricResult hs_fabric_leave(HSFabric* fabric, HSParticipant owner) {
    if (!fabric || owner.slot >= HS_FABRIC_PARTICIPANTS) return HS_FABRIC_INVALID;
    HSParticipantSlot* slot = &fabric->slots[owner.slot];
    if (pthread_mutex_trylock(&slot->mutex) != 0) return HS_FABRIC_BUSY;
    HSFabricResult result = ownership(slot, owner);
    if (result == HS_FABRIC_OK) slot->closing = true;
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
        if (slot->admitted == UINT64_MAX || frame->sequence != slot->admitted + 1)
            result = HS_FABRIC_SEQUENCE;
        else if (slot->count == HS_FABRIC_QUEUE) result = HS_FABRIC_FULL;
        else {
            HSParticipantFrame* dest = &slot->queue[(slot->head + slot->count) % HS_FABRIC_QUEUE];
            dest->sequence = frame->sequence;
            dest->count = frame->count;
            /* Canonicalize unused fields: no borrowed pointers/payload/state. */
            for (u32 i = 0; i < frame->count; i++)
                dest->draws[i] = (HSRenderCmd){.op = HS_RC_DRAW, .a = (u8)owner.slot};
            slot->count++;
            slot->admitted = frame->sequence;
        }
    }
    pthread_mutex_unlock(&slot->mutex);
    return result;
}

u32 hs_fabric_step(HSFabric* fabric, u32 budget) {
    if (!fabric || !budget) return 0;
    u32 events = 0;
    for (u32 visited = 0; visited < HS_FABRIC_PARTICIPANTS && events < budget; visited++) {
        u32 i = fabric->cursor;
        fabric->cursor = (i + 1) % HS_FABRIC_PARTICIPANTS;
        HSParticipantSlot* slot = &fabric->slots[i];
        if (pthread_mutex_trylock(&slot->mutex) != 0) continue;
        if (slot->active && slot->count) {
            HSParticipantFrame* source = &slot->queue[slot->head];
            HSParticipantView* view = &fabric->views[i];
            view->owner = (HSParticipant){i, slot->generation};
            view->active = true;
            view->frame.sequence = source->sequence;
            view->frame.count = source->count;
            memcpy(view->frame.draws, source->draws, source->count * sizeof(HSRenderCmd));
            slot->head = (slot->head + 1) % HS_FABRIC_QUEUE;
            slot->count--;
            fabric->applied++;
            events++;
        } else if (slot->active && slot->closing) {
            fabric->views[i].active = false;
            slot->active = false;
            fabric->retired++;
            events++;
        }
        pthread_mutex_unlock(&slot->mutex);
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
