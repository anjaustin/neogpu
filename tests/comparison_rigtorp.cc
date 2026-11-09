// Unmodified third-party queue; only the workload/retained-view adapter is ours.
#define _Static_assert static_assert
#include "comparison_transport.h"
#include <rigtorp/SPSCQueue.h>
#include <cstdlib>
#include <cstring>
#include <new>

static_assert(sizeof(HSRenderCmd) == 32 && sizeof(HSParticipantFrame) == 2064,
              "C++ reference must use the measured aarch64 C layouts");

struct Entry {
    HSParticipantFrame frame;
    explicit Entry(const HSParticipantFrame& input) noexcept {
        frame.sequence = input.sequence;
        frame.count = input.count;
        for (u32 i = 0; i < input.count; i++) {
            frame.draws[i] = {};
            frame.draws[i].op = HS_RC_DRAW;
            frame.draws[i].a = input.draws[i].a;
        }
    }
};
struct Lane {
    rigtorp::SPSCQueue<Entry> queue{HS_FABRIC_QUEUE};
    u64 admitted = 0; // single-producer owned
    HSParticipantFrame retained{}; // single-consumer owned
};
struct Context { Lane lanes[HS_FABRIC_PARTICIPANTS]; u32 cursor = 0; };
static void* create(u32 participants) {
    if (!participants || participants > HS_FABRIC_PARTICIPANTS) return nullptr;
    auto* context = new (std::nothrow) Context;
    if (context) for (const auto& lane : context->lanes)
        if (lane.queue.capacity() != HS_FABRIC_QUEUE) std::abort();
    return context;
}
static bool submit(void* pointer, u32 index, const HSParticipantFrame* input) {
    auto* context = static_cast<Context*>(pointer);
    if (index >= HS_FABRIC_PARTICIPANTS || input->count > HS_FABRIC_DRAWS) std::abort();
    for (u32 i = 0; i < input->count; i++) {
        const auto& draw = input->draws[i];
        if (draw.op != HS_RC_DRAW || draw.a != index || draw.payload_len) std::abort();
    }
    auto& lane = context->lanes[index];
    if (lane.admitted == UINT64_MAX || input->sequence != lane.admitted + 1) std::abort();
    if (!lane.queue.try_emplace(*input)) return false;
    lane.admitted = input->sequence;
    return true;
}
static u32 step(void* pointer) {
    auto* context = static_cast<Context*>(pointer);
    u32 applied = 0;
    for (u32 visited = 0; visited < HS_FABRIC_PARTICIPANTS; visited++) {
        auto& lane = context->lanes[context->cursor];
        context->cursor = (context->cursor + 1) % HS_FABRIC_PARTICIPANTS;
        const auto* entry = lane.queue.front();
        if (!entry) continue;
        lane.retained.sequence = entry->frame.sequence;
        lane.retained.count = entry->frame.count;
        std::memcpy(lane.retained.draws, entry->frame.draws, entry->frame.count * sizeof(HSRenderCmd));
        lane.queue.pop();
        applied++;
    }
    return applied;
}
static const HSParticipantFrame* view(void* pointer, u32 lane) {
    return &static_cast<Context*>(pointer)->lanes[lane].retained;
}
static void destroy(void* pointer) { delete static_cast<Context*>(pointer); }
extern "C" const ComparisonOps comparison_rigtorp = {"rigtorp", create, submit, step, view, destroy};
