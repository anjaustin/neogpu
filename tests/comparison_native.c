#include "comparison_transport.h"
#include <stdlib.h>

_Static_assert(sizeof(HSRenderCmd) == 32 && sizeof(HSParticipantFrame) == 2064,
               "Native/reference frame layouts must match");

typedef struct { HSFabric fabric; HSParticipant owners[HS_FABRIC_PARTICIPANTS]; } Context;
static void* create(u32 participants) {
    Context* context = calloc(1, sizeof(*context));
    if (!context || !hs_fabric_init(&context->fabric)) { free(context); return NULL; }
    for (u32 i = 0; i < participants; i++) {
        if (hs_fabric_join(&context->fabric, &context->owners[i]) != HS_FABRIC_OK) abort();
    }
    return context;
}
static bool submit(void* pointer, u32 lane, const HSParticipantFrame* input) {
    Context* context = pointer;
    HSFabricResult result = hs_fabric_submit(&context->fabric, context->owners[lane], input);
    if (result != HS_FABRIC_OK && result != HS_FABRIC_FULL && result != HS_FABRIC_BUSY) abort();
    return result == HS_FABRIC_OK;
}
static u32 step(void* pointer) {
    return hs_fabric_step(&((Context*)pointer)->fabric, HS_FABRIC_PARTICIPANTS);
}
static const HSParticipantFrame* view(void* pointer, u32 lane) {
    return &((Context*)pointer)->fabric.views[lane].frame;
}
static void destroy(void* pointer) {
    Context* context = pointer;
    hs_fabric_destroy(&context->fabric);
    free(context);
}
const ComparisonOps comparison_native = {"neogpu", create, submit, step, view, destroy};
