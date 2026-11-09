#ifndef COMPARISON_TRANSPORT_H
#define COMPARISON_TRANSPORT_H
#include "hs_fabric.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    const char* name;
    void* (*create)(u32 participants);
    bool (*submit)(void* context, u32 lane, const HSParticipantFrame* input);
    u32 (*step)(void* context);
    const HSParticipantFrame* (*view)(void* context, u32 lane);
    void (*destroy)(void* context);
} ComparisonOps;
extern const ComparisonOps comparison_native, comparison_rigtorp;
#ifdef __cplusplus
}
#endif
#endif
