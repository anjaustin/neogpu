#ifndef BEDROCK_SUPPORT_H
#define BEDROCK_SUPPORT_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct __attribute__((aligned(64))) {
    uint64_t attempts, busy, full, admitted, validated_draws, canonical_bytes;
    uint64_t visits, active_visits, inactive_visits, service_busy, empty, applied, copy_bytes, yields;
    int tid;
} BedrockCounters;
#ifdef __cplusplus
extern thread_local BedrockCounters* br_current;
extern thread_local int br_collect;
#else
extern _Thread_local BedrockCounters* br_current;
extern _Thread_local int br_collect;
#endif
extern unsigned br_participants;
void bedrock_thread(unsigned index);
void bedrock_start(void);
void bedrock_control(const char* command);
void bedrock_emit(const char* implementation, void* context, unsigned participants);
#ifdef __cplusplus
}
#endif
#define BR_ADD(field, value) do { if (br_collect && br_current) br_current->field += (value); } while (0)
#define BR_INC(field) BR_ADD(field, 1)
#endif
