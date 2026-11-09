"""Generate the ROI driver from checked-in source, not historical artifacts."""
if not __debug__:
    raise RuntimeError('Bedrock evidence requires Python assertions enabled')


def roi_driver(source):
    edits = [
        ('#include "comparison_transport.h"', '#include "comparison_transport.h"\n#include "bedrock_support.h"'),
        ('Producer* producer = pointer;', 'Producer* producer = pointer;\n    bedrock_thread(producer->id + 1);'),
        ('while (!atomic_load_explicit(&go, memory_order_acquire)) sched_yield();',
         'while (!atomic_load_explicit(&go, memory_order_acquire)) atomic_signal_fence(memory_order_seq_cst);\n    bedrock_start();'),
        ('pin(0);', 'pin(0);\n    bedrock_thread(0); br_participants = participants;'),
        ('u64 start = ns(), applied = 0;', 'bedrock_control("enable"); bedrock_start();\n    u64 start = ns(), applied = 0;'),
        ('u64 elapsed = ns() - start;', 'u64 elapsed = ns() - start;\n    bedrock_control("disable");'),
        ('ops->destroy(context);', 'bedrock_emit(ops->name, context, participants);\n    ops->destroy(context);'),
    ]
    for before, after in edits:
        if source.count(before) != 1:
            raise ValueError('ROI driver anchor missing/ambiguous: '+before)
        source = source.replace(before, after)
    return source
