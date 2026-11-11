# LMM: Explicit Channel QoS (NODES)

## Current Implementation
- Per-channel budgets in hs_step(): RT=4096, RENDER=16384, TELEM=1024
- Per-channel block policy: RT=true, RENDER=true, TELEM=false
- Channel assignment based on opcode (hardcoded in hs_submit_enqueue)
- RT ops: OP_FENCE, OP_QUERY_*, OP_SET_*
- RENDER ops: OP_FRAME_*, OP_PRESENT
- TELEM: everything else (default)

## What's Missing for Explicit QoS

### 1. Priority Scheduling
Currently all channels processed in order RT->RENDER->TELEM
- Could add: process RT until empty, then RENDER, then TELEM
- Current: budget-based round-robin

### 2. TELEM Drops Under Pressure
Currently TELEM doesn't block but also doesn't auto-drop
- Add: when TELEM queue full, drop oldest
- Add: submit_full for TELEM triggers drops

### 3. Min Bandwidth Guarantees
- RT: min 4096 msgs/step (already set)
- RENDER: min 16384 msgs/step (already set)  
- TELEM: best-effort remainder

### 4. Explicit Priority Flag
```c
typedef enum {
    HS_CHAN_QOS_RT,        // Critical, no drops
    HS_CHAN_QOS_RENDER,    // Interactive, limited drops
    HS_CHAN_QOS_TELEM,     // Background, aggressive drops
} HSChannelQOS;
```

## Implementation Plan
1. Add HSChannelQOS enum
2. Map CHAN_RT/QOS_RT, etc.
3. Add TELEM auto-drop when queue full
4. Document behavior
