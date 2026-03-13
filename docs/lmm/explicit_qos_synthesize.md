# LMM: Explicit Channel QoS (SYNTHESIZE)

## P0 Spec: Explicit RT/RENDER/TELEM Semantics

### 1. Add QoS Priority Enum
```c
typedef enum {
    HS_CHAN_QOS_CRITICAL = 0,  // RT: never drops
    HS_CHAN_QOS_INTERACTIVE = 1, // RENDER: limited drops
    HS_CHAN_QOS_BACKGROUND = 2, // TELEM: aggressive drops
} HSChannelQOS;
```

### 2. Map Channels to QoS
- CHAN_RT -> HS_CHAN_QOS_CRITICAL
- CHAN_RENDER -> HS_CHAN_QOS_INTERACTIVE
- CHAN_TELEM -> HS_CHAN_QOS_BACKGROUND

### 3. TELEM Auto-Drop Behavior
In hs_spsc_push():
```c
if (ch == CHAN_TELEM && !hs_spsc_try_push(...)) {
    // Drop: don't fall through to MPSC
    atomic_fetch_add(&sys->telem_dropped, 1);
    return true;  // "delivered" (as drops)
}
```

### 4. Track TELEM Drops
Add to HSSystem:
```c
atomic_uint telem_dropped[CHAN_COUNT];
```

### 5. Expose in OP_QUERY_FABRIC
Add telem_dropped to fabric stats

## Acceptance Criteria
- [ ] TELEM drops counted when queue full
- [ ] OP_QUERY_FABRIC shows telem_dropped
- [ ] RT always processed first (budget)
- [ ] RENDER gets fair share
- [ ] All existing tests pass
