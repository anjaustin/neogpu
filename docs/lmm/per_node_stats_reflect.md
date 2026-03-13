# LMM: Per-Node Statistics (REFLECT)

## Edge Cases
1. **High-water wraparound**: atomic_uint wraps at 2^32
   - Solution: Use larger type or accept rare wrap
2. **Performance impact**: Adding max() on hot path
   - Solution: Only track on submit path (not spsc), and use relaxed atomics
3. **32-bit overflow**: Very high message counts
   - Accept: unlikely to hit in practice

## Failure Modes
- High-water never resets (intentional - shows peak load)
- Need explicit reset op if desired

## API Design
```c
// In HSSystem
atomic_uint submit_hw[CHAN_COUNT];

// Query via IPC - extend OP_QUERY_FABRIC
struct ChannelStats {
    u32 spsc_ok;
    u32 spsc_full; 
    u32 mpsc_ok;
    u32 submit_full;
    u32 submit_hw;  // NEW
};
```

## Test Cases
1. Send messages, verify hw increases
2. Flood channel, verify hw reflects peak
3. Clear system, verify hw persists (or resets)
