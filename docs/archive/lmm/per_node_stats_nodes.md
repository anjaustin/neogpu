# LMM: Per-Node Statistics (NODES)

## Current Stats Available
- spsc_ok[CHAN_COUNT] - successful SPSC sends per channel
- spsc_full[CHAN_COUNT] - SPSC full events per channel  
- mpsc_ok[CHAN_COUNT] - MPSC fallback successful per channel
- submit_full[CHAN_COUNT] - submit queue full per channel
- toolbus_dropped - toolbus overflow count

## What's Missing
1. **High-water marks**: Max queue depth ever reached
2. **Per-node visibility**: Which node is producing what?
3. **Latency tracking**: Time from send to apply

## Implementation Plan
Add to HSSystem:
- `atomic_uint submit_hw[CHAN_COUNT]` - high-water mark for submit queue
- Update on each enqueue: `hw = max(hw, depth)`

## Use via IPC
- Extend OP_QUERY_STATS to include high-water marks
- Or new OP_QUERY_CHANNEL_STATS

## Design Decision
Keep it simple: just add high-water tracking to existing submit queues
