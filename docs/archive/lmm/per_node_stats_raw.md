# LMM: Per-Node Statistics (RAW)

## Current State
- NODE_SYSTEM returns aggregate stats (total messages, drops, etc.)
- No per-node visibility into channel fabric health
- Can't see which channel (RT/RENDER/TELEM) has issues

## Need
- Per-channel message counts
- Per-channel drop counts  
- Per-channel queue depth / high-water marks
- Per-node processing latency

## Use Cases
1. Debug: which channel is dropping?
2. Tune: are budgets too tight?
3. Monitor: is RT channel starving?
4. Debug: why is frame taking so long?

## Questions for NODES
- Track stats in HSSystem or per-node?
- Granularity: per-channel (RT/RENDER/TELEM) or per-node (each node instance)?
- Reset semantics: when do stats clear?
