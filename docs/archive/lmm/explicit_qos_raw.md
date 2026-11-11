# LMM: Explicit Channel QoS (RAW)

## Current State
- RT/RENDER/TELEM channels exist conceptually
- Default channel based on opcode
- Budget per channel
- Drop semantics implicit

## Need
- Explicit priority levels:
  - RT (CHAN_RT): Strict priority, never drops
  - RENDER (CHAN_RENDER): Weighted fair, limited drops
  - TELEM (CHAN_TELEM): Background, aggressive drops

## QoS Requirements
1. **RT**: Guaranteed bandwidth, no drops
2. **RENDER**: Fair share, graceful degradation
3. **TELEM**: Best-effort, first to drop

## Current Issues
- RT can be blocked by RENDER if budgets misconfigured
- No explicit priority scheduling
- TELEM not actually dropped under pressure

## Questions for NODES
- Implement priority queue for RT?
- Drop policy: TELEM drops first, then RENDER?
- Min bandwidth guarantees per channel?
