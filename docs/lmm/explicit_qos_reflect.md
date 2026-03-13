# LMM: Explicit Channel QoS (REFLECT)

## Edge Cases

### 1. RT Channel Starvation
- If RT keeps filling, RENDER/TELEM never get processed
- Solution: max RT budget per tick, spill over

### 2. TELEM Queue Full
- Currently blocks or returns false
- Desired: drop oldest TELEM messages
- Implementation: when spsc_full, drop and count

### 3. Mixed Workloads
- Heavy RT + heavy RENDER + heavy TELEM
- RT should always get its 4096
- RENDER gets up to 16384
- TELEM gets remainder (could be 0)

### 4. Budget = 0
- Currently allowed
- RT with budget=0: would never process
- Should enforce min budget for RT?

### 5. Backward Compatibility  
- Existing code uses channels implicitly
- Don't break existing message flows

## Failure Modes
- RT with 0 budget: hangs waiting (avoid)
- TELEM drops: count in stats (need)
- Priority inversion: ensure RT first (already done)

## Stats Needed
- telem_drops (new)
- rt_starved_count (new?)
