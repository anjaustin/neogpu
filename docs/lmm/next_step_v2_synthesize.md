# LMM: Next Step-Charge for NeoGPU (SYNTHESIZE)

## Decision: Ship a Running Game Demo

The next step-change is to **ship a simple game that actually runs on NeoGPU**, demonstrating that the architecture works end-to-end.

### Why This Wins

1. **First-mover advantage**: PicoGPU has no shipped games - we can be first
2. **Proves architecture**: Shows message-passing works for real-time workloads
3. **Sets stage for ML**: Even at 1 token/sec, having game + ML proves the concept
4. **Low risk**: Simple game = low implementation cost, high validation value
5. **Narrative control**: "NeoGPU: the game engine with built-in LLM inference"

### The Game: Pong

**Why Pong:**
- Simplest game that proves the engine works
- Requires: rendering, input, game loop, collision
- Does NOT require: audio, complex sprites, fancy shaders
- Recognizable: everyone knows what Pong is
- Impressive: "we built a game from scratch"

### Implementation Spec

#### Phase 1: The Demo (1-2 days)

**Deliverable**: Pong running on Pi4 via NeoGPU game engine

**Must have:**
- [ ] Render ball and paddles
- [ ] Input: arrow keys / D-pad control
- [ ] Ball physics (bounce off walls, paddle)
- [ ] Score display
- [ ] Game loop at 30+ FPS

**Don't need (yet):**
- Audio
- High-res graphics
- Multiple levels

#### Phase 2: The ML Integration (parallel)

**Deliverable**: ML inference running alongside game

**Approach:**
- Run game at 60 FPS on one thread
- Run ML inference on separate thread  
- Use message-passing to coordinate
- Show ML output on screen (text)

**For demo purposes:**
- Use slow 1 tok/sec - it's fine for proof-of-concept
- Show "AI opponent" generating text
- Note: "GPU acceleration path exists, not yet enabled"

#### Phase 3: Enable GPU (after demo)

**Deliverable**: GPU-accelerated ML enabled by default

**Steps:**
1. Fix any issues from Phase 2
2. Enable GPU path in loader
3. Benchmark actual speedup
4. Document in release notes

### The Narrative

**Today:**
- "NeoGPU has a game engine and ML inference but nothing runs on it"

**After Phase 1:**
- "NeoGPU runs Pong - here's the code, here's the binary"

**After Phase 2:**
- "NeoGPU runs Pong WITH LLM inference - the first game engine with built-in AI"

**After Phase 3:**
- "NeoGPU runs it all with GPU acceleration"

### Success Criteria

- [ ] Pong compiles and runs on Pi4
- [ ] Game is playable (30+ FPS, responsive input)
- [ ] Can show ML inference running alongside
- [ ] Source code available
- [ ] Demo video/screenshots for announcement

### Timeline

| Phase | Effort | Duration |
|-------|--------|----------|
| Phase 1: Pong | Medium | 1-2 days |
| Phase 2: ML integration | Medium | 2-3 days |
| Phase 3: Enable GPU | Low | 1 day |

**Total: 1 week to shipped demo**

### Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Pong is too hard | Start with moving triangle |
| ML too slow | Accept for POC, note GPU path |
| Pi4 unavailable | Test on emulator or CI |
| No one cares | Ship anyway - it's about proving it works |
