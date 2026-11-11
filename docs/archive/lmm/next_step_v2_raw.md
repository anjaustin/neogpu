# LMM: Next Step-Charge for NeoGPU (RAW)

## Unfiltered Dump

Game engine and ML code exist separately but neither has a "killer app" that proves the architecture.

**Game engine:**
- Has render, input, audio stubs, textures - works
- BUT: no actual game runs on it
- PicoGPU has 5 demos, no shipped games either
- Could NeoGPU be FIRST to ship a real game?

**ML code:**
- BitNet inference: 1 token/sec
- Works end-to-end with GGUF loader
- BUT: too slow for real-time game interaction
- Would need ~10x faster for NPC dialogue in games

**The differentiation:**
- Message-passing substrate is unique
- But how do you SHOW it off?
- Need something that runs and impresses

**What about:**
- A simple game (Pong, Space Invaders) - validates game engine works
- ML already works - just needs to be faster
- Batched prefill could help but still not real-time
- GPU integration is READY but not enabled

**Barriers:**
- Audio is stub - need ALSA integration
- GPU path needs weight preloading for speed
- No one has SHIPPED a game on either PicoGPU or NeoGPU

## Raw Questions

- Should we ship a simple game first to validate the engine?
- Is the ML fast enough for ANY real-time use?
- What's the minimum viable demo?
- Should we focus on game OR ML OR both?
- Why is GPU disabled by default?

## Raw Fears

- Game demo might be boring, doesn't show off the architecture
- ML at 1 token/sec is embarrassing
- We're ahead of PicoGPU on paper but behind on shipped demos
- All this infrastructure might be for nothing if nothing runs on it
