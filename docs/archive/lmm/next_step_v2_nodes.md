# LMM: Next Step-Charge for NeoGPU (NODES)

## Key Points Extracted from RAW

### What's Done
1. Game engine headers exist (~600 LOC)
2. BitNet inference pipeline works (1 token/sec)
3. Message-passing substrate is production-quality
4. GPU acceleration path exists (disabled)
5. Lock-free queues, IPC tooling all working
6. PicoGPU has no shipped games either

### What's Missing
1. No game actually runs on the engine
2. ML too slow for real-time (1 tok/s << 10+ tok/s needed)
3. Audio output is stub (no ALSA)
4. GPU path not integrated into main inference
5. No demo that proves the architecture works

### Tensions

| Tension | Why It Matters |
|---------|----------------|
| Game demo vs ML speed | Game would validate engine but ML too slow for NPCs |
| Ship fast vs ship right | Quick demo vs fully integrated GPU |
| PicoGPU competition | They're older but no shipped games either |
| Audio needed | Game needs sound to be playable |
| GPU disabled | Acceleration exists but unused |

### Constraints

| Constraint | Limit |
|------------|-------|
| Pi4 CPU | 4 cores @ 1.8GHz - ML is memory bandwidth limited |
| Pi4 GPU | V3D ~28 GFLOPS - not enough for heavy ML |
| Time | Need quick win to show momentum |
| Attention | Don't spread too thin |

### Leverage Points

- **HIGH leverage**: Ship a game demo (validates engine, beats PicoGPU)
- **HIGH leverage**: Enable GPU by default (immediate ~2x speedup)
- **MEDIUM leverage**: ALSA audio (needed for playable game)
- **MEDIUM leverage**: Batched prefill (helps but not game-changing)
- **LOW leverage**: Larger models (no hardware to run them fast)

### Real Questions to Answer

1. Can we build a simple game in < 1 day that runs on NeoGPU?
2. Is 1 token/sec acceptable for a "proof of concept" demo?
3. What's the minimum audio needed (beeps? samples? streaming?)?
4. Is GPU path stable enough to enable by default?

## Reference: Existing Haxe Game Demos

### PicoGPU Samples (5 demos)
- Start.gpu - basic rotating cube
- CubeRT.gpu - render targets
- DrawInstanced.gpu - instancing
- Sound.gpu - audio synthesis
- TextHelloWorld.gpu - text rendering

### Haxe Pong on GitHub

| Repo | Framework | Stars |
|------|-----------|-------|
| dstrekelj/kha-pong | Kha | 9 |
| HaxeFlixel-Pong | HaxeFlixel | 5 |
| taras42/Pong | Haxe+OpenFL | 2 |
| N1ckn1ght/HaxePong | Heaps | 1 |

**Key insight**: Kha targets native platforms including ARM - could be a reference for NeoGPU game structure!
