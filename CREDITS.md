# Credits & Lineage

NeoGPU did not spring from nowhere. It stands on prior work, and this document
records that debt plainly, because a project should know and honor where its
ideas came from.

## PicoGPU — the origin

**[PicoGPU](https://github.com/ncannasse/picogpu)** by **Nicolas Cannasse**
(creator of Haxe) is the project NeoGPU was born from.

- **Repository**: https://github.com/ncannasse/picogpu
- **Live demo**: https://ncannasse.github.io/picogpu
- **What it is**: a ~300 KB software GPU written in Haxe/HScript, compiling to
  multiple targets including JavaScript/WebGL and HashLink.

PicoGPU proved the thesis that NeoGPU is built on: **a complete GPU runtime can
be implemented in software, kept tiny, and still do real work** — rendering,
sound, the lot. Without that proof-of-concept and the inspiration it provided,
NeoGPU would not exist.

### What NeoGPU inherited from PicoGPU

The debt is concrete, not vague:

- **The core idea** — a compact, self-contained software GPU runtime, small
  enough to understand end to end.
- **The feature surface** NeoGPU mirrors: vertex & fragment buffers; blending,
  culling, depth, stencil, color mask, and clipping; render targets and
  instancing; matrix / vector / quaternion math; textured UV mapping.
- **The audio model** — 4-channel, 48 kHz sound.

### Where NeoGPU diverges

Divergence is the inspiration succeeding, not fading. NeoGPU took the idea in
its own direction:

| | PicoGPU | NeoGPU |
|---|---|---|
| Language | Haxe / HScript | C (header-oriented) |
| Target | WebGL, HashLink | Raspberry Pi 4 V3D (bare-metal DRM/GBM) |
| Architecture | GPU runtime | message-native runtime substrate |
| Compute | graphics + audio | graphics + tooling/IPC + experimental ML (BitNet) |
| Optimization | cross-target VM | native ARM/NEON, Cortex-A72 |

The architectures have diverged substantially. That is a measure of how far a
good idea travels — not a reason to drop the acknowledgment.

## Thanks

To **Nicolas Cannasse**, for PicoGPU and for showing that a tiny GPU is both
possible and worth building. NeoGPU is a tribute to that work.

---

*If you build on NeoGPU in turn, consider carrying this lineage forward: credit
what you drew from, specifically. It costs nothing and it keeps the record
honest.*
