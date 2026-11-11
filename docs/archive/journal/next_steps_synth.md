# Synthesis: Possible Next Step-Changes for NeoGPU

## Working Thesis
NeoGPU should evolve toward a clearer identity as a message-native runtime substrate, and its next major technical step should be a semantic split in the fabric rather than more universal-path micro-optimizations.

## Possibility Space

### Path A: Semantic Fabric Redesign
Introduce explicit message classes, such as:
- overwriteable state updates
- durable control messages
- payload-bearing commands
- telemetry / lossy observation traffic

Potential effect:
- lower contention for state-style traffic
- cleaner fast paths without branchy heuristics
- clearer performance model

Risks:
- deep architectural change
- requires strong tests and tooling to remain safe

### Path B: Runtime Identity Consolidation
Reframe NeoGPU around one dominant story:
- a tiny message-native runtime where rendering, IPC, and future inference are coordinated nodes

Potential effect:
- makes subsystem priorities clearer
- simplifies documentation and demos
- gives architectural decisions a stronger center of gravity

Risks:
- may force deferral or containment of currently ambiguous features

### Path C: Make Graphics The Canonical Proof Surface
Elevate visible demos, test scenes, and hardware inspection tooling as first-class repo outputs.

Potential effect:
- strengthens trust and distinctiveness
- makes regressions easier to catch
- gives humans a concrete handle on the system's value

Risks:
- can delay less visible substrate work if done without discipline

### Path D: Constrain ML Until It Is Real
Keep ML clearly marked experimental unless it is promoted to full operational quality.

Potential effect:
- protects repo identity from blur
- reduces strategic diffusion

Risks:
- may disappoint ambition if the goal was immediate unification

## Recommended Sequence
1. Clarify NeoGPU's center-of-gravity narrative in docs and demos.
2. Define semantic message classes and the invariants for each class.
3. Prototype one structural fast path for overwriteable state traffic outside the current universal path.
4. Keep ML explicitly experimental until it reaches the same reality standard as graphics/tooling.
5. Use visible demos and tooling queries as acceptance criteria for major changes.

## Success Criteria
- [ ] A new reader can describe what NeoGPU is in one coherent sentence.
- [ ] The fabric distinguishes at least two message semantic classes explicitly.
- [ ] Performance work targets the right class semantics, not just local heuristics.
- [ ] Visible demos remain a first-class proof surface.
- [ ] Experimental subsystems are labeled honestly and scoped intentionally.

## One-Sentence Summary
The next big leap is to make NeoGPU more itself: clearer in identity, truer in message semantics, and stronger in the visible proof of what the runtime can do.
