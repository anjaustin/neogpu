# NeoGPU Remediation Plan - 2026-03-14

## Purpose

Turn NeoGPU from a repo with several partially-competing futures into a more coherent message-native runtime with a clearer center of gravity, safer extension paths, and more truthful subsystem boundaries.

## Guiding Thesis

The next step-change for NeoGPU is not another isolated micro-optimization. It is a coordinated remediation across:

1. repo identity
2. fabric semantics
3. graphics proof surface
4. ML scope honesty

These are coupled. Each one reduces confusion in the others.

## Workstream 1 - Clarify NeoGPU's Center Of Gravity

### Goal

Make it obvious what NeoGPU is for a reader, contributor, and user.

### Current Problem

The repo currently reads as a mixture of:

- an ARM/NEON GPU message layer
- a benchmark and concurrency lab
- a runtime with IPC and tooling
- an experimental ML subsystem

All are present, but the repo does not yet rank them clearly.

### Remediation

- rewrite top-level narrative around NeoGPU as a message-native runtime substrate
- describe rendering, IPC/tooling, and future inference as node classes inside one runtime
- explicitly mark what is operationally real versus experimental
- make the proof surface visible in README and docs

### Deliverables

- updated `README.md`
- updated/added architecture overview doc
- cross-links to canonical proof demos and tooling docs

### Success Criteria

- a new reader can describe NeoGPU in one coherent sentence
- the repo's dominant identity is clearer than its side experiments
- docs stop over-claiming maturity where maturity is not yet real

## Workstream 2 - Define Semantic Message Classes

### Goal

Stop treating all messages as if they deserve identical routing and synchronization cost.

### Current Problem

The profiling work suggests structural mismatch:

- overwriteable state updates
- durable control messages
- payload-bearing commands
- telemetry traffic

are all funneled through nearly the same queue discipline and routing structure.

### Remediation

- define semantic message classes in docs first
- specify invariants for each class
- identify which classes may be lossy, overwriteable, durable, or ordered
- design a structural fast path for at least one class instead of adding heuristics to the universal path

### Deliverables

- semantic message class spec
- migration plan from current opcode/channel model to class-aware routing
- one candidate prototype target for structural implementation

### Success Criteria

- at least two message classes are defined explicitly
- class semantics are tied to queue/routing behavior, not just prose
- future performance work targets class semantics rather than ad hoc shortcuts

## Workstream 3 - Strengthen Graphics As Canonical Proof Surface

### Goal

Use visible graphics behavior as the clearest proof that NeoGPU is real, inspectable, and worth extending.

### Current Problem

The graphics demos are the strongest artifacts in the repo, but they are not yet consistently treated as canonical proof surfaces.

### Remediation

- define canonical graphics demos and tests
- ensure test lifecycle and build rules preserve those demos safely
- emphasize hardware-observable correctness, not just exit status
- link performance and system identity back to visible demos

### Deliverables

- canonical demo list
- stable build/run instructions for those demos
- test-source preservation rules and safer cleanup rules
- doc references from README into the canonical demos

### Success Criteria

- the graphics proof surface is easy to find and run
- user-visible correctness is part of acceptance, not an afterthought
- the repo has at least one unmistakable hardware-visible demo path

## Workstream 4 - Constrain ML Until It Is Operationally Real

### Goal

Protect repo coherence by making ML status honest and bounded.

### Current Problem

ML is partially scaffolded, but not yet real to the same standard as graphics/tooling. That creates conceptual blur.

### Remediation

- explicitly mark ML as experimental where appropriate
- separate current ML scaffolding from operational claims
- define what would have to become true for ML to be promoted to first-class operational status

### Deliverables

- updated ML wording in README and docs
- explicit promotion criteria for ML maturity
- scoped boundary between experimental and operational subsystems

### Success Criteria

- docs do not imply mature ML functionality where it does not exist
- contributors can tell what is experimental immediately
- future ML work has a clear graduation standard

## Execution Order

1. Workstream 1: center of gravity narrative
2. Workstream 2: semantic message class specification
3. Workstream 3: graphics proof surface strengthening
4. Workstream 4: ML scoping and maturity boundary

This order is intentional:

- identity first, so the rest has a center
- semantics second, so performance work has a real target
- proof surface third, so the repo demonstrates its identity clearly
- ML scoping fourth, so ambiguity is reduced after the center is established

## Working Rules

For each workstream:

1. implement
2. red-team
3. document
4. commit
5. push

Red-team means:

- validate correctness against the real outcome for that workstream
- check for regressions or contradictions in repo messaging
- confirm that the change improves coherence, not just local wording or mechanics

## Out Of Scope For This Plan

- broad new subsystem additions
- premature ML promotion
- more routing heuristics without a semantic class model
- cleanup shortcuts that endanger source-of-truth test files
