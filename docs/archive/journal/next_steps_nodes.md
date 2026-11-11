# Nodes of Interest: Next Step-Changes for NeoGPU

## Node 1: Multiple Futures Are Present
NeoGPU currently contains at least three futures: graphics/runtime substrate, heterogeneous compute playground, and performance research bench.
Why it matters: ambiguity here creates energy but also diffusion.

## Node 2: The Repo's Hidden Strength Is Legibility
The message fabric, tooling ops, and visible tests make the system inspectable and reason-about-able on hardware.
Why it matters: this may be a stronger differentiator than raw speed alone.

## Node 3: Abstraction Must Become Cheap Enough To Disappear
The fabric must remain light enough that the message model feels elegant rather than costly.
Tension with Node 2: legibility adds structure, which can add cost.

## Node 4: Local Performance Heuristics Are Mostly Exhausted
Recent profiling cycles showed that shallow scheduler and routing shortcuts often regress.
Why it matters: the next performance gains are likely architectural.

## Node 5: ML Is Not Yet Equally Real
Graphics and tooling are operationally real; ML still feels experimental or scaffold-like.
Tension: promoting ML too early could blur the repo's identity.

## Node 6: Semantic Message Classes Could Be A Step-Change
Some messages are overwriteable state, some are streams, some are batch/transactional. Treating them identically may be the core mismatch.
Dependency: this could reduce unnecessary queue contention and routing cost.

## Node 7: Framing Is An Architectural Act
Clarifying what NeoGPU is for humans is not just documentation; it determines what gets optimized and what remains experimental.
Why it matters: identity drives systems decisions.

## Node 8: Visible Demos Are Strategic, Not Decorative
The spheres demo and graphics tests are not peripheral. They are the clearest proof of the repo's character.
Why it matters: one strong demo can anchor identity better than many abstract claims.

## Node 9: Developer Trust Accelerates Iteration
Safe cleanup, reliable tests, accurate docs, and tooling clarity reduce the cognitive tax of experimentation.
Dependency: repo trust can create a step-change in velocity.

## Node 10: A Deep Fabric Change May Matter More Than More Tuning
The biggest performance step-change may require changing queue semantics or introducing a second class of message path.
Tension with Node 9: deep changes can destabilize the repo if not framed and tested clearly.

## Node 11: The Repo May Want A Tiny Runtime OS Story
A plausible unifying identity is NeoGPU as a message-native runtime where rendering, IPC, and inference are coordinated nodes.
Why it matters: this could gather the existing pieces into one coherent narrative.

## Node 12: The Repo Needs A Center Of Gravity
Without a center, every subsystem competes equally for design priority.
Why it matters: a step-change may be as much about subtraction and ranking as addition.
