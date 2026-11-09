# Participant fabric: first vertical slice

## Acceptance before implementation

Purpose: independently submitting **threads** contribute complete immutable draw
lists to one renderer, without resetting or mutating another participant's list.
This is an additive protocol; legacy message semantics and evidence stay intact.

Success requires FIFO application, explicit overload rejection, rotating service,
join/leave/rejoin with stale-handle rejection, retained last-complete contributions,
and pixel evidence on the Pi. Count every accepted frame through application;
retirement must drain accepted work before releasing a slot. A deliberately busy
participant slot must not stop other slots. Preserve all prior regression gates.

## Deliberately narrow contract

- Eight slots, four queued frames per slot, at most 64 draws per frame. Storage
  is bounded and owned by the fabric; no producer pointers survive admission.
- A handle is a slot plus a nonwrapping 64-bit generation. Join claims a free
  slot; leave marks closing, rejects further submissions, drains accepted frames,
  and retires on the consumer. Only retirement makes the slot reusable.
- Sequence numbers start at 1 and must advance exactly by one per accepted frame.
  A full/busy rejection consumes no sequence. Concurrent callers sharing one
  handle must coordinate their sequence allocation or retry explicit rejection.
- Each participant owns one draw bank equal to its slot number. Only `HS_RC_DRAW`
  is accepted; global clears/state/frame markers and borrowed payloads are not.
  The renderer owns/preloads immutable vertex buffers. Uploads, arbitrary state,
  textures, and resource snapshots are **not** implemented by this slice.
- Admission and service use per-slot `pthread_mutex_trylock`, never a global
  fallback queue. This is **not lock-free**. The consumer skips busy slots and
  retains their last applied contribution. A process dying while holding a
  mutex is not recovered; this is not cross-process crash isolation.
- One externally serialized consumer owns step, retained views, composition and
  backend execution. One call visits each slot at most once, applying at most
  one event per visit; successful events advance the rotating cursor. Under
  uncontended slots, budget-one steps service each backlogged slot within eight
  successful events. Mutex/scheduler contention has no hard-real-time bound.
  A zero return from step means no progress that pass, not proof that every
  producer queue is empty. A busy slot can still contain work.
- Empty contributions atomically hide that participant; no new submission means
  its last complete contribution remains. Composition observes one consumer
  boundary, not a globally synchronized application simulation epoch.
  FIFO application is not presentation acknowledgement: several steps before a
  composition can supersede intermediate views. Retirement can remove a drained
  contribution before any presentation. Present-every-submission is not promised.
- Producer code does not execute on the consumer. Blocking backend/consumer work
  still blocks presentation; priority/preemption is not claimed or fixed here.
- Init/destroy are externally serialized; stop/join all callers before destroy.
  An initialized fabric must not be copied or reinitialized while alive. Handles
  are scoped to that initialization lifetime; they are not capabilities against
  malicious in-process code and must not survive destroy/reinit.

## Evidence required

Headless: saturation/FIFO and input-copy ownership, budget-one fairness, held-lock
isolation, invalid/global/cross-bank command rejection, retirement drain, stale
handles, more than eight lifecycle participants, and concurrent exact delivery.
Record actual elapsed time, retries and per-participant applied sequences; report
frames/s separately from old payload-free messages/s. Run ASan/UBSan/leak checks
and TSan. This is a functionality change, not an optimization or speedup claim.

Graphics: two producer threads, renderer-owned banks, incomplete private edits,
stall/leave/rejoin; verify expected colored pixels before presentation and retain
captures. No real-frame claim follows solely from a decoded list.

Remaining uncertainty: production resource/version ownership, process transport,
hard latency limits, third-party queue comparisons and equivalent graphics-cost
comparisons. Do not infer these from this restricted slice.
