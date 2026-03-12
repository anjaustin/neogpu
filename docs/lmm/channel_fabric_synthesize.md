# SYNTHESIZE: Channelized Fabric Execution Plan

## Goal
Implement a channelized fabric that preserves determinism and improves latency control under load.

## P0 Decisions

1) Start with 2 channels:
- `CHAN_RT`: input/audio/time-critical control
- `CHAN_RENDER`: render state + draw commands

2) Deterministic scheduler:
- drain `CHAN_RT` first (bounded budget)
- drain `CHAN_RENDER` next (bounded budget)

3) Backpressure policy:
- `CHAN_RT`: block for essential ops; optional drop-oldest for sampled input events
- `CHAN_RENDER`: block (pristine) + optional coalescing for state setters

4) Capture policy:
- capture `CHAN_RENDER` by default
- optionally capture input subset from `CHAN_RT`

## Implementation Sketch

### Step 1: Channel spec (doc)
- Add `docs/CHANNEL_FABRIC.md` with:
  - channel list
  - drain order + budget constants
  - per-channel backpressure policy
  - capture inclusion

### Step 2: Add channel classification
Choose one:
- A) add `Message.channel` (explicit, self-describing)
- B) keep `Message` unchanged and use `op->channel` table (ABI stable)

### Step 3: Duplicate transport per channel
- `producers[channel][pid]` SPSC lanes
- `submit[channel]` fallback MPSC

### Step 4: Step-thread scheduling
- deterministic drain order
- deterministic per-channel budgets

### Step 5: Coalesce state setters (render channel)
- in the drain stage (before routing), coalesce redundant state messages
- keep draw commands uncoalesced

### Step 6: Falsification
- add stress tests:
  - saturate telemetry (should not block RT/render)
  - saturate render and ensure RT still meets budget
  - capture/replay determinism with channel schedule

## Success Criteria

- RT latency stays bounded under render spam.
- Render capture/replay remains deterministic.
- Backpressure behavior is per-channel and documented.
- Telemetry cannot deadlock the system.
