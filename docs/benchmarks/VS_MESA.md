# NeoGPU against the reference stack, 2026-11-16

Raspberry Pi 4 ("darkcrystal"), kernel 7.0.0-1023-raspi, revision `ff8289d`
plus `tools/bench_vs_mesa.c`. Every GPU run sat inside the GPU lock with the
performance governor. GPU resets: 205 before, 205 after.

| path | stack |
|---|---|
| **NeoGPU** | `hs_qpu_kernels` → `hs_qpu` (own encoder) → `hs_v3d` → kernel v3d driver. No Mesa, no libdrm, no GLES |
| **Mesa** (reference) | OpenGL ES 3.1 compute, Mesa 26.0.8 (V3D 4.2.14.0), EGL on GBM on the same render node → Mesa's v3d driver → the same kernel driver |
| **CPU** | 4 × Cortex-A72, huge pages; 1 or 4 threads, prefetch on or off; the best is reported |

Both GPU paths ran the same work on the same GPU, checked every output word on
every timed trial, and interleaved trials. Each timed gather followed an untimed
run of the same job, which keeps the GPU at full clock (HARDWARE_MAP_2026-11-15.md,
"GPU idle clock"). Times are wall time on the calling thread:
- NeoGPU: from submit until the BO wait returns.
- Mesa: from `glDispatchCompute` until `glFinish` returns.

Evidence (Pi only, gitignored): `tests/evidence-bench-20261116/`
(`run1.txt`, `run2.txt`, `lp-*.txt`, `ab-*.txt`, `cold.txt`, `cold-idle.txt`,
`cold-gap.txt`, `mailbox.txt`).
Reproduce it with:

```
make bench-vs-mesa        # exits 1 if any path's output is wrong
make tools/tmuop_probe
tools/bench_vs_mesa cold neo; tools/bench_vs_mesa cold mesa
flock build/gpu.lock tools/tmuop_probe M 2 3
```

## Summary

| workload | NeoGPU | Mesa | NeoGPU vs Mesa | best CPU | NeoGPU vs CPU |
|---|---|---|---|---|---|
| Cold start to first checked result, GPU awake | 0.46–0.48 ms | 80.8–82.5 ms | **~172×** | — | — |
| Cold start to first checked result, GPU runtime-suspended | 10.40–10.43 ms | 90.6–92.2 ms | **8.8×** | — | — |
| Single-job round trip, p50 (12 runs, two modes) | 61.3–62.1 or 73.5–74.0 µs | 83.9–86.0 or 101.8–105.1 µs | 1.35–1.43× | — | — |
| Pipelined small jobs, J = 1000 (12 runs) | 24.2k–27.2k jobs/s | 13.0k–14.1k or 24.2k–27.1k jobs/s | 0.98–1.87× | — | — |
| Gather, sequential indices | 612.8 M/s | 345.2 M/s | **1.78×** | 245.2 M/s | 2.50× |
| Gather, random, 512 KiB table | 74.6 M/s | 73.3 M/s | 1.02× | 229.6 M/s | 0.32× |
| Gather, random, 64 MiB table | 36.3 M/s | 34.0 M/s | 1.07× | 42.2 M/s | 0.86× |
| CPU ↔ GPU message inside a running job (probe, not shipped) | 0.76 µs RTT p50, 1.20–1.23 M msg/s | 83.9–105.1 µs per round-trip job; 13.0k–27.1k jobs/s pipelined | **110–138×** latency, **44–92×** rate | — | — |

**Where NeoGPU is ahead by 10× or more:**
- **Cold start, GPU awake: ~172×.** Most of Mesa's 81 ms is its own
  start-up: loading the driver, compiling the shader, creating the context.
- **CPU ↔ GPU messaging: 110–138× on latency, 44–92× on rate.** These figures come
  from the in-job mailbox probe, which runs on NeoGPU's own encoder and
  runtime. It is not yet a shipped API.

**Where NeoGPU is ahead, but by less than 10×:**
- **Bandwidth-bound gather: 1.78×** against Mesa's best configuration.
- **Cold start from a suspended GPU: 8.8×.** Both paths pay the same
  10.4 ms power-up.

**Where NeoGPU and Mesa are level:**
- **Per-job dispatch rate, in Mesa's fast mode.** Both pipelined paths then
  reach ~24–27 k jobs/s, about 37 µs per job, which is the kernel driver's
  per-job floor. In 8 of 12 runs Mesa sat at 13.0–14.1 k jobs/s instead, and
  NeoGPU led by 1.82–1.87×. See "Run to run" below.
- **Random gather.** It is bound by DRAM or the TMU. The CPU wins when the
  table fits its 1 MiB L2: 4 cores with prefetch reach 229.6 M/s.

## Cold start: process start to first checked result

`bench_vs_mesa cold neo|mesa` runs as a fresh process. It times from `main`
to reading back a 16-lane job's write and checking it:
- NeoGPU: open, 3 BOs, encode, load, dispatch, wait, read.
- Mesa: EGL/GBM, an ES 3.1 context, shader compile, SSBO, dispatch, map.

| GPU state before the process | NeoGPU | Mesa | ratio |
|---|---|---|---|
| active (back-to-back processes; `cold.txt`, runs 2–5) | 0.459–0.476 ms | 80.8–82.5 ms | ~172× |
| runtime-suspended (1 s idle before each; `cold-idle.txt`, 5 + 5) | 10.40–10.43 ms | 90.6–92.2 ms | 8.8× |

### New finding: the ~10 ms "first job" is the GPU powering up

The v3d device uses runtime power management. Its autosuspend delay is
`autosuspend_delay_ms = 100` in `/sys/class/drm/card0/device/power/`.

`cold-gap.txt` swept idle gaps of 50, 80, 120, 150 and 300 ms, twice each.
It read `runtime_status` just before each cold NeoGPU process. The cost
followed the GPU's state in all 10 runs:

| state before the process | cold NeoGPU |
|---|---|
| `suspended` (7 runs) | 10.41–10.46 ms |
| `active` (3 runs) | 0.46–0.51 ms |

So the ~10–12 ms that earlier docs called the "first job per process" cost
is a runtime resume after more than 100 ms of GPU idle, not a per-process
cost.

The idle-clock threshold (between 50 and 200 ms, HARDWARE_MAP) brackets this
100 ms delay. Whether the slow first job comes from that same suspend is not
yet measured.

Changing the delay (for example `power/control = on`) is a sysfs setting
that needs root. It was not changed here.

## Single-job round trip

One 16-lane job writes 1 to a word, and the CPU waits for completion. Each
path ran 2000 jobs, in alternating blocks of 100, after 200 warm-up jobs. The
word was checked afterwards: `ok` on both paths.

| path | min | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| NeoGPU | 56.4 µs | 62.1 µs | 63.9 µs | 72.9 µs | 80.0 µs |
| Mesa | 71.1 µs | 83.9 µs | 85.5 µs | 97.9 µs | 111.2 µs |

Mesa adds ~22 µs per job on top of NeoGPU's path, and both paths share the
kernel's submit-and-wait cost. That table is run 1. Across 12 runs the p50
fell into one of two modes; see "Run to run".

## Pipelined small jobs

J store_one jobs were submitted back to back with one wait. Each figure is the
median of 7 interleaved trials.

| J | NeoGPU jobs/s | Mesa jobs/s | ratio |
|---|---|---|---|
| 10 | 25,208 | 22,055 | 1.14× |
| 100 | 27,470 | 26,330 | 1.04× |
| 1000 | 27,156 | 27,087 | 1.00× |

That table is run 1. Across 12 runs Mesa's rate at J = 1000 was bimodal;
see "Run to run". The ceiling (~36.8 µs per job) belongs to the kernel's
job path. Neither
path's userspace adds to it once jobs queue. More throughput per job
submission needs more work per job, or no job per message at all, which is
what the mailbox does.

## Run to run: the dispatch figures have two modes per process

The dispatch sections ran 12 times as separate processes:
- `run1.txt` and `run2.txt` (the second through `make bench-vs-mesa`, right
  after `test-hardware`);
- `lp-quiet-{1,2,3}.txt` and `lp-sampled.txt`;
- `ab-{quiet,sampler}-{1,2,3}.txt`, interleaved, with and without a
  background reader of the clock tree every 250 ms.

| measure | fast mode | slow mode | runs fast / slow |
|---|---|---|---|
| NeoGPU round trip p50 | 61.3–62.1 µs | 73.5–74.0 µs | 3 / 9 |
| Mesa round trip p50 | 83.9–86.0 µs | 101.8–105.1 µs | 3 / 9 (the same runs as NeoGPU's) |
| Mesa pipelined, J = 1000 | 24.2–27.1 k jobs/s | 13.0–14.1 k jobs/s | 4 / 8 |
| NeoGPU pipelined, J = 1000 | 24.2–27.2 k jobs/s (one band) | — | — |

What was measured:
- The round-trip mode is shared. In each run both paths were fast or both
  were slow, and NeoGPU led by 1.35–1.43× in every run.
- The pipelined mode does not follow the round-trip mode.
- Neither mode followed the background reader (`ab-*`), and the v3d clock
  read 750 MHz throughout a run (sampled every 250 ms).
- Nothing else held a DRM node (`fuser`), and the SoC was at 41–42 °C.

Not yet traced. This is likely the same thing as the bimodal ~17 µs
dispatch step in HARDWARE_MAP. All v3d interrupts land on CPU 0, where the
bench thread is pinned; that is a lead, not a measured cause.

## Gather: out[i] = table[idx[i]], 4M × 4 B

Each figure is the median of 7 interleaved trials, with every output word
checked on every trial.

NeoGPU swept 1 or 8 batches with k = 1, 2, 4 or 8 loads in flight. Mesa swept
local sizes 16, 64, 128 and 256, with 1, 4 or 16 elements per invocation;
each element was indexed `inv + j * total`, so lanes stay coalesced.

256 is the largest local size ES 3.1 allows on this GPU, and Mesa's best was
at 256 in every case. Its SSBO limit is 128 MiB, so the 64 MiB table fits.

| case | NeoGPU best | Mesa best | CPU best |
|---|---|---|---|
| sequential idx, 64 MiB table | 612.8 M/s, 6.84 ms (8 batches, k=8) | 345.2 M/s, 12.15 ms (local 256, 16/inv) | 245.2 M/s (1 core) |
| random idx, 512 KiB table | 74.6 M/s (8 batches, k=8) | 73.3 M/s (local 256, 16/inv) | 229.6 M/s (4 cores, prefetch) |
| random idx, 64 MiB table | 36.3 M/s (8 batches, k=1) | 34.0 M/s (local 256, 1/inv) | 42.2 M/s (4 cores) |

- **Sequential:** NeoGPU's line-owning layout (QPU_ENCODER.md) moves 7.4 GB/s
  of table, index and output traffic (12 B per element). Mesa's best moves
  4.1 GB/s.
- **Random:** all three paths sit near the shared DRAM ceiling measured
  before (42.2–43.7 M/s system-wide, HARDWARE_MAP). The GPU paths are 14%
  (NeoGPU) and 19% (Mesa) under it, and the CPU reaches it.
- **Random, L2-sized table:** the CPU's 1 MiB L2 serves the 512 KiB table, so
  the CPU wins 3×. The GPU's L2T does not hold it (HARDWARE_MAP measured
  75.0 M/s there, matching this run).

## CPU ↔ GPU messaging inside one job (measured probe, not shipped)

`tools/tmuop_probe M 2 3` ran 3 × 1000 messages, 0 lost and 0 resets. The
CPU writes a sequence number and the GPU sees it with TMU cache op 2 and acks:
- RTT p50 0.76 µs, p99 1.02 µs, max 1.09–9.72 µs;
- 1.20–1.23 M msgs/s.

This matches HARDWARE_MAP (red team 4: 0.76 µs, 1.20–1.22 M/s).

The reference stack has no equivalent: each message is a job. Against Mesa:
- the single-job round trip of 83.9–105.1 µs is 110–138× the mailbox's
  0.76 µs RTT;
- the pipelined rate of 13.0–27.1 k jobs/s is 1/92 to 1/44 of the mailbox's
  rate.

Shipping the mailbox needs three things (open work):
- encoder support for cache-op lookups that return no data;
- a config uniform per op;
- a lease or hand-off scheme under the ~1 s job reset.

## Limits of this comparison

- Mesa is used through its public API at its best measured configuration.
  The comparison does not cover Mesa features NeoGPU lacks (graphics,
  general shaders).
- Gather and store_one are the kernels NeoGPU ships today. Other workloads
  are not compared yet.
- Wall times include each stack's own userspace cost. That cost is the
  thing being compared.
