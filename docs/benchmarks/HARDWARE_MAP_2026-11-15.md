# Hardware map: Raspberry Pi 4, measured 2026-11-15

This is a map of what the board's hardware does when driven directly, measured
to find where the existing stacks leave capability unused. Every number is a
measurement on darkcrystal; none is a datasheet figure or an estimate.

## Reproducing

```sh
make hwprobe hwprobe-progs            # probes; QPU programs into build/qpu
sudo tools/hwprobe_cpu 2              # CPU side
tools/hwprobe_v3d 2                   # V3D identity, BO costs, CPU<->BO bandwidth
tools/hwprobe_qpu build/qpu 0         # QPU sections a m d c g G (all by default)
make qpu-progs                        # NeoGPU-encoded programs -> build/qpu/neo
tools/hwprobe_qpu build/qpu/neo 0 adgG  # the same sections from NeoGPU's own encoder
```

Sections m and c use the reference kernels (summation, memset, scopy), which
only the oracle assembles. The others run from NeoGPU's own encoder
([QPU_ENCODER.md](QPU_ENCODER.md)), and its programs are word-identical.

- Set the CPU governor to `performance` for the run, then restore it.
- Make sure the daily `aide` scan is not running. One run overlapped it, and
  its 4-core figure was wrong (see below).
- `hwprobe_v3d` and `hwprobe_qpu` talk to the kernel's v3d driver directly: no
  Mesa, no libdrm, no GLES. They need the `render` group, not root.
- The QPU programs are assembled with py-videocore6 (GPL-2), used as an
  instrument only. The assembled programs stay out of git, and nothing from
  py-videocore6 enters NeoGPU.

Raw outputs live on the Pi in `tests/evidence-measure-20261115/hwprobe/`,
which is gitignored like the other evidence directories:
- `clean/`: quiet system, performance governor;
- `gather/`;
- the earlier `cpu-run*`, `qpu-run*` and `v3d-run*`.

Where an earlier run overlapped aide, `clean/` supersedes it.

## What this means for NeoGPU

- **Compute:** the four A72 cores (57.3 GFLOP/s) out-compute the whole QPU
  array (44.5).
- **Bandwidth:** the GPU's advantage is DRAM bandwidth, 11.6 GB/s against the
  CPU cluster's 3.9–4.5.
- **Random access:** capped near 43 M accesses/s for the whole system,
  whichever side issues the reads. Work that is random has to be made coherent
  (sort, bin, tile) before the GPU's bandwidth helps.
- **Launching GPU jobs:** the kernel submit path costs 22–75 µs per job, and
  the GPU work itself takes a few µs. That path is the largest overhead
  measured.
- **Reading results back:** the CPU reads GPU buffers at 1.48 GB/s, because
  the mapping is uncached. GPU results should stay on the GPU, unless a
  cacheable return path is measured to be faster.
- **Two QPU rules,** both found by bisection and both required of any QPU code
  NeoGPU generates:
  1. A program never ends with TMU writes in flight.
  2. A batch of TMU loads ends with a thread switch before `ldtmu`.

## Compute ceilings

| unit | measured |
|---|---|
| A72 NEON fmla.4s, 1 core | 14.32 GFLOP/s (1 fmla per cycle at 1.8 GHz) |
| A72 NEON, 4 cores at once | **57.34 GFLOP/s** (14.33 per core, in clean runs 1 and 2) |
| QPU fadd+fmul dual issue, 1 QPU | 5.56 GFLOP/s (4.000 cycles per instruction at 749.8 MHz) |
| QPU, all 8 QPUs | 44.48 GFLOP/s |

The aide-overlapped run showed 35.0 GFLOP/s on 4 cores, with core 0 at 8.75;
that was contamination. On this board the four CPU cores have more
floating-point throughput than the whole QPU array.

## Memory

| path | GB/s |
|---|---|
| CPU DRAM read, 1 core / 4 cores | 4.5 / 3.9 |
| CPU DRAM write, 1 core | 4.1 |
| QPU TMU read, 8 QPUs (summation) | 11.64 |
| QPU TMU read, 1 QPU | 2.22 |
| QPU TMU write, 1 or 8 QPUs (memset) | 5.67 |
| QPU copy, 8 QPUs, read+write | 6.98 |
| GPU reading while 1 / 3 CPU threads read | 8.25 / 7.55 (CPU gets 2.97 / 2.83) |
| CPU read of a v3d BO mapping, ldp q | **1.476, the same at 16K, 256K and 16M** (the mapping is uncached) |
| CPU write to a v3d BO mapping | 4.16 |

The GPU reads DRAM at 2.6x the rate of all four CPU cores. A CPU must not
bulk-read GPU BOs through the v3d mapping: even a 16K buffer reads at 1.48 GB/s.

## CPU ↔ GPU dispatch (kernel CSD path, empty or 1-word jobs)

- Submit to WAIT_BO return: p50 63–75 µs. Back to back: 24–27k jobs/s (37–42 µs each).
- Submit to the CPU seeing a GPU-written word: p50 **22.6–27 µs or 39.5–44 µs**.
  The minimum is about 21.5 µs.
- The two states are about 17 µs apart. Which one a process gets does not
  depend on the CPU the probe is pinned to (0, 1, 2 and 3 were all tested), and
  it can change partway through a process. The v3d IRQ is always on cpu0. Jobs
  go through unbound `kworker/u16:*-v3d_csd` workers, whose CPU changes from run
  to run. The cause is **not yet measured** (next step: ftrace the submit path).
- In every case the GPU's own work is a few µs. Almost all of the 20–75 µs is
  the kernel submit and complete path.

## GPU hang: a shader must not end with TMU writes in flight

dmesg showed `Resetting GPU for hang. V3D_ERR_STAT: 0x00001000` once per
probe run.

| sequence (one process) | GPU hangs |
|---|---|
| dispatch only (end jobs, then tmuwt "done" jobs) | 0, 0, 0 |
| ALU, then dispatch | 0, 0 |
| summation, then dispatch | 0 |
| scopy, then dispatch (flag inside and outside scopy's range) | 0, 0, 0 |
| **reference memset, then dispatch** | **1, 1, 1** (the first done job hangs; its word is never written) |
| memset + `barrierid(syncb, sig=thrsw)` spliced before the end | 0, 0 |
| memset + `tmuwt` spliced before the end | 0, 0 |
| reference memset, then done *without* tmuwt | 0, 0 |

The reference memset ends right after its TMU store loop. Summation and scopy
end with `barrierid(syncb, sig=thrsw)`, and their source comments that this sync
"is needed between the last TMU operation and the program end". When a shader
ends with writes still in flight, the **next job's `tmuwt` never completes**,
and the scheduler's timeout resets the GPU. Adding the sync does not change
memset bandwidth (5.65–5.67 GB/s).

**Rule for NeoGPU's QPU code generator:** every shader that issues TMU writes
ends with `tmuwt` or a `syncb` barrier before the thread-end sequence.

## CPU micro-architecture (core 2, 1.8 GHz)

Latency in cycles:

| op | cycles |
|---|---|
| add | 1 |
| mul | 5 |
| L1 load | 4 |
| fmadd | 7 |
| NEON fmla | 3 |
| ldar | 7 |
| stlr | 19 |
| ldxr/stxr | 20 |
| dmb | 5 |
| dsb | 30 |
| isb | 27 |

Pointer-chase latency:

| working set | latency |
|---|---|
| ≤16K | 2.2 ns |
| 64–512K | 11–16 ns |
| DRAM, huge pages | 124 ns |
| DRAM, 4K pages | 217 ns |

Other:
- Core to core: 70.0 ns one way, the same for all six pairs.
- Kernel crossings:

| crossing | time |
|---|---|
| getppid syscall | 883–899 ns |
| futex wait/wake round trip | 14.0–16.5 µs |
| vDSO clock_gettime | 46 ns |

v3d BO lifecycle:

| size | create | free |
|---|---|---|
| 4K | 20 µs | 23 µs |
| 16M | 4.8 ms | 0.7 ms |
| 64M | 19.9 ms | 3.4 ms |

Allocate BOs up front; never allocate one per frame.

## Gather: out[i] = table[idx[i]], 4 M elements of 4 B

Raw outputs: `gather/gather-2.txt`, `gather/concurrent-1.txt` and `concurrent-2.txt`.
`gather/gather-1-hung.txt` is the first attempt, whose QPU load form hung the GPU (see
the rule below). Every output word is checked against the CPU result. Rates are
in M elements/s; the GPU number is the best of 1/8 QPUs with 1–8 loads in flight.

| case | best CPU | best GPU | GPU/CPU |
|---|---|---|---|
| random, 512 KiB table (fits CPU L2, not GPU L2T) | 225.6 (4 cores, huge pages, prefetch) | 75.0 (8 QPU) | 0.33 |
| random, 64 MiB table (DRAM) | 43.1 (4 cores, huge pages) | 37.0 (8 QPU, 1 in flight) | 0.86 |
| sequential idx, 64 MiB table | 301.4 (1 core, 4K pages) | 608.8 (8 QPU, 8 in flight; 581.4 before the layout fix) | 2.02 |

- **Random DRAM access has one system-wide ceiling of about 43 M accesses/s,**
  shared by the CPU and GPU:

  | run | random gathers, M/s |
  |---|---|
  | GPU alone | 36.9 |
  | CPU, 3 or 4 cores | 43.0–43.6 |
  | CPU and GPU together | 43.1–43.3 total |

  Two runs agree to within 0.1. Three CPU cores already reach the ceiling. The
  GPU adds no random-access capacity: it only moves the work.
- **Coherent (sequential) access is where the GPU's bandwidth shows:** 2.02x the
  best CPU with the shipped kernel. The original layout slowed down at k=4
  (19.7–21.3 ms, with 2.5–2.7x the DRAM reads). That was root-caused and
  fixed on 2026-11-16 (QPU_ENCODER.md, "The k = 3..5 gather slowdown"). k=4
  now takes 7.87 ms.
- **Reading back cancels the win today.** Consuming the 16 MiB result from the
  GPU BO takes 11.4 ms (1.47 GB/s), against 3.7–3.8 ms from CPU memory. In the
  sequential case: GPU 7.2 + 11.4 = 18.6 ms end to end, against CPU 13.9 + 3.8 =
  17.7 ms. A GPU result must stay on the GPU, or reach the CPU through a
  cacheable path (not yet measured).

## QPU rule: a batch of TMU loads must end with a thread switch

Measured with `ld_*` in `tools/hwprobe_qpu_progs.py`: 1 QPU, one 16-lane load, then `ldtmu`.

| load form | result |
|---|---|
| `mov(tmua, a, sig=thrsw)`; 2 nops; `ldtmu` | correct |
| `mov(tmua, a)`; 2 nops; `ldtmu` | GPU hang |
| `thrsw`, 2 nops before; `mov(tmuau, a)` (config 0xFFFFFFFF); `ldtmu` | GPU hang |

A gather that ends each batch of k ≤ 8 loads with `thrsw` runs correctly (0
hangs in all 24 configurations). The reference summation and scopy loop without
a thread switch on each load; what makes that safe for them was not isolated.
NeoGPU's code generator follows the measured form.

## CPU ↔ GPU inside one running job (mailbox feasibility), measured 2026-11-16

`tools/hwprobe_mailbox` is built only on NeoGPU's own encoder and runtime.
Raw outputs: `tests/evidence-measure-20261116/mailbox/` on the Pi. The
question: can a long-running QPU program exchange messages with the CPU
through shared memory, skipping the 22–75 µs submit path?

| | measured | runs |
|---|---|---|
| **A**: GPU stores, CPU watches the word | The CPU sees **only the final value, at job end**. `tmuwt` does not make a store visible; the stores stay in the GPU's caches until the kernel cleans them after the job | 3 × 100 ticks |
| **B**: GPU polls a word the CPU then writes | The GPU **never sees** the write: 2 M polls over 192 ms at ~96 ns per poll, which is cache-hit speed. Re-run with the write 30 ms after dispatch, safely past start-up: still never seen | 8 + 8, resets counted: 0 |
| **E**: eviction sweep of F KiB between polls | **F ≤ 192 KiB: never seen** (each job polled ~0.74 s after the write). F = 256 / 512 KiB: seen. Latency 222 / 216 µs in one run, 292 / 263 µs in the other | 2 runs; resets counted per row (0) in the second, by dmesg around the first |
| **E**: sweep of up to 512 KiB after each store | Still only the final value reaches the CPU | 9 sizes |
| **C**: one job running longer and longer | 1000 ms ran to completion. A 1500 ms job was **reset at 1012 ms, and all its writes were lost** | 1 series |
| **O**: TMU op code in the `tmuc` config byte (bits 7:4), each op as poll and as store-then-op, with and without `ldtmu` | see below | 2 rounds |

TMU op results:
- **Ops 0–3 hang** in every form tried (6 resets per op per round).
- **Op 1 made stores visible mid-job:** ticks 1 and 2 arrived 1.000 ms apart, at 11 and 12 ms. Then the job hung.
- **Op 4 returns corrupt data.**
- **Op 15** (the default) is unchanged.
- The idea that bits 7:4 hold an op was recalled from Mesa's V3D 4.1 headers and has not been verified. These semantics are not decoded.

What this means:
- **Within a job, the GPU's caches are not coherent with the CPU in either
  direction.** A plain shared-memory mailbox cannot work.
- Eviction gives CPU→GPU coherence only at 216–292 µs per message, worse
  than launching a job. That is 4 samples. Each includes the job's
  teardown, because the CPU sees the ack only at job end, and depends on
  where the write lands in a ~390 µs sweep.
- Nothing measured gives GPU→CPU visibility before job end, except the
  undecoded op 1, which then hangs.
- Any persistent program must also end within ~1 s.

The lead worth taking is decoding the TMU operation semantics, starting with
op 1's mid-job write-back, from Mesa's MIT-licensed hardware description:
read it as documentation, then measure each claim.

## TMU cache operations: a working mid-job mailbox, decoded 2026-11-16

A decoding agent read Mesa's MIT-licensed hardware description as
documentation and measured every claim on the Pi with `tools/tmuop_probe.c`.
The central results were then reproduced independently. Raw outputs are on
the Pi in `tests/evidence-tmuop-20261116/` (the agent's) and
`tests/evidence-measure-20261116/` (the reproduction).

**Setting the config.** Writing an address to `tmuau` takes the next uniform
as that access's config word:

    0xFFFFFF00 | perpixel<<7 | op<<3 | type

- type 7 is 32-bit; op 15 is a regular access.
- The op field is **bits 6:3** (measured): op 8 returned the old value and
  incremented memory, 9 decremented, 10 inverted, and 15 read normally.
- The existing kernels' 0xFC bytes decode as per-pixel, op 15, vec4.
- Every op consumes one uniform.

The earlier `tmuc` experiment (above) wrote `op<<4 | 0xF`. That is per-quad
mode with ops 1, 3, 5, 7 and 9, not 0–4. Its "op 1" was really op 3, which
publishes stores, as it saw. Why those runs hung was not re-tested.

| op | returns data | CPU write → GPU sees it | GPU store → CPU sees it mid-job | other |
|---|---|---|---|---|
| 0 | no | no | no | |
| 1 | no | **yes** | no | discards the GPU's own unwritten store |
| 2 | no | **yes** | **yes** | publishes, then reloads |
| 3 | no | no | **yes** | publishes only |
| 4 | no | no | no | full L1 clear: the stale copy is held beyond L1 |
| 8 / 9 / 10 | old value | n/a | n/a | atomic inc / dec / not |

How the table was measured:
- 2 runs per cell, 0 resets.
- Reproduced independently for ops 1–3: 2 runs each, 0 resets. With op 2
  the CPU saw a GPU store at 0.089–0.091 ms into a 60 ms job; with op 3 at
  0.082–0.094 ms.
- **Reach of an op:**
  - Publishing covers at least the 64-byte line.
  - Reloading covers only the words the op itself addressed, so each lane
    runs the op on its own word.
  - Whether the stale data sits in the TMU's write combiner or in the L2T
    was inferred, not isolated.

**Recipe:**
- **GPU → CPU:** store, `tmuwt`, then op 2 or 3 on the same address via
  `tmuau`.
- **CPU → GPU:** op 2 or 1 on the word via `tmuau`, then a normal load.
- **Never use op 1 on data the GPU has written:** it discards it.

**Ping-pong inside one running job:** the CPU writes a sequence number, the
GPU sees it and acks, the CPU sees the ack.

| | messages | RTT p50 | p99 | max | rate |
|---|---|---|---|---|---|
| decoding agent: 1 word with or without `tmuwt`, and a 16-word payload | 70,000, 0 lost, 0 torn of 30,000 payloads | 0.67–0.98 µs | 1.15–2.41 µs | 31–284 µs | 0.42–1.20 M/s |
| reproduction: op 2, 1 word, `tmuwt` | 3 × 10,000, 0 lost, 0 resets | **0.76 µs** | 1.02 µs | 34–143 µs | 0.55–1.20 M/s |

That is about **30–100x faster than a kernel submit** (22–75 µs).

What it costs on the GPU, and what still binds:
- A coherent poll costs ~330 ns on the GPU, against ~96 ns for a stale,
  cache-hit poll.
- A job is still reset at ~1 s, so a persistent program must finish and be
  relaunched within that.
- Every op consumes a uniform. Rewinding the uniform stream with a
  branch's `ub` field is untested.
- The message rate varied 0.42–1.2 M/s between runs, for an unexplained
  reason. The first run in the reproduction was the slow one.

## The GPU clock drops when the GPU idles (measured 2026-11-16)

Found while red-teaming `hwprobe_qpu`. Raw outputs:
`tests/evidence-measure-20261116/redteam3/`.
- After the GPU has been idle for 200 ms or more, the next job runs on a low
  clock. The scopy kernel took the same 14.41 M GPU cycles either way:
  19.23 ms at 749.6 MHz after a short idle, but 29.2 ms (a 492.7–493.4 MHz
  average) after 200 ms or 1 s of idle.
- Idle gaps of 0, 10 and 50 ms kept the full clock. The threshold lies
  between 50 and 200 ms and was not narrowed further.
- **A 1 ms warm-up job just before restores the full clock**: 749.2–749.3 MHz
  at warm-ups of 1, 5, 20, 50 and 100 ms.
- The destination's prior contents had no effect: fills of 0x00, 0x01, 0xEE
  and 0xFF gave the same times.

What it means:
- The first job after an idle spell pays up to ~1.5x in time. A latency-
  sensitive NeoGPU path should keep the GPU warm, or account for the cost.
- `hwprobe_qpu` now runs a 1 ms ALU warm-up before every timed memory and
  gather job (`HWPROBE_WARM_MS`, default 1, 0 disables). With every word
  checked, it re-measured the published figures unchanged: scopy
  6.98 GB/s, memset 5.68, summation 11.63; gather 612.6 M/s sequential
  (k=8) and 36.0–36.3 M/s random.
- They had been steady-clock numbers all along. The sampled checks used
  before were short enough to keep the GPU warm.
