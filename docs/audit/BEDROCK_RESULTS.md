# Transport bedrock: measured costs and remaining gaps

Baseline: `2f72c91`. Production `src/` and `include/` were unchanged during this
measurement batch; subsequent optimization is recorded in [BEDROCK_P0.md](BEDROCK_P0.md). These are
CPU transport/application measurements, not GPU completion, IPC or presentation
measurements. See [the measurement contract](BEDROCK_PLAN.md).

## Main result

The largest demonstrated structural expense is repeated mutex acquisition,
including consumer probes of empty/inactive lanes. Kernel-entering producer
retries are another measured expense. Neither observation licenses removing
lifecycle guarantees or replacing the retry policy without further tests.

Primary workload: producer pinned to CPU 1, consumer to CPU 0, eight lane
positions, capacity four, one draw/frame. Both implementations apply the same
valid work FIFO with canonical contents. The reference is unmodified Rigtorp
SPSCQueue revision `c8e9008dc4639220cc9bcf4c84abd2ff0d344aa5`; its static adapter
does **not** implement the native participant lifecycle.

### Aggregate CPU work per applied frame

Three one-million-frame runs per implementation, ROI-gated counters, without
per-yield tracepoints; all counters below ran at 100%. Cycles count both threads,
including kernel execution, and are not a single-call latency.

| Metric | Native median [range] | Reference median [range] |
|---|---:|---:|
| Cycles/frame | 4002.753 [3733.361–4413.055] | 806.421 [789.328–835.402] |
| Instructions/frame | 3904.440 [3687.354–4207.299] | 661.506 [648.947–669.845] |
| Branch misses/frame | 16.682 [15.658–18.595] | 0.550 [0.523–0.699] |

Source: `tests/evidence-bedrock-untraced/pmu.json`, `core-untraced`, control.
These differences are measured total work, not an additive attribution of costs.

## Ranked findings

### 1. Lock traffic is proven; its complete latency cost is not

The instrumented workload reconciles:

* attempts = admitted + BUSY + FULL;
* retries = BUSY + FULL = producer yields;
* consumer visits = busy + empty + applied;
* inactive visits = seven times active visits in this one-participant workload.

In simultaneous software-count/PMU runs of 100,000 frames, native successful
mutex acquisitions predict two successful exclusive stores each:

| Predicted mutex stores | Measured successful exclusive stores | Residual |
|---:|---:|---:|
| 4,106,042 | 4,106,050 | 8 |
| 3,451,552 | 3,451,560 | 8 |
| 3,248,202 | 3,248,210 | 8 |

Reference runs have zero queue mutex acquisitions and eight successful exclusive
stores each. The common residual is **unattributed**, not rounded away.
Count instrumentation changes pacing; its timings are not throughput evidence.
Separate uninstrumented exclusive-event runs measured median 33.439 successful
exclusive stores/frame native versus eight total reference. Those separate runs
must not be substituted into the exact same-run accounting above.

The Pi lacks HWCAP_ATOMICS (HWCAP 2183); LSE is unavailable. Exclusive events
are speculative PMU events, not high-level rejection counts or proof of which
cache line caused a failure. Empty-lane locking is a priority for a future
correctness-preserving experiment, not an installed fix.

### 2. Yield calls cost work but usually do not switch tasks

A corrected CPU-0/1 systemwide trace recorded native producer 7,375 yields and
12 switch-outs; reference producer 924 yields and five switch-outs. No observed
migrations or reported lost events; trace yield counts match diagnostic counts.
Switch-ins need not equal switch-outs at ROI boundaries.

Isolated, genuinely untraced wall controls (three runs, one million iterations)
measured median `getpid()` **873.424 ns/call**, `sched_yield()` **1635.348 ns/call**.
These are loop-average costs on this machine, not queue cost partitions. ARM
`YIELD` and empty-loop averages sit near the loop floor; they do not establish
individual instruction latency. Tracepoint-instrumented syscall timings are
kept separately because tracing itself adds measurable work.

Replacing scheduler yield with ARM `YIELD` is **not a general fix**:

| Native diagnostic | Scheduler yield median fps | ARM YIELD median fps |
|---|---:|---:|
| One draw | 937,026 | 673,476 |
| 64 draws | 377,066 | 758,767 |

Larger one-million-frame PMU trials also exposed severe spinning costs. No
production backoff change was retained.

### 3. Refills are measured; the kernel-path explanation remains incomplete

Separate user-mode PMU groups measured median native/reference events per frame:

| Event | Native | Reference |
|---|---:|---:|
| L1D TLB refill | 4.775973 | 0.013430 |
| L1I cache refill | 4.234981 | 0.012585 |
| L1I TLB refill | 9.748996 | 0.000735 |
| L2D refill | 0.003197 | 0.002489 |

The low L2 refill rate does not support treating the gap as principally DRAM
traffic. Syscall and backoff controls test kernel-entry involvement, but do not
identify every refill's cause. Aggregate bus/cache counters do not identify
individual coherence ownership transfers. Kernel symbol access was restricted.

Kernel configuration includes `CONFIG_UNMAP_KERNEL_AT_EL0=y` and
`CONFIG_ARM64_SW_TTBR0_PAN=y`; configuration alone does not establish active
runtime paths. Sysfs reports Spectre v2 and speculative-store-bypass Vulnerable,
Meltdown Not affected. No mitigation or frequency setting was changed.

### 4. Driver work interacts with queue pacing

Prebuilding inputs improved native one-draw median from 901,731 to 1,113,101 fps
but reduced reference from 4,508,745 to 2,220,852 fps in the same collection.
Removing work can worsen this coupled producer/consumer system. Observer/input
ablation differences cannot be summed into an exact decomposition. Future
optimization acceptance needs workload and run-length controls, not one median.

## Hardware, layout and measurement floor

* Cortex-A72: 64-byte lines, private 32 KiB L1D/48 KiB L1I, shared 1024 KiB L2.
* Architectural `CNTFRQ_EL0`: **54,000,000 Hz**. A successful `CNTVCT_EL0` probe
  compared 546,727 ticks with 10,104,778 ns monotonic elapsed; the brackets differ.
  `clock_getres()` reports 1 ns, not physical counter granularity or accuracy.
* Capacity probes scheduled cycles plus six listed events at 100%; adding a
  seventh event made the group unschedulable. This is an observed configuration
  limit, not a universal PMU guarantee. Main measurements use smaller groups.
* Inventory glob probes found no exposed power/energy sensors, ARM SPE or
  CoreSight ETM PMU devices at the queried paths. No energy estimate is claimed.
* `sizeof(HSFabric)=83416`, slot stride 8336; mutex/active/admitted/head/count/queue
  offsets 0/48/64/72/76/80. Observed slot-zero base modulo 64 was 32.
  Layout establishes placement, not measured false-sharing cost.

Actual native/reference executable and libc disassembly, libc build identity and
SHA-256 are retained in `tests/evidence-bedrock-inventory-final/`.

## Verification, corrections and reproduction

Generated diagnostic source copies leave production untouched. ASan/UBSan/leak
and TSan qualification passed for instrumented native/reference paths. Exact
delivery/count reconciliation, decoder controls, `make check-sources` and
`make test` passed. These are transport diagnostics; this batch does not claim
new graphics acceptance or CI coverage.

Two instrumentation mistakes remain part of the evidence trail:

1. Initial ROI runs aborted because this perf version sends an ACK with a
   trailing NUL. The helper now consumes the full four/five-byte response.
   `initial`/`v2` failed artifacts are not accepted PMU measurements.
2. The original scheduler parser expected keyed PIDs; perf printed pretty
   `task:PID … ==> task:PID`. Cached zero switch counts are invalid. The report
   re-decodes raw scripts with keyed/pretty/colon-name/rejection controls.
3. Subsequent red-team review found valid `-1` perf event-context headers on
   task exit. The payload still carries the real task PID. Decoding these adds
   one producer switch-out to each validation trace (12 native, five reference),
   correcting the previously reported 11/four. See [the red-team report](BEDROCK_REDTEAM.md).

`tests/test_bedrock_report.py` recomputes the report from raw collections:

```sh
python3 tests/test_bedrock_report.py \
  --primary tests/evidence-bedrock-final \
  --deep tests/evidence-bedrock-deep \
  --backoff tests/evidence-bedrock-backoff \
  --syscalls tests/evidence-bedrock-syscalls-final \
  --inventory tests/evidence-bedrock-inventory \
  --extra tests/evidence-bedrock-untraced \
  --output tests/evidence-bedrock-report-recomputed.json
```

The later `inventory-final` supplements rather than rewrites original provenance.
Early collections have less complete environment/header provenance than later
ones. Commands, statuses, generated sources, hashes, counter scaling and raw
traces remain on the Pi; a local compressed archive preserves the collections.

## Next experiments, not conclusions

First test eliminating inactive-lane mutex probes while preserving publication,
retirement, stale-handle rejection and fair service. Then test a bounded retry
policy across one/64 draws, producer counts and longer runs against fresh
controls. Measure exact accounting, sanitizers and full lifecycle behavior before
throughput acceptance. A static SPSC win is not proof of a drop-in lifecycle-safe
replacement. No optimization, commit, push or CI work is included in this batch.
