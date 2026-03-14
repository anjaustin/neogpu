# NeoGPU Atomics Profile - 2026-03-13

**Context:** Pi 4 profiling pass focused on message fabric atomics, queueing, and step-thread drain cost.

## Goal

Identify where compute time is being spent in the channel fabric so optimization work can be applied in a measurable, reversible order.

## Profiling Setup

- Host: Raspberry Pi 4
- Build: `-O2 -g -pg`
- Binary: `neogpu_prof`
- Tool: `gprof`
- Workload: full demo/test binary with producer benchmarks enabled

## Flat Profile Highlights

Top self-time samples:

- `__aarch64_ldadd4_relax`: 30.8%
- `hs_step`: 21.5%
- `hs_route_immediate`: 19.2%
- `__aarch64_cas4_relax`: 8.1%
- `mq_pop`: 4.1%
- `shader_node_process`: 4.1%
- `hs_render_record`: 2.9%
- `hs_send`: 1.7%

## Main Finding

The hottest cost is not domain logic. It is shared atomic bookkeeping.

A large fraction of CPU time is spent in AArch64 atomic read-modify-write instructions:

- `ldadd` from frequent counter increments on hot send paths
- `cas` from queue claiming and high-water tracking on fallback MPSC paths

## Code Paths Implicated

### 1. Hot-path stats updates in `hs_send`

`src/hs_core.c:660-677`

Per-message atomic increments are done for:

- `spsc_ok`
- `spsc_ok_by_prod`
- `spsc_full`
- `spsc_full_by_prod`
- `telem_dropped`

These are on the send fast path and scale directly with message rate.

### 2. Fallback MPSC enqueue path

`src/hs_core.c:265-307`

Costs include:

- CAS loop on `enqueue_pos`
- atomic increment of `mpsc_ok`
- extra CAS loop for `submit_hw`
- atomic increment of `submit_full`

This is especially expensive under contention or when producers spill out of SPSC lanes.

### 3. Step-thread SPSC scanning

`src/hs_core.c:1141-1188`

`hs_step()` repeatedly:

- loads `producer_count`
- scans each producer lane for each channel
- atomically loads `head` and `tail`
- routes messages one at a time into `hs_route_immediate()`

This is not the top atomic bottleneck, but it is the biggest non-intrinsic runtime bucket once the atomic instructions themselves are accounted for.

## Producer Benchmark Interpretation

The producer benchmark in `src/main.c:1296-1358` amplifies the exact hot paths we care about:

- many calls to `hs_send`
- heavy SPSC success-path accounting
- fallback MPSC contention when SPSC lanes saturate or spill

So the profile is representative for compute spent in the fabric.

## Ordered Optimization Plan

### Item 1 - Reduce hot-path atomic counter overhead

First target:

- move hot send counters off the per-message shared atomic path
- prefer per-thread or per-producer accumulation with periodic aggregation
- if needed, add a lower-overhead perf mode for detailed fabric counters

Expected payoff:

- lower `ldadd` overhead immediately
- better scaling in the common SPSC fast path

### Item 2 - Reduce submit high-water CAS overhead

Second target:

- stop updating `submit_hw` via CAS on every fallback enqueue
- update it by sampling, batching, or step-thread-side observation instead

Expected payoff:

- lower `cas` overhead in the MPSC fallback path

### Item 3 - Reduce step-thread scan cost

Third target:

- avoid scanning inactive producer/channel lanes every step
- consider active producer bitmaps or per-channel active masks

Expected payoff:

- reduce `hs_step()` cost under mixed producer counts

## Red-Team Validation Plan

After each optimization item:

1. rebuild and run the full test/demo binary
2. rerun the producer benchmark profile build
3. compare throughput and flat-profile percentages
4. record regressions, edge cases, and telemetry correctness concerns
5. commit and push only after the red-team pass is documented

## Success Criteria

- lower atomic intrinsic share in `gprof`
- equal or better producer benchmark throughput
- no regression in tests or IPC/tooling behavior
- no loss of correctness in queue semantics or result delivery

## Item 1 Result - Reduce Hot-Path Atomic Counter Overhead

### Change

Implemented in:

- `src/hs_core.c`
- `src/hs_nodes.c`
- `src/main.c`

Details:

- removed shared per-channel `spsc_ok` and `spsc_full` atomic increments from the send fast path
- kept per-producer SPSC counters as the source of truth
- changed producer-owned SPSC counter updates from atomic fetch-add to producer-local running totals published with atomic store
- updated fabric query and benchmark reporting to aggregate from per-producer counters instead of shared channel counters

### Red-Team Validation

Validation steps:

- rebuilt `neogpu_demo`
- ran full test/demo binary successfully
- rebuilt `neogpu_prof`
- reran `gprof`
- compared benchmark throughput against the earlier baseline run

### Profiling Delta

Before item 1:

- `__aarch64_ldadd4_relax`: 30.8%
- `__aarch64_cas4_relax`: 8.1%

After item 1:

- `__aarch64_ldadd4_relax`: 15.4%
- `__aarch64_cas4_relax`: 31.3%

Interpretation:

- shared atomic add pressure was reduced substantially
- once that cost dropped, the fallback queue CAS path became the dominant bottleneck
- this validates the original item ordering: item 2 should now target the MPSC high-water/CAS path directly

### Throughput Delta

Reference baseline from earlier run:

- 1 thr: 8.66M msg/s
- 2 thr: 8.17M msg/s
- 4 thr: 9.97M msg/s
- 8 thr: 8.78M msg/s
- 16 thr: 5.83M msg/s

After item 1:

- 1 thr: 11.13M msg/s
- 2 thr: 16.85M msg/s
- 4 thr: 7.75M msg/s
- 8 thr: 6.87M msg/s
- 16 thr: 6.14M msg/s

### Red-Team Conclusion

Item 1 is a mixed result, but still useful:

- clear win at low producer counts
- slight win at 16 threads
- regression in the 4-thread and 8-thread ranges because contention shifts harder into the fallback CAS path

Conclusion:

- keep item 1 because it removes a proven hot atomic-add cost
- proceed immediately to item 2 to attack the newly exposed dominant CAS bottleneck

## Item 2 Result - Remove Per-Enqueue `submit_hw` CAS Tracking

### Change

Implemented in:

- `src/hs_core.c`

Details:

- removed `submit_hw` compare-exchange updates from `hs_submit_enqueue()`
- moved submit queue depth observation into the single step thread
- high-water tracking is now updated from `hs_step()` with load/store semantics instead of producer-side CAS contention

### Red-Team Validation

Validation steps:

- rebuilt `neogpu_demo`
- reran the benchmark suite
- rebuilt `neogpu_prof`
- reran `gprof`
- compared item 2 results against item 1

### Throughput Delta vs Item 1

Item 1:

- 1 thr: 11.13M msg/s
- 2 thr: 16.85M msg/s
- 4 thr: 7.75M msg/s
- 8 thr: 6.87M msg/s
- 16 thr: 6.14M msg/s

Item 2:

- 1 thr: 11.04M msg/s
- 2 thr: 15.57M msg/s
- 4 thr: 8.49M msg/s
- 8 thr: 7.18M msg/s
- 16 thr: 6.23M msg/s

Interpretation:

- near-neutral at 1 thread
- slight regression at 2 threads
- solid recovery at 4, 8, and 16 threads
- this matches the hypothesis that item 2 helps once fallback queue contention becomes the bottleneck

### Profiling Delta vs Item 1

Item 1 profile:

- `__aarch64_cas4_relax`: 31.3%
- `__aarch64_ldadd4_relax`: 15.4%
- `hs_step`: 13.9%

Item 2 profile:

- `__aarch64_cas4_relax`: 32.6%
- `__aarch64_ldadd4_relax`: 16.4%
- `hs_step`: 10.2%

Interpretation:

- flat-profile percentages remain noisy because the remaining fallback queue CAS operations still dominate the profile
- step-thread cost fell materially, which is consistent with moving high-water bookkeeping off the producer path
- benchmark throughput is the more convincing signal for this item: medium/high contention improved

### Red-Team Conclusion

Item 2 is a net win and should be kept.

- it improves the contention-heavy ranges that item 1 exposed
- it simplifies producer-side hot code
- it does not change queue semantics, only where telemetry-style high-water accounting is maintained

Conclusion:

- keep item 2
- proceed to item 3 to reduce repeated step-thread scanning work

## Item 3 Result - Active Producer Masks in `hs_step()`

### Change Attempted

Attempted in:

- `include/hs_core.h`
- `src/hs_core.c`

Approach:

- track per-channel active producer masks
- mark producers active on successful SPSC push
- have `hs_step()` scan only the active producers instead of all producer slots

### Red-Team Validation

Validation steps:

- rebuilt `neogpu_demo`
- reran the benchmark suite
- rebuilt `neogpu_prof`
- reran `gprof`

### Result

This change regressed throughput badly and was reverted.

Observed benchmark result for the attempted version:

- 1 thr: 7.49M msg/s
- 2 thr: 9.34M msg/s
- 4 thr: 6.07M msg/s
- 8 thr: 5.43M msg/s
- 16 thr: 4.70M msg/s

Profile signal from the rejected version:

- new hotspot: `__aarch64_ldset4_relax` at 16.1%
- `hs_step()` share did not improve enough to justify the new atomic bitset churn

### Failure Analysis

The active-mask idea introduced new atomic set/clear traffic on every producer activation and idle transition.

That replaced scan overhead with another contested synchronization path:

- producer-side `fetch_or`
- step-side `fetch_and`
- race-repair logic that re-set the bit when a producer became active again during drain

Net effect:

- more atomic churn
- worse throughput across all tested thread counts

### Conclusion

Reject item 3 in this form.

Do not keep the active producer mask implementation.

## Next Investigation Direction

A better step-thread optimization should avoid introducing new shared atomics.

Promising alternatives:

- increase SPSC burst size adaptively when a producer is clearly hot
- reduce per-message work in `hs_route_immediate()` and node processing
- revisit benchmark topology to reduce fallback MPSC pressure before adding more coordination state

## Item 4 Result - Adaptive SPSC Burst Sizing

### Change Attempted

Attempted in:

- `src/hs_core.c`

Approach:

- increase the SPSC drain burst size in `hs_step()` based on queue depth
- use larger bursts for visibly hot producer lanes instead of the fixed `32` message chunk size

### Red-Team Validation

Validation steps:

- rebuilt `neogpu_demo`
- reran the producer benchmark suite
- rebuilt `neogpu_prof`
- reran `gprof`

### Result

This change regressed throughput severely and was reverted.

Observed benchmark result for the attempted version:

- 1 thr: 2.65M msg/s
- 2 thr: 3.03M msg/s
- 4 thr: 2.19M msg/s
- 8 thr: 1.90M msg/s
- 16 thr: 1.18M msg/s

Profile signal from the rejected version:

- `hs_step` rose to 21.8%
- `hs_route_immediate` remained the dominant worker cost at 30.2%
- fallback CAS pressure was still present, but the larger bursts made scheduling/fairness behavior much worse under this benchmark

### Failure Analysis

The larger SPSC bursts let hot producers monopolize more of the channel budget before fallback queues were drained.

Net effect:

- worse fairness across producers and queue sources
- more time stranded in the step loop before competing work got service
- significantly lower overall throughput under the benchmark topology used here

### Conclusion

Reject item 4 in this form.

Do not keep adaptive burst sizing as a simple queue-depth-to-burst heuristic.

## Next Investigation Direction

The strongest remaining lever is likely inside `hs_route_immediate()` and nearby per-message work rather than queue-drain geometry.

Promising next candidates:

- reduce validation/logging/recording branches inside `hs_route_immediate()`
- split the render fast path from the fully general path
- reduce node inbox traffic or collapse trivial node processing for benchmark-style ops
