# Transport bedrock: measurement contract

Runtime baseline: `2f72c91`. Success is an evidence-backed cost model, not a
predetermined optimization. Production src/include must remain byte-identical.
All accepted work must apply FIFO with canonical contents; no probe may silently
relax lifecycle/ownership guarantees or discard failed experiments.

## Questions, uncertainty and verification

1. **Hardware/OS boundary:** record actual CPU/HWCAP, cache topology, affinity,
   governor/frequency/thermal snapshots, PMU scheduling capacity, permitted raw
   events and tracepoints. Confirm the outlined-atomic path rather than assuming
   LSE support. Distinguish unsupported/permission-denied counters from zero.
2. **Unmodified timing/cost:** compare native and pinned optimized SPSC reference
   with the same driver, pinned placement, input, observation and exact delivery.
   Record cycles, instructions, cache/TLB refills, branch misses, exclusive
   events where available, context switches, migrations and faults. Use small
   event groups and preserve time-enabled/running so multiplexed estimates are
   not mistaken for exact counts. Normalize per applied contribution, not retry.
3. **Rejection causes and service behavior:** generated instrumented source
   copies count producer BUSY vs FULL, active/inactive consumer probes, busy vs
   empty visits, successful applications, and driver yields per thread. No shared
   hot instrumentation counters: writer-owned counters are read after joins.
   Count-only timings are not performance estimates. Reconcile admissions,
   applications, attempts and yields, including the reference's full-queue path.
4. **Scheduler:** trace actual sched_yield calls and actual switch/migration
   events separately where accessible. A yield does not imply a context switch.
   Check trace loss; do not count inherited task events as solely worker events.
5. **Layout/ISA:** retain offsetof/sizeof/cache-line offsets, allocation addresses
   in the diagnostic, disassembly of actual adapter/driver/libc code and libc
   identity. Separate true sharing (mutex/count) from accidental sharing. A
   refill is not a direct measurement of coherence ownership transfers.
6. **Ablations:** use generated copies only, preserve full object layouts, and
   test at least active-lane pacing / backoff and common observer/input-building
   costs. Interventions must be named and measured against a fresh same-harness
   control/reference. Include failed/time-limited runs; qualify correctness
   before collecting rates. No independent timing differences are assumed to
   add into an exact fivefold decomposition.

## Scope and guardrails

Primary case: one producer / one draw / four queued frames / eight lane slots.
Use larger-draw and/or multi-producer controls only to test specific hypotheses,
not to explode the matrix indiscriminately. Preserve raw JSONL, source hashes,
exact commands/exit statuses, event scaling and environment. Cap experiments
with explicit timeouts; earlier single-slot starvation is evidence, not a fix.
Keep the optimized Rigtorp reference and all common correctness checks unless a
specifically labelled observer-cost diagnostic replaces them with an independently
validated observer. No production optimization is retained in this batch.

Unavailable bus/coherence tracing, exclusive-failure causes, or exact per-cycle
attribution remain explicit gaps. PMU EXCL failure events may aggregate several
causes and cannot alone identify a particular lock/cache line. This is CPU
transport/application analysis, not GPU, IPC or physical panel throughput.
