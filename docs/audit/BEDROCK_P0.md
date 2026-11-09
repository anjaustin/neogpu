# P0: bedrock optimization acceptance

Goal: reduce measured transport CPU work without weakening FIFO delivery,
canonical copies, sequence rejection, generation ownership, draining retirement,
retained views or budget-one fair service. CI remains deferred.

| P0 | Acceptance | Status |
|---|---|---|
| Synchronization overhead | Less measured CPU work and sustained throughput improvement without lifecycle regressions | Split publication promoted; production recheck passed |
| Unnecessary lane probes | Avoid wasted probes without sacrificing fairness or worsening small-frame work | Steady-state consumer mutex probes removed; atomic scanning remains |
| Admission retry cost | Reduce kernel-entry/retry work across load and frame sizes without uncontrolled spinning | Retry frequency reduced; scheduler policy unchanged; FULL retries remain |

P0 denotes priority, not permission to bypass evidence or merge a tradeoff as a
universal improvement. The SPSC reference remains a control, not a replacement
for the participant lifecycle.

First candidate: mutex-protected producers publish an atomic per-lane work hint;
the consumer skips lanes whose hint is clear. The hint must cover queued frames
and pending retirement. Every update is under the existing slot mutex, so a
consumer clearing the hint cannot overwrite a concurrent producer publication.
The hint never authorizes access to non-atomic state: the mutex remains required.
False negatives during a concurrent publication defer work to another pass;
zero progress already does not prove emptiness. No sleeping/wakeup protocol is
introduced. No busy-wait replacement for scheduler yield is proposed.

Unknowns: whether hint sharing offsets avoided lock traffic; sparse versus fully
occupied lane behavior; interaction with input/observer costs; sustained-run
variance. A diagnostic copy must first pass lifecycle/concurrency tests,
ASan/UBSan/leak and TSan, then compare fresh baseline/candidate/reference runs
at one/two/eight producers and one/64 draws. Retain raw runs, source hashes and
failed experiments. Do not infer savings from instruction count alone. Promote
only after correctness and performance evidence; no lifecycle-free substitution.

## Candidate 1: work hint — rejected for general use

`tests/evidence-bedrock-p0-hint/` preserves generated sources, commands,
lifecycle tests, ASan/UBSan/leak, TSan and 54 timing runs. All correctness gates
passed, but the timing matrix failed acceptance. Median applied frames/second:

| Producers | Draws | Baseline | Hint candidate | Pinned SPSC reference |
|---:|---:|---:|---:|---:|
| 1 | 1 | 804,953 | 569,584 | 4,506,066 |
| 1 | 64 | 331,489 | 930,809 | 1,095,568 |
| 2 | 1 | 1,179,891 | 768,894 | 4,523,758 |
| 2 | 64 | 598,006 | 741,986 | 1,112,784 |
| 8 | 1 | 572,470 | 398,562 | 2,840,711 |
| 8 | 64 | 541,451 | 647,440 | 1,026,753 |

Three runs per cell; 200,000 frames per producer. Order reverses on the middle
pair. Work and exact accepted/applied totals match; reference lifecycle remains
less complete. These are same-driver measurements, not production guarantees.
Faster empty-lane scanning is not sufficient evidence of faster end-to-end
transport. Exact causes of the one-draw regression are not established.

## Candidate 2: atomic exclusion gate — rejected

Retain the existing per-lane critical sections and BUSY rejection semantics,
but replace pthread trylock/unlock with a lock-free 32-bit atomic gate. A relaxed
read rejects already-held gates; a strong acquire compare-exchange claims a
free gate; a release store relinquishes it. There is no explicit spin loop,
blocking fallback, changed scheduler policy or lifecycle bypass. Compiler
LL/SC retry behavior is not a fixed instruction-count guarantee.

All authoritative slot state remains guarded, with the same acquire/release
happens-before edges. Join, submit, leave and consumer step use the same gate.
The static lock-free assertion rejects targets needing a library lock. Smaller
slot layout is part of the intervention, so results cannot be attributed solely
to removing unlock's exclusive operation. The held-slot fixture is translated
to acquire/release the diagnostic gate; its isolation assertions are unchanged.
The original baseline and rejected generated sources remain preserved.

### Gate correctness argument to verify against tests

* A successful acquire CAS is the sole entry to the critical section; release
  store is the sole normal exit. The same gate protects lifecycle and data.
* A relaxed read of a held gate may return BUSY immediately; it never grants
  ownership. A failed CAS likewise grants no access to protected state.
* Acquire/release publishes the guarded queue, ownership and closing state.
  Failed calls consume neither sequence nor accepted work.
* Initialization precedes sharing, and destruction still requires external
  quiescence. No dynamic synchronization resource remains to destroy.
* The gate word's operations must be lock-free; the **fabric is still
  lock-based**, not a lock-free queue. A stalled owner can still stall its lane.
* Replacing a public internal field changes the source/layout ABI; all clients
  must rebuild. There is no promised stable or shared-memory fabric ABI.

The original benchmark fixture is pinned with the baseline revision; generated
candidate fixtures change only how the deliberately held critical section is
acquired/released. `test_bedrock_hint.c` separately stresses 1,024 empty
retirements and 50,000 zero-draw drain/republication transitions, without relying
on either internal locking representation.

## Candidates 2 and 3 rejected; next: split publication

The gate alone passed correctness/sanitizers but regressed one-draw medians:
913,431→774,233 fps (one producer), 1,204,750→1,057,903 (two), and
865,713→684,486 (eight). Its one-draw ROI instruction median fell
4653.738→3204.018/frame while cycles were 4635.299→4530.154/frame in a separate
one-million-frame collection. Fewer instructions did not establish a general
throughput win. Artifacts: `evidence-bedrock-p0-gate{,-pmu}`.

Combining gate and hint also failed: one-draw baseline/candidate medians were
911,711/481,485, 1,264,566/722,463, and 979,296/604,223 fps for one/two/eight
producers. Large-frame wins do not justify small-frame regressions. Artifacts:
`evidence-bedrock-p0-gate-hint`. Both remain diagnostic-only.

### Split publication contract (candidate 4)

Retain pthread exclusion among join/submit/leave and retirement, but remove the
consumer from the steady-state slot critical section. Producers publish a ring
write cursor with release after copying; the single consumer acquires it before
reading. The consumer releases a read cursor only after copying; producers
acquire it before reusing storage. Multiple submitting threads remain serialized
by the producer mutex: this is not a lifecycle-free SPSC substitution.

Read/write cursors are monotonic modulo 2^32 and **never reset on join**. Resetting
them would let an idle consumer observe an incoherent reuse epoch. Capacity four
divides the cursor modulus; add a static power-of-two assertion and explicitly
test wraparound. Generation is set before publishing accepted frames and cannot
change until this same consumer retires that drained generation. Closing is an
atomic notification; an empty consumer acquires the lifecycle mutex and rechecks
closing/emptiness before retiring. Queued frames always take precedence.

Invariants to verify: no queue-storage reuse before the consumer's release;
no generation reuse before retirement; no admitted frame skipped on leave;
unchanged validation/sequence error order; zero-draw publication and empty
retirement; multi-submitter serialization; budget-one fairness. Failure or a
workload regression rejects this candidate as well. This contract was written
before its experiments; the results and promotion are recorded below.

## Accepted candidate: split publication

The exact `src/hs_fabric.c` and `include/hs_fabric.h` bytes from
`tests/evidence-bedrock-p0-split-final/candidate/` were promoted after the
extended matrix and targeted sanitizer qualification passed. No scheduler retry
policy, GPU implementation, message transport or input handling was changed.
This optimizes the opt-in participant fabric, not every NeoGPU subsystem.

### Sustained same-driver comparison

One million frames **per producer**, three paired runs per cell. Producer CPUs
are 1–3 (shared when eight producers run); consumer CPU is 0. Native validation,
canonical input copies, per-frame FIFO observation and periodic latency sampling
remain enabled. The pinned reference still lacks full native lifecycle handling.

| Producers | Draws | Baseline median fps [range] | Promoted median fps [range] | Reference median fps |
|---:|---:|---:|---:|---:|
| 1 | 1 | 957,320 [814,204–981,191] | 3,218,302 [3,189,470–3,223,666] | 4,512,312 |
| 1 | 64 | 368,320 [316,470–375,195] | 1,004,516 [977,755–1,015,221] | 1,098,455 |
| 2 | 1 | 1,353,764 [1,231,539–1,355,151] | 3,761,200 [3,560,526–4,510,059] | 3,688,682 |
| 2 | 64 | 573,538 [512,724–582,312] | 1,039,621 [855,656–1,070,231] | 1,032,563 |
| 8 | 1 | 755,704 [727,900–787,056] | 2,587,208 [1,892,996–2,845,954] | 2,930,702 |
| 8 | 64 | 443,063 [368,759–513,539] | 933,411 [811,531–1,042,138] | 940,447 |

All 18 candidate/baseline pairs improved throughput; paired ratios ranged
1.669–3.917. All accepted frames applied with the same canonical contents and
FIFO sequence. This collection applied **198 million** accepted contributions
across all three implementations, not displayed frames. Reference ranges and
per-pair results remain in the machine-readable report; overlapping native and
reference observations do not establish a general win against the reference.

Primary median retries/applied frame fell **0.415450→0.000509**. Eight-producer,
64-draw retries remained **0.279495/frame**, down from 0.961146. Backpressure
has not disappeared, and these aggregate retry counts do not classify FULL/BUSY.

### Latency tradeoff, not a blanket no-regression claim

The report pools periodic samples (every 128th sequence), using nearest-rank
quantiles; these are **not all-frame worst-case latency bounds**. Primary sampled
application p99 fell **6018→760 ns**; sampled application p99 improved in all six
cells. However, two-producer/64-draw application p50 rose **6148→6722 ns** while
its p99 fell 14222→9111 ns. Throughput improved in that cell. This is a measured
latency tradeoff, not a universal improvement in every statistic.

### Separate one-million-frame ROI PMU controls

Three runs per cell, all listed counters running at 100%, no per-yield
tracepoints. Aggregate cycles include producer and consumer, not critical-path
latency. Instrumentation and timing differences are not additive partitions.

| One producer | Baseline | Split publication | Reference |
|---|---:|---:|---:|
| One draw, cycles/frame | 4143.512 | 1120.664 | 798.893 |
| One draw, instructions/frame | 4124.174 | 955.668 | 664.032 |
| One draw, successful exclusive stores/frame | 36.340252 | 2.000558 | 0.000008 |
| 64 draws, cycles/frame | 9835.133 | 3616.523 | 3476.766 |
| 64 draws, instructions/frame | 10302.166 | 3878.540 | 3751.323 |

Source: `tests/evidence-bedrock-p0-split-pmu/manifest.json`. These PMU runs are
separate from the sustained matrix, and have their own fresh baseline/reference.
Consumer locking is structurally removed from normal application; producer and
retirement mutexes deliberately remain. No cache-line ownership-transfer count
or exact energy saving is inferred.

### Verification and reproduction

Before promotion, optimized tests, ASan/UBSan/leak and TSan passed for the
existing FIFO/lifetime/copy/validation/fairness/concurrent-churn suite plus:

* 1,024 retirements with no submitted frame;
* 50,000 zero-draw drain/republication transitions;
* ring cursor wrap through UINT32_MAX, full rejection, FIFO and generation reuse;
* four submitting threads sharing one participant, 20,000 exact FIFO applications.

The production regression gate now includes the two new behavioral tests, and
`make fabric-sanitize` includes both under ASan/UBSan and TSan. A later fixture
also verifies that an already-published entry can apply while its producer mutex
is held, while leave correctly reports BUSY. Production checks are recorded
separately rather than inferred from diagnostic success.

### Post-promotion production checks

`make check-sources`, `make test`, `make fabric-sanitize` and the instrumented
driver's ASan/UBSan/leak/TSan qualification passed against the promoted source.
Runtime source/header bytes matched the extended-matrix candidate at promotion.
The subsequent red-team pass added compile-only atomic/capacity guards and
verified identical compiled `.text`; see [BEDROCK_REDTEAM.md](BEDROCK_REDTEAM.md).

Three fresh same-run 100,000-frame software-count/PMU reconciliations measured:

| Twice successful producer mutex acquisitions | Successful exclusive stores | Residual |
|---:|---:|---:|
| 200,796 | 200,804 | 8 |
| 201,934 | 201,942 | 8 |
| 202,200 | 202,208 | 8 |

Native BUSY admissions were zero in these one-producer runs; FULL admissions
were 398, 967 and 1,100 respectively. Yield tracepoint counts match the exact
software retry counts. Reference measured the same residual eight stores.
The residual remains unattributed. These instrumented timings are not used for
rate claims. Sources: `tests/evidence-bedrock-p0-accounting/`.

Actual GLES checks also passed: six retained-view/retirement/rejoin stages,
18 presentations, 36 RGB checks, five applications and one retirement. The
direct-GLES comparison verified **1,536,000 identical RGBA bytes** across the
whole 800×480 frame. New captures use a fresh evidence directory rather than
overwriting earlier graphics evidence. Logs/captures are in
`tests/evidence-bedrock-p0-production/`. These checks establish rendering
correctness for those scenes, not an end-to-end graphics speedup.
The six new captures were also visually inspected in a local contact sheet:
red+green, red only, red+green, green only, green only, then blue+green.

Raw P0 artifacts (including all rejected candidates) were archived in the local
snapshot as `tests/evidence-bedrock-p0-archive.tar.gz`: 69,883,620 bytes, 869
regular files, SHA-256
`90f41cd71d397f7dfec7bc75d1dae532dbbc95f2d88893a50bd2964fdfeae28a`.
The archive was fully decompressed for verification. Final documentation and
the contact sheet are retained in a separate final supplement.

```sh
python3 tests/test_bedrock_p0.py --candidate split --frames 1000000 \
  --output tests/evidence-bedrock-p0-repeat
python3 tests/test_bedrock_p0_report.py tests/evidence-bedrock-p0-repeat
python3 tests/test_bedrock_p0_pmu.py --source tests/evidence-bedrock-p0-repeat \
  --output tests/evidence-bedrock-p0-repeat-pmu
make test fabric-sanitize
```

The baseline source/held-lock fixture is frozen at `2f72c91`; output directories
must be fresh. Final timing runs retain per-command frequency/governor/thermal
snapshots and common-source hashes. Earlier rejected gate experiments have less
complete environment provenance. Raw sources, failed candidates, counts,
commands and capture evidence remain separate. No commit, push or CI work is
included in this batch.

For a fresh checkout without the pinned reference, first run
`python3 tests/test_comparison_evidence.py --output tests/evidence-comparison-final --runs 3 --frames 100000`.
Alternatively provide a preserved reference root with `--reference PATH` to both
P0 runners; header and license hashes are verified. The PMU ROI driver is now
generated from checked-in source, so no historical generated-driver directory
is required. Subsequent red-team remediation/commit work is recorded in
[BEDROCK_REDTEAM.md](BEDROCK_REDTEAM.md).
