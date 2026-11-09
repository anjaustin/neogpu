# Bedrock red-team acceptance

Scope: all uncommitted bedrock diagnostics, documentation and participant-fabric
optimization, on top of `2f72c91`. Preserve unrelated work and failed experiments.
No CI/security/governor changes or history rewriting. Commit verified changes,
then attempt the ordinary push; authentication restrictions are not bypassed.

Success: no unresolved findings within the reviewed scope; deterministic
publication/reuse/retirement race tests with mutation controls; regression,
strict compilation, sanitizer and actual GLES evidence; repeatable performance
comparison and trustworthy unsupported-counter handling. This is not a proof
against all schedules or a hard-real-time latency guarantee.

Invariants: copied FIFO delivery, validation/sequence ordering, multiple callers
per handle, generation isolation, drain-before-retirement, storage reuse only
after consumer release, and budget-one fairness. Production changes require
fresh evidence or demonstrated executable equivalence for compile-only guards.

Open questions: untested partial perf ACK reads, malformed scheduler headers,
PMU scaling acceptance, cached timing/provenance consistency, and adversarial
interleavings between cursor snapshots and retirement/publication. Findings and
their verification will be recorded below before commit.

## Findings and remediations

1. **FIFO ACK framing:** a short read could reject a valid response. Read the
   byte stream through newline, tolerate one optional trailing NUL from the
   preceding ACK, and reject partial environment configuration. Seven valid/
   malformed fragmented-response cases plus missing-environment controls verify
   behavior. Neither missing nor malformed control silently disables ROI gating.
2. **Skipped scheduler records:** reject malformed recognized-event headers,
   but accept perf's valid `-1` event-context TID. Decode real task PIDs from the
   payload. Positive/negative header controls exposed an additional historical
   undercount: native producer switch-outs are **12**, not 11; reference **five**,
   not four, for 7,375/924 yields. Duplicate/missing thread metadata is rejected.
3. **PMU scaling/availability:** require reported 100.00% scheduling before
   accepting an unmultiplexed counter. Reject nonfinite/negative values, zero
   runtime, malformed/duplicate rows and missing required P0 events. Unsupported
   values remain unavailable, never zero. Reported precision is not a claim
   about an unrounded kernel scheduling fraction.
4. **Report trust:** recompute cached rates from integer work/time, require a
   complete paired matrix and equal frame work, cross-check raw command status
   and summary fields. Synthetic wrong-rate, contradictory raw-summary,
   nonfinite metadata and incomplete-matrix controls must fail. Hash raw trace
   text and parser helpers in the derived evidence manifest.
5. **Assertions disabled:** evidence scripts relying on assertions now refuse
   optimized Python execution instead of silently bypassing verification.
6. **Evidence overwrite:** completed bedrock phases and existing command records
   cannot be overwritten by re-running into an old collection.
7. **Hardware inventory width:** declare `getauxval`'s unsigned-long argument and
   return types, avoiding ctypes' default int truncation of high HWCAP bits.
8. **Reproduction dependency:** generate ROI drivers from checked-in source
   through shared exact-anchor transformations, not an ignored historical output.
   P0 runners accept an explicit reference location and pin both header/license.
9. **Concurrency coverage:** force pauses immediately before publication, after
   an empty snapshot but before closing/retirement, and after copying but before
   releasing queue storage. The correct implementation passes optimized,
   ASan/UBSan/leak and TSan runs. Deliberately releasing storage early or omitting
   the retirement emptiness recheck must fail the intended assertions (not time
   out or fail to compile). Existing wrap/reuse/shared-owner tests remain.
10. **Compile-time bounds:** assert positive bounded power-of-two capacity and
    lock-free closing notification, in addition to cursor atomics. These guards
    do not alter runtime instructions: qualified and current fabric `.text`
    both hash to `a01648601fe3ed293128c98afbe1f81a39916f758eb1b7b5aa0416d405265608`.

No data race or lifecycle defect was found in the reviewed runtime protocol.
That is a scoped audit result, not a proof of all possible schedules. Three
earlier performance-regressing alternatives remain rejected and preserved.

## Repeated measurements

Fresh one-million-frame-per-producer runs, three pairs per cell, retain FIFO,
canonical-copy observation and the SHA-pinned optimized SPSC reference:

| Producers | Draws | Baseline median fps | Optimized median fps | Reference median fps |
|---:|---:|---:|---:|---:|
| 1 | 1 | 911,180 | 3,189,914 | 4,565,272 |
| 1 | 64 | 327,564 | 1,013,719 | 1,104,597 |
| 2 | 1 | 1,845,327 | 5,064,510 | 5,435,073 |
| 2 | 64 | 781,890 | 1,152,904 | 1,058,858 |
| 8 | 1 | 1,575,851 | 4,429,618 | 4,070,011 |
| 8 | 64 | 970,052 | 1,119,989 | 1,004,856 |

All 18 pairs improved throughput (candidate/baseline ratios 1.113–4.214), with
198 million accepted contributions applied across the matrix. Between-collection
variation is substantial; this does not establish a universal multiplier or
general superiority over the reference. Full ranges, paired rates, sampled
latencies and retries: `tests/evidence-bedrock-redteam-performance-report.json`.

The latency tradeoff persists: two-producer/64-draw sampled application p50 rose
6296→7648 ns, and eight-producer/64-draw p50 rose 25834→30370 ns. The latter's
admission p50 rose 907→944 ns. Sampled application p99 improved in every cell,
including 13203→9259 ns and 44537→40426 ns for those two workloads. These are
periodic pooled samples, not all-frame latency or worst-case guarantees. The
optimization is retained for the demonstrated throughput/CPU-work improvements,
not claimed to improve every latency statistic; queue capacity and admission
semantics were not weakened to make the latency numbers look better.

Fresh simultaneous software-count/PMU accounting again measured producer-mutex
budgets 200,172/200,144/200,280 versus successful exclusive stores
200,180/200,152/200,288: residual eight each. Reference again measured eight.
Yield counts match software retries. Instrumented timings are not rate evidence.
Historical PMU/count/ablation/syscall numeric summaries are unchanged under the
strict parser; only the explicitly corrected scheduler exit counts changed.

The repaired self-contained ROI runner completed 36 fresh PMU executions, all
required counters reporting 100.00% scheduling. Primary aggregate cycles/frame:
3978.614 baseline, 1112.557 optimized, 795.984 reference; instructions/frame:
3887.347, 954.898, 653.761. Successful exclusive stores/frame: 34.287654,
2.001820, 0.000008. These are separate counter collections, not an additive cost
decomposition or an end-to-end graphics speedup.

## Final gate

Final strict C compilation, `make test`, `make fabric-sanitize`, Python syntax,
tool negative controls and adversarial mutation controls passed. Reference
license tampering was rejected by both P0 runners; a repeated completed phase
was rejected without changing its manifest. The generated ROI driver matches
the previously measured driver's bytes without requiring that artifact as input.
The accepted split candidate is now the P0 runner's default; rejected candidates
remain explicitly selectable for reproduction.

Actual GLES again passed six lifecycle stages/18 presentations/36 RGB checks,
and the direct comparison verified 1,536,000 identical full-frame RGBA bytes.
Raw results: `tests/evidence-bedrock-redteam-production/`,
`tests/evidence-bedrock-redteam-adversarial/`,
`tests/evidence-bedrock-redteam-reproduction/`,
`tests/evidence-bedrock-redteam-pmu/` and the performance/report collections
above. Evidence archival and repository checks precede commit. The ordinary
push result is reported explicitly; token permission failure is not bypassed by
changing workflows, credentials or history.

All six new graphics captures were visually inspected as a contact sheet and
match the expected retention/retirement/rejoin sequence. The complete red-team
archive in the local snapshot is `tests/evidence-bedrock-redteam-archive.tar.gz`
(37,739,719 bytes, 854 regular files), SHA-256
`c9d9c33eaecbb33b977d4c09a2ff0a711ccdfe048c314b98e9a241482ae39390`.
It was fully decompressed for verification. The final report/contact sheet are
preserved separately so the archive does not need to contain its own hash.
