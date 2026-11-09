# Frame-runtime integration: second red-team pass

## Review contract

Success means no confirmed unresolved defect in the new lifecycle/composition
contract, and evidence that rejects mislabeled or corrupted measurements.
Preserve legacy APIs, FIFO/generation/retirement guarantees, bounded service,
builder restoration, overflow rejection and producer/consumer ownership.
Never copy or mutate a live fabric, change historical measurements, or treat
sanitizer success as proof that all schedules are safe.

Questions: can concurrent churn break composition rather than just service?
Do all lanes at maximum draw count and near-capacity legacy tails compose
correctly? Can invalid wrapper calls change accounting? Can the reducer accept
wrong invocation metadata or changed source snapshots despite matching summary
numbers? Verification requires reproducible negative controls, model-based
composition checks, sanitizer/regression runs, and a recorded source comparison.

## Confirmed findings and remediation

### Medium: report did not bind summary data to the invocation

Before remediation, replacing the raw command's argv with
`wrong-binary native 8 1 64` while retaining a valid legacy summary produced a
successful report. Raw-summary equality alone did not detect mislabeled work.
The reducer now requires the exact mode/producer/frame/draw arguments, the
expected variant binary within the collection, and a corresponding successful
manifest command record. Negative controls include both wrong arguments and
cross-variant binary substitution.

### Medium: snapshot hashes were recorded but not enforced by the reducer

Changing a recorded source hash to all zeroes still produced a successful
report. Manual hash verification had passed previously, so this is a missing
automated gate, not evidence that the original collection used wrong sources.
The reducer now verifies immutable in-collection snapshot hashes and requires
the driver, common harness/adapter/header, and variant runtime snapshots.
Mutable live-tree aliases are deliberately not used as reproduction snapshots.
Missing snapshots, hash changes and changed snapshot contents are rejected.

Both pre-remediation reproductions are preserved in
`tests/evidence-frame-runtime-redteam/reducer-before.json`. Sixteen synthetic
controls now cover these cases and the earlier report failures; they are added
to `make test`. The controls also explicitly reject assertion-disabled Python.

## Runtime review and expanded verification

No new production-runtime defect has been confirmed in this pass. Review
rechecked live-count RMW subtraction during rejoin, monotonic bound publication,
readiness lifetime, retirement ordering, capacity arithmetic, suffix restoration,
partial-init cleanup, link wiring and source-preserving cleanup rules.

New tests cover:

* An independent, seeded composition oracle over 1,024 near-capacity cases with
  all eight lanes, varying draw counts, terminal suffixes and preexisting
  overflow: 643 valid compositions and 381 rejected frames. It verifies exact
  commands, callback counts, original builder restoration and repeated dispatch.
* Concurrent eight-producer churn through actual end-frame composition, not
  only queue service: 800 accepted/applied/retired contributions with changing
  draw counts, including empty frames, canonical slot order and restored legacy
  prefix/suffix. The original service-only 800-contribution stress also remains.
* Null/invalid/stale/closing wrapper calls and sequence rejection without
  accounting changes, including operations after quiescent shutdown.
* A deterministic busy retirement: zero-progress process and end-frame passes
  retain the prior draw and live count until the retirement mutex is available.

Header comments explicitly prohibit backend callbacks from reentering consumer
APIs, clarify that zero progress is not retirement completion, and forbid copying
a live HSGpu. These clarify existing ownership constraints; no new guard or
production-runtime instruction was added.

The upgraded reducer verifies 283 immutable snapshot files and reproduces every
original numeric timing result exactly. `src/hs_gpu.c` remains byte-identical to
the final measured candidate; header non-comment tokens also match. Accordingly,
this pass does not claim new performance measurements or erase the measured
legacy/1p cost. The final report's existing limitations still apply.

## Completed gates and disposition

The expanded default regression, source-preservation checks, ASan/UBSan with
leak detection, TSan and sixteen report controls passed on the Pi. Fresh normal
HSGpu GLES verification passed 18 presentations / 36 lifecycle pixel checks and
full-frame equality of 1,536,000 RGBA bytes against direct GLES. The standalone
fabric and general-message sources remain unchanged.

Evidence is in `tests/evidence-frame-runtime-redteam/` and
`tests/evidence-frame-runtime-graphics-redteam/`. Original timing collections
remain immutable. The prior mixed performance result, extra object storage,
shared-global-state limitation, and message-only capture/fence scope remain
explicit acceptance tradeoffs, not defects silently declared resolved.

Disposition: the two confirmed evidence-validation findings are remediated and
covered by negative controls. No confirmed runtime blocker remains from this
pass; this is not proof against every possible concurrent schedule. Changes
remain uncommitted, with no new commit or push requested.

Subsequent user decision: the measured tradeoffs were accepted and a commit was
approved. Participant checks are now wired into the standard regression,
ASan/UBSan, TSan and hardware targets, rather than relying on dedicated targets
alone. Push-permission remediation and CI remain separate, deferred work.
