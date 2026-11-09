# Shutdown latency: measurement contract

Baseline: committed `3aaab60`. Success means lower complete shutdown wall/CPU
cost while retaining resource release, pixels, capture defaults and normal
rendering behavior. Returning early, deferring cleanup to exit, suppressing leaks
or changing driver/security/frequency settings is not an accepted optimization.

First locate time with gated, per-call wall/process-CPU observers and a separate
syscall trace. Preserve uninstrumented controls: nested/instrumented intervals
are diagnostic, not additive savings or accepted throughput measurements.
Unknowns include driver worker joins, deferred GPU work, surface/context cleanup,
EGL display destruction, GBM teardown and device close.

Test diagnostic source copies first. Compare equivalent lifecycle work under
alternating-order repetitions, including initialized-but-idle, one draw,
multi-draw and outstanding-work cases. Check context release, handles/file
descriptors, ASan/UBSan/leaks and framebuffer equality. A fast incorrect cleanup
is a failed candidate. Do not promote a change that merely moves the same wait
into another measured phase or introduces an unaccepted rendering regression.
Preserve raw output, source/binary hashes, traces, failed attempts and environment
metadata. New work remains uncommitted unless separately approved.

## Initial diagnosis

Five uninstrumented controls and five separately instrumented runs per scene
completed. Single-draw control/profile shutdown medians were 42.429/40.835 ms;
retained-draw medians were 37.717/37.913 ms. Instrumented final `close()` medians
were 35.266 ms (single) and 30.810 ms (retained), with about 0.93–0.97 ms process
CPU. `eglReleaseThread` was about 6 microseconds, not the dominant wait. The
separate syscall trace confirms a 25.518 ms final `close(3)` after EGL/GBM cleanup.
These are diagnostic intervals, not a causal/additive performance budget.

First bounded experiment: remove only the explicit `drmDropMaster` immediately
before final close. Closing the last owned descriptor already releases master
ownership; the hypothesis is that the explicit transition creates redundant
kernel/display work. It is not yet a demonstrated speedup. Retain every EGL
cleanup operation and the actual final close. Check balanced fd counts, closed
graphics fd, released context, successful subsequent master reacquisition, and
both reacquisition and complete ownership-probe intervals. Reject apparent
savings that merely move the wait into the next device use.

The close-only candidate failed: paired shutdown ratios mixed widely and the
subsequent ownership probe removed the apparent single-scene median advantage.
The early-RmFB candidate also failed 21-pair qualification: shutdown CPU rose
in 21/21 single-draw pairs and 20/21 retained pairs, with inconsistent wall-time
results. Neither candidate is promoted.

A final diagnostic captures the complete original CRTC mode/framebuffer and
all connectors attached to that CRTC, then restores them before resource
destruction. Its saved-state-only control isolates query/allocation costs from
the restore call. Missing/invalid snapshot data or failed restoration must fall
back to full baseline cleanup, never skip close or EGL release. Capture is
per-instance and its allocations are freed. Actual mode-set return values and
restoration of the captured framebuffer are observed separately; an unexercised
or failed restore path cannot count as a performance success.

Full restore was rejected after 21 pairs per scene. Single-draw baseline/restore
shutdown wall medians were 35.962/47.821 ms, process CPU 6.367/9.752 ms, and
shutdown-plus-probe 52.402/65.375 ms. Retained medians were 44.379/41.711 ms wall,
7.659/9.510 ms CPU, and 60.682/59.423 ms combined. CPU regressed in every pair;
saved-state connector probing also increased cold-frame cost.

The remaining bounded candidate uses cached connector queries and requires
identical timing fields, one identical connector, and zero saved viewport
offsets before attempting a page flip to the captured original framebuffer.
It waits synchronously for a matching completion event (100 ms bound), then
uses full SetCrtc restoration on ineligibility, submission failure, or incomplete
event processing. EGL/GBM destruction and final close remain unconditional.
The event user-data points into the graphics instance, which remains alive
through event processing and final descriptor close; the kernel does not invoke
the userspace callback asynchronously. On timeout, no later event dispatch is
performed and final close discards pending descriptor events before instance
reset. A successful submission alone is not accepted as completion.

The observer separately records page-flip attempts, submissions, callbacks,
target framebuffer, and full-mode restoration. Performance evidence is accepted
only when each candidate run actually completes the intended flip without
fallback. Correct fallback paths require separate qualification if timing
warrants promotion. No candidate has yet earned a production change.

## Final disposition

No production change is accepted. The first flip collection stopped at its
intended-path assertion: submission failed and full-mode fallback succeeded.
Preserve that incomplete collection (`evidence-shutdown-flip-restore`) rather
than reclassifying it as successful. A fresh three-pair diagnostic per scene
(`evidence-shutdown-flip-diagnostic`) recorded the error and fallback separately:
all 6 attempts returned `EINVAL` (22), zero submissions succeeded, and zero flip
events completed. All six full-mode fallbacks restored the captured framebuffer,
released the EGL context, balanced fd counts, closed the original graphics fd,
and permitted subsequent master acquisition. The rejection's precise driver
cause is not established; matching timings alone does not prove flip support.
No page-flip performance result is claimed, and no further candidate is promoted
on the basis of fallback-only timing. Successful-event, timeout, leak-sanitizer,
reinitialization, idle/outstanding-work, and actual RGBA qualification of this
candidate remain unperformed because the intended path did not pass its gate.

The reducer now distinguishes collection validity from intended-path execution:
`flip_path.intended_path_exercised_every_run` is false in the diagnostic report.
It verifies raw/cached concordance, invocation, source/binary hashes when present,
frame sequences, context release, descriptor balance, flip targets/completion,
and observed mode fallback. Isolated-copy negative controls accepted the valid
restore/flip diagnostic collections and rejected respectively 6 and 8 corrupted
inputs, including raw mismatch, live context, fd leak, wrong invocation, missing
mode audit, source hash mismatch, fabricated flip completion, and wrong target.
Original evidence was not edited by these controls.

The demonstrated bottleneck remains a wait inside final DRM `close()` on this
Pi stack. This does not prove a universal or irreducible kernel/display latency
floor, nor attribute the exact wait to one kernel subsystem. It does show that
the bounded user-space candidates tested here did not establish an acceptable
improvement over complete committed cleanup. Keep `3aaab60` as the production
path, including EGL release and final fd close; keep all new diagnostics and
this report uncommitted pending user review.

Final checks: `git diff --exit-code 3aaab60` confirmed no tracked production
changes; `make check-sources`, diagnostic Python syntax checks, and `make test`
passed. These checks do not substitute for unperformed hardware/sanitizer
qualification of a promoted candidate; none was promoted.
