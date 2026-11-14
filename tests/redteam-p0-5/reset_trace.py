#!/usr/bin/env python3
"""Summarise a reset_trace.bt trace: the timeout/reset events and what the
renderer's jobs did around them. Usage: reset_trace.py <trace> [window_ms]"""
import sys, json

ev = []
for line in open(sys.argv[1]):
    p = line.split()
    if not p or not p[0].isdigit():
        continue
    ev.append((int(p[0]), p[1], p[2], p[3] if len(p) > 3 else "", " ".join(p[4:])))
win = float(sys.argv[2]) if len(sys.argv) > 2 else 250.0
key = [e for e in ev if e[1] in ("reset_begin", "reset_end", "increase_karma") or "timedout" in e[1]]
resets = [e[0] for e in ev if e[1] == "reset_begin"]
rcl = [e[0] for e in ev if e[1] == "rcl_irq"]
out = {"events": len(ev), "resets": len(resets), "render_jobs_done": len(rcl)}
out["timeouts"] = [{"ms": round(e[0] / 1e6, 3), "event": e[1], "detail": e[4]} for e in key]
if rcl:
    # render jobs finished per window, and the longest gap between two
    t0, t1 = rcl[0], rcl[-1]
    n = int((t1 - t0) / 1e6 / win) + 1
    hist = [0] * n
    for t in rcl:
        hist[int((t - t0) / 1e6 / win)] += 1
    out["render_jobs_per_s_by_window"] = {"window_ms": win, "first_window_starts_ms": round(t0 / 1e6, 1),
                                          "rates": [round(h * 1000 / win, 1) for h in hist]}
    gaps = sorted(((b - a, a) for a, b in zip(rcl, rcl[1:])), reverse=True)[:5]
    out["longest_gaps_between_render_jobs_ms"] = [{"gap_ms": round(g / 1e6, 2), "starts_ms": round(a / 1e6, 2)} for g, a in gaps]
for r in resets:
    # jobs the kernel ran again after this reset (same job pointer run twice), and jobs skipped
    end = next((e[0] for e in ev if e[1] == "reset_end" and e[0] >= r), r)
    after = [e for e in ev if end <= e[0] <= end + 2_000_000 and e[1] in ("render_run", "bin_run", "csd_run_enter")]
    seen_before = {}
    for e in ev:
        if e[0] < r and e[1] in ("render_run", "bin_run", "csd_run_enter"):
            seen_before[e[4].split()[0]] = e[0]
    rerun = [{"event": e[1], "ms_after_reset_end": round((e[0] - end) / 1e6, 3), "detail": e[4],
              "first_started_ms_before_reset": round((r - seen_before[e[4].split()[0]]) / 1e6, 2)}
             for e in after if e[4].split()[0] in seen_before]
    out.setdefault("jobs_run_again_after_reset", []).append({"reset_ms": round(r / 1e6, 3), "reset_takes_us": round((end - r) / 1e3, 1), "jobs": rerun})
    nxt = next((t for t in rcl if t > end), None)
    prv = max((t for t in rcl if t < r), default=None)
    if nxt and prv:
        out.setdefault("render_job_end_before_and_after_reset_ms", []).append([round((r - prv) / 1e6, 2), round((nxt - end) / 1e6, 2)])
# compute jobs longer than 20 ms: what the render jobs did while each ran
starts = {}
long_jobs = []
flight = {"0": None, "1": None}  # bin / render job on the hardware: its start time
for e in ev:
    if e[1] == "cl_hw_start":
        flight[e[4].split()[0].split("=")[1]] = e[0]
    elif e[1] == "bcl_irq":
        flight["0"] = None
    elif e[1] == "rcl_irq":
        flight["1"] = None
    if e[1] == "csd_hw_start":
        starts[e[4]] = (e[0], (flight["0"] is not None, flight["1"] is not None))
    elif e[1] == "csd_irq" and e[4] in starts:
        s, who = starts.pop(e[4])
        if e[0] - s > 20_000_000:
            inside = [t for t in rcl if s <= t <= e[0]]
            edges = [s] + inside + [e[0]]
            long_jobs.append({"ms": round((e[0] - s) / 1e6, 2), "starts_ms": round(s / 1e6, 1),
                              "bin_job_in_flight_at_start": who[0], "render_job_in_flight_at_start": who[1],
                              "render_jobs_started_inside": sum(1 for x in ev if x[1] == "cl_hw_start" and "render=1" in x[4] and s <= x[0] <= e[0]),
                              "render_jobs_done_inside": len(inside),
                              "render_jobs_per_s_inside": round(len(inside) * 1e9 / (e[0] - s), 1),
                              "longest_time_without_a_render_job_ms": round(max(b - a for a, b in zip(edges, edges[1:])) / 1e6, 2)})
# per submitter of compute jobs: the render jobs finished between its first and last submit
subs = {}
for e in ev:
    if e[1] == "csd_ioctl":
        f = subs.setdefault(e[3], [e[0], e[0], 0])
        f[1] = e[0]
        f[2] += 1
out["render_jobs_per_s_while_submitting"] = {
    c: {"compute_jobs": n, "span_ms": round((b - a) / 1e6, 1),
        "render_jobs_per_s": round(sum(1 for t in rcl if a <= t <= b) * 1e9 / (b - a), 1) if b > a else None,
        "longest_time_without_a_render_job_ms": round(max((y - x for x, y in zip([a] + [t for t in rcl if a <= t <= b], [t for t in rcl if a <= t <= b] + [b])), default=0) / 1e6, 2)}
    for c, (a, b, n) in subs.items()}
if long_jobs:
    out["compute_jobs_over_20ms"] = long_jobs
print(json.dumps(out, indent=1))
