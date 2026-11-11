#!/usr/bin/env python3
"""Summarise pacing_trace.so output (tests/pacing_trace.c).

motion_ms_sd: spread of the animation time advanced per refresh, counting
a repeated refresh as 0 (ideal 0: every refresh advances one refresh).
repeated: refreshes that kept the previous frame, overall and per second
after the takeover flip. step_ms: animation time (swap time of the frame
shown) advanced per refresh between consecutive new frames; uniform steps
mean even motion. latency_ms: flip completion minus that frame's swap."""
import json, sys

def load(path):
    flips, swaps = [], []
    for line in open(path):
        p = line.split()
        if p[0] == 'F': flips.append((int(p[1]), float(p[2]), int(p[3]) if len(p) > 3 else -1))
        elif p[0] == 'S': swaps.append(float(p[1]))
    return flips, swaps

def q(a, x):
    return round(a[min(len(a) - 1, int(len(a) * x))], 2) if a else None

def hist(values, width=4.0):
    h = {}
    for v in values:
        b = int(v // width * width)
        h[b] = h.get(b, 0) + 1
    return {f"{k}-{k + int(width)}": h[k] for k in sorted(h)}

def summarise(path, content=None):
    flips, swaps = load(path)
    if content:
        # Animation time of each swap's frame, as logged by the application.
        # The application logs only the frames it presents; swaps before its
        # first logged frame (e.g. during init) are aligned out at the front.
        logged = [float(l) for l in open(content)]
        if len(logged) > len(swaps):
            raise SystemExit(f"{content}: more logged frames than swaps")
        swaps = [float("nan")] * (len(swaps) - len(logged)) + logged
    if len(flips) < 2:
        return {"trace": path, "flips": len(flips)}
    t0 = flips[0][1]
    secs = int((flips[-1][1] - t0) / 1e6) + 1
    per_sec = [0] * secs
    for i in range(1, len(flips)):
        per_sec[int((flips[i][1] - t0) / 1e6)] += flips[i][0] - flips[i - 1][0] - 1
    gaps = [flips[i][0] - flips[i - 1][0] for i in range(1, len(flips))]
    sw = sorted(x for x in (swaps[i] - swaps[i - 1] for i in range(1, len(swaps))) if x == x)
    shown = [(f, swaps[f[2] - 1]) for f in flips if 0 < f[2] <= len(swaps) and swaps[f[2] - 1] == swaps[f[2] - 1]]
    steps = sorted((shown[i][1] - shown[i - 1][1]) / 1e3 / (shown[i][0][0] - shown[i - 1][0][0])
                   for i in range(1, len(shown)))
    # Motion per refresh: animation time advanced at every refresh, 0 on a
    # refresh that repeated the previous frame. Ideal: one refresh each time.
    motion = []
    for i in range(1, len(shown)):
        gap = shown[i][0][0] - shown[i - 1][0][0]
        motion += [0.0] * (gap - 1) + [(shown[i][1] - shown[i - 1][1]) / 1e3]
    mm = sum(motion) / len(motion) if motion else 0
    motion_sd = (sum((x - mm) ** 2 for x in motion) / len(motion)) ** 0.5 if motion else 0
    lat = sorted((f[1] - t) / 1e3 for f, t in shown)
    mean = sum(steps) / len(steps) if steps else 0
    sd = (sum((x - mean) ** 2 for x in steps) / len(steps)) ** 0.5 if steps else 0
    return {"trace": path, "flips": len(flips), "refreshes": flips[-1][0] - flips[0][0],
            "repeated": sum(gaps) - len(gaps), "repeated_first_s": per_sec[0], "repeated_per_s": per_sec,
            "max_gap": max(gaps), "swaps": len(swaps),
            "swap_ms_median": q([x / 1e3 for x in sw], 0.5), "swap_ms_p99": q([x / 1e3 for x in sw], 0.99),
            "swap_ms_max": round(sw[-1] / 1e3, 2) if sw else None, "mapped": len(shown),
            "motion_ms_sd": round(motion_sd, 2), "motion_hist": hist(motion), "step_ms_mean": round(mean, 2), "step_ms_sd": round(sd, 2), "step_ms_p1": q(steps, 0.01),
            "step_ms_p99": q(steps, 0.99), "latency_ms_median": q(lat, 0.5), "latency_ms_p99": q(lat, 0.99)}

if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--content":  # --content <trace> <content-file>
        print(json.dumps(summarise(args[1], args[2])))
    else:
        for path in args:
            print(json.dumps(summarise(path)))
