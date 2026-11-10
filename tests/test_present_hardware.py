"""Hardware gate: presented frames must actually be scanned out.

Runs tests/test_present_flip under tests/scanout_probe, which watches the CRTC
from outside the process. Self-reported counters are not enough: the gate
requires the display controller to switch framebuffers once per presented
frame and to scan out every colour of the rendered sequence, in order.

Usage (root, display attached):
    sudo python3 -E tests/test_present_hardware.py [--present BIN] [--probe BIN]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PALETTE = ["ff0000", "00ff00", "0000ff", "ffffff"]


def observe(present, probe, args, probe_ms, pixel_ms):
    watcher = subprocess.Popen([probe, str(probe_ms), str(pixel_ms)], stdout=subprocess.PIPE, text=True)
    time.sleep(0.3)
    run = subprocess.run([present, *map(str, args)], capture_output=True, text=True, timeout=60)
    out, _ = watcher.communicate(timeout=probe_ms / 1000 + 10)
    report = json.loads(run.stdout.strip().splitlines()[-1])
    return run.returncode, report, json.loads(out)


def timeline(scan, console_fb):
    seq = []
    for _, fb, pixel in scan["pixels"]:
        if fb == console_fb or pixel is None:
            continue
        if not seq or seq[-1] != (fb, pixel):
            seq.append((fb, pixel))
    return seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--present", default=str(ROOT / "tests/test_present_flip"))
    parser.add_argument("--probe", default=str(ROOT / "tests/scanout_probe"))
    opts = parser.parse_args()
    failures = []

    idle = json.loads(subprocess.run([opts.probe, "200", "100"], capture_output=True, text=True).stdout)
    console_fb = idle["pixels"][0][1] if idle["pixels"] else 0

    # 1. Throughput at vsync: one hardware flip per presented frame, ~refresh rate.
    rc, report, scan = observe(opts.present, opts.probe, [240, 1], 5000, 7)
    flips = scan["fb_changes"]
    colours = {pixel for _, fb, pixel in scan["pixels"] if fb != console_fb}
    print(json.dumps({"case": "vsync", "presented": report.get("presented"), "fps": report.get("fps"),
                      "hardware_flips": flips, "colours": sorted(c for c in colours if c)}))
    if rc != 0 or not report.get("passed"):
        failures.append("vsync run failed its own contract")
    # +1: the first frame replaces the console framebuffer via the modeset.
    if not report.get("presented") or abs(flips - report["presented"]) > 1:
        failures.append(f"hardware flips {flips} != presented {report.get('presented')}")
    if not set(PALETTE) <= colours:
        failures.append(f"not every colour reached the display: {sorted(c for c in colours if c)}")

    # 2. Order: with frames held, the scanout must follow the palette exactly
    #    and never show the buffer being rendered (alternating framebuffers).
    rc, report, scan = observe(opts.present, opts.probe, [8, 1, 300], 4000, 40)
    seq = timeline(scan, console_fb)
    expected = [PALETTE[i % 4] for i in range(8)]
    pixels = [pixel for _, pixel in seq]
    fbs = [fb for fb, _ in seq]
    alternating = all(a != b for a, b in zip(fbs, fbs[1:]))
    print(json.dumps({"case": "order", "scanned": pixels, "fbs": fbs}))
    if rc != 0 or pixels != expected:
        failures.append(f"scanout order {pixels} != rendered {expected}")
    if not alternating:
        failures.append("consecutive frames shared a framebuffer (rendering into the scanned-out buffer)")

    # 3. vsync off: never blocks on vblank, never shows more than one flip per vblank.
    rc, report, scan = observe(opts.present, opts.probe, [240, 0], 2000, 50)
    print(json.dumps({"case": "no-vsync", "fps": report.get("fps"), "presented": report.get("presented"),
                      "dropped": report.get("dropped"), "hardware_flips": scan["fb_changes"]}))
    if rc != 0 or not report.get("passed"):
        failures.append("vsync-off run failed its own contract")
    if abs(scan["fb_changes"] - report.get("presented", -99)) > 1:
        failures.append("vsync-off hardware flips do not match presented frames")

    # 4. Display handed back to the console after exit.
    time.sleep(0.5)
    after = json.loads(subprocess.run([opts.probe, "200", "100"], capture_output=True, text=True).stdout)
    back = after["pixels"] and after["pixels"][-1][1] == console_fb
    print(json.dumps({"case": "handback", "console_fb": console_fb, "after_fb": after["pixels"][-1][1] if after["pixels"] else None}))
    if not back:
        failures.append("console framebuffer not restored after exit")

    for failure in failures:
        print("FAIL:", failure)
    print("present hardware gate:", "passed" if not failures else "FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
