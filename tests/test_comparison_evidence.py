"""Pinned optimized-queue comparison. Output stays in a fresh tests/evidence-* directory."""
import argparse
import datetime
import glob
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
REVISION = "c8e9008dc4639220cc9bcf4c84abd2ff0d344aa5"
DEPENDENCIES = {
    "include/rigtorp/SPSCQueue.h": "62d9f783862666be5c7ca3c8fbcec2e45d9920167c4b9e689bda4957f66dd86e",
    "LICENSE": "21d7e737765bc0c986ee78f16a3e25ee3cb5ad424feece274b64a10654c31912",
}


def command(argv, env=None):
    start = time.monotonic_ns()
    result = subprocess.run([str(a) for a in argv], text=True, capture_output=True,
                            timeout=120, env=env)
    return {"command": [str(a) for a in argv], "exit_status": result.returncode,
            "elapsed_ns": time.monotonic_ns() - start,
            "stdout": result.stdout, "stderr": result.stderr}


def environment():
    values = {}
    for pattern in ("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor",
                    "/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq",
                    "/sys/class/thermal/thermal_zone*/temp"):
        for name in sorted(glob.glob(pattern)):
            try:
                values[name] = Path(name).read_text().strip()
            except OSError as error:
                values[name] = str(error)
    return {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "affinity": sorted(os.sched_getaffinity(0)), "sysfs": values,
            "throttle": command(["vcgencmd", "get_throttled"]) if shutil.which("vcgencmd") else None}


parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--output", required=True)
parser.add_argument("--runs", type=int, default=5)
parser.add_argument("--frames", type=int, default=100000)
parser.add_argument("--sanitizers", action="store_true")
args = parser.parse_args()
if args.runs < 3 or not 128 <= args.frames <= 1000000:
    parser.error("at least 3 pairs and 128..1000000 frames are required")
output = Path(args.output)
if not output.name.startswith("evidence-") or not output.resolve().is_relative_to((ROOT / "tests").resolve()):
    parser.error("output must be a fresh tests/evidence-* directory")
output.mkdir(exist_ok=False)
assert set(range(4)).issubset(os.sched_getaffinity(0)), "CPUs 0..3 are required"
sources = [Path("Makefile")]
for directory in ("include", "src", "tests"):
    sources.extend(p for p in Path(directory).iterdir() if p.suffix in (".c", ".h", ".cc", ".py"))
manifest = {"completed": False, "baseline": command(["git", "rev-parse", "HEAD"]),
            "working_tree": command(["git", "status", "--short"]), "reference_revision": REVISION,
            "dependencies": DEPENDENCIES, "compiler_c": command(["gcc", "--version"]),
            "compiler_cpp": command(["g++", "--version"]), "kernel": command(["uname", "-a"]),
            "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)},
            "before": environment(), "commands": [], "runs": []}


def checked(argv, label, env=None):
    result = command(argv, env)
    (output / f"{label}.json").write_text(json.dumps(result, indent=2))
    manifest["commands"].append({key: value for key, value in result.items() if key != "stdout"})
    assert result["exit_status"] == 0, result
    return result["stdout"]


def build(variant):
    directory = output / variant
    directory.mkdir()
    flags = ["-march=armv8-a+simd", "-mtune=cortex-a72", "-Wall", "-Wextra", "-Werror", "-DNDEBUG", "-Iinclude", "-Itests"]
    flags += ["-O3", "-ffast-math", "-funroll-loops"] if variant == "optimized" else ["-O1", "-g", "-fno-omit-frame-pointer", "-fsanitize=" + ("address,undefined" if variant == "asan" else "thread")]
    objects = []
    for source in ("src/hs_fabric.c", "tests/comparison_native.c", "tests/test_comparison_transport.c", "tests/comparison_rigtorp.cc"):
        obj = directory / (Path(source).stem + ".o")
        cpp = source.endswith(".cc")
        checked(["g++" if cpp else "gcc", *flags, *(["-std=c++23", "-isystem", output / "reference/include"] if cpp else []),
                 "-c", source, "-o", obj], variant + "-build-" + Path(source).stem)
        objects.append(obj)
    binary = directory / "comparison"
    checked(["g++", *flags, *objects, "-lpthread", "-lm", "-o", binary], variant + "-link")
    return binary


try:
    for name, digest in DEPENDENCIES.items():
        data = urllib.request.urlopen(f"https://raw.githubusercontent.com/rigtorp/SPSCQueue/{REVISION}/{name}", timeout=30).read()
        assert hashlib.sha256(data).hexdigest() == digest, name
        destination = output / "reference" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    if args.sanitizers:
        for variant in ("asan", "tsan"):
            binary = build(variant)
            env = {**os.environ, "ASAN_OPTIONS": "detect_leaks=1:halt_on_error=1",
                   "UBSAN_OPTIONS": "halt_on_error=1", "TSAN_OPTIONS": "halt_on_error=1"}
            checked([binary, "--selftest"], variant + "-selftest", env)
            for implementation in ("neogpu", "rigtorp"):
                for participants in (1, 8):
                    checked([binary, implementation, participants, 1024, 64], f"{variant}-{implementation}-{participants}", env)
    binary = build("optimized")
    checked([binary, "--selftest"], "optimized-selftest")
    summaries = []
    for participants in (1, 2, 4, 8):
        for draws in (1, 64):
            for implementation in ("neogpu", "rigtorp"):
                checked([binary, implementation, participants, 4096, draws], f"warm-{implementation}-{participants}-{draws}")
            for pair in range(1, args.runs + 1):
                order = ("neogpu", "rigtorp") if pair % 2 else ("rigtorp", "neogpu")
                for implementation in order:
                    before = environment()
                    label = f"pair-{pair}-{implementation}-{participants}-{draws}"
                    raw = checked([binary, implementation, participants, args.frames, draws], label)
                    (output / f"{label}.jsonl").write_text(raw)
                    records = [json.loads(line) for line in raw.splitlines()]
                    summary = records[-1]
                    assert summary["accepted"] == summary["applied"] == participants * args.frames
                    samples = records[:-1]
                    assert len(samples) == participants * (args.frames // 128)
                    latencies = sorted(sample["applied_ns"] - sample["attempt_ns"] for sample in samples)
                    assert latencies[0] >= 0
                    summary.update(pair=pair, frames_per_second=summary["applied"] * 1e9 / summary["elapsed_ns"],
                                   latency_samples=len(latencies),
                                   latency_p99_ns=latencies[(len(latencies) * 99 + 99) // 100 - 1])
                    summaries.append(summary)
                    manifest["runs"].append({"label": label, "before": before, "after": environment(), "summary": summary})
            selected = [s for s in summaries if s["participants"] == participants and s["draws"] == draws]
            medians = {implementation: statistics.median(s["frames_per_second"] for s in selected if s["implementation"] == implementation)
                       for implementation in ("neogpu", "rigtorp")}
            print(json.dumps({"participants": participants, "draws": draws, "medians_frames_per_second": medians,
                              "neogpu_over_rigtorp": medians["neogpu"] / medians["rigtorp"]}), flush=True)
    (output / "summaries.json").write_text(json.dumps(summaries, indent=2))
    manifest["completed"] = True
finally:
    manifest["after"] = environment()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
