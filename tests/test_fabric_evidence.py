"""Collect reproducible participant-protocol correctness runs, not a speedup claim."""
import argparse
import datetime
import glob
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)


def command(args):
    result = subprocess.run(args, text=True, capture_output=True, timeout=120)
    return {"command": args, "exit_status": result.returncode,
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
parser.add_argument("--runs", type=int, default=3)
args = parser.parse_args()
if args.runs < 3:
    parser.error("at least three repetitions are required")
output = Path(args.output)
if not output.resolve().is_relative_to((ROOT / "tests").resolve()):
    parser.error("output must be under tests/")
output.mkdir(exist_ok=False)
sources = [Path("Makefile")]
for directory in ("include", "src", "tests"):
    sources.extend(p for p in Path(directory).iterdir() if p.suffix in (".c", ".h", ".py"))
manifest = {"completed": False, "baseline": command(["git", "rev-parse", "HEAD"]),
            "working_tree": command(["git", "status", "--short"]),
            "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)},
            "compiler": command(["gcc", "--version"]), "kernel": command(["uname", "-a"]),
            "cpuinfo": Path("/proc/cpuinfo").read_text(), "before": environment(),
            "build_recipe": command(["make", "-n", "-B", "tests/test_fabric"]), "runs": []}
try:
    build = command(["make", "tests/test_fabric"])
    (output / "build.json").write_text(json.dumps(build, indent=2))
    assert build["exit_status"] == 0, build
    for run in range(1, args.runs + 1):
        before = environment()
        start = time.monotonic_ns()
        result = command(["./tests/test_fabric"])
        elapsed = time.monotonic_ns() - start
        raw = result.pop("stdout")
        (output / f"fabric-{run}.jsonl").write_text(raw)
        manifest["runs"].append({"run": run, "elapsed_ns": elapsed,
                                 "before": before, "after": environment(), **result})
        assert result["exit_status"] == 0, result
        records = [json.loads(line) for line in raw.splitlines()]
        assert len(records) == 5 and all(record["passed"] for record in records)
        delivery = records[3]
        assert delivery["accepted"] == delivery["applied"] == 40000
        lifecycle = records[4]
        assert lifecycle["accepted"] == lifecycle["applied"] == lifecycle["retired"] == 512
        print(json.dumps({"run": run, **delivery}))
    manifest["completed"] = True
finally:
    manifest["after"] = environment()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
