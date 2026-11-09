"""Run in-tree probes; retain raw records, environment and source provenance.

Usage: python3 tests/test_architecture_evidence.py --output tests/evidence-RUN
Exit zero means valid experiment execution, not that hypotheses were supported.
"""
import argparse
import datetime
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
    patterns = ("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor",
                "/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq",
                "/sys/class/thermal/thermal_zone*/temp")
    import glob
    sysfs = {}
    for pattern in patterns:
        for name in sorted(glob.glob(pattern)):
            try:
                sysfs[name] = Path(name).read_text().strip()
            except OSError as error:
                sysfs[name] = str(error)
    return {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "affinity": sorted(os.sched_getaffinity(0)), "sysfs": sysfs,
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
metadata = {
    "baseline": command(["git", "rev-parse", "HEAD"]),
    "working_tree": command(["git", "status", "--short"]),
    "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)},
    "compiler": command(["gcc", "--version"]), "kernel": command(["uname", "-a"]),
    "cpuinfo": Path("/proc/cpuinfo").read_text(),
    "build_recipe": command(["make", "-n", "-B", "tests/test_architecture", "tests/test_architecture_latency"]),
    "environment_before": environment(), "runs": [],
}
try:
    build = command(["make", "tests/test_architecture", "tests/test_architecture_latency"])
    (output / "build.json").write_text(json.dumps(build, indent=2))
    assert build["exit_status"] == 0, build["stderr"]
    for run in range(1, args.runs + 1):
        for label, invocation in (
                ("probes", ["./tests/test_architecture"]),
                ("latency", ["./tests/test_architecture_latency"] + (["--reverse"] if run % 2 == 0 else []))):
            before = environment()
            start = time.monotonic_ns()
            result = command(invocation)
            elapsed = time.monotonic_ns() - start
            (output / f"{label}-{run}.jsonl").write_text(result.pop("stdout"))
            metadata["runs"].append({"run": run, "label": label, "elapsed_ns": elapsed,
                                     "before": before, "after": environment(), **result})
            assert result["exit_status"] == 0, result
            records = [json.loads(line) for line in (output / f"{label}-{run}.jsonl").read_text().splitlines()]
            if label == "probes":
                assert len(records) == 7
            else:
                summaries = [r for r in records if r.get("record") == "summary"]
                assert len(summaries) == 3
                for summary in summaries:
                    samples = [r for r in records if r.get("record") == "sample" and r["workload"] == summary["workload"]]
                    assert len(samples) == summary["samples"] == summary["rt_applied"]
                    assert summary["background_applied"] == summary["background_admitted"]
                    values = sorted(r["applied_ns"] - r["attempt_ns"] for r in samples)
                    for percentile in (50, 95, 99):
                        assert values[(len(values) * percentile + 99) // 100 - 1] == summary[f"attempt_to_apply_p{percentile}_ns"]
                    assert values[-1] == summary["attempt_to_apply_max_ns"]
                    print(json.dumps({"run": run, **summary}))
    metadata["completed"] = True
finally:
    metadata["environment_after"] = environment()
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2))
