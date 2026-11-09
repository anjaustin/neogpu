"""Equivalent retained two-triangle scene: direct GLES versus participant executor."""
import argparse
import ast
import datetime
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)


def command(argv):
    start = time.monotonic_ns()
    result = subprocess.run([str(a) for a in argv], text=True, capture_output=True, timeout=120)
    return {"command": [str(a) for a in argv], "exit_status": result.returncode,
            "elapsed_ns": time.monotonic_ns() - start, "stdout": result.stdout, "stderr": result.stderr}


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


def shader(source, name):
    match = re.search(r"const char\* " + name + r"\s*=\s*(.*?);\n", source, re.S)
    assert match, name
    return "".join(ast.literal_eval(part) for part in re.findall(r'"(?:\\.|[^"\\])*"', match[1]))


parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--output", required=True)
args = parser.parse_args()
output = Path(args.output)
if not output.name.startswith("evidence-") or not output.resolve().is_relative_to((ROOT / "tests").resolve()):
    parser.error("output must be a fresh tests/evidence-* directory")
output.mkdir(exist_ok=False)
sources = [Path("Makefile")]
for directory in ("include", "src", "tests"):
    sources.extend(p for p in Path(directory).iterdir() if p.suffix in (".c", ".h", ".cc", ".py"))
manifest = {"completed": False, "baseline": command(["git", "rev-parse", "HEAD"]),
            "working_tree": command(["git", "status", "--short"]), "before": environment(),
            "compiler": command(["gcc", "--version"]), "kernel": command(["uname", "-a"]),
            "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)},
            "build_recipe": command(["make", "-n", "-B", "tests/test_comparison_graphics"])}
try:
    backend = Path("src/hs_backend_gles.c").read_text()
    test = Path("tests/test_comparison_graphics.c").read_text()
    for name, original in (("vertex", "vs"), ("fragment", "fs")):
        assert shader(test, name) == shader(backend, original), name + " shader mismatch"
    manifest["shader_sources_identical"] = True
    build = command(["make", "tests/test_comparison_graphics"])
    (output / "build.json").write_text(json.dumps(build, indent=2))
    assert build["exit_status"] == 0, build
    result = command(["sudo", "-n", "./tests/test_comparison_graphics", output / "scene"])
    (output / "execution.json").write_text(json.dumps(result, indent=2))
    (output / "frames.jsonl").write_text(result["stdout"])
    assert result["exit_status"] == 0, result
    records = [json.loads(line) for line in result["stdout"].splitlines()]
    assert sum(r.get("identical") is True for r in records) == 1
    manifest["captures_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.ppm")}
    assert len(set(manifest["captures_sha256"].values())) == 1
    frames = [r for r in records if r["record"] == "frame"]
    blocks = [r for r in records if r["record"] == "block"]
    assert len(frames) == 6000 and len(blocks) == 10
    summaries = []
    for block in blocks:
        selected = [f for f in frames if f["pair"] == block["pair"] and f["path"] == block["path"]]
        assert len(selected) == block["frames"] == 600
        summary = dict(block)
        for key in ("cpu_submit_ns", "completion_ns", "finish_wait_ns", "swap_ns", "total_ns"):
            values = sorted(f[key] for f in selected)
            summary[key + "_p50"] = statistics.median(values)
            summary[key + "_p99"] = values[(len(values) * 99 + 99) // 100 - 1]
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    (output / "summaries.json").write_text(json.dumps(summaries, indent=2))
    manifest["completed"] = True
finally:
    manifest["after"] = environment()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
