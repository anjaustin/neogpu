"""Diagnostic ablation only: one-slot service is INVALID for multi-participant use.

Preserve structures, capacity, ownership and frame work; remove seven inactive
lane probes in generated source copies, for both adapters. Never edit runtime.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--reference", default="tests/evidence-comparison-final/reference", type=Path)
args = parser.parse_args()
if not args.output.name.startswith("evidence-") or not args.output.resolve().is_relative_to((ROOT / "tests").resolve()):
    parser.error("output must be a fresh tests/evidence-* directory")
args.output.mkdir(exist_ok=False)
header = args.reference / "include/rigtorp/SPSCQueue.h"
assert hashlib.sha256(header.read_bytes()).hexdigest() == "62d9f783862666be5c7ca3c8fbcec2e45d9920167c4b9e689bda4957f66dd86e"
names = ["src/hs_fabric.c", "tests/comparison_native.c", "tests/comparison_rigtorp.cc", "tests/test_comparison_transport.c"]
original = {name: Path(name).read_bytes() for name in names}
manifest = {"completed": False, "warning": "Diagnostic single-participant-only scheduler, not a production optimization",
            "original_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in original.items()},
            "commands": [], "results": []}


def checked(command, label):
    begin = time.monotonic_ns()
    result = subprocess.run([str(a) for a in command], text=True, capture_output=True, timeout=120)
    record = {"command": [str(a) for a in command], "exit_status": result.returncode,
              "elapsed_ns": time.monotonic_ns() - begin, "stdout": result.stdout, "stderr": result.stderr}
    (args.output / (label + ".json")).write_text(json.dumps(record, indent=2))
    manifest["commands"].append({k: v for k, v in record.items() if k != "stdout"})
    assert result.returncode == 0, record
    return result.stdout


try:
    manifest["baseline"] = checked(["git", "rev-parse", "HEAD"], "baseline-revision").strip()
    flags = ["-O3", "-g", "-march=armv8-a+simd", "-mtune=cortex-a72", "-ffast-math", "-funroll-loops",
             "-Wall", "-Wextra", "-Werror", "-DNDEBUG", "-Iinclude", "-Itests"]
    binaries = {}
    for variant in ("baseline", "single-slot"):
        directory = args.output / variant
        directory.mkdir()
        objects = []
        for name in names:
            source = original[name].decode()
            if variant == "single-slot" and name.endswith("hs_fabric.c"):
                edits = {"visited < HS_FABRIC_PARTICIPANTS && events < budget": "visited < 1 && events < budget",
                         "fabric->cursor = (i + 1) % HS_FABRIC_PARTICIPANTS;": "fabric->cursor = 0;"}
                for before, after in edits.items():
                    assert source.count(before) == 1
                    source = source.replace(before, after)
            if variant == "single-slot" and name.endswith("comparison_rigtorp.cc"):
                edits = {"visited < HS_FABRIC_PARTICIPANTS; visited++": "visited < 1; visited++",
                         "context->cursor = (context->cursor + 1) % HS_FABRIC_PARTICIPANTS;": "context->cursor = 0;"}
                for before, after in edits.items():
                    assert source.count(before) == 1
                    source = source.replace(before, after)
            destination = directory / Path(name).name
            destination.write_text(source)
            obj = destination.with_suffix(".o")
            cpp = name.endswith(".cc")
            checked(["g++" if cpp else "gcc", *flags,
                     *(["-std=c++23", "-isystem", args.reference / "include"] if cpp else []),
                     "-c", destination, "-o", obj], variant + "-compile-" + Path(name).stem)
            objects.append(obj)
        binary = directory / "comparison"
        checked(["g++", *flags, *objects, "-lpthread", "-lm", "-o", binary], variant + "-link")
        checked([binary, "--selftest"], variant + "-selftest")
        binaries[variant] = binary
    for pair in range(1, 6):
        order = [(v, i) for v in binaries for i in ("neogpu", "rigtorp")]
        if pair % 2 == 0:
            order.reverse()
        for variant, implementation in order:
            label = f"pair-{pair}-{variant}-{implementation}"
            raw = checked([binaries[variant], implementation, 1, 1000000, 1], label)
            (args.output / (label + ".jsonl")).write_text(raw)
            record = json.loads(raw.splitlines()[-1])
            assert record["applied"] == record["accepted"] == 1000000
            record.update(variant=variant, pair=pair,
                          frames_per_second=record["applied"] * 1e9 / record["elapsed_ns"])
            manifest["results"].append(record)
    for variant in binaries:
        for implementation in ("neogpu", "rigtorp"):
            selected = [r["frames_per_second"] for r in manifest["results"] if r["variant"] == variant and r["implementation"] == implementation]
            print(json.dumps({"variant": variant, "implementation": implementation, "median_frames_per_second": statistics.median(selected),
                              "min_frames_per_second": min(selected), "max_frames_per_second": max(selected)}), flush=True)
    assert all(Path(name).read_bytes() == data for name, data in original.items())
    manifest["runtime_unchanged"] = True
    manifest["completed"] = True
finally:
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
