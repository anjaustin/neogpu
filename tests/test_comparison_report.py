"""Recompute cross-path summaries from completed raw comparison collections."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--transport", required=True, type=Path)
parser.add_argument("--graphics", required=True, nargs="+", type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
if not args.output.name.startswith("evidence-") or not args.output.resolve().is_relative_to((ROOT / "tests").resolve()):
    parser.error("output must be a fresh tests/evidence-* file")
report = {"transport": [], "graphics": {}, "manifest_sha256": {},
          "report_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def manifest(directory):
    path = directory / "manifest.json"
    data = path.read_bytes()
    result = json.loads(data)
    assert result["completed"], directory
    report["manifest_sha256"][str(path)] = hashlib.sha256(data).hexdigest()
    return result


transport = manifest(args.transport)
assert transport["reference_revision"] == "c8e9008dc4639220cc9bcf4c84abd2ff0d344aa5"
for name, digest in transport["dependencies"].items():
    assert hashlib.sha256((args.transport / "reference" / name).read_bytes()).hexdigest() == digest
summaries = []
for run in transport["runs"]:
    records = [json.loads(line) for line in (args.transport / (run["label"] + ".jsonl")).read_text().splitlines()]
    summary = records[-1]
    assert summary["accepted"] == summary["applied"] and summary["frame_bytes"] == 2064 and summary["command_bytes"] == 32
    samples = records[:-1]
    latencies = sorted(sample["applied_ns"] - sample["attempt_ns"] for sample in samples)
    assert latencies and latencies[0] >= 0
    summary["latency_p99_ns"] = latencies[(len(latencies) * 99 + 99) // 100 - 1]
    summary["fps"] = summary["applied"] * 1e9 / summary["elapsed_ns"]
    summaries.append(summary)
for participants in (1, 2, 4, 8):
    for draws in (1, 64):
        record = {"participants": participants, "draws": draws}
        for implementation in ("neogpu", "rigtorp"):
            selected = [s for s in summaries if s["implementation"] == implementation and s["participants"] == participants and s["draws"] == draws]
            assert len(selected) >= 3
            rates = [s["fps"] for s in selected]
            record[implementation] = {"runs": len(selected), "median_fps": statistics.median(rates),
                                      "min_fps": min(rates), "max_fps": max(rates),
                                      "median_sampled_p99_ns": statistics.median(s["latency_p99_ns"] for s in selected)}
        record["neogpu_over_rigtorp"] = record["neogpu"]["median_fps"] / record["rigtorp"]["median_fps"]
        report["transport"].append(record)

blocks = []
for directory in args.graphics:
    metadata = manifest(directory)
    assert metadata["shader_sources_identical"]
    records = [json.loads(line) for line in (directory / "frames.jsonl").read_text().splitlines()]
    assert sum(r.get("identical") is True for r in records) == 1
    assert len(metadata["captures_sha256"]) == 2
    for name, digest in metadata["captures_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    for block in (r for r in records if r["record"] == "block"):
        frames = [r for r in records if r["record"] == "frame" and r["path"] == block["path"] and r["pair"] == block["pair"]]
        assert len(frames) == block["frames"]
        summary = dict(block, collection=str(directory))
        for key in ("cpu_submit_ns", "completion_ns", "finish_wait_ns", "swap_ns", "total_ns"):
            values = sorted(frame[key] for frame in frames)
            summary[key + "_p50"] = statistics.median(values)
            summary[key + "_p99"] = values[(len(values) * 99 + 99) // 100 - 1]
        blocks.append(summary)
for path in ("direct", "neogpu"):
    selected = [b for b in blocks if b["path"] == path]
    report["graphics"][path] = {"blocks": len(selected), "frames": sum(b["frames"] for b in selected)}
    for key in ("cpu_submit_ns", "completion_ns", "finish_wait_ns", "swap_ns", "total_ns"):
        medians = [b[key + "_p50"] for b in selected]
        tails = [b[key + "_p99"] for b in selected]
        report["graphics"][path][key] = {"median_of_block_medians": statistics.median(medians),
                                         "block_median_range": [min(medians), max(medians)],
                                         "block_p99_range": [min(tails), max(tails)]}
report["graphics"]["blocks"] = blocks
with args.output.open("x") as destination:
    destination.write(json.dumps(report, indent=2))
print(json.dumps({"transport": report["transport"], "graphics": {k: v for k, v in report["graphics"].items() if k != "blocks"}}, indent=2))
