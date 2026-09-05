"""Require real replay rows on both chips; never mix compile/eager rows into attribution."""

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import sys


def analyze(rows, trace_id, steps):
    sessions = defaultdict(list)
    for row in rows:
        if row.get("METAL TRACE ID") != str(trace_id):
            continue
        replay = row.get("METAL TRACE REPLAY SESSION ID", "")
        if not replay:
            continue
        duration = float(row["DEVICE KERNEL DURATION [ns]"])
        if not math.isfinite(duration) or duration <= 0:
            raise AssertionError("Invalid traced kernel duration")
        sessions[(row["DEVICE ID"], int(replay))].append(row)
    devices = sorted({device for device, replay in sessions})
    if len(devices) != 2:
        raise AssertionError("Missing both-chip trace replay evidence")
    results = []
    for device in devices:
        selected = sorted(replay for chip, replay in sessions if chip == device)[-steps:]
        if len(selected) != steps:
            raise AssertionError("Incomplete decode replay sequence")
        totals = defaultdict(list)
        cores = defaultdict(set)
        counts = []
        for replay in selected:
            operations = sessions[(device, replay)]
            counts.append(len(operations))
            durations = defaultdict(float)
            for row in operations:
                durations[row["OP CODE"]] += float(row["DEVICE KERNEL DURATION [ns]"])
                cores[row["OP CODE"]].add(row.get("CORE COUNT", ""))
            for name, duration in durations.items():
                totals[name].append(duration)
        if len(set(counts)) != 1 or any(len(values) != steps for values in totals.values()):
            raise AssertionError("Operation coverage changes between replays")
        results.append(dict(device=device, replay_sessions=selected, operations_per_replay=counts[0],
            operations=sorted([dict(op=name, median_summed_kernel_ns=statistics.median(values),
                                    core_counts=sorted(cores[name])) for name, values in totals.items()],
                              key=lambda item: -item["median_summed_kernel_ns"])))
    return results


def measured_log(log):
    if log.count("QWEN_PROFILE_MEASURE_BEGIN") != 1 or log.count("QWEN_PROFILE_MEASURE_END") != 1:
        raise AssertionError("Missing unambiguous profiling boundaries")
    measured = log.split("QWEN_PROFILE_MEASURE_BEGIN", 1)[1].split("QWEN_PROFILE_MEASURE_END", 1)[0]
    if "markers were dropped" in measured:
        raise AssertionError("Profiler dropped markers during measured decode")
    return log.split("QWEN_PROFILE_MEASURE_BEGIN", 1)[0].count("markers were dropped")


def main(root):
    generation = json.loads((root / "generation.json").read_text())
    if not generation["passed"]:
        raise AssertionError("Profile generation correctness failed")
    warmup_warnings = measured_log((root / "console.log").read_text(errors="replace"))
    reports = list(root.rglob("*ops_perf_results*.csv"))
    if len(reports) != 1:
        raise AssertionError("Expected exactly one operation report")
    with reports[0].open(newline="") as stream:
        devices = analyze(csv.DictReader(stream), generation["decode_trace_id"], len(generation["steps"]))
    result = dict(passed=True, devices=devices, excluded_warmup_drop_warnings=warmup_warnings,
        scope="Last 15 host-sampling decode trace replays; summed kernel durations are not critical-path wall time or serving throughput")
    (root / "attribution.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
