"""Attribute the frozen B1/eight-slot recurrent-state copy chain in measured traces."""

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import sys


def shape(row, tensor):
    return tuple(row[f"{tensor}_{axis}_PAD[LOGICAL]"] for axis in "WZYX")


def summarize(rows, native_rows, trace_id, replay_ids):
    def select(records):
        return [row for row in records if row.get("METAL TRACE ID") == str(trace_id)
                and row.get("METAL TRACE REPLAY SESSION ID") in replay_ids]
    selected, native = select(rows), select(native_rows)
    def coverage(records, operation):
        return Counter((row["DEVICE ID"], row["METAL TRACE REPLAY SESSION ID"],
                        row["GLOBAL CALL COUNT"], row[operation]) for row in records)
    if not selected or coverage(selected, "OP CODE") != coverage(native, "OP NAME"):
        raise AssertionError("Native operation identity/coverage mismatch")
    sessions = defaultdict(list)
    for row in selected:
        duration = float(row["DEVICE KERNEL DURATION [ns]"])
        if not math.isfinite(duration) or duration <= 0:
            raise AssertionError("Invalid kernel duration")
        sessions[(row["DEVICE ID"], row["METAL TRACE REPLAY SESSION ID"])].append(row)
    if set(sessions) != {(device, replay) for device in ("0", "1") for replay in replay_ids}:
        raise AssertionError("Missing chip or replay")
    full = ("8[8]", "24[24]", "128[128]", "128[128]")
    active = ("1[1]", "24[24]", "128[128]", "128[128]")
    expected = ("SliceDeviceOperation", "DecodeGatedDeltaRuleDeviceOperation",
                "InterleavedToShardedDeviceOperation", "SliceWriteDeviceOperation")
    results = []
    for (device, replay), records in sorted(sessions.items()):
        records.sort(key=lambda row: int(row["GLOBAL CALL COUNT"]))
        positions = [index for index, row in enumerate(records) if row["OP CODE"] == expected[1]]
        if len(positions) != 48:
            raise AssertionError("Expected all 48 GDN layers")
        totals = defaultdict(float)
        for index in positions:
            chain = records[index - 1:index + 3] if index else []
            if tuple(row["OP CODE"] for row in chain) != expected:
                raise AssertionError("Unexpected state-chain adjacency")
            checks = ((0, "INPUT_0", full), (0, "OUTPUT_0", active),
                      (1, "INPUT_5", active), (1, "OUTPUT_1", active),
                      (2, "INPUT_0", active), (2, "OUTPUT_0", active),
                      (3, "INPUT_0", active), (3, "INPUT_1", full), (3, "OUTPUT_0", full))
            for offset, tensor, dimensions in checks:
                if shape(chain[offset], tensor) != dimensions or chain[offset][f"{tensor}_DATATYPE"] != "BFLOAT16":
                    raise AssertionError("State shape/precision changed")
            for row in chain:
                totals[row["OP CODE"]] += float(row["DEVICE KERNEL DURATION [ns]"]) / 1e6
        copy_ms = sum(totals[name] for name in expected if name != expected[1])
        total_ms = sum(float(row["DEVICE KERNEL DURATION [ns]"]) for row in records) / 1e6
        results.append(dict(device=device, replay=replay, layers=48, kernel_ms=dict(totals),
                            copy_ms=copy_ms, all_kernel_ms=total_ms, copy_fraction=copy_ms / total_ms))
    devices = []
    for device in ("0", "1"):
        entries = [entry for entry in results if entry["device"] == device]
        devices.append(dict(device=device, replays=len(entries), layers_per_replay=48,
                            kernel_ms={name: statistics.median(entry["kernel_ms"][name] for entry in entries)
                                       for name in expected},
                            median_copy_ms=statistics.median(entry["copy_ms"] for entry in entries),
                            min_copy_ms=min(entry["copy_ms"] for entry in entries),
                            max_copy_ms=max(entry["copy_ms"] for entry in entries),
                            median_copy_fraction=statistics.median(entry["copy_fraction"] for entry in entries)))
    return dict(passed=True, devices=devices, sessions=results,
                scope="Summed measured kernel durations, not critical-path time or predicted speedup; do not sum chips")


def main(root):
    attribution = json.loads((root / "attribution.json").read_text())
    generation = json.loads((root / "generation.json").read_text())
    if not attribution["passed"] or not generation["passed"]:
        raise AssertionError("Source trace gates failed")
    replays = {str(value) for value in attribution["devices"][0]["replay_sessions"]}
    path, = root.glob("reports/**/*ops_perf_results*.csv")
    native_path = root / "metadata/cpp_device_perf_report.csv"
    if not native_path.is_file():
        native_path = root / ".logs/cpp_device_perf_report.csv"
    with path.open(newline="") as report, native_path.open(newline="") as native:
        result = summarize(csv.DictReader(report), csv.DictReader(native), generation["decode_trace_id"], replays)
    (root / "gdn-state-attribution.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(passed=result["passed"], devices=result["devices"], scope=result["scope"]), indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
