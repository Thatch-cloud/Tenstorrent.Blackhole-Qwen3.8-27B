"""Cross-check native replay coverage and rank traced matrix shapes."""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys


def summarize(rows, runtime_rows, trace_id, replay_ids):
    selected = [row for row in rows if row.get("METAL TRACE ID") == str(trace_id)
                and row.get("METAL TRACE REPLAY SESSION ID") in replay_ids]
    native = [row for row in runtime_rows if row.get("METAL TRACE ID") == str(trace_id)
              and row.get("METAL TRACE REPLAY SESSION ID") in replay_ids]
    def coverage(records, operation):
        return Counter((row["DEVICE ID"], row["METAL TRACE REPLAY SESSION ID"], row[operation]) for row in records)
    if not selected or coverage(selected, "OP CODE") != coverage(native, "OP NAME"):
        raise AssertionError("Reported replay operation counts disagree with native C++ profiler")
    grouped = defaultdict(lambda: defaultdict(list))
    for row in selected:
        if row["OP CODE"] != "MatmulDeviceOperation":
            continue
        key = tuple(row[field] for field in ("DEVICE ID", "INPUT_1_Y_PAD[LOGICAL]", "INPUT_1_X_PAD[LOGICAL]",
                                             "CORE COUNT", "INPUT_1_DATATYPE"))
        grouped[key][row["METAL TRACE REPLAY SESSION ID"]].append(float(row["DEVICE KERNEL DURATION [ns]"]))
    result = []
    for key, sessions in grouped.items():
        if set(sessions) != replay_ids:
            raise AssertionError("Matrix shape missing from some measured replays")
        result.append(dict(device=key[0], weight_y=key[1], weight_x=key[2], cores=key[3], dtype=key[4],
                           calls_per_replay=statistics.median(len(values) for values in sessions.values()),
                           median_summed_kernel_ms=statistics.median(sum(values) for values in sessions.values()) / 1e6))
    return sorted(result, key=lambda entry: (entry["device"], -entry["median_summed_kernel_ms"]))


def main(root):
    attribution = json.loads((root / "attribution.json").read_text())
    generation = json.loads((root / "generation.json").read_text())
    replays = {str(value) for value in attribution["devices"][0]["replay_sessions"]}
    report_path, = root.glob("reports/**/*ops_perf_results*.csv")
    runtime_path = root / "metadata/cpp_device_perf_report.csv"
    if not runtime_path.is_file():
        runtime_path = root / ".logs/cpp_device_perf_report.csv"
    with report_path.open(newline="") as report, runtime_path.open(newline="") as runtime:
        shapes = summarize(csv.DictReader(report), csv.DictReader(runtime), generation["decode_trace_id"], replays)
    result = dict(passed=True, native_operation_coverage_matches=True, matrix_shapes=shapes,
                  scope="Padded/logical weight dimensions and per-replay summed kernel durations, not critical-path wall time")
    (root / "matrix-shapes.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
