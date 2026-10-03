"""Check profiler completeness separately from the underlying numerical test."""

import json
import csv
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def check(root):
    document = ET.parse(root / "tests.xml")
    suites = list(document.iter("testsuite"))
    tests = sum(int(suite.get("tests", 0)) for suite in suites)
    failures = sum(int(suite.get("failures", 0)) + int(suite.get("errors", 0)) +
                   int(suite.get("skipped", 0)) for suite in suites)
    logs = (root / "console.log").read_text(errors="replace")
    dropped = "markers were dropped" in logs
    reports = [path for path in root.rglob("*.csv") if "ops_perf_results" in path.name and path.stat().st_size > 0]
    operations = {}
    for path in reports:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                try:
                    duration = float(row["DEVICE KERNEL DURATION [ns]"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not math.isfinite(duration) or duration <= 0:
                    continue
                key = (row["DEVICE ID"], row["OP CODE"])
                operation = operations.setdefault(key, dict(device=key[0], op=key[1], kernel_ns=[],
                    core_counts=set(), available_worker_counts=set()))
                operation["kernel_ns"].append(duration)
                operation["core_counts"].add(row.get("CORE COUNT", "unavailable"))
                operation["available_worker_counts"].add(row.get("AVAILABLE WORKER CORE COUNT", "unavailable"))
    for operation in operations.values():
        operation["core_counts"] = sorted(operation["core_counts"])
        operation["available_worker_counts"] = sorted(operation["available_worker_counts"])
    devices = sorted({key[0] for key in operations})
    report = dict(passed=tests > 0 and failures == 0 and not dropped and len(devices) == 2,
                  tests=tests, failed_or_skipped=failures, dropped_markers=dropped,
                  profiled_devices=devices, operations=list(operations.values()),
                  reports=[str(path.relative_to(root)) for path in reports],
                  scope="Eager layer attribution; not full-model traced decode speed")
    (root / "profile-summary.json").write_text(json.dumps(report, indent=2))
    if not report["passed"]:
        raise AssertionError(report)
    print(json.dumps(report))


if __name__ == "__main__":
    check(Path(sys.argv[1]))
