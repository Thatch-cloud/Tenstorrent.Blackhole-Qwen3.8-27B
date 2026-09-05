"""Summarize client estimates and observed cache gauges without inventing engine timing."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics


METRIC = re.compile(r'^vllm:(kv_cache_usage_perc|gpu_cache_usage_perc|num_requests_running|num_requests_waiting|num_preemptions_total)(?:\{.*\})?\s+([0-9.eE+-]+)(?:\s|$)')


def summarize(root):
    runs = defaultdict(list)
    for path in root.glob("run-*-length-*-batch-*-user-*.json"):
        result = json.loads(path.read_text())
        parts = result["label"].split("-")
        if not result.get("passed"):
            raise AssertionError(f"Failed request: {path.name}")
        runs[(int(parts[3]), int(parts[5]))].append(result)
    groups = []
    for (length, batch), results in sorted(runs.items()):
        rates = [result["client_decode_estimate_tok_s"] for result in results
                 if result["client_decode_estimate_tok_s"] is not None]
        gaps = [result["event_gap_p99_s"] for result in results if result["event_gap_p99_s"] is not None]
        users = defaultdict(set)
        for result in results:
            fingerprint = result.get("token_ids_sha256", result.get("token_strings_sha256"))
            if fingerprint is None:
                raise AssertionError("Missing output fingerprint")
            users[result["label"].split("-")[-1]].add(fingerprint)
        groups.append(dict(target_prompt_tokens=length, batch=batch, requests=len(results),
            actual_prompt_tokens=sorted({result["prompt_tokens"] for result in results}),
            client_decode_median_tok_s=statistics.median(rates) if rates else None,
            client_decode_min_tok_s=min(rates) if rates else None,
            client_decode_max_tok_s=max(rates) if rates else None,
            ttft_median_s=statistics.median(result["ttft_s"] for result in results),
            event_gap_p99_max_s=max(gaps) if gaps else None,
            same_output_across_repeats=all(len(values) == 1 for values in users.values()),
            engine_committed_tok_s=None))
    gauges = defaultdict(list)
    scrape_errors = 0
    busy_samples, busy_zero_cache = 0, 0
    with (root / "metrics.jsonl").open() as stream:
        for line in stream:
            sample = json.loads(line)
            if "error" in sample:
                scrape_errors += 1
                continue
            values = defaultdict(list)
            for metric_line in sample["metrics"].splitlines():
                match = METRIC.match(metric_line)
                if match:
                    name, value = match.group(1), float(match.group(2))
                    gauges[name].append(value)
                    values[name].append(value)
            if any(value > 0 for value in values["num_requests_running"]):
                busy_samples += 1
                cache = values["kv_cache_usage_perc"] or values["gpu_cache_usage_perc"]
                if cache and all(value == 0 for value in cache):
                    busy_zero_cache += 1
    summary = json.loads((root / "baseline-summary.json").read_text())
    report = dict(passed=summary.get("passed", False), workloads=groups,
        gauges={name: dict(samples=len(values), minimum=min(values), maximum=max(values)) for name, values in gauges.items()},
        busy_samples=busy_samples, busy_zero_cache_samples=busy_zero_cache, scrape_errors=scrape_errors,
        scope="Client estimates and raw exporter observations; not 200-token/s certification or full cache-lifecycle validation",
        cache_classification=("zero observed while running: investigate exporter/allocator timing" if busy_zero_cache else
            "cache occupancy observed nonzero" if any(value > 0 for name in ("kv_cache_usage_perc", "gpu_cache_usage_perc") for value in gauges[name])
            else "insufficient active occupancy evidence; do not assume disabled cache"))
    if not groups:
        report["passed"] = False
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = summarize(args.root)
    (args.root / "analysis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)
