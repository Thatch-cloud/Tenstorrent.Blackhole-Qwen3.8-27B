"""Fail closed on missing reports or output divergence across continuation arms."""

import json
from pathlib import Path
import sys


def compare(control, arm):
    expected = [f"boundary-{length}" for length in (63, 64, 65, 2047, 2048, 2049, 4096, 8193)] + ["after-cancel"]
    if any(list(root.glob("isolated-2049-*.json")) for root in (control, arm)):
        expected = [f"isolated-2049-{repeat}" for repeat in range(3)] + expected
    for root in (control, arm):
        summary = json.loads((root / "interleave-summary.json").read_text())
        if not summary.get("passed") or summary.get("results") != expected:
            raise AssertionError(f"Incomplete arm: {root.name}")
    for name in expected:
        reports = [json.loads((root / f"{name}.json").read_text()) for root in (control, arm)]
        if not all(report.get("passed") for report in reports):
            raise AssertionError(f"Failed request: {name}")
        for key in ("prompt_tokens", "requested_tokens", "output_sha256", "token_ids_sha256"):
            if reports[0][key] != reports[1][key]:
                raise AssertionError(f"Continuation divergence: {name}, {key}")
    mixed_checks = 0
    if any((root / "mixed-traffic.json").exists() for root in (control, arm)):
        mixed = [json.loads((root / "mixed-traffic.json").read_text()) for root in (control, arm)]
        if not all(report.get("passed") for report in mixed):
            raise AssertionError("Failed mixed-traffic arm")
        for name in ("B0", "A", "C"):
            for key in ("tokens", "text_sha256"):
                if mixed[0]["baseline"][name][key] != mixed[1]["concurrent"][name][key]:
                    raise AssertionError(f"Mixed-traffic cross-arm divergence: {name}, {key}")
            mixed_checks += 1
    result = dict(passed=True, checks=len(expected), mixed_checks=mixed_checks, arm=arm.name,
                  scope="Exact output token-ID hashes; not full state certification")
    (arm / "whole-chunked-comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    compare(Path(sys.argv[1]), Path(sys.argv[2]))
