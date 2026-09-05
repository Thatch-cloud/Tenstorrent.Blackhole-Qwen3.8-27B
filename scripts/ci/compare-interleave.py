"""Fail closed on missing reports or output divergence across continuation arms."""

import json
from pathlib import Path
import sys


def compare(control, arm):
    expected = [f"boundary-{length}" for length in (63, 64, 65, 2047, 2048, 2049, 4096, 8193)] + ["after-cancel"]
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
    result = dict(passed=True, checks=len(expected), arm=arm.name,
                  scope="Exact output token-ID hashes; not full state certification")
    (arm / "whole-chunked-comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    compare(Path(sys.argv[1]), Path(sys.argv[2]))
