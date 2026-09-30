"""Emit a tuning-experiment-template.md record from the spec and evidence."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).with_name("spec")))
from validate_spec import parse_simple_yaml  # noqa: E402

import gates  # noqa: E402


def render(spec, phase_id, evidence):
    phase = next((p for p in spec.get("phases", []) if p.get("id") == phase_id), None)
    if phase is None:
        raise ValueError(f"Phase {phase_id!r} not in registry")
    gate_report = gates.evaluate_phase(phase, evidence)
    lines = [
        f"# {phase_id}: {phase.get('title', 'untitled')}",
        "",
        f"Status: **{'gates passed' if gate_report['all_passed'] else 'gates failed / not promoted'}.** "
        "Serving defaults unchanged.",
        "",
        "## Question and decision",
        "",
        "| Field | Record |",
        "|---|---|",
        f"| ID, date, owner | {phase_id}, {datetime.now(timezone.utc).date().isoformat()}, Not recorded |",
        f"| Status | {'Decision ready' if gate_report['all_passed'] else 'Not promoted'} |",
        f"| Single changed variable | {phase.get('variable', 'measurement only')} |",
        f"| Tool | `{phase.get('tool', 'Not recorded')}` |",
        "",
        "## Gates",
        "",
        "| Gate | Passed | Evidence |",
        "|---|---|---|",
    ]
    for result in gate_report["results"]:
        lines.append(f"| {result['gate']} | {'Pass' if result['passed'] else 'FAIL'} | {result['reason']} |")
    lines += ["", "## Metrics", "", "```json", json.dumps(evidence.get("metrics", {}), indent=2, sort_keys=True), "```"]
    if not gate_report["all_passed"]:
        lines += ["", "This candidate is **not promoted**; a failed gate is a recorded result, "
                      "not a gap in the record."]
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=str(Path(__file__).with_name("spec") / "phases.yaml"))
    parser.add_argument("--phase", required=True)
    parser.add_argument("--evidence", required=True, help="evidence JSON for gates.evaluate_phase")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    spec = parse_simple_yaml(Path(args.spec).read_text(encoding="utf-8"))
    evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    record = render(spec, args.phase, evidence)
    if args.out:
        Path(args.out).write_text(record, encoding="utf-8")
    else:
        print(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
