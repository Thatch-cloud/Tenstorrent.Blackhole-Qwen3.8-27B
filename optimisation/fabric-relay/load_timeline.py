"""P0.3 loader timeline: parse instrumentation events into the stage table.

Input is a JSON list of {stage, card, started_s, ended_s} events emitted by
the instrumented load_target_once path. Stages follow the plan:
read, deserialize, convert, upload_card0, upload_card1.

Pure analysis; the instrumented loader itself ships with the runtime harness
changes for experiment P0.3.
"""

import argparse
import json
from pathlib import Path

REQUIRED_STAGES = ["read", "deserialize", "convert", "upload_card0", "upload_card1"]


def timeline(events, tolerance_s=1.0):
    per_stage = {}
    for event in events:
        stage = event["stage"]
        if stage not in REQUIRED_STAGES:
            raise ValueError(f"Unknown load stage {stage!r}")
        duration = float(event["ended_s"]) - float(event["started_s"])
        if duration < 0:
            raise ValueError(f"Negative duration for {stage!r}")
        per_stage.setdefault(stage, 0.0)
        per_stage[stage] += duration
    missing = [stage for stage in REQUIRED_STAGES if stage not in per_stage]
    if missing:
        raise ValueError(f"Missing stages: {', '.join(missing)}")
    total = float(max(event["ended_s"] for event in events) - min(event["started_s"] for event in events))
    summed = sum(per_stage.values())
    if abs(total - summed) > tolerance_s:
        raise ValueError(f"Stage sum {summed:.2f}s exceeds span {total:.2f}s beyond tolerance")
    card1_upload_share = per_stage["upload_card1"] / total if total else 0.0
    return {
        "per_stage_s": {stage: round(per_stage[stage], 3) for stage in REQUIRED_STAGES},
        "total_s": round(total, 3),
        "card1_upload_share": round(card1_upload_share, 4),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, help="loader events JSON")
    parser.add_argument("--tolerance-s", type=float, default=1.0)
    args = parser.parse_args(argv)
    events = json.loads(Path(args.events).read_text(encoding="utf-8"))
    print(json.dumps(timeline(events, args.tolerance_s), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
