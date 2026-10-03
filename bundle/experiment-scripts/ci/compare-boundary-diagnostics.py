"""Attribute paired diagnostic differences without relaxing output equivalence."""

import argparse
import json
from pathlib import Path


def load(path):
    grouped = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        grouped.setdefault(record["request_id"], []).append(record)
    return list(grouped.values())


def compare(control, arm):
    if len(control) != len(arm) or not control:
        raise ValueError("Incomplete paired diagnostic requests")
    results = []
    for index, (left, right) in enumerate(zip(control, arm)):
        slots = [[record for record in records if record["kind"] == "slot"] for records in (left, right)]
        if any(len(records) != 1 for records in slots):
            raise ValueError("Expected exactly one final slot write per request")
        prefill_inputs = [[record for record in records if record["kind"] == "input" and record["phase"] == "prefill"]
                          for records in (left, right)]
        if any(not records for records in prefill_inputs):
            raise ValueError("Missing prefill inputs")
        final_inputs = [records[-1] for records in prefill_inputs]
        if final_inputs[0]["tokens"] != final_inputs[1]["tokens"]:
            raise ValueError("Paired prompts differ")
        changes = {}
        for field in ("recurrent", "convolution"):
            states = [records[0][field] for records in slots]
            if len(states[0]) != len(states[1]):
                raise ValueError("Layer counts differ")
            changes[field] = [layer for layer, (first, second) in enumerate(zip(*states)) if first != second]
        outputs = []
        for phase, step in (("prefill", 0), ("decode", 0), ("decode", 1), ("decode", 2)):
            matched = [[record for record in records if record["kind"] == "output" and
                        record["phase"] == phase and record["decode_step"] == step] for records in (left, right)]
            if any(not records for records in matched):
                continue
            first, second = (records[-1] for records in matched)
            outputs.append(dict(phase=phase, step=step, equal=first["logits"] == second["logits"],
                                control_ids=first["top_ids"], arm_ids=second["top_ids"],
                                control_values=first["top_values"], arm_values=second["top_values"]))
        decode_inputs = [[dict(positions=record["positions"], tokens=record["tokens"])
                          for record in records if record["kind"] == "input" and record["phase"] == "decode"]
                         for records in (left, right)]
        results.append(dict(index=index, prompt_lens=final_inputs[0]["prompt_lens"],
                            changed_state_layers=changes, outputs=outputs, decode_inputs=decode_inputs))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("arm", type=Path)
    args = parser.parse_args()
    result = compare(load(args.control / "boundary-diagnostics.jsonl"), load(args.arm / "boundary-diagnostics.jsonl"))
    (args.arm / "boundary-attribution.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
