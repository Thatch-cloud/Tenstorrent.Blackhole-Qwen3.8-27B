"""Validate the fabric-relay phase registry against the plan's invariants.

Host-only test companion and CLI. The YAML subset used by phases.yaml is
indentation-based mappings/lists with inline scalars; no external YAML
dependency is required.
"""

import argparse
import json
import re
import sys
from pathlib import Path

REQUIRED_BASELINE_IDS = ["P0.1", "P0.2", "P0.3", "P0.4"]
REQUIRED_METRIC_FIELDS = ["id", "title", "tool", "gates", "metrics", "records"]
KNOWN_RULES = {
    "at_least_repeats", "abba_ordering", "candidate_not_worse_than",
    "all_stages_present", "sum_matches_total", "all_cards_present",
    "weight_digests_equal", "exact_output_state", "added_latency_below",
    "all_blocks_not_worse", "gain_repeats_beyond_variability",
}
KNOWN_RECORDS = {"baseline", "candidate"}
PLAN_RELAY_VARIABLES = {
    "P1": "card-1 weight delivery path",
    "P2.a": "card-1 dispatch core type",
    "P2.b": "card-1 result readback path",
}


def _strip_comment(text):
    """Drop a trailing ' #...' comment (not preceded by content in quotes)."""
    index = text.find(" #")
    return text if index < 0 else text[:index]


def parse_scalar(text):
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        body = text[1:-1].strip()
        return [parse_scalar(part) for part in body.split(",")] if body else []
    if text.startswith("{") and text.endswith("}"):
        result = {}
        for part in _split_top_level(text[1:-1]):
            key, _, value = part.partition(":")
            result[parse_scalar(key)] = parse_scalar(value)
        return result
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    try:
        return float(text) if re.fullmatch(r"-?\d+\.\d+", text) else text
    except ValueError:
        return text


def _split_top_level(text):
    parts, depth, current = [], 0, []
    for char in text:
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current))
    return [part for part in (piece.strip() for piece in parts) if part]


def parse_simple_yaml(text):
    """Parse the mapping/list subset used by phases.yaml."""
    class _Deferred:
        """Placeholder whose container type resolves to the first child seen."""

        def __init__(self):
            self.value = None

        def become(self, container):
            self.value = container

    root = {}
    # kind: "root", "map" (deferred value of a key, at the key's indent),
    # "entry" (list-item mapping, at its content indent)
    stack = [(-1, root, "root")]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip()
        while True:
            top_indent, _, kind = stack[-1]
            if (kind == "entry" and indent < top_indent) or \
                    (kind in ("map", "root") and indent <= top_indent):
                stack.pop()
            else:
                break
        parent_indent, parent, kind = stack[-1]
        if isinstance(parent, _Deferred) and parent.value is None:
            parent.become([] if line.startswith("- ") else {})
        if isinstance(parent, _Deferred):
            parent = parent.value
        if line.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"List item outside a list: {raw!r}")
            item_text = line[2:].strip()
            item_text = _strip_comment(item_text).strip()
            key, sep, value = item_text.partition(":")
            value = _strip_comment(value).strip()
            if sep and not item_text.startswith(("\"", "'")):
                entry = {key.strip(): parse_scalar(value)}
                stack.append((indent + 2, entry, "entry"))
                parent.append(entry)
            else:
                parent.append(parse_scalar(item_text))
            continue
        if not isinstance(parent, dict):
            raise ValueError(f"Mapping key inside a list: {raw!r}")
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"Expected key: value on {raw!r}")
        key, value = key.strip(), _strip_comment(value).strip()
        if value == "":
            child = _Deferred()
            parent[key] = child
            stack.append((indent, child, "map"))
        else:
            parent[key] = parse_scalar(value)

    def resolve(node):
        if isinstance(node, _Deferred):
            return resolve(node.value)
        if isinstance(node, dict):
            return {key: resolve(value) for key, value in node.items()}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(root)


def validate(spec):
    errors = []
    phases = spec.get("phases", [])
    by_id = {}
    for phase in phases:
        phase_id = phase.get("id")
        if not phase_id:
            errors.append("phase missing id")
            continue
        if phase_id in by_id:
            errors.append(f"duplicate phase id {phase_id}")
        by_id[phase_id] = phase
        for field in REQUIRED_METRIC_FIELDS:
            if field not in phase:
                errors.append(f"{phase_id}: missing field {field}")
        if phase.get("records") not in KNOWN_RECORDS:
            errors.append(f"{phase_id}: unknown record kind {phase.get('records')!r}")
        for gate in phase.get("gates", []):
            if gate.get("rule") not in KNOWN_RULES:
                errors.append(f"{phase_id}: gate {gate.get('id')!r} has unknown rule {gate.get('rule')!r}")
        for dependency in phase.get("depends", []):
            if dependency not in by_id:
                errors.append(f"{phase_id}: dependency {dependency} not yet declared")
    for required in REQUIRED_BASELINE_IDS:
        if required not in by_id:
            errors.append(f"required baseline phase {required} missing")
    for phase_id, variable in PLAN_RELAY_VARIABLES.items():
        declared = str(by_id.get(phase_id, {}).get("variable", ""))
        if variable not in declared:
            errors.append(f"{phase_id}: single changed variable must name {variable!r}")
    ordered = [phase.get("id") for phase in phases]
    if [pid for pid in ordered if pid and pid.startswith("P0.")] != REQUIRED_BASELINE_IDS:
        errors.append("baseline phases P0.1..P0.4 must be declared first, in order")
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=str(Path(__file__).with_name("phases.yaml")))
    parser.add_argument("--json", action="store_true", help="emit the parsed registry as JSON")
    args = parser.parse_args(argv)
    spec = parse_simple_yaml(Path(args.spec).read_text(encoding="utf-8"))
    if args.json:
        print(json.dumps(spec, indent=2))
        return 0
    errors = validate(spec)
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    count = len(spec.get("phases", []))
    print(f"OK: {count} phases valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
