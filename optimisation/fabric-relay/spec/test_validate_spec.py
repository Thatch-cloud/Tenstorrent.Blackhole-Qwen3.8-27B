"""Host-only tests for the fabric-relay phase registry validator."""

import unittest
from pathlib import Path

from validate_spec import parse_simple_yaml, validate

SPEC_PATH = Path(__file__).with_name("phases.yaml")


class ParserTests(unittest.TestCase):
    def test_scalar_types(self):
        self.assertEqual(parse_simple_yaml("a: 3"), {"a": 3})
        self.assertEqual(parse_simple_yaml("a: 1.5"), {"a": 1.5})
        self.assertEqual(parse_simple_yaml("a: hello"), {"a": "hello"})
        self.assertEqual(parse_simple_yaml("a: [1, 2]"), {"a": [1, 2]})
        self.assertEqual(parse_simple_yaml("a: {x: 1, y: s}"), {"a": {"x": 1, "y": "s"}})

    def test_nested_mapping_and_list(self):
        text = "phases:\n  - id: P0.1\n    gates:\n      - id: repeats\n        params: {count: 9}\n"
        spec = parse_simple_yaml(text)
        phase = spec["phases"][0]
        self.assertEqual(phase["gates"][0]["params"]["count"], 9)


class RegistryTests(unittest.TestCase):
    def test_shipped_registry_is_valid(self):
        spec = parse_simple_yaml(SPEC_PATH.read_text(encoding="utf-8"))
        self.assertEqual(validate(spec), [])

    def test_missing_baseline_fails(self):
        spec = parse_simple_yaml(SPEC_PATH.read_text(encoding="utf-8"))
        spec["phases"] = [p for p in spec["phases"] if p["id"] != "P0.2"]
        self.assertTrue(any("P0.2" in e for e in validate(spec)))

    def test_unknown_gate_rule_fails(self):
        spec = parse_simple_yaml(SPEC_PATH.read_text(encoding="utf-8"))
        spec["phases"][0]["gates"][0]["rule"] = "vibes"
        self.assertTrue(any("unknown rule" in e for e in validate(spec)))

    def test_candidate_dependency_ordering_enforced(self):
        spec = parse_simple_yaml(SPEC_PATH.read_text(encoding="utf-8"))
        first_candidate = next(p for p in spec["phases"] if p["id"] == "P1")
        moved = dict(first_candidate)
        moved["id"] = "P0.0"
        moved["depends"] = ["P9"]
        spec["phases"] = [moved] + spec["phases"]
        self.assertTrue(any("not yet declared" in e for e in validate(spec)))


if __name__ == "__main__":
    unittest.main()
