"""Cross-arm comparisons must not accept missing or divergent evidence."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location("comparison", Path(__file__).with_name("compare-interleave.py"))
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class CompareTests(unittest.TestCase):
    def compare(self, mutation=None, isolated=False, mixed=False):
        with tempfile.TemporaryDirectory() as directory, mock.patch("builtins.print"):
            control, arm = Path(directory) / "control", Path(directory) / "arm"
            expected = [f"boundary-{length}" for length in (63, 64, 65, 2047, 2048, 2049, 4096, 8193)] + ["after-cancel"]
            if isolated:
                expected = [f"isolated-2049-{repeat}" for repeat in range(3)] + expected
            for root in (control, arm):
                root.mkdir()
                (root / "interleave-summary.json").write_text(json.dumps(dict(passed=True, results=expected)))
                for name in expected:
                    (root / f"{name}.json").write_text(json.dumps(dict(passed=True, prompt_tokens=100,
                        requested_tokens=32, output_sha256="output", token_ids_sha256="tokens")))
                if mixed:
                    requests = {name: dict(tokens=[1, 2], text_sha256="text") for name in ("B0", "A", "C")}
                    (root / "mixed-traffic.json").write_text(json.dumps(dict(passed=True, baseline=requests, concurrent=requests)))
            if mutation:
                mutation(arm)
            comparison.compare(control, arm)
            return json.loads((arm / "whole-chunked-comparison.json").read_text())

    def test_all_nine_checks_required(self):
        self.assertEqual(self.compare()["checks"], 9)

    def test_mixed_outputs_also_match_control(self):
        self.assertEqual(self.compare(mixed=True)["mixed_checks"], 3)
        def mutate(arm):
            path = arm / "mixed-traffic.json"
            report = json.loads(path.read_text())
            report["concurrent"]["B0"]["tokens"] = [1, 3]
            path.write_text(json.dumps(report))
        with self.assertRaisesRegex(AssertionError, "Mixed-traffic cross-arm divergence"):
            self.compare(mutate, mixed=True)

    def test_diagnostic_repeats_are_also_equivalence_gated(self):
        self.assertEqual(self.compare(isolated=True)["checks"], 12)
        with self.assertRaises(FileNotFoundError):
            self.compare(lambda arm: (arm / "isolated-2049-2.json").unlink(), isolated=True)

    def test_missing_request_fails(self):
        with self.assertRaises(FileNotFoundError):
            self.compare(lambda arm: (arm / "boundary-63.json").unlink())

    def test_divergent_output_fails(self):
        def mutate(arm):
            path = arm / "boundary-63.json"
            report = json.loads(path.read_text())
            report["output_sha256"] = "different"
            path.write_text(json.dumps(report))
        with self.assertRaisesRegex(AssertionError, "Continuation divergence"):
            self.compare(mutate)

    def test_incomplete_summary_fails(self):
        with self.assertRaisesRegex(AssertionError, "Incomplete arm"):
            self.compare(lambda arm: (arm / "interleave-summary.json").write_text('{"passed":true,"results":[]}'))


if __name__ == "__main__":
    unittest.main(verbosity=2)
