"""Profiler attribution rejects eager rows and incomplete replay evidence."""

import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("profile_check", Path(__file__).with_name("check-model-profile.py"))
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class ProfileTests(unittest.TestCase):
    def rows(self):
        return [dict(zip(("DEVICE ID", "METAL TRACE ID", "METAL TRACE REPLAY SESSION ID",
                         "DEVICE KERNEL DURATION [ns]", "OP CODE", "CORE COUNT"),
                        (device, "7", str(replay), "100", "Matmul", "32")))
                for device in ("0", "1") for replay in range(4)]

    def test_only_selected_trace_replays(self):
        rows = self.rows() + [{"METAL TRACE ID": "8"}, {"METAL TRACE ID": "7", "METAL TRACE REPLAY SESSION ID": ""}]
        result = checker.analyze(rows, 7, 3)
        self.assertEqual(result[0]["replay_sessions"], [1, 2, 3])
        self.assertEqual(result[1]["operations"][0]["median_summed_kernel_ns"], 100)

    def test_incomplete_and_inconsistent_evidence_fails(self):
        for rows in (self.rows()[:4], self.rows()[:-2], self.rows() + [self.rows()[-1]]):
            with self.assertRaises(AssertionError):
                checker.analyze(rows, 7, 3)

    def test_drop_warning_scope_requires_explicit_boundaries(self):
        clean = "QWEN_PROFILE_MEASURE_BEGIN\nstep\nQWEN_PROFILE_MEASURE_END"
        self.assertEqual(checker.measured_log("markers were dropped\n" + clean), 1)
        for log in ("", clean.replace("step", "markers were dropped"), clean + clean):
            with self.assertRaises(AssertionError):
                checker.measured_log(log)


if __name__ == "__main__":
    unittest.main()
