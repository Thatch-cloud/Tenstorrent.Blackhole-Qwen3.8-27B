"""Profiler attribution rejects eager rows and incomplete replay evidence."""

import importlib.util
import ast
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

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

    def test_report_staging_only_selects_measured_trace(self):
        spec = importlib.util.spec_from_file_location("stage_profile", Path(__file__).with_name("stage-model-profile.py"))
        stager = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stager)
        source = subprocess.check_output(["git", "-C", "/opt/ttsim/tt-metal", "show",
                                          "HEAD:tools/tracy/process_ops_logs.py"], text=True)
        helper = Path(__file__).resolve().parents[2] / "optimisation/sim/stage-continuation.py"
        with patch.dict(os.environ, QWEN_PROFILE_STAGE_HELPER=str(helper)):
            tree = ast.parse(stager.transform(source))
        function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "process_ops")
        selected = next(node for node in function.body if isinstance(node, ast.Assign) and isinstance(node.value, ast.DictComp))
        namespace = dict(generation={"decode_trace_id": 7}, ops={
            1: {"metal_trace_id": None}, 2: {"metal_trace_id": 6}, 3: {"metal_trace_id": 7}, 4: {"metal_trace_id": "7"}})
        exec(compile(ast.Module(body=[selected], type_ignores=[]), "select", "exec"), namespace)
        self.assertEqual(set(namespace["ops"]), {3, 4})


if __name__ == "__main__":
    unittest.main()
