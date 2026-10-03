import ast
from pathlib import Path
import unittest


class TraceRestoreTests(unittest.TestCase):
    def test_no_prefill_between_candidate_capture_and_replay(self):
        tree = ast.parse(Path(__file__).with_name("full-prefix.py").read_text())
        function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "batched")
        calls = [ast.unparse(node) for node in ast.walk(function) if isinstance(node, ast.Call)]
        self.assertIn("save(replay_initial)", calls)
        self.assertIn("restore(replay_initial)", calls)
        self.assertFalse(any(call.startswith("prefill(") for call in calls))

    def test_timing_prefill_precedes_fixture_allocations(self):
        tree = ast.parse(Path(__file__).with_name("full_batch_timing.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "measure")
        prefill = [index for index, node in enumerate(function.body) if ast.unparse(node) == "prefill()"]
        candidate = [index for index, node in enumerate(function.body)
                     if isinstance(node, ast.Assign) and ast.unparse(node).startswith("candidate = ModelBatch(")]
        self.assertEqual(len(prefill), 1)
        self.assertEqual(len(candidate), 1)
        self.assertLess(prefill[0], candidate[0])


if __name__ == "__main__":
    unittest.main()
