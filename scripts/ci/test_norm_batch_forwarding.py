import ast
from pathlib import Path
import unittest


class NormBatchForwardingTests(unittest.TestCase):
    def test_shared_masks_reach_both_retained_replay_boundaries(self):
        root = Path(__file__).resolve().parent
        for filename, called, expression in (
                ('full-prefix.py', 'verify_replay', 'options.attention_mask_once'),
                ('full_replay.py', 'ModelBatch', 'attention_mask_once')):
            tree = ast.parse((root / filename).read_text())
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == called]
            self.assertEqual(len(calls), 1)
            options = {keyword.arg: ast.unparse(keyword.value) for keyword in calls[0].keywords}
            self.assertEqual(options['attention_mask_once'], expression)

    def test_all_harness_boundaries_forward_the_explicit_option(self):
        root = Path(__file__).resolve().parent
        contracts = {
            'full_device_selection.py': {'ModelBatch': 1},
            'full_replay.py': {'ModelBatch': 1},
            'full_request.py': {'VerifierEngine': 1},
            'full-prefix.py': {'measure_request': 1, 'measure_selection': 1, 'verify_replay': 1},
        }
        for filename, expected in contracts.items():
            tree = ast.parse((root / filename).read_text())
            for name, count in expected.items():
                calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Name) and node.func.id == name]
                self.assertEqual(len(calls), count, (filename, name))
                for call in calls:
                    options = [keyword.value for keyword in call.keywords if keyword.arg == 'norm_batch']
                    self.assertEqual(len(options), 1, (filename, name))
                    self.assertIn(ast.unparse(options[0]), ('norm_batch', 'options.norm_batch'))


if __name__ == '__main__':
    unittest.main()
