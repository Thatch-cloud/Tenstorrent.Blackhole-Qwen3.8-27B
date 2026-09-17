import ast
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location('attention_replay_cli', Path(__file__).with_name('attention-replay.py'))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class ReplayGateTests(unittest.TestCase):
    def test_wide_gate_requires_compact_native_scratch_before_device_import(self):
        with patch.dict('os.environ', {'QWEN_HARDWARE_TESTS': '1', 'QWEN_CARDS_ALLOCATED': '1'}, clear=True), patch(
                'sys.argv', ['attention-replay.py', '--max-group-rows', '8']):
            with self.assertRaisesRegex(ValueError, 'compact native scratch'):
                gate.main()

    def test_selected_group_width_reaches_prepared_reader(self):
        tree = ast.parse(Path(__file__).with_name('attention-replay.py').read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'ReplayAttentionReader']
        self.assertEqual(len(calls), 1)
        options = {keyword.arg: ast.unparse(keyword.value) for keyword in calls[0].keywords}
        self.assertEqual(options['max_group_rows'], 'options.max_group_rows')
