import ast
from pathlib import Path
import unittest
from unittest.mock import patch

from compact_score_hardware_sources import GUARD, HARDWARE_GUARD, payloads, transform


class HardwareSourcesTests(unittest.TestCase):
    def test_only_environment_guard_and_import_binding_change(self):
        directory = Path(__file__).parent
        outputs = payloads(directory)
        for kind, name in (('device', 'compact_score_device.py'), ('markov', 'compact_markov.py')):
            original = (directory / name).read_text()
            changed = outputs[f'compact_score_hardware_{kind}.py']
            original_tree, changed_tree = ast.parse(original), ast.parse(changed)
            for tree in (original_tree, changed_tree):
                tree.body = [node for node in tree.body if not isinstance(node, ast.ImportFrom)
                             or node.module not in ('compact_score_device', 'compact_score_hardware_device')]
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef) and node.name in ('execute', 'execute_local_winners', 'reduce_winners'):
                        node.body = node.body[1:]
            self.assertEqual(ast.dump(original_tree), ast.dump(changed_tree))
            with self.assertRaises(ValueError):
                transform(original.replace(GUARD, HARDWARE_GUARD), kind)

    def test_generated_device_refuses_unallocated_or_simulator_execution(self):
        source = payloads(Path(__file__).parent)['compact_score_hardware_device.py']
        namespace = {'__file__': __file__}
        exec(compile(source, 'hardware-device', 'exec'), namespace)
        enabled = dict(QWEN_COMPACT_SCORE_HARDWARE='1', QWEN_CARDS_ALLOCATED='1',
                       QWEN_HARDWARE_TESTS='1', QWEN_FROZEN_COMBINED_RUNTIME='1')
        cases = [{}, *[{key: value for key, value in enabled.items() if key != missing} for missing in enabled],
                 dict(enabled, TT_METAL_SIMULATOR='test'), dict(enabled, QWEN_SIM_ONLY='1')]
        for environment in cases:
            with patch.dict('os.environ', environment, clear=True):
                with self.assertRaisesRegex(ValueError, 'Allocated'):
                    namespace['execute_local_winners'](None, None, None, None, 0, None)
                with self.assertRaisesRegex(ValueError, 'Allocated'):
                    namespace['reduce_winners'](None, None, None, 64, None)
