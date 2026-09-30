import ast
from pathlib import Path
import unittest
from unittest.mock import patch

from gdn_direct_window_hardware_sources import batch, device, payloads


class DirectHardwareSourcesTests(unittest.TestCase):
    def test_device_math_and_work_partition_unchanged(self):
        source = Path(__file__).with_name('gdn_direct_window_device.py').read_text()
        original, changed = ast.parse(source), ast.parse(device(source))
        for tree in (original, changed):
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'execute')
            function.body = function.body[1:]
        self.assertEqual(ast.dump(original), ast.dump(changed))
        namespace = {'__file__': __file__}
        exec(device(source), namespace)
        with patch.dict('os.environ', {}, clear=True), self.assertRaisesRegex(ValueError, 'Allocated'):
            namespace['execute'](None, None, None, None, None, None, None, None)

    def test_recurrence_and_checkpoint_lifecycle_are_not_replaced(self):
        source = Path(__file__).with_name('gdn_batched_conv.py').read_text()
        changed = batch(source)
        suffix = '        prefixes = [None] * rows\n'
        self.assertEqual(source[source.index(suffix):], changed[changed.index(suffix):])
        self.assertIn('packed, windows = direct[:3], direct[3:]', changed)
        self.assertNotIn('owned.extend(packed)', changed)
        self.assertNotIn('build_windows(mesh, projected, conv_states)', changed)
        with self.assertRaises(ValueError):
            batch(source + '\n')
        self.assertEqual(len(payloads(Path(__file__).parent)), 2)
