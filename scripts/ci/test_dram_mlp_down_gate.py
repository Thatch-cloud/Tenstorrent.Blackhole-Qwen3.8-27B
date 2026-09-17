import json
from pathlib import Path
import unittest
from unittest.mock import patch

import dram_mlp_down_gate as gate


class DownGateTests(unittest.TestCase):
    def test_recorded_repeat_qualifies_only_matching_sources(self):
        root = Path(__file__).parent
        native_root = Path('/mock-native')
        native = json.loads((root / 'dram-mlp-down-hardware.json').read_text())['native_sources']
        original_digest = gate.digest
        def digest(path):
            if path.is_relative_to(native_root):
                return native[path.relative_to(native_root).as_posix()]
            return original_digest(path)
        with patch.object(gate, 'digest', side_effect=digest):
            result = gate.qualify_repeated(root, native_root)
        self.assertTrue(result['passed'])
        self.assertEqual(len(set(result['hardware_sha256'])), 2)
        for changed in ('dram_mlp_down.py', 'dram-mlp-hardware.py'):
            with patch.object(gate, 'digest', side_effect=lambda path: '0' * 64
                    if path.name == changed else digest(path)):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    gate.qualify_repeated(root, native_root)
        with patch.object(gate, 'digest', side_effect=digest), patch.object(gate, 'qualify_hardware',
                return_value=dict(eligible_for_full_model_gate=False)):
            with self.assertRaises(ValueError):
                gate.qualify_repeated(root, native_root)


if __name__ == '__main__':
    unittest.main()
