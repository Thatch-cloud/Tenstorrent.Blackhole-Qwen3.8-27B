from pathlib import Path
import unittest
from unittest.mock import patch

import dspark_native_fixed_gate as gate


class NativeFixedGateTests(unittest.TestCase):
    def test_current_pinned_report_and_sources_qualify(self):
        self.assertEqual(gate.qualify(Path(__file__).parent), {gate.REPORT: gate.SHA256})

    def test_changed_adapter_fails_closed(self):
        original = gate.digest
        with patch.object(gate, 'digest', side_effect=lambda path:
                'changed' if path.name == 'dspark_native_full_attention.py' else original(path)):
            with self.assertRaisesRegex(ValueError, 'source changed'):
                gate.qualify(Path(__file__).parent)

    def test_changed_report_fails_closed(self):
        with patch.object(gate, 'digest', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'Pinned successful'):
                gate.qualify(Path(__file__).parent)
