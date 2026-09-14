import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dspark_attention_64k_gate import qualify


class HardwareGateTests(unittest.TestCase):
    def test_unpinned_report_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'report.json'
            report.write_text('{"passed": true}')
            with self.assertRaisesRegex(ValueError, 'Exact retained'):
                qualify(directory, report)

    @unittest.skipUnless(os.environ.get('QWEN_64K_REPORT'), 'Retained hardware artifact required')
    def test_real_report_and_dependency_mutation(self):
        directory = Path(__file__).parent
        report = Path(os.environ['QWEN_64K_REPORT'])
        evidence = qualify(directory, report)
        self.assertTrue(evidence['component_qualified'])
        self.assertFalse(evidence['runtime_admitted'])
        self.assertFalse(evidence['performance_qualified'])
        original = Path.read_bytes

        def modified(path):
            data = original(path)
            return data + b'\n' if path.name == 'dspark_ladder_normalization.py' else data

        with patch.object(Path, 'read_bytes', modified):
            with self.assertRaisesRegex(ValueError, 'dependency changed'):
                qualify(directory, report)
