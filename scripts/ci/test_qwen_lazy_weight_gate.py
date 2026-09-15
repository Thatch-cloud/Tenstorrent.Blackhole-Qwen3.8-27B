import os
from pathlib import Path
import tempfile
import unittest

from qwen_lazy_weight_gate import qualify


@unittest.skipUnless(os.environ.get('QWEN_LAZY_AUDIT_REPORT'), 'Retained hardware report required')
class LazyGateTests(unittest.TestCase):
    def test_full_retained_hardware_gate(self):
        evidence = qualify(Path(__file__).parent, os.environ['QWEN_LAZY_AUDIT_REPORT'])
        self.assertEqual(evidence['lazy_loader_audit_run'], 34936162975)
        self.assertEqual(evidence['lazy_loader']['records'][0]['materializations'], 0)

    def test_report_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'changed.json'
            report.write_bytes(Path(os.environ['QWEN_LAZY_AUDIT_REPORT']).read_bytes() + b' ')
            with self.assertRaises(ValueError):
                qualify(Path(__file__).parent, report)


if __name__ == '__main__':
    unittest.main()
