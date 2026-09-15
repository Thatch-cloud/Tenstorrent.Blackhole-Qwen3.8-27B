import os
from pathlib import Path
import unittest

from dspark_64k_shared_gate import qualify


class SharedGateTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('QWEN_SHARED_AUDIT_REPORT'), 'Retained hardware audit required')
    def test_retained_combined_audit(self):
        evidence = qualify(Path(__file__).parent, os.environ['QWEN_SHARED_AUDIT_REPORT'])
        self.assertEqual(len(evidence['shared_audit']['loads']), 96)
        self.assertTrue(evidence['correctness_screen_qualified'])
        self.assertFalse(evidence['performance_qualified'])
