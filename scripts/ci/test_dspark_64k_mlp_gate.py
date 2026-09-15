import os
from pathlib import Path
import unittest

from dspark_64k_mlp_gate import qualify


class MlpGateTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('QWEN_MLP_AUDIT_REPORT'), 'Retained combined hardware audit required')
    def test_real_combined_audit(self):
        result = qualify(Path(__file__).parent, os.environ['QWEN_MLP_AUDIT_REPORT'])
        self.assertTrue(result['correctness_screen_qualified'])
        self.assertFalse(result['performance_qualified'])
        self.assertEqual(len(result['mlp_audit']['hits']), 64)
