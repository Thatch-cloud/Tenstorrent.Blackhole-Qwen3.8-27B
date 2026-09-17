import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dspark_64k_score_gate import qualify


@unittest.skipUnless(os.environ.get('QWEN_SCORE_AUDIT_REPORT'), 'Retained combined hardware audit required')
class ScoreGateTests(unittest.TestCase):
    def test_retained_combined_audit(self):
        evidence = qualify(Path(__file__).parent, os.environ['QWEN_SCORE_AUDIT_REPORT'])
        self.assertTrue(evidence['correctness_screen_qualified'])
        self.assertFalse(evidence['performance_qualified'])
        self.assertEqual(evidence['score_audit']['calls'], 5)

    def test_changed_report_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'changed.json'
            path.write_bytes(Path(os.environ['QWEN_SCORE_AUDIT_REPORT']).read_bytes() + b' ')
            with self.assertRaises(ValueError):
                qualify(Path(__file__).parent, path)

    def test_nested_identity_uses_score_audit_with_stubbed_numerical_gate(self):
        from dspark_64k_score_timed import qualify as qualify_timing
        import dspark_center_fill_timed as center
        from dspark_64k_score_gate import SCREEN_RUN, SCREEN_SHA256
        expected = qualify(Path(__file__).parent, os.environ['QWEN_SCORE_AUDIT_REPORT'])['request']
        def numerical(directory, reports):
            self.assertEqual(center.SCREEN_RUN, SCREEN_RUN)
            self.assertEqual(center.SCREEN_SHA256, SCREEN_SHA256)
            return expected
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dspark-sfpu-request-screen.json'
            path.write_bytes(Path(os.environ['QWEN_SCORE_AUDIT_REPORT']).read_bytes())
            with patch.object(center, 'qualify', numerical):
                evidence = qualify_timing(Path(__file__).parent, directory)
            self.assertEqual(evidence, expected)
