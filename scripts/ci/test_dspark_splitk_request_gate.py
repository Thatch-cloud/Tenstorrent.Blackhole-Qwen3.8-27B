from pathlib import Path
import os
import tempfile
import unittest

from dspark_splitk_request_gate import qualify


class RequestGateTests(unittest.TestCase):
    def test_fabricated_report_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / 'report.json'
            report.write_text('{"passed": true}')
            with self.assertRaisesRegex(ValueError, 'Exact completed'):
                qualify(temporary, report)

    @unittest.skipUnless(os.environ.get('QWEN_SPLITK_REQUEST_REPORT'), 'Retained hardware report required')
    def test_retained_hardware_audit_and_changed_source(self):
        import shutil

        directory = Path(__file__).parent
        report = Path(os.environ['QWEN_SPLITK_REQUEST_REPORT'])
        result = qualify(directory, report)
        self.assertTrue(result['correctness_screen_qualified'])
        self.assertFalse(result['performance_qualified'])
        with tempfile.TemporaryDirectory() as temporary:
            for name in result['combined_sources']:
                shutil.copyfile(directory / name, Path(temporary) / name)
            changed = Path(temporary) / 'dspark_splitk_combined_runtime.py'
            changed.write_text('changed runtime')
            with self.assertRaisesRegex(ValueError, 'integration changed'):
                qualify(temporary, report)
