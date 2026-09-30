import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

from draft_kv_slide_gate import DIRECT_REPORT_SHA256, REPORT_SHA256, validate_record


class PublicationAdmissionTests(unittest.TestCase):
    def admission(self, report):
        directory = Path(__file__).parent
        return dict(passed=True, checks=120, report_sha256=report, sources={
            name: hashlib.sha256((directory / filename).read_bytes()).hexdigest()
            for name, filename in (('history-append-probe.py', 'draft-kv-slide-probe.py'),
                ('draft_kv_slide.py', 'draft_kv_slide.py'), ('draft_kv_slide.cpp', 'draft_kv_slide.cpp'))})

    def test_original_admission_still_valid(self):
        validate_record(self.admission(REPORT_SHA256))

    def test_new_report_cannot_admit_original_kernel(self):
        with self.assertRaises(ValueError):
            validate_record(self.admission(DIRECT_REPORT_SHA256))

    def test_direct_requires_matching_staged_kernel_and_report(self):
        direct = Path(__file__).with_name('draft_kv_slide_direct.cpp').read_bytes()
        original_read = Path.read_bytes

        def staged_read(path):
            return direct if path.name == 'draft_kv_slide.cpp' else original_read(path)

        with patch.object(Path, 'read_bytes', staged_read):
            validate_record(self.admission(DIRECT_REPORT_SHA256))
            with self.assertRaises(ValueError):
                validate_record(self.admission(REPORT_SHA256))

    def test_unknown_report_and_incomplete_matrix_rejected(self):
        admission = self.admission('0' * 64)
        with self.assertRaises(ValueError):
            validate_record(admission)
        admission = self.admission(REPORT_SHA256)
        admission['checks'] = 119
        with self.assertRaises(ValueError):
            validate_record(admission)
