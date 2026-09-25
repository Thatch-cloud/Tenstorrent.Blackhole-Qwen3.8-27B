import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from mlp_block_stream_pipeline_gate import source_record, validate_weights, qualify, READER_SHA256, HELPER_SHA256


class PipelineAdmissionTests(unittest.TestCase):
    def test_current_sources_reproduce_executed_pipeline_manifest(self):
        result = source_record(Path(__file__).parent)
        self.assertEqual(result['candidate_reader_sha256'], READER_SHA256)
        self.assertEqual(result['helper_sha256'], HELPER_SHA256)
        self.assertEqual(result['before'], '4b826527eec61ae0a24757c51cc4aed982d17157cb46b39d4d234dc73f80fdd9')
        self.assertEqual(result['after'], '8f4d3d3d9d3720ab03d3a7a22490ea95440d84ffb315da52f979b438dbb93e09')

    def test_all_weight_comparisons_are_required(self):
        checks = [dict(chip=chip, pages=43520, mismatched_words=0, exact=True,
            workers=64, source_exact=True, projection=projection)
            for projection in ('gate', 'up') for chip in (0, 1)]
        validate_weights(dict(weight_checks=checks))
        for mutate in (lambda value: value.pop(), lambda value: value[0].update(source_exact=False),
                lambda value: value[0].update(mismatched_words=1), lambda value: value[0].update(pages=1)):
            changed = copy.deepcopy(checks)
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_weights(dict(weight_checks=changed))

    def test_serial_admission_failure_is_not_bypassed(self):
        with patch('mlp_block_stream_pipeline_gate.qualify_serial', side_effect=ValueError('serial rejected')):
            with self.assertRaisesRegex(ValueError, 'serial rejected'):
                qualify('.', '.', '.', {})

    def test_other_report_cannot_be_relabelled_as_pipeline(self):
        with patch('mlp_block_stream_pipeline_gate.qualify_serial', return_value={}), \
                patch('mlp_block_stream_pipeline_gate.Path.read_bytes', return_value=b'{}'):
            with self.assertRaisesRegex(ValueError, 'Exact reviewed'):
                qualify('.', '.', '.', {})
