import os
import unittest
from unittest.mock import patch

import dspark_64k_admission as admission


class AdmissionTests(unittest.TestCase):
    def test_exact_request_shape(self):
        admission.validate_request(65536, 256)
        for context, output in ((8192, 256), (65536, 1024), (True, 256), (65536, True)):
            with self.assertRaises(ValueError):
                admission.validate_request(context, output)

    def test_validates_both_evidence_sources_and_resets_on_failure(self):
        arguments = dict(context=65536, output_tokens=256, factory_root='/native', build_path='/build')
        with patch.dict(os.environ, {'QWEN_LADDER_BACKEND': 'hardware'}), \
                patch.object(admission, 'qualify', return_value={'report_sha256': 'report'}) as component, \
                patch.object(admission, 'validate_build', return_value={
                    'backend': 'hardware', 'passed': True, 'factory_sha256': 'factory'}) as build:
            with self.assertRaisesRegex(RuntimeError, 'abort'):
                with admission.admitted_request('/scripts', '/report', **arguments) as evidence:
                    component.assert_called_once_with('/scripts', '/report')
                    build.assert_called_once_with('/native', '/build', '/scripts')
                    self.assertEqual(admission.current_admission()['capacity'], 66560)
                    evidence['capacity'] = 1
                    self.assertEqual(admission.current_admission()['capacity'], 66560)
                    with self.assertRaisesRegex(ValueError, 'Nested'):
                        with admission.admitted_request('/scripts', '/report', **arguments):
                            pass
                    raise RuntimeError('abort')
            self.assertIsNone(admission.current_admission())

    def test_no_admission_when_native_build_validation_fails(self):
        with patch.dict(os.environ, {'QWEN_LADDER_BACKEND': 'hardware'}), \
                patch.object(admission, 'qualify', return_value={'report_sha256': 'report'}), \
                patch.object(admission, 'validate_build', side_effect=ValueError('binary changed')):
            with self.assertRaisesRegex(ValueError, 'binary changed'):
                with admission.admitted_request('/scripts', '/report', context=65536,
                        output_tokens=256, factory_root='/native', build_path='/build'):
                    self.fail('Invalid build admitted')
            self.assertIsNone(admission.current_admission())
