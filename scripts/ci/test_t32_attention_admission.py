import os
import unittest
from unittest.mock import patch

from t32_attention_admission import PATCHED, require_active


class AdmissionTests(unittest.TestCase):
    def test_validated_runtime_is_not_full_request_qualification(self):
        environment = dict(QWEN_SIM_ONLY='1', QWEN_PRECISE_DRAFT_ACTIVE='1', QWEN_T32_SFPU_SUM='1',
                           TT_METAL_HOME='/runtime')
        with patch.dict(os.environ, environment, clear=True), \
                patch('t32_attention_admission.audit_active_kernel', return_value=dict(patched=PATCHED)), \
                patch('t32_attention_admission.fingerprints', return_value={'runtime': 'verified'}):
            self.assertFalse(require_active()['full_request_qualified'])
            for name, value in (('QWEN_T32_SFPU_SUM', '0'), ('QWEN_T32_NUMERATOR_TAP', '2'),
                                ('QWEN_CARDS_ALLOCATED', '1'), ('QWEN_T32_FP32_BUILD', '1')):
                with patch.dict(os.environ, {name: value}), self.assertRaises(ValueError):
                    require_active()

    def test_missing_policy_fails_before_runtime_read(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch('t32_attention_admission.audit_active_kernel') as audit, self.assertRaises(ValueError):
            require_active()
        audit.assert_not_called()
