import unittest

import native_draft_sdpa
from dspark_direct_fp32_screen import summarize_screen
from test_dspark_sfpu_request_screen import request


class DirectScreenTests(unittest.TestCase):
    def test_actual_kernel_identity_and_existing_audits_required(self):
        expected = dict.fromkeys(native_draft_sdpa.SOURCE_HASHES, 'candidate')
        value = request()
        value.update(target_attention_t16=True, attention_replay=True, family_routing=True, capture_count=5,
            native_attention_kernel=dict(original=native_draft_sdpa.SOURCE_HASHES, patched=dict(expected)))
        options = dict(expected=expected, admission={'numerical_qualified': True}, sources={'candidate.py': 'hash'})
        result = summarize_screen([value], **options)
        self.assertTrue(result['direct_fp32_stage'])
        self.assertFalse(result['performance_qualified'])
        value['native_attention_kernel']['patched']['compute_common.hpp'] = 'old-kernel'
        with self.assertRaises(ValueError):
            summarize_screen([value], **options)
        value['native_attention_kernel']['patched'] = dict(expected)
        value['state_exact'] = False
        with self.assertRaises(ValueError):
            summarize_screen([value], **options)


if __name__ == '__main__':
    unittest.main()
