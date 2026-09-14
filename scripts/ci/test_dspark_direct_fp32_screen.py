import unittest
from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
from unittest.mock import patch

import native_draft_sdpa
from dspark_direct_fp32_screen import summarize_screen
from test_dspark_sfpu_request_screen import request


class DirectScreenTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('TT_NATIVE_TEST_ROOT'), 'Pinned native sources required')
    def test_actual_runtime_source_requires_early_staging_scope(self):
        import dspark_64k_scope
        from dspark_direct_fp32_stage import staging_scope
        from dspark_sum_sfpu import sum_scope
        from dspark_score_sfpu import kernel_scope
        from dspark_score_bitwise import bitwise_infinity_checks
        from dspark_mask_bits import mask_scope

        root = Path(os.environ['TT_NATIVE_TEST_ROOT'])
        original = {name: (root / native_draft_sdpa.KERNEL_DIRECTORY / name).read_bytes()
            for name in native_draft_sdpa.SOURCE_HASHES}

        @contextmanager
        def admission(*args, **kwargs):
            yield {'context': 65536, 'capacity': 66560}

        for early in (False, True):
            with ExitStack() as stack:
                stack.enter_context(patch.object(dspark_64k_scope, 'admitted_request', admission))
                if early:
                    stack.enter_context(staging_scope())
                for scope in (sum_scope(), kernel_scope(), bitwise_infinity_checks(),
                        dspark_64k_scope.runtime_scope('.', '.', context=65536, output_tokens=256,
                            factory_root=root, build_path='unused'), mask_scope()):
                    stack.enter_context(scope)
                if not early:
                    stack.enter_context(staging_scope())
                source = native_draft_sdpa.patched_sources(original)['compute_common.hpp']
                self.assertEqual(source.count(b'qwen_stage_score_tile(in0_cb, QWEN_SCORE_SCRATCH_CB, true);'),
                    1 if early else 0)

    def test_staging_installed_before_runtime_arithmetic_contexts(self):
        entry = Path(__file__).with_name('dspark_64k_entry.py').read_text()
        self.assertIn('with direct_candidate_scope, hardware_candidate_scope, runtime_scope(', entry)
        screen = Path(__file__).with_name('dspark_direct_fp32_screen.py').read_text()
        self.assertNotIn('with staging_scope()', screen)

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
