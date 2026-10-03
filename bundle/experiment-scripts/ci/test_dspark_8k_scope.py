from contextlib import nullcontext
import unittest
from unittest.mock import patch

import dspark_8k_scope as candidate


class ScopeTests(unittest.TestCase):
    def test_all_runtime_bindings_restore_after_error(self):
        import dspark_full_attention
        import dspark_native_cached_layer
        import dspark_native_fixed_gate
        import dspark_stable_history
        import native_draft_sdpa
        from dspark_attention_chunk_trial import execute

        originals = (dspark_full_attention.MAX_CONTEXT, dspark_native_cached_layer.attend,
            dspark_native_fixed_gate.qualify, dspark_stable_history.StableHistoryKV,
            native_draft_sdpa.replacements)
        with patch.object(candidate, 'admitted_request', return_value=nullcontext({'component': 'test'})):
            with self.assertRaisesRegex(RuntimeError, 'request abort'):
                with candidate.runtime_scope('.', context=8192, output_tokens=256,
                        factory_root='.', build_evidence={}):
                    self.assertEqual(dspark_full_attention.MAX_CONTEXT, 8448)
                    self.assertIs(dspark_native_cached_layer.attend, execute)
                    self.assertIsNot(dspark_stable_history.StableHistoryKV, originals[3])
                    self.assertIn('QWEN_DRAFT_EXP_APPROX ||', native_draft_sdpa.replacements()['sdpa.cpp'][0][1])
                    with self.assertRaises(ValueError):
                        dspark_native_fixed_gate.qualify('unrelated-directory')
                    raise RuntimeError('request abort')
        self.assertEqual((dspark_full_attention.MAX_CONTEXT, dspark_native_cached_layer.attend,
            dspark_native_fixed_gate.qualify, dspark_stable_history.StableHistoryKV,
            native_draft_sdpa.replacements), originals)

    def test_extended_history_cannot_allocate_without_admission(self):
        from dspark_stable_history import StableHistoryKV
        extended = candidate.stable_history_class(StableHistoryKV)
        with self.assertRaises(ValueError):
            extended(None, None, None, None, None, None, None, position=8192, capacity=8448)


if __name__ == '__main__':
    unittest.main()
