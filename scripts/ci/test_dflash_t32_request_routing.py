from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from full_dflash_request import measure_dflash_request


class ReachedVerifier(RuntimeError):
    pass


class T32RequestRoutingTests(unittest.TestCase):
    def options(self):
        return dict(fixtures=(None, None, None, None), prefill=Mock(), decode=Mock(),
            live_digest=Mock(), kv_digest=Mock(), inactive_digest=Mock(), eos_ids=[],
            max_new_tokens=256, block_rows=32, proposal_capture=True, commit_only_gdn=True,
            fused_convolution=True, cache_history=True, native_proposal_attention=True,
            target_combined_t32=True)

    def test_t32_routes_parallel_verifier_and_defers_proposal_capture(self):
        measure = Mock(side_effect=ReachedVerifier())
        modules = {'full_request': SimpleNamespace(measure_request=measure),
            'models.tt_transformers.tt.ccl': SimpleNamespace(TT_CCL=Mock())}
        with patch.dict('sys.modules', modules), \
                patch('dflash_t32_native_scope.require_active', return_value={'block_rows': 32}) as admission:
            with self.assertRaises(ReachedVerifier):
                measure_dflash_request(None, None, None, [1] * 4096, None, None, **self.options())
        admission.assert_called_once()
        options = measure.call_args.kwargs
        self.assertEqual(options['lookup_max_rows'], 32)
        self.assertTrue(options['attention_replay'] and options['family_routing'])
        self.assertNotIn('target_attention_t16', options)
        self.assertTrue(callable(options['verifier_before_capture']))

    def test_partial_or_bare_flag_configuration_never_reaches_verifier(self):
        for changed in ({'target_combined_t32': False}, {'block_rows': 16},
                {'target_attention_t16': True}, {'cache_history': False},
                {'native_proposal_attention': False}, {'commit_only_gdn': False}):
            measure = Mock(side_effect=ReachedVerifier())
            modules = {'full_request': SimpleNamespace(measure_request=measure),
                'models.tt_transformers.tt.ccl': SimpleNamespace(TT_CCL=Mock())}
            with patch.dict('sys.modules', modules), self.assertRaises(ValueError):
                measure_dflash_request(None, None, None, [1] * 4096, None, None,
                    **{**self.options(), **changed})
            measure.assert_not_called()


if __name__ == '__main__':
    unittest.main()
