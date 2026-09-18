from contextlib import contextmanager
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import dflash_combined_request as combined


class CombinedDFlashTests(unittest.TestCase):
    def exercise(self, fail=False, native=False, block_stream=False, runtime_only=False):
        active = []
        audits = []

        @contextmanager
        def scoped(name, audit):
            active.append(name)
            audit['restored'] = False
            audits.append(audit)
            try:
                yield audit
            finally:
                self.assertEqual(active.pop(), name)
                audit['restored'] = True

        target = dict(direct=dict(hits=48, restored=True), down=dict(hits=[1] * 64, restored=True))
        shared = dict(loads=[{}] * 48, released=True, admission=dict(report_sha256=combined.NORM_SHA256))
        fusion = SimpleNamespace(audit=dict(hits=[1] * 64), install=lambda: scoped('fusion', {}))
        module = SimpleNamespace(build=object(), scoped_shared_qk=lambda *args: scoped('shared', shared))
        model = SimpleNamespace(layers=[SimpleNamespace(feed_forward=SimpleNamespace(
            weights=SimpleNamespace(w_gate_up=index))) for index in range(64)])

        def measure(*args, **kwargs):
            self.assertEqual(active, (['native'] if native else []) +
                ['target', 'stream' if block_stream else 'register', 'shared', 'fusion'])
            self.assertIs(kwargs.get('native_proposal_attention', False), native)
            for key in ('proposal_capture', 'commit_only_gdn', 'fused_convolution', 'cache_history', 'target_attention_t16'):
                self.assertIs(kwargs[key], True)
            self.assertEqual(kwargs['block_rows'], 16)
            for index in range(48):
                module.build()
            if fail:
                raise RuntimeError('device request failed')
            return dict(selected_drafter='dflash2')

        with patch.dict(sys.modules, {
                'fused_t16_scope': SimpleNamespace(FusedT16Arm=lambda *args: fusion),
                'gdn_shared_qk_scope': module,
                'gdn_shared_qk_gate': SimpleNamespace(qualify=object()),
                'models.tt_transformers.tt.ccl': SimpleNamespace(tt_all_reduce=object())}), \
                patch('full_dflash_request.measure_dflash_request', side_effect=measure) as benchmark, \
                patch('dflash_t16_native_scope.scoped_native_t16', side_effect=lambda *args: scoped('native', {})), \
                patch('mlp_block_stream_runtime.scoped_block_stream',
                    side_effect=lambda *args, **kwargs: scoped('stream', dict(constructions=64, calls=64))), \
                patch('mlp_block_stream_request.validate_request') as stream_validation, \
                patch.object(combined, 'qualify_windows', return_value={}), \
                patch.object(combined, 'qualify_down', return_value={}), \
                patch.object(combined, 'qualify_norm', return_value={}), \
                patch.object(combined, 'scoped_cumulative_t16', side_effect=lambda *args, **kwargs: scoped('target', target)), \
                patch.object(combined, 'scoped_register_epilogue', side_effect=lambda *args, **kwargs: scoped('register', {})), \
                patch.object(combined, 'scatter_build', return_value=[1, 2, 3]), \
                patch.object(combined, 'validate_request'), patch.object(combined, 'validate_fusion_policy'):
            try:
                if runtime_only:
                    with combined.combined_runtime(None, model, directory='.', runtime_root='.',
                            native_attention_evidence='evidence',
                            block_stream=dict(evidence='evidence', streams=tuple(range(64)))) as runtime:
                        self.assertEqual(active, ['native', 'target', 'stream', 'shared', 'fusion'])
                        for index in range(48):
                            module.build()
                        if fail:
                            raise RuntimeError('serving request failed')
                    benchmark.assert_not_called()
                    self.assertEqual(runtime['builds'], [3] * 48)
                    self.assertTrue(runtime['stream_audit']['restored'])
                    return runtime
                result = combined.measure_combined_dflash(None, model, None, [1] * 4096, None, None,
                    directory='.', runtime_root='.', max_new_tokens=256,
                    **(dict(block_stream=dict(evidence='evidence', streams=tuple(range(64)))) if block_stream else {}),
                    **(dict(native_attention_evidence='evidence') if native else {}))
                self.assertEqual(stream_validation.call_count, int(block_stream))
                return result
            finally:
                self.assertEqual(active, [])
                self.assertTrue(all(audit['restored'] for audit in audits))

    def test_all_target_routes_wrap_complete_t16_request(self):
        result = self.exercise()
        self.assertEqual(result['norm_reader']['builds'], 48)
        self.assertTrue(result['register_epilogue']['register_resident'])

    def test_native_candidate_retains_all_target_scopes(self):
        self.exercise(native=True)
        with self.assertRaisesRegex(RuntimeError, 'device request failed'):
            self.exercise(native=True, fail=True)

    def test_failure_unwinds_every_target_route(self):
        with self.assertRaisesRegex(RuntimeError, 'device request failed'):
            self.exercise(fail=True)

    def test_block_stream_keeps_native_draft_and_other_target_routes(self):
        result = self.exercise(native=True, block_stream=True)
        self.assertEqual(result['fused_t16_mlp']['extra_weight_allocations'], 64)
        self.assertTrue(result['block_stream']['restored'])
        with self.assertRaisesRegex(RuntimeError, 'device request failed'):
            self.exercise(native=True, block_stream=True, fail=True)

    def test_serving_runtime_uses_same_scopes_without_benchmark_generation(self):
        self.exercise(runtime_only=True)
        with self.assertRaisesRegex(RuntimeError, 'serving request failed'):
            self.exercise(runtime_only=True, fail=True)
