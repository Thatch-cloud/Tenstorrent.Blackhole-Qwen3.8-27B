from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
import io
import json
import sys
import unittest
import torch
from unittest.mock import Mock, patch

import serving_runtime
from test_serving_fast_policy import FastPolicyTests


class RuntimeAttachmentTests(unittest.TestCase):
    def exercise(self, fail=False):
        events = []
        model = SimpleNamespace(args=object(), mesh_device=object(),
            layers=[SimpleNamespace(is_full_attention=False, attention=object()) for _ in range(48)])
        worker = SimpleNamespace(vllm_config=FastPolicyTests().fixture(),
            model_runner=SimpleNamespace(model=SimpleNamespace(model=[model])))

        @contextmanager
        def combined(*args, **kwargs):
            self.assertEqual(kwargs['block_stream'], {'streams': 'serial', 'evidence': 'stream'})
            self.assertEqual(kwargs['kv_publication_evidence'], 'dma')
            events.append('runtime_enter')
            try:
                yield {'admitted': True}
            finally:
                events.append('runtime_exit')

        lifecycle = SimpleNamespace(close=Mock(side_effect=lambda: events.append('lifecycle_close')))
        pool = SimpleNamespace(close=Mock(side_effect=lambda: events.append('pool_close')),
                               describe=Mock(return_value=dict(users=1)))
        weights = SimpleNamespace(close=Mock(side_effect=lambda: events.append('weights_close')), tensors=[],
                                  describe=Mock(return_value=dict(tensors=0, weights=[])))
        operations = Mock()
        fixtures = ('manifests', ['layer'] * 5, {'fc.weight': 'projection'}, {'norm.weight': 'selector'})

        def build_pool(*args, **kwargs):
            events.append('pool_build')
            return pool

        def build_weights(*args, **kwargs):
            events.append('weights_build')
            return weights

        with patch.dict(sys.modules, {
                'models.common.sampling.generator': SimpleNamespace(SamplingGenerator=Mock(return_value=Mock())),
                'models.tt_transformers.tt.ccl': SimpleNamespace(TT_CCL=Mock()),
                'gdn_snapshot': SimpleNamespace(ActiveSnapshot=Mock())}), \
                patch('dflash_combined_request.combined_runtime', side_effect=combined), \
                patch.object(serving_runtime, 'ServingCacheOwner'), \
                patch.object(serving_runtime, 'ServingBufferPool', side_effect=build_pool) as pooled, \
                patch.object(serving_runtime, 'PreparedDraftWeights', side_effect=build_weights) as prepared, \
                patch('sampling_link_policy.sampler_links', side_effect=lambda *args: nullcontext()) as links, \
                patch.object(serving_runtime, 'FastServingLifecycle', return_value=lifecycle) as install, \
                patch('sys.stdout', new_callable=io.StringIO) as out:
            try:
                with serving_runtime.attach_combined_runtime(worker, operations, directory='.', runtime_root='.',
                        fixtures=fixtures, native_attention_evidence='native',
                        block_stream={'streams': 'serial', 'evidence': 'stream'},
                        kv_publication_evidence='dma', eos_ids=(99,), cancelled=lambda: False) as attached:
                    # One stage line carrying both the pool and the named shared weights.
                    lines = [json.loads(line) for line in out.getvalue().splitlines() if line.startswith('{')]
                    self.assertEqual(lines, [dict(stage='serving_buffer_pool', users=1,
                                                  draft_weights=dict(tensors=0, weights=[]))])
                    self.assertIs(attached['lifecycle'], lifecycle)
                    self.assertEqual(links.call_args.args[1], 4)
                    self.assertFalse(attached['serving_qualified'])
                    self.assertTrue(callable(install.call_args.kwargs['capture_factory']))
                    self.assertTrue(callable(install.call_args.kwargs['bridge_factory']))
                    # Sized for the scheduler and allocated before anything else in the attachment,
                    # with the verifier geometry every request will ask for: the 48 GDN helpers,
                    # the one page-table width, the capture widths with multiplicity for a
                    # 256-token budget under a T16 cap, the five feature taps and the rotary
                    # builder (which imports the native construction only when called).
                    pooled.assert_called_once()
                    self.assertEqual(pooled.call_args.args, (operations, model.mesh_device))
                    options = pooled.call_args.kwargs
                    self.assertEqual(options['users'], 1)
                    self.assertEqual(len(options['helpers']), 48)
                    self.assertEqual(options['page_width'], 68)
                    self.assertEqual(options['bucket_rows'], (1, 2, 4, 8, 8, 16, 16))
                    self.assertEqual(options['feature_taps'], 5)
                    self.assertTrue(callable(options['rope']))
                    self.assertEqual(set(options), {'users', 'helpers', 'page_width', 'bucket_rows', 'feature_taps', 'rope'})
                    # The device geometry from_prefill asks for, prepared once inside the
                    # admitted runtime and before any request.
                    prepared.assert_called_once_with(operations, model.mesh_device, fixtures[1], fixtures[2], fixtures[3],
                        block_rows=16, live_query_qk=False, native_proposal_attention=True)
                    events.append('request')
                    if fail:
                        raise RuntimeError('request failed')
            finally:
                # Pool first and closed last; weights inside the admitted runtime; both
                # outlive the lifecycle that lends them to devices.
                self.assertEqual(events, ['pool_build', 'runtime_enter', 'weights_build', 'request',
                                          'lifecycle_close', 'weights_close', 'runtime_exit', 'pool_close'])

    def test_combined_recipe_lives_until_request_traces_are_closed(self):
        self.exercise()

    def test_request_failure_still_closes_lifecycle_before_recipe(self):
        with self.assertRaisesRegex(RuntimeError, 'request failed'):
            self.exercise(fail=True)
