from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
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
        operations = Mock()
        with patch.dict(sys.modules, {
                'models.common.sampling.generator': SimpleNamespace(SamplingGenerator=Mock(return_value=Mock())),
                'models.tt_transformers.tt.ccl': SimpleNamespace(TT_CCL=Mock()),
                'gdn_snapshot': SimpleNamespace(ActiveSnapshot=Mock())}), \
                patch('dflash_combined_request.combined_runtime', side_effect=combined), \
                patch.object(serving_runtime, 'ServingCacheOwner'), \
                patch.object(serving_runtime, 'ServingBufferPool', return_value=pool) as pooled, \
                patch('sampling_link_policy.sampler_links', side_effect=lambda *args: nullcontext()) as links, \
                patch.object(serving_runtime, 'FastServingLifecycle', return_value=lifecycle) as install:
            try:
                with serving_runtime.attach_combined_runtime(worker, operations, directory='.', runtime_root='.',
                        fixtures=(), native_attention_evidence='native',
                        block_stream={'streams': 'serial', 'evidence': 'stream'},
                        kv_publication_evidence='dma', eos_ids=(99,), cancelled=lambda: False) as attached:
                    self.assertIs(attached['lifecycle'], lifecycle)
                    self.assertEqual(links.call_args.args[1], 4)
                    self.assertFalse(attached['serving_qualified'])
                    self.assertTrue(callable(install.call_args.kwargs['capture_factory']))
                    self.assertTrue(callable(install.call_args.kwargs['bridge_factory']))
                    # Sized for the scheduler and allocated before any request could trace.
                    pooled.assert_called_once_with(operations, model.mesh_device, users=1)
                    events.append('request')
                    if fail:
                        raise RuntimeError('request failed')
            finally:
                # The pool outlives the lifecycle that lends its slots to devices.
                self.assertEqual(events, ['runtime_enter', 'request', 'lifecycle_close', 'runtime_exit', 'pool_close'])

    def test_combined_recipe_lives_until_request_traces_are_closed(self):
        self.exercise()

    def test_request_failure_still_closes_lifecycle_before_recipe(self):
        with self.assertRaisesRegex(RuntimeError, 'request failed'):
            self.exercise(fail=True)
