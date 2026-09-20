from contextlib import contextmanager, nullcontext
from functools import partial
from types import SimpleNamespace
import io
import json
import sys
import unittest
import torch
from unittest.mock import Mock, patch

from packed_verifier import m1_shape
import serving_packed_step
import serving_runtime
import serving_sequential_step
from test_serving_fast_policy import FastPolicyTests


class RuntimeAttachmentTests(unittest.TestCase):
    def exercise(self, fail=False, packed=False):
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
        block = SimpleNamespace(close=Mock(side_effect=lambda: events.append('block_close')),
                                describe=Mock(return_value=dict(name='packed-block')))
        operations = Mock()
        fixtures = ('manifests', ['layer'] * 5, {'fc.weight': 'projection'}, {'norm.weight': 'selector'})
        generator = Mock(return_value=Mock())

        def build_pool(*args, **kwargs):
            events.append('pool_build')
            return pool

        def build_weights(*args, **kwargs):
            events.append('weights_build')
            return weights

        def build_block(*args, **kwargs):
            events.append('block_build')
            return block

        with patch.dict('os.environ', {'QWEN_FAST_PACKED_STEP': '1' if packed else '0'}), \
                patch.dict(sys.modules, {
                'models.common.sampling.generator': SimpleNamespace(SamplingGenerator=generator),
                'models.tt_transformers.tt.ccl': SimpleNamespace(TT_CCL=Mock()),
                'gdn_snapshot': SimpleNamespace(ActiveSnapshot=Mock())}), \
                patch('dflash_combined_request.combined_runtime', side_effect=combined), \
                patch.object(serving_runtime, 'ServingCacheOwner'), \
                patch.object(serving_runtime, 'ServingBufferPool', side_effect=build_pool) as pooled, \
                patch.object(serving_runtime, 'PreparedDraftWeights', side_effect=build_weights) as prepared, \
                patch('packed_verifier.PackedVerifierEngine', side_effect=build_block) as packed_engine, \
                patch('sampling_link_policy.sampler_links', side_effect=lambda *args: nullcontext()) as links, \
                patch.object(serving_runtime, 'FastServingLifecycle', return_value=lifecycle) as install, \
                patch('sys.stdout', new_callable=io.StringIO) as out:
            try:
                with serving_runtime.attach_combined_runtime(worker, operations, directory='.', runtime_root='.',
                        fixtures=fixtures, native_attention_evidence='native',
                        block_stream={'streams': 'serial', 'evidence': 'stream'},
                        kv_publication_evidence='dma', eos_ids=(99,), cancelled=lambda: False) as attached:
                    # One stage line carrying the pool, the named shared weights and the
                    # device step that serves the rounds - with the block when it was built.
                    lines = [json.loads(line) for line in out.getvalue().splitlines() if line.startswith('{')]
                    step = (dict(serving_packed_step.describe(), block=dict(name='packed-block')) if packed
                            else serving_sequential_step.describe())
                    self.assertEqual(lines, [dict(stage='serving_buffer_pool', users=1,
                                                  draft_weights=dict(tensors=0, weights=[]), device_step=step)])
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
                    packed_step = install.call_args.kwargs['packed_step']
                    if packed:
                        # The packed block: over the same 48 helpers and the pinned sampler, the
                        # pool it restores from, the uploaded draft weights, the M1 shape at the
                        # pool's page width and the five feature taps; the step is bound to it.
                        packed_engine.assert_called_once()
                        args, options = packed_engine.call_args.args, packed_engine.call_args.kwargs
                        self.assertEqual((args[0], args[1], len(args[2])), (operations, model, 48))
                        self.assertIs(args[3], generator.return_value)
                        self.assertEqual(options, dict(pool=pool, shared_weights=weights, shape=m1_shape(68),
                                                       feature_taps=(5, 19, 33, 47, 61)))
                        self.assertIsInstance(packed_step, partial)
                        self.assertIs(packed_step.func, serving_packed_step.packed_device_step)
                        self.assertEqual((packed_step.args, packed_step.keywords), ((), dict(block=block)))
                    else:
                        # The serving default: no block is built, the sequential step is wired.
                        packed_engine.assert_not_called()
                        self.assertIs(packed_step, serving_sequential_step.sequential_packed_step)
                    events.append('request')
                    if fail:
                        raise RuntimeError('request failed')
            finally:
                # Pool first and closed last; weights inside the admitted runtime; both
                # outlive the lifecycle that lends them to devices. The block, when built,
                # comes after the weights and before the lifecycle, and closes between them.
                self.assertEqual(events, ['pool_build', 'runtime_enter', 'weights_build',
                                          *(['block_build'] if packed else []), 'request', 'lifecycle_close',
                                          *(['block_close'] if packed else []), 'weights_close', 'runtime_exit',
                                          'pool_close'])

    def test_combined_recipe_lives_until_request_traces_are_closed(self):
        self.exercise()

    def test_request_failure_still_closes_lifecycle_before_recipe(self):
        with self.assertRaisesRegex(RuntimeError, 'request failed'):
            self.exercise(fail=True)

    def test_the_packed_block_is_built_after_the_weights_and_before_the_lifecycle_only_when_asked(self):
        self.exercise(packed=True)

    def test_a_request_failure_closes_the_packed_block_after_the_lifecycle_and_before_the_weights(self):
        with self.assertRaisesRegex(RuntimeError, 'request failed'):
            self.exercise(fail=True, packed=True)
