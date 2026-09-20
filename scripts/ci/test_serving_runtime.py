from contextlib import contextmanager, nullcontext
from functools import partial
from types import SimpleNamespace
import io
import json
import sys
import unittest
import torch
from unittest.mock import Mock, patch

from packed_shapes import m1_shape, m3_shape
import serving_packed_step
import serving_runtime
import serving_sequential_step
from test_serving_fast_policy import FastPolicyTests

# The block the runtime builds per scheduler request count under QWEN_FAST_PACKED_STEP=1
# (packed_shapes.serving_shape): two take M1, four take M3, any other count none.
SHAPES = {2: m1_shape(68), 4: m3_shape(68)}


class RuntimeAttachmentTests(unittest.TestCase):
    ATTACH_FAILED = ('[PINDIAG] attach failed with RuntimeError: attach failed; closing the attach scopes now (the '
                     'packed block if built, the draft weights, the admitted combined runtime from its shared-QK '
                     'scope, the sampler links, the pool) - their exits fence the device, and a hung device blocks '
                     'the first fence')

    def exercise(self, fail=False, packed=False, users=1, attach_fail=False, probe=None):
        events = []

        def diag(template, *values):
            # the attach-failed line lands among the events, so its place before the scope
            # closes is checked; every other line is inspected through the mock
            if template.startswith('[PINDIAG] attach failed'):
                events.append(('diag', template.format(*values)))
        shape = SHAPES.get(users) if packed else None
        built = shape is not None
        # beside the four-user block the per-request captures are trimmed to (1, 2, 4)
        trimmed = built and shape.users == 4
        model = SimpleNamespace(args=object(), mesh_device=object(),
            layers=[SimpleNamespace(is_full_attention=False, attention=object()) for _ in range(48)])
        config = FastPolicyTests().fixture()
        config.scheduler_config.max_num_seqs = users
        # more than one request installs the one-in-flight scheduler unless one is named
        config.scheduler_config.scheduler_cls = 'named'
        worker = SimpleNamespace(vllm_config=config, model_runner=SimpleNamespace(model=SimpleNamespace(model=[model])))

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
                               describe=Mock(return_value=dict(users=users)))
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
                patch.object(serving_runtime, 'pindiag', side_effect=diag) as diagnostic, \
                patch('packed_verifier.PackedVerifierEngine', side_effect=build_block) as packed_engine, \
                patch('sampling_link_policy.sampler_links', side_effect=lambda *args: nullcontext()) as links, \
                patch.object(serving_runtime, 'FastServingLifecycle', return_value=lifecycle,
                             side_effect=RuntimeError('attach failed') if attach_fail else None) as install, \
                patch('sys.stdout', new_callable=io.StringIO) as out:
            try:
                with serving_runtime.attach_combined_runtime(worker, operations, directory='.', runtime_root='.',
                        fixtures=fixtures, native_attention_evidence='native',
                        block_stream={'streams': 'serial', 'evidence': 'stream'},
                        kv_publication_evidence='dma', eos_ids=(99,), cancelled=lambda: False) as attached:
                    # One stage line carrying the pool, the named shared weights and the
                    # device step that serves the rounds - with the block when it was built,
                    # and why none was when the switch asked for one.
                    lines = [json.loads(line) for line in out.getvalue().splitlines() if line.startswith('{')]
                    if built:
                        step = dict(serving_packed_step.describe(), block=dict(name='packed-block'))
                    elif packed:
                        step = dict(serving_sequential_step.describe(),
                                    packed_block_skipped='no packed block shape for %d scheduler requests' % users)
                    else:
                        step = serving_sequential_step.describe()
                    self.assertEqual(lines, [dict(stage='serving_buffer_pool', users=users,
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
                    # builder (which imports the native construction only when called) - and,
                    # only when a block is built over it, the one packed shape that block takes.
                    pooled.assert_called_once()
                    self.assertEqual(pooled.call_args.args, (operations, model.mesh_device))
                    options = pooled.call_args.kwargs
                    self.assertEqual(options['users'], users)
                    self.assertEqual(len(options['helpers']), 48)
                    self.assertEqual(options['page_width'], 68)
                    self.assertEqual(options['bucket_rows'], (1, 2, 4) if trimmed else (1, 2, 4, 8, 8, 16, 16))
                    self.assertEqual(options['feature_taps'], 5)
                    self.assertTrue(callable(options['rope']))
                    expected = {'users', 'helpers', 'page_width', 'bucket_rows', 'feature_taps', 'rope'}
                    if built:
                        self.assertEqual(options['packed_shapes'], ((shape.users, shape.rows_per_user),))
                        expected.add('packed_shapes')
                    self.assertEqual(set(options), expected)
                    # The device geometry from_prefill asks for, prepared once inside the
                    # admitted runtime and before any request.
                    prepared.assert_called_once_with(operations, model.mesh_device, fixtures[1], fixtures[2], fixtures[3],
                        block_rows=16, live_query_qk=False, native_proposal_attention=True)
                    packed_step = install.call_args.kwargs['packed_step']
                    if built:
                        # The packed block: over the same 48 helpers and the pinned sampler, the
                        # pool it restores from, the uploaded draft weights, the shape the request
                        # count picks at the pool's page width and the five feature taps; the
                        # step is bound to it.
                        packed_engine.assert_called_once()
                        args, options = packed_engine.call_args.args, packed_engine.call_args.kwargs
                        self.assertEqual((args[0], args[1], len(args[2])), (operations, model, 48))
                        self.assertIs(args[3], generator.return_value)
                        self.assertEqual(options, dict(pool=pool, shared_weights=weights, shape=shape,
                                                       feature_taps=(5, 19, 33, 47, 61)))
                        self.assertIsInstance(packed_step, serving_packed_step.PackedStep)
                        self.assertIs(packed_step.block, block)
                    else:
                        # No block: the sequential step is wired.
                        packed_engine.assert_not_called()
                        self.assertIs(packed_step, serving_sequential_step.sequential_packed_step)
                    # The [PINDIAG] lines of the attach, in order: the no-block line when the
                    # switch asked at a count no shape serves, the trim of the per-request
                    # captures beside the four-user block, and always the allocator after
                    # everything the attach allocated (the fake pool has no device to read).
                    expected = []
                    if packed and not built:
                        expected.append(('[PINDIAG] QWEN_FAST_PACKED_STEP=1 builds no packed block for {} scheduler requests '
                                         '(two take the 32-row M1 block, four the 64-row M3 block); the sequential step '
                                         'serves the rounds', users))
                    if trimmed:
                        expected.append(('[PINDIAG] per-request captures trimmed to widths {} for the four-user block', (1, 2, 4)))
                    expected.append(('[PINDIAG] dram after attach: {}', 'unavailable (pool without device statistics)'))
                    self.assertEqual([call.args for call in diagnostic.call_args_list], expected)
                    if probe is not None:
                        probe(install, diagnostic)
                    events.append('request')
                    if fail:
                        raise RuntimeError('request failed')
            finally:
                # Pool first and closed last; weights inside the admitted runtime; both
                # outlive the lifecycle that lends them to devices. The block, when built,
                # comes after the weights and before the lifecycle, and closes between them.
                # An attach that fails logs the failure BEFORE any scope closes (their exits
                # fence the device, which a failed attach may have left hung: run 35507675630).
                if attach_fail:
                    self.assertEqual(events, ['pool_build', 'runtime_enter', 'weights_build',
                                              *(['block_build'] if built else []), ('diag', self.ATTACH_FAILED),
                                              *(['block_close'] if built else []), 'weights_close', 'runtime_exit',
                                              'pool_close'])
                else:
                    self.assertEqual(events, ['pool_build', 'runtime_enter', 'weights_build',
                                              *(['block_build'] if built else []), 'request', 'lifecycle_close',
                                              *(['block_close'] if built else []), 'weights_close', 'runtime_exit',
                                              'pool_close'])

    def test_combined_recipe_lives_until_request_traces_are_closed(self):
        self.exercise()

    def test_request_failure_still_closes_lifecycle_before_recipe(self):
        with self.assertRaisesRegex(RuntimeError, 'request failed'):
            self.exercise(fail=True)

    def test_the_packed_block_is_built_after_the_weights_and_before_the_lifecycle_only_when_asked(self):
        self.exercise(packed=True, users=2)

    def test_a_request_failure_closes_the_packed_block_after_the_lifecycle_and_before_the_weights(self):
        with self.assertRaisesRegex(RuntimeError, 'request failed'):
            self.exercise(fail=True, packed=True, users=2)

    def test_four_scheduler_requests_take_the_sixty_four_row_block(self):
        self.exercise(packed=True, users=4)

    def test_the_bridge_factory_caps_the_engines_captures_only_beside_the_four_user_block_and_logs_the_allocator(self):
        for users, packed, expected in ((4, True, dict(capture_rows=4)), (2, True, {}), (1, False, {})):
            seen = {}

            def probe(install, diagnostic):
                bridge_factory = install.call_args.kwargs['bridge_factory']
                state = SimpleNamespace(req_id='request-1', block_ids=([3, 4],))
                request = SimpleNamespace(engine=object(), close=Mock())
                serving_runtime.ServingCacheOwner.return_value.physical_pages = 100
                with patch.object(serving_runtime, 'from_prefill', return_value=request) as factory, \
                        patch.object(serving_runtime, 'VerifierPageBinding'), \
                        patch.object(serving_runtime, 'FastRunnerBridge', return_value='bridge'):
                    self.assertEqual(bridge_factory(state, 'capture'), 'bridge')
                options = factory.call_args.kwargs
                seen['capture'] = {name: value for name, value in options.items() if name == 'capture_rows'}
                seen['engine_lines'] = [call.args for call in diagnostic.call_args_list
                                        if call.args[0].startswith('[PINDIAG] dram after engine')]

            with self.subTest(users=users):
                self.exercise(packed=packed, users=users, probe=probe)
                self.assertEqual(seen['capture'], expected)
                self.assertEqual(seen['engine_lines'],
                                 [('[PINDIAG] dram after engine {}: {}', 'request-1', 'unavailable (pool without device statistics)')])

    def test_an_attach_failure_is_logged_before_the_scopes_whose_exits_fence_the_device_close(self):
        for packed, users in ((True, 4), (False, 1)):
            with self.subTest(packed=packed, users=users), self.assertRaisesRegex(RuntimeError, 'attach failed'):
                self.exercise(packed=packed, users=users, attach_fail=True)

    def test_a_request_count_no_block_serves_stays_sequential_and_says_so(self):
        for users in (1, 3, 8):
            with self.subTest(users=users):
                self.exercise(packed=True, users=users)

    def test_the_switch_unset_builds_no_block_at_any_request_count(self):
        for users in (2, 4):
            with self.subTest(users=users):
                self.exercise(packed=False, users=users)
