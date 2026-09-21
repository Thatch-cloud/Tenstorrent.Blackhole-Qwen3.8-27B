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
# (packed_shapes.serving_shape): two take M1, four take M3 - UNLESS four scheduler
# requests take QWEN_FAST_FOUR_AS_TWO's pair of 32-row M1 blocks instead, which is the
# default at four requests (serving_runtime.py).
SHAPES = {2: m1_shape(68), 4: m3_shape(68)}


class RuntimeAttachmentTests(unittest.TestCase):
    ATTACH_FAILED = ('[PINDIAG] attach failed with RuntimeError: attach failed; closing the attach scopes now (the '
                     'packed block if built, the draft weights, the admitted combined runtime from its shared-QK '
                     'scope, the sampler links, the pool) - their exits fence the device, and a hung device blocks '
                     'the first fence')

    def exercise(self, fail=False, packed=False, users=1, attach_fail=False, probe=None, four_as_two=None,
                 replay_group_rows=None):
        events = []

        def diag(template, *values):
            # the attach-failed line lands among the events, so its place before the scope
            # closes is checked; every other line is inspected through the mock
            if template.startswith('[PINDIAG] attach failed'):
                events.append(('diag', template.format(*values)))
        # QWEN_FAST_FOUR_AS_TWO defaults ON at four scheduler requests (serving_runtime.py):
        # `four_as_two=None` leaves that default in force (two 32-row M1 blocks), True sets
        # it explicitly, False keeps the single 64-row M3 block. At any other request count
        # the switch does nothing.
        two_blocks = packed and users == 4 and four_as_two is not False
        if two_blocks:
            shapes = (m1_shape(68), m1_shape(68))
        else:
            shape = SHAPES.get(users) if packed else None
            shapes = (shape,) if shape is not None else ()
        built = bool(shapes)
        # Beside the four-user block - one 64-row block or two 32-row ones, the same total
        # rows either way - the per-request captures are trimmed to (1, 2, 4).
        trimmed = packed and users == 4
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

        env = {'QWEN_FAST_PACKED_STEP': '1' if packed else '0'}
        if four_as_two is not None:
            env['QWEN_FAST_FOUR_AS_TWO'] = '1' if four_as_two else '0'
        if replay_group_rows is not None:
            env['QWEN_FAST_REPLAY_GROUP_ROWS'] = str(replay_group_rows)
        with patch.dict('os.environ', env), \
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
                    # device step that serves the rounds - with the block(s) when built,
                    # and why none was when the switch asked for one.
                    lines = [json.loads(line) for line in out.getvalue().splitlines() if line.startswith('{')]
                    if built and two_blocks:
                        step = dict(serving_packed_step.describe(), blocks=[dict(name='packed-block')] * len(shapes))
                    elif built:
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
                    # only when a block is built over it, the packed shape(s) that block takes.
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
                        # `packed_shapes` names each distinct shape ONCE (four_as_two's pair of
                        # 32-row blocks share one (2, 16) shape); the pool lends each block its
                        # own set through the separate `packed_replicas` count.
                        self.assertEqual(options['packed_shapes'],
                                         ((2, 16),) if two_blocks else ((shapes[0].users, shapes[0].rows_per_user),))
                        expected.add('packed_shapes')
                        if two_blocks:
                            self.assertEqual(options['packed_replicas'], {(2, 16): 2})
                            expected.add('packed_replicas')
                        # QWEN_FAST_REPLAY_GROUP_ROWS (packed_verifier.replay_group_rows),
                        # read only here beside a block that is actually built: 4 unless
                        # named, matching the flag packed_verifier.py itself now reads.
                        self.assertEqual(options['packed_replay_group_rows'],
                                         replay_group_rows if replay_group_rows is not None else 4)
                        expected.add('packed_replay_group_rows')
                    self.assertEqual(set(options), expected)
                    # The device geometry from_prefill asks for, prepared once inside the
                    # admitted runtime and before any request.
                    prepared.assert_called_once_with(operations, model.mesh_device, fixtures[1], fixtures[2], fixtures[3],
                        block_rows=16, live_query_qk=False, native_proposal_attention=True)
                    packed_step = install.call_args.kwargs['packed_step']
                    if built:
                        # The packed block(s): over the same 48 helpers and the pinned sampler,
                        # the pool it restores from, the uploaded draft weights, the shape the
                        # request count picks at the pool's page width and the five feature
                        # taps; the step is bound to it (or, under the switch, to both, each
                        # claiming its own disjoint pool slots in scheduler order).
                        self.assertEqual(packed_engine.call_count, len(shapes))
                        for index, shape in enumerate(shapes):
                            call = packed_engine.call_args_list[index]
                            self.assertEqual((call.args[0], call.args[1], len(call.args[2])), (operations, model, 48))
                            self.assertIs(call.args[3], generator.return_value)
                            expected_options = dict(pool=pool, shared_weights=weights, shape=shape,
                                                    feature_taps=(5, 19, 33, 47, 61))
                            if two_blocks:
                                expected_options['pool_slots'] = tuple(range(2 * index, 2 * index + 2))
                            self.assertEqual(call.kwargs, expected_options)
                        self.assertIsInstance(packed_step, serving_packed_step.PackedStep)
                        if two_blocks:
                            self.assertEqual(packed_step.blocks, (block, block))
                            self.assertIsNone(packed_step.block, 'no single block owns a two-block round')
                        else:
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
                # outlive the lifecycle that lends them to devices. Each block, when built,
                # comes after the weights and before the lifecycle, and closes between them,
                # one entry per configured shape. An attach that fails logs the failure
                # BEFORE any scope closes (their exits fence the device, which a failed
                # attach may have left hung: run 35507675630).
                block_built = ['block_build'] * len(shapes)
                block_closed = ['block_close'] * len(shapes)
                if attach_fail:
                    self.assertEqual(events, ['pool_build', 'runtime_enter', 'weights_build',
                                              *block_built, ('diag', self.ATTACH_FAILED),
                                              *block_closed, 'weights_close', 'runtime_exit',
                                              'pool_close'])
                else:
                    self.assertEqual(events, ['pool_build', 'runtime_enter', 'weights_build',
                                              *block_built, 'request', 'lifecycle_close',
                                              *block_closed, 'weights_close', 'runtime_exit',
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

    def test_four_scheduler_requests_take_two_thirty_two_row_blocks_by_default(self):
        self.exercise(packed=True, users=4)

    def test_four_scheduler_requests_take_two_thirty_two_row_blocks_with_the_switch_explicitly_on(self):
        self.exercise(packed=True, users=4, four_as_two=True)

    def test_four_scheduler_requests_take_the_sixty_four_row_block_with_the_switch_off(self):
        self.exercise(packed=True, users=4, four_as_two=False)

    def test_two_scheduler_requests_take_one_thirty_two_row_block_regardless_of_the_switch(self):
        for four_as_two in (None, True, False):
            with self.subTest(four_as_two=four_as_two):
                self.exercise(packed=True, users=2, four_as_two=four_as_two)

    def test_a_request_failure_closes_both_packed_blocks_after_the_lifecycle_and_before_the_weights(self):
        with self.assertRaisesRegex(RuntimeError, 'request failed'):
            self.exercise(fail=True, packed=True, users=4)

    def test_the_pool_receives_the_replay_group_rows_flag_when_a_block_is_built(self):
        self.exercise(packed=True, users=2, replay_group_rows=8)

    def test_the_pool_receives_the_default_replay_group_rows_without_the_flag(self):
        self.exercise(packed=True, users=4)

    def test_the_flag_is_inert_when_no_block_is_built(self):
        # users=3 takes neither shape (packed_shapes.serving_shape): no block is built, and
        # exercise()'s own pool-options check only expects 'packed_replay_group_rows' when
        # `built` is True, so this proves the runtime's `if packed_shapes:` guard keeps the
        # kwarg (and the packed_verifier import behind it) out of the pool call here, flag
        # set or not.
        self.exercise(packed=True, users=3, replay_group_rows=8)

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
