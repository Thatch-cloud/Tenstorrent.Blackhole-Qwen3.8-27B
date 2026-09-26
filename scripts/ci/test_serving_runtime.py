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

    STREAM = {'streams': 'serial', 'evidence': 'stream'}

    def exercise(self, fail=False, packed=False, users=1, attach_fail=False, probe=None, four_as_two=None,
                 replay_group_rows=None, block_stream=STREAM, extra_env=None, refused=False, padded=None,
                 admission=None, statistics_refused=False):
        events = []
        # S2 W7 (QWEN_FAST_EXTENT_REPLAY=1): `admission` sets the flag and records packed_any_admission.admit
        # and admit_statistics as events ('admit', 'statistics'); its 'admit' / 'statistics' entries are the
        # exceptions they raise. None leaves the flag and both functions alone.
        self.admission_calls = dict(admit=[], statistics=[])

        def admission_call(name):
            def call(*args, **kwargs):
                events.append(name)
                self.admission_calls[name].append((args, kwargs))
                if admission.get(name) is not None:
                    raise admission[name]
                return {}
            return call

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
        # C2-any with no block built captures the same sequential widths (serving_runtime.py).
        no_block_any_request = not built and (extra_env or {}).get('QWEN_FAST_ANY_REQUEST') == '1'
        trimmed = (packed and users == 4) or no_block_any_request
        model = SimpleNamespace(args=object(), mesh_device=object(),
            layers=[SimpleNamespace(is_full_attention=False, attention=object()) for _ in range(48)])
        config = FastPolicyTests().fixture()
        config.scheduler_config.max_num_seqs = users
        # more than one request installs the one-in-flight scheduler unless one is named
        config.scheduler_config.scheduler_cls = 'named'
        worker = SimpleNamespace(vllm_config=config, model_runner=SimpleNamespace(model=SimpleNamespace(model=[model])))

        @contextmanager
        def combined(*args, **kwargs):
            # The recipe startup handed over, unchanged: the serial stream by default, None
            # only where register_reader_reason admits the register-epilogue reader.
            self.assertEqual(kwargs['block_stream'], block_stream)
            self.assertEqual(kwargs['kv_publication_evidence'], 'dma')
            # Only the admitted single gate/up shape adds a keyword; otherwise the call's
            # keywords are exactly the ones it always had.
            self.combined_keywords = sorted(kwargs)
            self.single_gateup_admitted = kwargs.get('single_gateup_admitted')
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
        if admission is not None:
            env['QWEN_FAST_EXTENT_REPLAY'] = '1'
        env.update(extra_env or {})
        with patch.dict('os.environ', env), \
                (patch('packed_any_admission.admit', side_effect=admission_call('admit'))
                 if admission is not None else nullcontext()), \
                (patch('packed_any_admission.admit_statistics', side_effect=admission_call('statistics'))
                 if admission is not None else nullcontext()), \
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
                        block_stream=block_stream,
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
                            if padded is not None:
                                # QWEN_FAST_PADDED_BLOCK: the one keyword it adds, only where admitted.
                                expected_options['padded_min_users'] = padded
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
                    if trimmed and no_block_any_request:
                        expected.append(('[PINDIAG] per-request captures trimmed to widths {} for C2-any with no packed block', (1, 2, 4)))
                    elif trimmed:
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
                # Under the extent flag the admission runs first, and the statistics check right after the pool.
                admitted = ['admit'] if admission is not None else []
                statistics = ['statistics'] if admission is not None else []
                if refused:
                    # Refused before the pool, the runtime or anything else was built.
                    self.assertEqual(events, admitted)
                elif statistics_refused:
                    # Only the pool was built, and it closes with the scopes after the failure is logged.
                    failure = admission['statistics']
                    self.assertEqual(events, [*admitted, 'pool_build', *statistics, ('diag', self.ATTACH_FAILED.replace(
                        'RuntimeError: attach failed', '%s: %s' % (type(failure).__name__, failure))), 'pool_close'])
                elif attach_fail:
                    self.assertEqual(events, [*admitted, 'pool_build', *statistics, 'runtime_enter', 'weights_build',
                                              *block_built, ('diag', self.ATTACH_FAILED),
                                              *block_closed, 'weights_close', 'runtime_exit',
                                              'pool_close'])
                else:
                    self.assertEqual(events, [*admitted, 'pool_build', *statistics, 'runtime_enter', 'weights_build',
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

    # --- A1 / C1: the register-epilogue reader in place of the serial block stream --------

    SKIP = {'QWEN_FAST_SKIP_BLOCK_STREAM': '1'}

    def test_the_sixty_four_row_block_admits_the_register_reader_with_the_skip_flag(self):
        # QWEN_FAST_SKIP_BLOCK_STREAM=1 at exactly the M3 shape: four scheduler requests,
        # QWEN_FAST_FOUR_AS_TWO=0, QWEN_FAST_PACKED_STEP=1. combined_runtime receives
        # block_stream=None (its scoped_register_epilogue branch) and the attach completes.
        self.exercise(packed=True, users=4, four_as_two=False, block_stream=None, extra_env=self.SKIP)

    def test_the_serial_stream_is_still_accepted_with_the_skip_flag_set(self):
        # The flag only ADMITS None at the M3 shape; a stream handed over is used as ever.
        self.exercise(packed=True, users=4, four_as_two=False, extra_env=self.SKIP)

    def test_without_the_skip_flag_the_serial_stream_is_still_required(self):
        # Flag unset: exactly the old refusal, at the M3 shape and everywhere else.
        for users, packed, four_as_two in ((4, True, False), (4, True, None), (2, True, None), (1, False, None)):
            with self.subTest(users=users, packed=packed, four_as_two=four_as_two), \
                    self.assertRaisesRegex(ValueError, 'serial weight-stream recipe required'):
                self.exercise(packed=packed, users=users, four_as_two=four_as_two, block_stream=None, refused=True)

    def test_the_skip_flag_admits_no_other_shape(self):
        # Two 32-row blocks at four users (the default), two users, one user, or the M3
        # count without the packed step: the stream is still mandatory.
        for users, packed, four_as_two in ((4, True, None), (4, True, True), (2, True, None), (1, False, None),
                                           (4, False, False)):
            with self.subTest(users=users, packed=packed, four_as_two=four_as_two), \
                    self.assertRaisesRegex(ValueError, 'serial weight-stream recipe required'):
                self.exercise(packed=packed, users=users, four_as_two=four_as_two, block_stream=None,
                              extra_env=self.SKIP, refused=True)

    def test_a_pipelined_stream_recipe_is_still_refused(self):
        with self.assertRaisesRegex(ValueError, 'serial weight-stream recipe required'):
            self.exercise(packed=True, users=4, four_as_two=False, extra_env=self.SKIP,
                          block_stream=dict(self.STREAM, pipeline_evidence='pipe'), refused=True)

    def test_every_flag_unset_passes_combined_runtime_exactly_the_old_keywords(self):
        self.exercise(packed=True, users=4, four_as_two=False)
        self.assertEqual(self.combined_keywords, ['block_stream', 'directory', 'kv_publication_evidence',
                                                  'native_attention_evidence', 'runtime_root'])
        self.exercise(packed=True, users=4, four_as_two=False, block_stream=None, extra_env=self.SKIP)
        self.assertIsNone(self.single_gateup_admitted)

    def test_the_single_gate_up_flag_admits_the_register_reader_at_the_sixty_four_row_block(self):
        self.exercise(packed=True, users=4, four_as_two=False, block_stream=None,
                      extra_env={'QWEN_FAST_SINGLE_GATEUP': '1'})
        self.assertIs(self.single_gateup_admitted, True)

    def test_the_single_gate_up_flag_is_admitted_for_c2_any_with_no_block(self):
        with patch.object(serving_runtime, 'attach_source_check'):
            self.exercise(packed=False, users=4, block_stream=None,
                          extra_env={'QWEN_FAST_SINGLE_GATEUP': '1', 'QWEN_FAST_ANY_REQUEST': '1'})
        self.assertIs(self.single_gateup_admitted, True)

    def test_the_single_gate_up_flag_is_refused_at_every_other_shape(self):
        # There FusedT16Arm serves the 16-row verify MLP (one user beside the M1 block, the
        # FOUR_AS_TWO rounds), and its BF4 register-epilogue projection is not bit-identical
        # to the native w1/w3 path that would replace it. Refused before anything is built,
        # whichever recipe startup handed over.
        for users, packed, four_as_two in ((1, False, None), (2, True, None), (4, True, None), (4, True, True),
                                           (4, False, False)):
            for block_stream in (None, self.STREAM):
                with self.subTest(users=users, packed=packed, four_as_two=four_as_two, stream=block_stream), \
                        self.assertRaisesRegex(ValueError, 'admitted only at the 64-row M3 block'):
                    self.exercise(packed=packed, users=users, four_as_two=four_as_two, block_stream=block_stream,
                                  extra_env={'QWEN_FAST_SINGLE_GATEUP': '1'}, refused=True)

    # --- Variable-user packed rounds M2: QWEN_FAST_PADDED_BLOCK ----------------------------

    def test_the_padded_block_flag_builds_the_sixty_four_row_block_with_its_minimum(self):
        # serving_runtime.padded_block_admission: the M3 block alone, built with the minimum asked for
        # (default 2); every other keyword of the block, the pool and the recipe unchanged.
        self.exercise(packed=True, users=4, four_as_two=False, extra_env={'QWEN_FAST_PADDED_BLOCK': '1'}, padded=2)
        self.exercise(packed=True, users=4, four_as_two=False, padded=3,
                      extra_env={'QWEN_FAST_PADDED_BLOCK': '1', 'QWEN_FAST_PADDED_BLOCK_MIN_USERS': '3'})
        self.assertEqual(self.combined_keywords, ['block_stream', 'directory', 'kv_publication_evidence',
                                                  'native_attention_evidence', 'runtime_root'])

    def test_the_padded_block_flag_is_refused_at_every_other_shape(self):
        # Two 32-row blocks at four users (the default), two users, one user, or the M3 count without
        # the packed step: refused before anything is built, as the single gate/up copy is.
        for users, packed, four_as_two in ((1, False, None), (2, True, None), (4, True, None), (4, True, True),
                                           (4, False, False)):
            with self.subTest(users=users, packed=packed, four_as_two=four_as_two), \
                    self.assertRaisesRegex(ValueError, 'QWEN_FAST_PADDED_BLOCK=1 is admitted only at the 64-row M3 block'):
                self.exercise(packed=packed, users=users, four_as_two=four_as_two,
                              extra_env={'QWEN_FAST_PADDED_BLOCK': '1'}, refused=True)

    def test_the_padded_block_minimum_alone_changes_nothing(self):
        # Unread while the flag is off: the attach is exactly today's.
        self.exercise(packed=True, users=4, four_as_two=False, extra_env={'QWEN_FAST_PADDED_BLOCK_MIN_USERS': '3'})


class RegisterReaderReasonTests(unittest.TestCase):
    """serving_runtime.register_reader_reason: the one predicate startup (skip the stream)
    and attach (admit block_stream=None) share, so the two can never disagree."""

    M3 = {'QWEN_FAST_SKIP_BLOCK_STREAM': '1', 'QWEN_FAST_PACKED_STEP': '1', 'QWEN_FAST_FOUR_AS_TWO': '0'}

    def reason(self, environ, requests=4):
        return serving_runtime.register_reader_reason(dict(scheduler_requests=requests), environ)

    def test_every_flag_unset_requires_the_stream(self):
        for requests in (1, 2, 3, 4, 8):
            self.assertIsNone(self.reason({}, requests))
        self.assertIsNone(self.reason({'QWEN_FAST_PACKED_STEP': '1', 'QWEN_FAST_FOUR_AS_TWO': '0'}))

    def test_the_skip_flag_names_the_sixty_four_row_block_only(self):
        self.assertEqual(self.reason(self.M3), ('the 64-row block', 'register-epilogue reader on native w_gate_up'))
        for name in ('QWEN_FAST_PACKED_STEP', 'QWEN_FAST_FOUR_AS_TWO'):
            with self.subTest(dropped=name):
                self.assertIsNone(self.reason({key: value for key, value in self.M3.items() if key != name}))
        self.assertIsNone(self.reason(dict(self.M3, QWEN_FAST_FOUR_AS_TWO='1')))
        self.assertIsNone(self.reason(dict(self.M3, QWEN_FAST_SKIP_BLOCK_STREAM='0')))
        for requests in (1, 2, 3, 8):
            with self.subTest(requests=requests):
                self.assertIsNone(self.reason(self.M3, requests))

    def test_the_policy_is_read_only_once_the_skip_flag_is_set(self):
        policy = Mock(return_value=dict(scheduler_requests=4))
        self.assertIsNone(serving_runtime.register_reader_reason(policy, {}))
        policy.assert_not_called()
        self.assertIsNotNone(serving_runtime.register_reader_reason(policy, self.M3))
        policy.assert_called_once_with()

    def test_the_single_gate_up_flag_applies_at_the_sixty_four_row_block_only(self):
        m3 = {'QWEN_FAST_SINGLE_GATEUP': '1', 'QWEN_FAST_PACKED_STEP': '1', 'QWEN_FAST_FOUR_AS_TWO': '0'}
        self.assertEqual(self.reason(m3), ('the single gate/up copy',
                                           'QWEN_FAST_SINGLE_GATEUP=1 builds no w_gate_up to stream'))
        # With the skip flag too, the single copy is the reason named.
        self.assertEqual(self.reason(dict(m3, QWEN_FAST_SKIP_BLOCK_STREAM='1'))[0], 'the single gate/up copy')
        refused = [(m3, 1), (m3, 2), (dict(m3, QWEN_FAST_FOUR_AS_TWO='1'), 4),
                   ({'QWEN_FAST_SINGLE_GATEUP': '1', 'QWEN_FAST_PACKED_STEP': '1'}, 4),
                   ({'QWEN_FAST_SINGLE_GATEUP': '1', 'QWEN_FAST_FOUR_AS_TWO': '0'}, 4)]
        for environ, requests in refused:
            with self.subTest(environ=environ, requests=requests), \
                    self.assertRaisesRegex(ValueError, 'admitted only at the 64-row M3 block .*, not users='):
                self.reason(environ, requests)

    def test_c2_any_with_no_packed_block_admits_both_flags(self):
        """The c2 profile: no block, engines capped at (1, 2, 4), so no 16-row MLP runs (run 36219636175)."""
        c2 = {'QWEN_FAST_ANY_REQUEST': '1', 'QWEN_FAST_PACKED_STEP': '0'}
        for requests in (1, 4):
            with self.subTest(requests=requests):
                self.assertEqual(self.reason(dict(c2, QWEN_FAST_SINGLE_GATEUP='1'), requests),
                                 ('the single gate/up copy', 'QWEN_FAST_SINGLE_GATEUP=1 builds no w_gate_up to stream'))
                self.assertEqual(self.reason(dict(c2, QWEN_FAST_SKIP_BLOCK_STREAM='1'), requests),
                                 ('C2-any with no packed block', 'register-epilogue reader on native w_gate_up'))
        self.assertIsNone(self.reason(c2))
        self.assertTrue(serving_runtime.c2_any_without_block({'QWEN_FAST_ANY_REQUEST': '1'}))
        # Not C2-any, or C2-any beside a block the switch builds: the old rule, refused off the M3 block.
        for environ, requests in (({'QWEN_FAST_PACKED_STEP': '0', 'QWEN_FAST_SINGLE_GATEUP': '1'}, 4),
                                  ({'QWEN_FAST_ANY_REQUEST': '0', 'QWEN_FAST_SINGLE_GATEUP': '1'}, 4),
                                  (dict(c2, QWEN_FAST_PACKED_STEP='1', QWEN_FAST_SINGLE_GATEUP='1'), 2),
                                  (dict(c2, QWEN_FAST_PACKED_STEP='1', QWEN_FAST_FOUR_AS_TWO='1',
                                        QWEN_FAST_SINGLE_GATEUP='1'), 4)):
            with self.subTest(environ=environ), \
                    self.assertRaisesRegex(ValueError, 'admitted only at the 64-row M3 block'):
                self.reason(environ, requests)

    def test_the_shape_description_names_what_is_configured(self):
        self.assertEqual(serving_runtime.m3_shape(dict(scheduler_requests=2), {'QWEN_FAST_PACKED_STEP': '1'}),
                         (False, 'users=2 FOUR_AS_TWO=unset PACKED_STEP=1'))
        self.assertEqual(serving_runtime.m3_shape(lambda: dict(scheduler_requests=4),
                                                  {'QWEN_FAST_PACKED_STEP': '1', 'QWEN_FAST_FOUR_AS_TWO': '0'}),
                         (True, 'users=4 FOUR_AS_TWO=0 PACKED_STEP=1'))


class MemoryLedgerHookTests(unittest.TestCase):
    """The attach's ledger hooks: every phase in order when a ledger is active, and nothing
    at all when it is not (the default)."""

    def run_attach(self, ledger_active):
        import memory_ledger

        calls, requests = [], []

        def record(phase, point=None, request=None, **walked):
            calls.append((phase, point, sorted(walked)))
            requests.append(request)

        def admitted(request_id, **walked):
            calls.append(('engine', request_id, sorted(walked)))

        test = RuntimeAttachmentTests()

        def probe(install, diagnostic):
            capture_factory = install.call_args.kwargs['capture_factory']
            bridge_factory = install.call_args.kwargs['bridge_factory']
            serving_runtime.ServingCacheOwner.return_value.physical_pages = 100
            capture_factory(4096)
            state = SimpleNamespace(req_id='request-1', block_ids=([3, 4],))
            with patch.object(serving_runtime, 'from_prefill', return_value=SimpleNamespace(engine=object(), close=Mock())), \
                    patch.object(serving_runtime, 'VerifierPageBinding'), \
                    patch.object(serving_runtime, 'FastRunnerBridge', return_value='bridge'):
                bridge_factory(state, 'capture')

        with patch.object(memory_ledger.MemoryLedger, 'phase', side_effect=AssertionError('no ledger is active')), \
                patch('dflash_prefill_window.PrefillWindowCapture'):
            if ledger_active:
                with patch.object(memory_ledger, 'record', side_effect=record), \
                        patch.object(memory_ledger, 'engine_admitted', side_effect=admitted):
                    test.exercise(packed=True, users=4, four_as_two=False, probe=probe)
            else:
                self.assertIsNone(memory_ledger.active())
                test.exercise(packed=True, users=4, four_as_two=False, probe=probe)
        self.requests = requests
        return calls

    def test_an_active_ledger_sees_every_attach_phase_in_order(self):
        calls = self.run_attach(True)
        # The prefill-after reading carries the full id for the JSON (the label a short one).
        self.assertEqual([request for request in self.requests if request is not None], ['request-1'])
        self.assertEqual([(phase, point) for phase, point, _ in calls], [
            ('P2', None), ('P3', None), ('P4', None), ('P5', None), ('P6', 'block1'), ('P7', 'after_attach'),
            ('prefill', 'before prompt=4096'), ('prefill', 'after req=request-1'), ('engine', 'request-1'),
            ('P13', 'before_shutdown')])
        self.assertEqual([walked for _, _, walked in calls], [
            ['buffer_pool'], ['gather_experiment', 'serving_collectives', 'serving_sampler'], ['combined_runtime'],
            ['draft_weights'], ['packed_block'], [], [], ['model_after_prefill'], ['engine_request'], []])

    def test_no_ledger_means_no_phase_runs(self):
        # memory_ledger.record is the real one here: with no active ledger it returns at
        # its first line, so MemoryLedger.phase (patched to fail the test) never runs.
        self.run_attach(False)


class PackedAnyAdmissionAttachTests(unittest.TestCase):
    """S2 W7: under QWEN_FAST_EXTENT_REPLAY=1 the attach admits the extent path (packed_any_admission.admit) on
    the host before anything is built, and requires the pool's DRAM statistics right after the pool; with
    the flag unset or '0' neither runs and the attach is exactly today's."""

    M3 = dict(packed=True, users=4, four_as_two=False)

    def test_the_flag_admits_before_anything_and_reads_the_statistics_after_the_pool(self):
        test = RuntimeAttachmentTests()
        test.exercise(admission={}, **self.M3)
        (args, kwargs), = test.admission_calls['admit']
        self.assertEqual(args, ('.',), 'the runtime root the binaries and kernels are read under')
        self.assertEqual(kwargs['m3'], (True, 'users=4 FOUR_AS_TWO=0 PACKED_STEP=1'))
        self.assertIsNone(kwargs['binary_record'], 'no override is requested in this environment')
        self.assertTrue(callable(kwargs['log']))
        (args, kwargs), = test.admission_calls['statistics']
        self.assertEqual(args[0].describe(), dict(users=4), 'the pool just built')

    def test_the_override_record_reaches_the_admission_so_the_binaries_are_hashed_once(self):
        record = dict(override='a' * 64, binaries={'build_Release/lib/_ttnncpp.so': 'a' * 64})
        test = RuntimeAttachmentTests()
        with patch('runtime_binary_override.install', return_value=record):
            test.exercise(admission={}, **self.M3)
        (_, kwargs), = test.admission_calls['admit']
        self.assertIs(kwargs['binary_record'], record)

    def test_a_refused_admission_builds_nothing(self):
        import packed_any_admission

        test = RuntimeAttachmentTests()
        with self.assertRaisesRegex(packed_any_admission.AdmissionRefused, 'CB2b'):
            test.exercise(admission=dict(admit=packed_any_admission.AdmissionRefused('CB2b: status PENDING')),
                          refused=True, **self.M3)
        self.assertEqual(test.admission_calls['statistics'], [])

    def test_unreadable_statistics_fail_the_attach_with_only_the_pool_built(self):
        import packed_any_admission

        test = RuntimeAttachmentTests()
        with self.assertRaisesRegex(packed_any_admission.AdmissionRefused, 'unavailable'):
            test.exercise(admission=dict(statistics=packed_any_admission.AdmissionRefused('statistics unavailable')),
                          statistics_refused=True, **self.M3)

    def test_the_flag_off_runs_neither(self):
        for extra_env in ({}, {'QWEN_FAST_EXTENT_REPLAY': '0'}):
            with self.subTest(extra_env=extra_env), \
                    patch('packed_any_admission.admit', side_effect=AssertionError('admit ran')), \
                    patch('packed_any_admission.admit_statistics', side_effect=AssertionError('statistics ran')):
                RuntimeAttachmentTests().exercise(extra_env=extra_env, **self.M3)
                RuntimeAttachmentTests().exercise(packed=False, users=1, extra_env=extra_env)

    def test_a_malformed_flag_is_refused_before_anything(self):
        for value in ('yes', 'true', '2', ''):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'QWEN_FAST_EXTENT_REPLAY must be 0 or 1'):
                RuntimeAttachmentTests().exercise(extra_env={'QWEN_FAST_EXTENT_REPLAY': value}, refused=True, **self.M3)

    def test_the_real_admission_refuses_this_checkout_s_runtime_and_names_why(self):
        """No fake admission: the attach under the flag reaches packed_any_admission.admit, which refuses a
        runtime root that is not K64j and the evidence whose CB2b has not passed - and nothing is built."""
        import packed_any_admission

        with patch.dict(packed_any_admission._STATE, clear=True), \
                self.assertRaises(packed_any_admission.AdmissionRefused) as caught:
            RuntimeAttachmentTests().exercise(extra_env={'QWEN_FAST_EXTENT_REPLAY': '1'}, refused=True, **self.M3)
        for words in ('QWEN_FAST_ANY_REQUEST=(unset), not 1', 'QWEN_FAST_RUNTIME_BINARY_SHA256=(unset)',
                      'the runtime is not K64j', 'CB2b: status PENDING, not PASS'):
            self.assertIn(words, str(caught.exception))
        self.assertFalse(packed_any_admission.admitted())
