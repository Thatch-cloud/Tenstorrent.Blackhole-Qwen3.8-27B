import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import torch
from unittest.mock import Mock, call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'speculative-decoding/harness'))

from greedy_session import GreedySession
from dflash_request_runtime import DFlashRequestRuntime, TARGET_TAPS
from serving_request_factory import from_prefill
import serving_request_factory


class RequestFactoryTests(unittest.TestCase):
    def fixture(self):
        sample = SimpleNamespace(temperature=0, n=1, max_tokens=256, min_tokens=0, ignore_eos=False,
            logprobs=None, prompt_logprobs=None, presence_penalty=0, frequency_penalty=0,
            repetition_penalty=1, stop=[], stop_token_ids=[])
        state = SimpleNamespace(req_id='request', prompt_token_ids=[1] * 4096,
            output_token_ids=[10], num_computed_tokens=0, sampling_params=sample,
            block_ids=(list(range(65)),))
        device = SimpleNamespace(position=4096, max_drafts=15, proposal_capture=None,
            propose=Mock(), prepare_publication=Mock(), commit_publication=Mock(),
            discard_publication=Mock(), close=Mock())
        engines = []

        def engine_factory(model, session, pages, helpers, **options):
            engine = SimpleNamespace(session=session, phase='idle', retain_feature_taps=TARGET_TAPS, close=Mock())
            options['before_capture'](engine)
            engines.append(engine)
            return engine

        components = SimpleNamespace(device=Mock(return_value=device), proposal=Mock(),
            runtime=DFlashRequestRuntime, session=GreedySession,
            engine=Mock(side_effect=engine_factory), collectives=Mock())
        # prefill_slot: the native GDN slot the batched prefill wrote; 0 is the first
        # user's, which is where the fast path reads, so there is nothing to adopt.
        capture = SimpleNamespace(outputs=Mock(return_value=('features',)), close=Mock(), prefill_slot=0)
        arguments = dict(state=state, capture=capture, fixtures=({}, [], {}, {}), eos_ids=(99,))
        return components, device, engines, arguments

    def helpers(self):
        # adopt_slot verifies its slices against a readback and reports the chip count.
        return [Mock(spec=['adopt_slot'], **{'adopt_slot.return_value': 2}) for _ in range(48)]

    def build(self, components, arguments, helpers=None, **extra):
        with patch('serving_request_factory.device_components', return_value=components):
            return from_prefill(object(), SimpleNamespace(args=SimpleNamespace(vocab_size=100),
                mesh_device=object()), object(), torch.tensor([list(range(65)) + [0] * 3], dtype=torch.int32),
                [object()] * 48 if helpers is None else helpers, **arguments, **extra)

    def test_the_engine_gets_the_runtimes_capture_cap_only_when_one_is_given(self):
        # the default: the engine is called exactly as before, no capture keyword at all
        components, device, engines, arguments = self.fixture()
        request = self.build(components, arguments)
        self.assertNotIn('capture_rows', components.engine.call_args.kwargs)
        self.assertEqual(components.engine.call_args.kwargs['max_verify_rows'], 16)
        request.close('request')
        # beside the four-user block the runtime caps the captures at 4; the qualified
        # verifier width the T16 gate reads stays 16
        components, device, engines, arguments = self.fixture()
        request = self.build(components, arguments, capture_rows=4)
        self.assertEqual(components.engine.call_args.kwargs['capture_rows'], 4)
        self.assertEqual(components.engine.call_args.kwargs['max_verify_rows'], 16)
        self.assertTrue(components.engine.call_args.kwargs['target_attention_t16'])
        request.close('request')

    def test_prefilled_seed_not_emitted_or_prefilled_twice(self):
        components, device, engines, arguments = self.fixture()
        request = self.build(components, arguments)
        self.assertEqual(request.session.emitted, [10])
        self.assertEqual(arguments['state'].output_token_ids, [10])
        self.assertEqual(request.session.position, 4096)
        self.assertEqual(arguments['state'].num_computed_tokens, 0)
        self.assertIs(request.runtime.engine, engines[0])
        components.proposal.assert_called_once_with(device, max_new_tokens=256)
        arguments['capture'].close.assert_called_once()
        request.close('request')
        engines[0].close.assert_called_once()
        device.close.assert_called_once()

    def test_the_engine_borrows_the_verifier_storage_of_the_slot_the_device_holds(self):
        # No slot (no pool): the engine allocates as before, no storage keyword at all.
        components, device, engines, arguments = self.fixture()
        self.build(components, arguments)
        self.assertNotIn('storage', components.engine.call_args.kwargs)
        # A slot without verifier storage (a pool built without the GDN helpers): likewise.
        components, device, engines, arguments = self.fixture()
        device.pool_slot = SimpleNamespace(verifier=None)
        self.build(components, arguments)
        self.assertNotIn('storage', components.engine.call_args.kwargs)
        # The slot's verifier storage, allocated at attach before any request's trace.
        components, device, engines, arguments = self.fixture()
        storage = object()
        device.pool_slot = SimpleNamespace(verifier=storage)
        request = self.build(components, arguments)
        self.assertIs(components.engine.call_args.kwargs['storage'], storage)
        self.assertEqual(components.engine.call_args.kwargs['retain_feature_taps'], TARGET_TAPS)
        request.close('request')

    def test_engine_failure_releases_owned_draft_and_features(self):
        components, device, _, arguments = self.fixture()
        components.engine.side_effect = RuntimeError('capture failed')
        with self.assertRaises(RuntimeError):
            self.build(components, arguments)
        device.close.assert_called_once()
        arguments['capture'].close.assert_called_once()

    def test_terminal_or_advanced_prefill_rejected_before_device_allocation(self):
        for tokens in ([99], [10, 11], []):
            components, _, _, arguments = self.fixture()
            arguments['state'].output_token_ids = tokens
            arguments['capture'].prefill_slot = 1
            helpers = self.helpers()
            with self.assertRaises(ValueError):
                self.build(components, arguments, helpers)
            components.device.assert_not_called()
            self.assertEqual([helper.adopt_slot.call_count for helper in helpers], [0] * 48,
                             'a refused request must not touch native GDN state')

    def test_the_prefill_slot_is_adopted_into_slot_zero_before_the_engine_saves_its_initial_state(self):
        # The batched prefill wrote the second user's GDN state into slot 1, and the
        # engine's initial save, its carry and every step read slot 0 (runs 35492676194,
        # 35493208438: the second user decoded from the first's state). The copy must
        # precede the engine - and the drafter, the first device work of admission.
        components, device, engines, arguments = self.fixture()
        arguments['capture'].prefill_slot = 1
        helpers, seen = self.helpers(), []
        engine_factory = components.engine.side_effect

        def adopted():
            return [helper.adopt_slot.call_args_list for helper in helpers]

        def device_factory(*args, **kwargs):
            seen.append(('device', adopted()))
            return device

        def engine_after_adoption(model, session, pages, given, **options):
            seen.append(('engine', adopted()))
            self.assertIs(given, helpers)
            return engine_factory(model, session, pages, given, **options)

        components.device.side_effect = device_factory
        components.engine.side_effect = engine_after_adoption
        with patch('serving_request_factory._log') as log:
            request = self.build(components, arguments, helpers)
        expected = [[call(1, layer=layer)] for layer in range(48)]
        self.assertEqual(seen, [('device', expected), ('engine', expected)])
        self.assertEqual([helper.adopt_slot.call_count for helper in helpers], [1] * 48, 'adopted exactly once')
        log.assert_called_once_with('[PINDIAG] adopted GDN slot {} into slot 0: {} layers, conv slices verified on {} for request {}',
                                    1, 48, 'both chips', 'request')
        self.assertIs(request.runtime.engine, engines[0])
        request.close('request')

    def test_layers_verified_on_differing_chips_are_refused_before_device_allocation(self):
        # Every helper must report the same chip count, and at least one chip: an
        # unverified layer is not the proof the gate run needs.
        for counts in ((2,) * 47 + (1,), (0,) * 48, (None,) * 48):
            components, _, _, arguments = self.fixture()
            arguments['capture'].prefill_slot = 1
            helpers = self.helpers()
            for helper, count in zip(helpers, counts):
                helper.adopt_slot.return_value = count
            with self.subTest(last=counts[-1]), self.assertRaisesRegex(ValueError, 'same chips'):
                self.build(components, arguments, helpers)
            components.device.assert_not_called()

    def test_slot_zero_and_a_single_sequence_prefill_adopt_nothing(self):
        # The first user prefills into slot 0, and a single-sequence prefill records no
        # slot at all: single-user behaviour is unchanged, no native copy is made.
        for slot in (0, None):
            components, _, engines, arguments = self.fixture()
            arguments['capture'].prefill_slot = slot
            helpers = self.helpers()
            request = self.build(components, arguments, helpers)
            with self.subTest(slot=slot):
                self.assertEqual([helper.adopt_slot.call_count for helper in helpers], [0] * 48)
                self.assertIs(request.runtime.engine, engines[0])
            request.close('request')

    def test_a_capture_without_a_recorded_slot_is_refused_before_device_allocation(self):
        components, _, _, arguments = self.fixture()
        arguments['capture'] = SimpleNamespace(outputs=Mock(return_value=('features',)), close=Mock())
        helpers = self.helpers()
        with self.assertRaisesRegex(ValueError, 'record the native slot'):
            self.build(components, arguments, helpers)
        components.device.assert_not_called()
        self.assertEqual([helper.adopt_slot.call_count for helper in helpers], [0] * 48)

    def test_a_refused_adoption_allocates_no_device_or_engine(self):
        components, _, _, arguments = self.fixture()
        arguments['capture'].prefill_slot = 9
        helpers = self.helpers()
        helpers[0].adopt_slot.side_effect = ValueError('outside the eight-slot batch')
        with self.assertRaisesRegex(ValueError, 'outside the eight-slot batch'):
            self.build(components, arguments, helpers)
        components.device.assert_not_called()
        components.engine.assert_not_called()

    def test_prompt_only_allocation_cannot_be_used_for_trace_warmup(self):
        components, _, _, arguments = self.fixture()
        arguments['state'].block_ids = (list(range(64)),)
        with self.assertRaisesRegex(ValueError, 'warmup pages'):
            self.build(components, arguments)
        components.device.assert_not_called()

    def test_a_frontier_at_or_past_the_prompt_is_not_a_fresh_prefill(self):
        """The boundary moved, and hardware is why.

        This asserted that ANY nonzero frontier is an advanced snapshot, which held
        while prefill happened in one step. Under chunked prefill the seed is emitted on
        the final chunk, so the frontier is legitimately the tokens the earlier chunks
        covered: run 35696842354 prefilled 32768 tokens in sixteen chunks and arrived
        here with 30720, and was refused for it.

        What the clause protects is that the request has not DECODED - the seed adopted
        must be this prefill's own first token. That boundary is the prompt length.
        """
        for frontier in (4096, 4097, 99999, -1, 'x'):
            components, _, _, arguments = self.fixture()
            arguments['state'].num_computed_tokens = frontier
            with self.subTest(frontier=frontier), self.assertRaisesRegex(
                    ValueError, 'frontier inside the prompt'):
                self.build(components, arguments)
            components.device.assert_not_called()

    def test_a_frontier_inside_the_prompt_is_a_chunked_prefill_and_is_allowed(self):
        """The shape run 35696842354 actually produced: fifteen chunks behind it and
        the sixteenth emitting the seed. Zero stays legal - that is the unchunked path,
        where nothing has run yet."""
        for frontier in (0, 1, 2048, 4095):
            components, _, _, arguments = self.fixture()
            arguments['state'].num_computed_tokens = frontier
            with self.subTest(frontier=frontier):
                self.build(components, arguments)
            components.device.assert_called_once()


EXTENT = {'QWEN_FAST_EXTENT_REPLAY': '1'}
MB = 10 ** 6


class Pool(object):
    """A pool whose allocator statistics read (free, largest_free) bytes per chip."""

    def __init__(self, *chips):
        self.chips = chips

    def dram_statistics(self):
        return [dict(chip=index, free=free, largest_free=largest) for index, (free, largest) in enumerate(self.chips)]


class ExtentMemoryTests(unittest.TestCase):
    """S2 W6 in the request factory, under QWEN_FAST_EXTENT_REPLAY=1 only: one 2048 proposal bucket (W6a), the
    post-prefill DRAM backstop and the hold's registration (W6b), the engine build's ledger point (W6d)."""

    def setUp(self):
        import serving_prefill_admission

        saved = sys.modules.pop(serving_prefill_admission.DRAM_KEY, None)

        def restore():
            sys.modules.pop(serving_prefill_admission.DRAM_KEY, None)
            if saved is not None:
                sys.modules[serving_prefill_admission.DRAM_KEY] = saved

        self.addCleanup(restore)

    def environ(self, on):
        """The flag set, or absent (the flag-off byte path)."""
        environ = {name: value for name, value in os.environ.items() if name != 'QWEN_FAST_EXTENT_REPLAY'}
        if on:
            environ.update(EXTENT)
        return patch.dict(os.environ, environ, clear=True)

    def short_request(self, pool=None):
        """RequestFactoryTests' fixture for a 60-token prompt, with a proposal that reads the ladder the way
        PreparedDFlashProposal does: dflash_proposal_trace.proposal_contexts at call time."""
        import dflash_proposal_trace

        components, device, engines, arguments = RequestFactoryTests().fixture()
        arguments['state'].prompt_token_ids = [1] * 60
        arguments['state'].block_ids = ([0, 1],)
        device.position = 60
        ladders = []
        components.proposal.side_effect = lambda drafter, max_new_tokens: ladders.append(
            dflash_proposal_trace.proposal_contexts(drafter.position, max_new_tokens))
        pages = torch.tensor([[0, 1] + [0] * 66], dtype=torch.int32)
        helpers = [Mock(spec=['adopt_slot'], **{'adopt_slot.return_value': 2}) for _ in range(48)]

        def build():
            with patch('serving_request_factory.device_components', return_value=components):
                return from_prefill(object(), SimpleNamespace(args=SimpleNamespace(vocab_size=100), mesh_device=object()),
                                    object(), pages, helpers, buffer_pool=pool, **arguments)

        return components, helpers, ladders, build

    def test_the_flag_is_strict(self):
        for value, expected in ((None, False), ('0', False), ('1', True)):
            with self.subTest(value=value):
                environ = {} if value is None else {'QWEN_FAST_EXTENT_REPLAY': value}
                self.assertIs(serving_request_factory.extent_replay_enabled(environ), expected)
        for value in ('true', '2', '', ' 1'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
                serving_request_factory.extent_replay_enabled({'QWEN_FAST_EXTENT_REPLAY': value})

    def test_inside_the_scope_every_position_takes_the_one_2048_bucket_and_outside_the_ladder_is_unchanged(self):
        import dflash_proposal_inputs
        import dflash_proposal_trace

        ladder = dflash_proposal_trace.proposal_contexts
        self.assertIs(ladder, dflash_proposal_inputs.proposal_contexts)
        with serving_request_factory.single_proposal_bucket():
            for position in range(1, 4097):
                for budget in (1, 256, 16383):
                    self.assertEqual(dflash_proposal_trace.proposal_contexts(position, budget), (2048,),
                                     (position, budget))
        self.assertIs(dflash_proposal_trace.proposal_contexts, ladder, 'restored after the scope')
        self.assertEqual(dflash_proposal_trace.proposal_contexts(60, 4096), (256, 512, 1024, 2048))
        with self.assertRaises(RuntimeError), serving_request_factory.single_proposal_bucket():
            raise RuntimeError('capture failed')
        self.assertIs(dflash_proposal_trace.proposal_contexts, ladder, 'restored after a failed capture')

    def test_the_single_bucket_keeps_the_ladders_refusals(self):
        for position, budget in ((True, 513), (0, 513), (262112, 1), (4096, 0), (4096, True), (262111, 2)):
            with self.subTest(position=position, budget=budget), self.assertRaises(ValueError):
                serving_request_factory.single_bucket_contexts(position, budget)

    def test_a_2048_bucket_under_2048_history_rows_is_a_mask_the_t16_gate_accepts(self):
        """proposal_inputs masks the rows past history_rows with -inf; the T16 validate_mask accepts the holes."""
        from dflash_proposal_inputs import proposal_inputs
        from dflash_t16_native_attention import validate_mask

        for position in (1, 60, 255, 2047):
            with self.subTest(position=position):
                mask = proposal_inputs(248044, position, min(position, 2048), 16, 2048)['mask']
                self.assertEqual(tuple(mask.shape), (1, 1, 32, 2080))
                validate_mask(mask)
                self.assertTrue(torch.isneginf(mask[..., position:2048]).all(), 'the holes are masked')

    def test_under_the_flag_the_engine_captures_one_2048_bucket_and_off_the_ladder(self):
        from dflash_proposal_inputs import proposal_contexts

        for on, expected in ((True, [(2048,)]), (False, [proposal_contexts(60, 256)])):
            with self.subTest(flag=on), self.environ(on):
                _, _, ladders, build = self.short_request()
                build().close('request')
                self.assertEqual(ladders, expected)
        self.assertEqual(proposal_contexts(60, 256), (256, 512), 'off, the short prompt takes two buckets')

    def test_the_ladder_line_names_the_single_bucket_under_the_flag(self):
        with self.environ(True):
            self.assertEqual(serving_request_factory._proposal_ladder(60, 4096), (2048,))
        with self.environ(False):
            self.assertEqual(serving_request_factory._proposal_ladder(60, 4096), (256, 512, 1024, 2048))

    def test_the_backstop_refuses_below_the_build_peak_and_the_reserve_on_the_smallest_largest_block(self):
        from serving_prefill_admission import backstop_need

        reserve = 256 * 2 ** 20
        need = backstop_need(reserve)
        log = Mock()
        self.assertEqual(serving_request_factory.dram_backstop(Pool((9_000 * MB, need), (9_000 * MB, need + 1)),
                                                               request_id='r', reserve=reserve, log=log), need)
        for pool in (Pool((9_000 * MB, need - 1), (9_000 * MB, need)),
                     Pool((40_000 * MB, 100 * MB), (40_000 * MB, 40_000 * MB))):
            with self.subTest(chips=pool.chips), self.assertRaisesRegex(serving_request_factory.RequestRefused,
                                                                        'DRAM backstop'):
                serving_request_factory.dram_backstop(pool, request_id='r', reserve=reserve, log=log)
        self.assertTrue(log.call_args_list[-1].args[0].startswith(serving_request_factory.DRAM_BACKSTOP_REFUSED))
        self.assertIsNone(serving_request_factory.dram_backstop(object(), request_id='r', reserve=reserve, log=log),
                          'an unreadable pool is a diagnostic, not a refusal')
        self.assertEqual(log.call_args_list[-1].args[1:], ('r', 'pool without device statistics'))

    def test_under_the_flag_a_short_pool_refuses_the_request_before_any_device_state(self):
        pool = Pool((9_000 * MB, 100 * MB), (9_000 * MB, 100 * MB))
        with self.environ(True):
            components, helpers, _, build = self.short_request(pool)
            with patch('serving_request_factory._log'), \
                    self.assertRaisesRegex(serving_request_factory.RequestRefused, 'DRAM backstop'):
                build()
            components.device.assert_not_called()
            components.engine.assert_not_called()
            self.assertEqual([helper.adopt_slot.call_count for helper in helpers], [0] * 48)
        with self.environ(False):
            components, _, _, build = self.short_request(pool)
            build().close('request')
            components.device.assert_called_once()

    def test_the_engine_build_is_a_ledger_before_point_under_the_flag_only(self):
        import memory_ledger
        from serving_prefill_admission import engine_build_peak

        for on in (True, False):
            with self.subTest(flag=on), self.environ(on):
                components, _, _, build = self.short_request()
                order = []
                components.device.side_effect = lambda *args, **kwargs: order.append('device') or components.device.return_value
                with patch.object(memory_ledger, 'before', side_effect=lambda *args, **kwargs: order.append(('before', args, kwargs))):
                    build().close('request')
                if on:
                    self.assertEqual(order, [('before', ('engine',), dict(estimate=engine_build_peak(), point='req=request',
                                                                          request='request')), 'device'])
                else:
                    self.assertEqual(order, ['device'])

    def test_registration_parks_the_hold_and_logs_what_it_reads(self):
        import serving_prefill_admission

        log = Mock()
        pool = Pool((9_000 * MB, 1_500 * MB), (9_000 * MB, 2_000 * MB))
        with self.environ(True):
            unregister = serving_request_factory.register_dram_admission(pool, log=log)
        holder = sys.modules[serving_prefill_admission.DRAM_KEY]
        reserve = 256 * 2 ** 20
        self.assertEqual(holder.admits(60), (True, dict(largest_free=1_500 * MB, need=1_000 * MB + reserve)))
        self.assertFalse(holder.admits(4096)[0], 'a long prompt needs 1.3 GB and the reserve')
        self.assertTrue(log.call_args.args[0].startswith(serving_request_factory.DRAM_REGISTERED))
        self.assertEqual(log.call_args.args[1:], (800 * MB, 200 * MB, 300 * MB, 2048, reserve, 1_500 * MB))
        unregister()
        self.assertNotIn(serving_prefill_admission.DRAM_KEY, sys.modules)
