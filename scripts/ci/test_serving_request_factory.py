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

    def build(self, components, arguments, helpers=None):
        with patch('serving_request_factory.device_components', return_value=components):
            return from_prefill(object(), SimpleNamespace(args=SimpleNamespace(vocab_size=100),
                mesh_device=object()), object(), torch.tensor([list(range(65)) + [0] * 3], dtype=torch.int32),
                [object()] * 48 if helpers is None else helpers, **arguments)

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
        log.assert_called_once_with('[PINDIAG] adopted GDN slot {} into slot 0: {} layers verified on {} for request {}',
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

    def test_cached_or_advanced_scheduler_snapshot_is_not_a_fresh_prefill(self):
        for frontier in (1, 2048, 4096):
            components, _, _, arguments = self.fixture()
            arguments['state'].num_computed_tokens = frontier
            with self.subTest(frontier=frontier), self.assertRaisesRegex(ValueError, 'pre-step frontier zero'):
                self.build(components, arguments)
            components.device.assert_not_called()
