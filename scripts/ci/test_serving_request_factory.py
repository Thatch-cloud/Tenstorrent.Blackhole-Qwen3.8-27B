from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import torch
from unittest.mock import Mock, patch

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
        capture = SimpleNamespace(outputs=Mock(return_value=('features',)), close=Mock())
        arguments = dict(state=state, capture=capture, fixtures=({}, [], {}, {}), eos_ids=(99,))
        return components, device, engines, arguments

    def build(self, components, arguments):
        with patch('serving_request_factory.device_components', return_value=components):
            return from_prefill(object(), SimpleNamespace(args=SimpleNamespace(vocab_size=100),
                mesh_device=object()), object(), torch.tensor([list(range(65)) + [0] * 3], dtype=torch.int32),
                [object()] * 48, **arguments)

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
            with self.assertRaises(ValueError):
                self.build(components, arguments)
            components.device.assert_not_called()

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
