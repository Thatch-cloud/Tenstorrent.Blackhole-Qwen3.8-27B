from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'speculative-decoding' / 'harness'))
from full_request import measure_request


class RequestPilotTests(unittest.TestCase):
    def run_fixture(self, *, seed=0, eos_ids=(), wrong=False):
        def decode(token, position, trace):
            logits = torch.zeros(1, 100)
            logits[0, (token + 1) % 3] = 1
            return logits

        def factory(model, session, pages, helpers, sampler):
            engine = SimpleNamespace(setup_ms=12.0, phase='idle', close=Mock())

            def verify(ticket):
                engine.phase = 'verified'
                return [99 if wrong else (token + 1) % 3 for token in ticket.tokens], dict(input_ms=0, verify_readback_ms=0)

            def publish(prefix):
                self.assertEqual(session.phase, 'committing')
                engine.phase = 'idle'

            engine.verify, engine.publish = verify, publish
            return engine

        with patch('full_request.VerifierEngine', side_effect=factory) as constructor:
            result = measure_request(SimpleNamespace(args=SimpleNamespace(vocab_size=100)), None,
                [0, 1, 2] * 12, None, [], prefill=lambda prompt: seed, decode=decode,
                live_digest=lambda: 'state', kv_digest=lambda position: position,
                inactive_digest=lambda: 'inactive', eos_ids=eos_ids, max_new_tokens=33)
        return result, constructor

    def test_actual_lookup_accounting_and_setup_are_separate(self):
        result, constructor = self.run_fixture()
        self.assertEqual(result['committed_decode_tokens'], 32)
        self.assertEqual(sum(block['committed'] for block in result['blocks']), 32)
        self.assertGreater(result['accepted'], 0)
        self.assertEqual(result['engine_setup_ms'], 12)
        self.assertEqual(result['post_seed_including_setup_ms'], result['decode_ms'] + 12)
        self.assertEqual(result['emitted'], [index % 3 for index in range(33)])
        constructor.assert_called_once()

    def test_terminal_prefill_has_no_decode_throughput(self):
        result, constructor = self.run_fixture(seed=2, eos_ids=(2,))
        self.assertEqual(result['committed_decode_tokens'], 0)
        self.assertIsNone(result['committed_tokens_per_second'])
        self.assertEqual(result['engine_setup_ms'], 0)
        constructor.assert_not_called()

    def test_accepted_terminal_state_matches_native_position(self):
        result, constructor = self.run_fixture(eos_ids=(2,))
        self.assertEqual(result['emitted'], [0, 1, 2])
        self.assertEqual(result['committed_decode_tokens'], 2)

    def test_wrong_target_output_cannot_produce_successful_result(self):
        with self.assertRaisesRegex(AssertionError, 'differs from native'):
            self.run_fixture(wrong=True)
