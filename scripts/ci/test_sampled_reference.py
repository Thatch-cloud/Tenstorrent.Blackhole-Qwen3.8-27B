"""The reference path of a sampled request (QWEN_FAST_SAMPLED, docs/tp4-sampled-verify.md section 4), pure CPU.

full_request.measure_request decodes a gold text one token at a time and checks every committed token of the speculative run against it ("0 mismatches"). For a sampled request the gold
is the keyed draw - sampled_accept.sample_logits at the token's absolute position, from the prefill logits for token 0 (position len(prompt)) - and the verifier's rows are the draws of
the engine's own logits (sampled_verify.sequential_predictions). Held here: the gold equals the plain sampler's text, the speculative run commits it with no mismatch, a wrong draw is
caught at its token, a sampled reference needs the prefill logits, and a greedy reference is the old function.
"""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / 'speculative-decoding' / 'harness'))

import full_request  # noqa: E402
import sampled_accept as sa  # noqa: E402
import sampled_test_support as support  # noqa: E402
import sampled_verify as sv  # noqa: E402
from tp_test_support import four_cards  # noqa: E402

PROMPT = [support.phi(token) for token in (5, 17, 3, 3, 40, 5, 17, 3, 3, 40, 5, 17, 3, 3)]
COUNT = 25


class Fixture(object):
    def __init__(self, sampling, wrong_at=None):
        self.sampling, self.wrong_at = sampling, wrong_at
        self.context = list(PROMPT)
        self.prefills = 0

    def prefill_logits(self, prompt):
        self.prefills += 1
        self.context = list(prompt)
        return torch.from_numpy(support.full_row(self.context)).reshape(1, 1, -1)

    def prefill(self, prompt):
        return int(self.prefill_logits(prompt).reshape(-1).argmax())

    def decode(self, token, position, trace):
        self.context.append(token)
        return torch.from_numpy(support.full_row(self.context)).reshape(1, -1)

    def factory(self, model, session, pages, helpers, sampler, **options):
        fixture = self
        engine = SimpleNamespace(setup_ms=1.0, phase='idle', close=mock.Mock(), buckets={}, pending_key=None, session=session, operations=support.Ops(), model=model)

        def verify(ticket):
            history = list(PROMPT) + list(session.emitted)
            rows = [support.full_row(history + list(ticket.tokens[1:index + 1])) for index in range(len(ticket.tokens))]
            engine.buckets = {1: dict(output=(support.logits_tensor(np_rows(rows)), None))}
            engine.pending_key = 1
            engine.phase = 'verified'
            engine.pending = ticket
            return [int(row.argmax()) for row in rows], dict(input_ms=0, verify_readback_ms=0)      # (the greedy rows; a sampled request's are replaced by its draws)

        def publish(prefix):
            engine.phase = 'idle'

        engine.verify, engine.publish = verify, publish
        return engine


def np_rows(rows):
    import numpy as np

    return np.stack(rows)


class Clean(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, support.clean_environment(), clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        context = four_cards()
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def measure(self, fixture, **options):
        model = SimpleNamespace(args=SimpleNamespace(vocab_size=support.VOCAB))
        with mock.patch('full_request.VerifierEngine', side_effect=fixture.factory):
            return full_request.measure_request(model, None, PROMPT, SimpleNamespace(shape=(1, 1024)), [], prefill=fixture.prefill, decode=fixture.decode,
                                                live_digest=lambda: 'state', kv_digest=lambda position: position, inactive_digest=lambda: 'inactive',
                                                eos_ids=(support.EOS,), max_new_tokens=COUNT, **options)


class SampledReferenceTests(Clean):
    def test_the_gold_is_the_plain_sampled_text_and_the_speculative_run_commits_it(self):
        for spec, seed in ((sa.SamplingSpec(0.7, 0.9, 8), 31), (sa.SamplingSpec(1.0, 1.0, 64), 2 ** 63 + 5), (sa.SamplingSpec(0.3, 1.0, 1), 1)):
            with self.subTest(spec=spec, seed=seed):
                fixture = Fixture(None)
                sampling = sv.Sampling(spec, seed)
                result = self.measure(fixture, sampling=sampling, prefill_logits=fixture.prefill_logits)
                self.assertEqual(result['emitted'], support.plain_run(PROMPT, COUNT, seed, spec))
                self.assertTrue(result['exact'])
                self.assertEqual(result['sampled'], dict(temperature=spec.temperature, top_p=spec.top_p, top_k=spec.top_k, seed=seed))
                self.assertEqual(fixture.prefills, 2, 'the gold prefill and the request\'s own')
                self.assertGreater(len(result['blocks']), 0)

    def test_the_first_token_is_the_draw_at_the_prompt_length_not_the_argmax(self):
        fixture = Fixture(None)
        spec = sa.SamplingSpec(1.0, 1.0, 64)
        firsts = set()
        for seed in range(8):
            result = self.measure(Fixture(None), sampling=sv.Sampling(spec, seed), prefill_logits=fixture.prefill_logits)
            expected = sa.sample_logits(support.full_row(PROMPT), spec, seed, len(PROMPT))
            self.assertEqual(result['emitted'][0], expected)
            firsts.add(expected)
        self.assertGreater(len(firsts), 1, 'the first token depends on the seed: it is a draw, not the argmax')

    def test_a_wrong_draw_is_caught_at_its_token(self):
        fixture = Fixture(None)
        real = sv.sequential_predictions

        def wrong(engine, output, ticket, sampling):
            rows = real(engine, output, ticket, sampling)
            rows[0] = (rows[0] + 1) % support.VOCAB
            return rows

        with mock.patch.object(sv, 'sequential_predictions', wrong), self.assertRaises(full_request.RequestMismatch) as caught:
            self.measure(fixture, sampling=sv.Sampling(sa.SamplingSpec(0.7, 0.9, 8), 31), prefill_logits=fixture.prefill_logits)
        self.assertEqual(caught.exception.evidence['kind'], 'committed-token-mismatch')

    def test_a_sampled_reference_needs_the_prefill_logits_and_a_greedy_one_must_not_have_them(self):
        sampling = sv.Sampling(sa.SamplingSpec(), 1)
        with self.assertRaisesRegex(ValueError, 'prefill_logits'):
            self.measure(Fixture(None), sampling=sampling)
        with self.assertRaisesRegex(ValueError, 'prefill_logits belongs'):
            self.measure(Fixture(None), prefill_logits=lambda prompt: None)

    def test_a_greedy_reference_is_the_old_function(self):
        fixture = Fixture(None)
        result = self.measure(fixture)
        self.assertNotIn('sampled', result)
        expected, context = [], list(PROMPT)
        while len(expected) < COUNT:
            token = int(support.full_row(context).argmax())
            expected.append(token)
            context.append(token)
            if token == support.EOS:
                break
        self.assertEqual(result['emitted'], expected)
        self.assertEqual(fixture.prefills, 2)

    def test_the_serving_gate_and_the_reference_use_one_function(self):
        """The audited gate's "0 mismatches": the verifier's draws and the gold are both sampled_accept's, on the same logits."""
        import inspect

        self.assertIn('sampled_accept.sample_logits', inspect.getsource(full_request.measure_request))
        self.assertIn('sa.draw_batch', inspect.getsource(sv.sequential_predictions))
        self.assertIn('sa.sample_logits', inspect.getsource(sv.FirstToken.__call__))


if __name__ == '__main__':
    unittest.main()
