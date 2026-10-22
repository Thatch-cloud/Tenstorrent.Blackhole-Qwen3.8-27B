"""The REAL verifier_engine_tp.VerifierEngine.verify for a sampled session (QWEN_FAST_SAMPLED, docs/tp4-sampled-verify.md), pure CPU.

test_verifier_engine_tp's engine (every device call a fake) is given the logits a 1/2/4-row verify trace leaves behind - the vocabulary shard on each chip - and a session that carries a
Sampling or not. verify() is unchanged around the hook: a sampled session's predictions are the keyed draws at the ticket's positions from the host slice of those logits; a greedy
session (no `sampling`, which is every session with the switch off) reads the pinned sampler's ids exactly as before."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sampled_accept as sa  # noqa: E402
import sampled_test_support as support  # noqa: E402
import sampled_verify as sv  # noqa: E402
import test_verifier_engine_tp as base  # noqa: E402
from tp_test_support import four_cards  # noqa: E402


class SequentialVerifyTests(unittest.TestCase):
    def setUp(self):
        sv.reset()
        self.addCleanup(sv.reset)
        patcher = mock.patch.dict(os.environ, support.clean_environment(), clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        context = four_cards()
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        logger = mock.patch.object(sv, 'log_line')
        logger.start()
        self.addCleanup(logger.stop)

    def engine(self, array, sampling=None, with_attribute=True):
        live, ticket, calls = base.verifying_engine(first=False, retained=False)
        ops = support.Ops()
        ops.execute_trace = lambda mesh, trace, cq_id, blocking: calls.append('execute')
        live.operations = ops
        live.model = SimpleNamespace(args=SimpleNamespace(vocab_size=support.VOCAB))
        live.buckets['key']['first'] = True
        live.buckets['key']['output'] = (support.logits_tensor(array), 'ids')
        if with_attribute:
            live.session.sampling = sampling
        live.position = 300
        live.restore_carry = lambda: True
        live.validate_bindings = lambda: None
        ticket = SimpleNamespace(tokens=tuple(range(array.shape[0])), position=300)
        return live, ticket, calls

    def rows(self, count, seed):
        rng = np.random.default_rng(seed)
        return (np.round(rng.normal(size=(count, support.VOCAB)) * 16) / 8).astype(np.float32)

    def verify(self, live, ticket):
        with mock.patch('verifier_engine_tp.stage_inputs', lambda *args: None):
            return live.verify(ticket)

    def test_a_sampled_session_gets_the_keyed_draws_of_the_read_back_logits(self):
        for rows in (1, 2, 4):
            array = self.rows(rows, 90 + rows)
            sampling = sv.Sampling(sa.SamplingSpec(0.8, 0.95, 20), 4242)
            live, ticket, calls = self.engine(array, sampling)
            predictions, timings = self.verify(live, ticket)
            bf16 = torch.from_numpy(array).bfloat16().float().numpy()
            self.assertEqual(predictions, [sa.sample_logits(bf16[row], sampling.spec, sampling.seed, 301 + row) for row in range(rows)], rows)
            self.assertEqual(live.phase, 'verified')
            self.assertIn('output_readback_host_ms', timings)

    def test_a_greedy_session_reads_the_pinned_sampler_ids_as_before(self):
        array = self.rows(4, 95)
        for session_has_attribute in (True, False):
            live, ticket, calls = self.engine(array, None, with_attribute=session_has_attribute)
            live.operations.to_torch = lambda part, mesh_composer=None: base.Host()
            live.operations.get_device_tensors = lambda tensor: [object()] * 4
            predictions, _ = self.verify(live, ticket)
            self.assertEqual(predictions, [7, 8, 9, 10], 'the ids the pinned sampler left, untouched by the sampled hook')

    def test_a_malformed_output_is_refused_not_served_as_argmax(self):
        array = self.rows(2, 96)
        live, ticket, calls = self.engine(array, sv.Sampling(sa.SamplingSpec(), 1))
        live.buckets['key']['output'] = (support.Tensor([torch.zeros(1, 1, 2, 123)] * 4), 'ids')
        with self.assertRaises(sv.SampledRefused):
            self.verify(live, ticket)
        self.assertEqual(live.phase, 'failed')


if __name__ == '__main__':
    unittest.main()
