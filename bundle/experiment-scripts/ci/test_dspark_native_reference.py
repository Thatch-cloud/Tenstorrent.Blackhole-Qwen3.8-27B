import unittest

import torch

from dspark_markov import greedy_proposals
from dspark_native_reference import NativeMarkovReference, audit_scores
from projection_rounding import grouped_projection_reference


class DSparkNativeReferenceTests(unittest.TestCase):
    def fixture(self):
        generator = torch.Generator().manual_seed(38259)
        return [(torch.randn(64, 256, generator=generator) / 8).bfloat16() for _ in range(2)]

    def test_chunking_keeps_the_exact_grouped_arithmetic(self):
        predecessor, successor = self.fixture()
        expected = grouped_projection_reference(predecessor[2:3], successor.T,
            destination_rounding=True, fidelity_span=32).float()
        for chunk in (32, 64, 2048):
            oracle = NativeMarkovReference(predecessor, successor, chunk_columns=chunk)
            self.assertTrue(torch.equal(oracle.bias(2, native=True), expected))
            self.assertIs(oracle.bias(2, native=True), oracle.bias(2, native=True))

    def test_fp32_trajectory_is_retained_independently_after_divergence(self):
        predecessor, successor = self.fixture()
        oracle = NativeMarkovReference(predecessor, successor)
        base = torch.zeros(1, 3, 64)
        base[:, 0] = -oracle.bias(2, native=False)
        expected_fp32 = greedy_proposals(base, torch.tensor([2]), predecessor.float(), successor.float())
        result = oracle.trajectory(base, 2)
        self.assertEqual([entry['fp32_token'] for entry in result], expected_fp32[0].tolist())
        self.assertNotEqual(result[0]['token'], result[0]['fp32_token'])
        self.assertEqual(result[1]['previous'], result[0]['token'])
        self.assertEqual(result[1]['fp32_previous'], result[0]['fp32_token'])
        for step, entry in enumerate(result):
            self.assertTrue(torch.equal(entry['same_input_fp32_scores'],
                base[:, step] + oracle.bias(entry['previous'], native=False)))

    def test_reference_owns_weight_snapshots(self):
        predecessor, successor = self.fixture()
        oracle = NativeMarkovReference(predecessor, successor)
        expected = predecessor[2:3].float() @ successor.float().T
        predecessor.zero_()
        successor.zero_()
        self.assertTrue(torch.equal(oracle.bias(2, native=False), expected))

    def test_score_audit_keeps_fp32_failure_and_rejects_native_drift(self):
        predecessor, successor = self.fixture()
        oracle = NativeMarkovReference(predecessor, successor)
        base = -oracle.bias(2, native=False).reshape(1, 1, 64)
        reference = oracle.trajectory(base, 2)[0]
        result = audit_scores(reference['scores'], reference['token'], reference)
        self.assertTrue(result['full_vocabulary_exact'])
        self.assertFalse(result['fp32_trajectory']['full_vocabulary_close'])
        self.assertGreater(result['fp32_trajectory']['mismatched'], 0)
        for scores, token in ((reference['scores'] + 1e-7, reference['token']),
                (reference['scores'], (reference['token'] + 1) % 64),
                (reference['scores'].double(), reference['token'])):
            with self.assertRaises(AssertionError):
                audit_scores(scores, token, reference)

    def test_native_score_audit_checks_float_bits_including_zero_sign(self):
        oracle = NativeMarkovReference(*[torch.zeros(64, 256, dtype=torch.bfloat16) for _ in range(2)])
        reference = oracle.trajectory(torch.zeros(1, 1, 64), 0)[0]
        scores = reference['scores'].clone()
        scores[0, 0] = -0.
        with self.assertRaises(AssertionError):
            audit_scores(scores, 0, reference)

    def test_ties_empty_or_invalid_parameters(self):
        weights = torch.zeros(64, 256, dtype=torch.bfloat16)
        oracle = NativeMarkovReference(weights, weights)
        observed = []
        result = oracle.trajectory(torch.zeros(1, 7, 64), 63, progress=observed.append)
        self.assertEqual([entry['token'] for entry in result], [0] * 7)
        self.assertEqual(observed, list(range(7)))
        for chunk in (True, 0, 16, 4097):
            with self.assertRaises(ValueError):
                NativeMarkovReference(weights, weights, chunk_columns=chunk)
        for token in (-1, 64, True):
            with self.assertRaises(ValueError):
                oracle.bias(token, native=True)
        for base in (torch.zeros(1, 0, 64), torch.zeros(2, 7, 64), torch.full((1, 7, 64), float('nan'))):
            with self.assertRaises(ValueError):
                oracle.trajectory(base, 0)


if __name__ == '__main__':
    unittest.main()
