import unittest

import torch

from dspark_markov import greedy_proposals


class DSparkMarkovTests(unittest.TestCase):
    def fixture(self):
        predecessor = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]], dtype=torch.float64)
        successor = torch.tensor([[-1., 0.], [2., -1.], [0., 2.]], dtype=torch.float64)
        return torch.zeros(2, 4, 3, dtype=torch.float64), torch.tensor([0, 1]), predecessor, successor

    def test_previous_proposal_drives_next_bias_not_repeated_anchor(self):
        logits, anchors, predecessor, successor = self.fixture()
        actual = greedy_proposals(logits, anchors, predecessor, successor)
        self.assertEqual(actual.tolist(), [[1, 2, 0, 1], [2, 0, 1, 2]])
        self.assertFalse(torch.equal(actual, logits.argmax(dim=-1)))

    def test_independent_scalar_reference_and_no_input_mutation(self):
        generator = torch.Generator().manual_seed(7)
        for width in (0, 7, 15):
            logits = torch.randn(2, width, 11, generator=generator, dtype=torch.float64)
            predecessor = torch.randn(11, 3, generator=generator, dtype=torch.float64)
            successor = torch.randn(11, 3, generator=generator, dtype=torch.float64)
            anchors = torch.tensor([2, 8])
            values = (logits, anchors, predecessor, successor)
            originals = [value.clone() for value in values]
            actual = greedy_proposals(*values)
            expected = torch.empty(2, width, dtype=torch.int64)
            for batch in range(2):
                previous = int(anchors[batch])
                for position in range(width):
                    scores = [float(logits[batch, position, token]) + sum(float(predecessor[previous, rank])
                        * float(successor[token, rank]) for rank in range(3)) for token in range(11)]
                    previous = max(range(11), key=lambda token: scores[token])
                    expected[batch, position] = previous
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(all(torch.equal(value, original) for value, original in zip(values, originals)))

    def test_ties_choose_lowest_full_vocabulary_token(self):
        logits, anchors, predecessor, successor = self.fixture()
        predecessor.zero_()
        self.assertEqual(greedy_proposals(logits, anchors, predecessor, successor).tolist(), [[0] * 4] * 2)

    def test_no_top_k_truncation_before_bias(self):
        logits = torch.tensor([[[4., 3., -10.]]])
        self.assertEqual(greedy_proposals(logits, torch.tensor([0]), torch.ones(3, 1),
            torch.tensor([[0.], [0.], [20.]])).tolist(), [[2]])

    def test_invalid_operands_fail(self):
        for mode in ('nan', 'inf', 'negative', 'oob', 'dtype', 'shape', 'overflow'):
            with self.subTest(mode=mode):
                logits, anchors, predecessor, successor = self.fixture()
                if mode in ('nan', 'inf'):
                    logits[0, 0, 0] = float(mode)
                elif mode == 'negative':
                    anchors[0] = -1
                elif mode == 'oob':
                    anchors[0] = 3
                elif mode == 'dtype':
                    predecessor = predecessor.float()
                elif mode == 'shape':
                    anchors = anchors[:, None]
                else:
                    predecessor.fill_(1e308)
                    successor.fill_(1e308)
                with self.assertRaises(ValueError):
                    greedy_proposals(logits, anchors, predecessor, successor)


if __name__ == '__main__':
    unittest.main()
