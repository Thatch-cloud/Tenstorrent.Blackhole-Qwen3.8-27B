import unittest

import torch

from dspark_confidence_reference import evaluate


class ConfidenceReferenceTests(unittest.TestCase):
    def inputs(self):
        return [torch.zeros(1, 3, 2), torch.tensor([0]), torch.tensor([[1, 2, 3]]),
            torch.arange(4, dtype=torch.float32)[:, None],
            torch.tensor([[0., 0., 1.]]), torch.zeros(1), torch.tensor(1.)]

    def test_predecessor_alignment_including_token_zero(self):
        result = evaluate(*self.inputs())
        torch.testing.assert_close(result['predecessors'], torch.tensor([[0, 1, 2]]))
        torch.testing.assert_close(result['logits'], torch.tensor([[0., 1., 2.]]))
        self.assertEqual(result['probabilities'][0, 0].item(), 0.5)

    def test_calibration_and_prefix_survival(self):
        inputs = self.inputs()
        inputs[-1] = torch.tensor([1., 2., 4.])
        result = evaluate(*inputs)
        expected = torch.sigmoid(torch.tensor([[0., 0.5, 0.5]]))
        torch.testing.assert_close(result['probabilities'], expected)
        torch.testing.assert_close(result['survival'], expected.cumprod(1))

    def test_single_position_tail_uses_anchor(self):
        inputs = self.inputs()
        inputs[0] = inputs[0][:, :1]
        inputs[2] = inputs[2][:, :1]
        result = evaluate(*inputs)
        self.assertEqual(result['survival'].tolist(), [[0.5]])

    def test_invalid_inputs_rejected(self):
        for index, value in ((1, torch.tensor([-1])), (2, torch.tensor([[1, 2, 4]])),
                (5, torch.tensor([float('nan')])), (6, torch.tensor(0.)),
                (6, torch.tensor([1., 1.])), (1, torch.tensor([0.]))):
            inputs = self.inputs()
            inputs[index] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                evaluate(*inputs)

    def test_embedding_rounds_to_hidden_dtype_before_projection(self):
        inputs = self.inputs()
        inputs[0] = inputs[0].bfloat16()
        inputs[3] = torch.tensor([[0.1001], [0.2001], [0.3001], [0.4001]])
        result = evaluate(*inputs)
        expected = inputs[3][:3, 0].bfloat16().float()[None, :]
        torch.testing.assert_close(result['logits'], expected, rtol=0, atol=0)

    def test_finite_inputs_with_overflow_rejected(self):
        inputs = self.inputs()
        inputs[0].fill_(torch.finfo(torch.float32).max)
        inputs[4].fill_(2.)
        with self.assertRaises(ValueError):
            evaluate(*inputs)
