from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import dspark_markov_score_layout as candidate


class MarkovLayoutTests(unittest.TestCase):
    def test_feedback_and_ownership(self):
        operations = Mock()
        mesh = SimpleNamespace(shape=[1, 2])
        anchor, base, predecessor, successor = (object() for _ in range(4))
        embeddings = [object() for _ in range(3)]
        latents = [object() for _ in range(3)]
        biases = [object() for _ in range(3)]
        scores = [object() for _ in range(3)]
        tokens = [object() for _ in range(3)]
        feedback = [object() for _ in range(3)]
        operations.embedding.side_effect = embeddings
        operations.to_layout.side_effect = latents
        operations.matmul.side_effect = biases
        operations.argmax.side_effect = tokens
        operations.reshape.side_effect = feedback
        owned, observed = [], []

        def layout(actual_operations, actual_mesh, actual_base, bias, step, retain):
            self.assertIs(actual_operations, operations)
            self.assertIs(actual_mesh, mesh)
            self.assertIs(actual_base, base)
            self.assertIs(bias, biases[step])
            return retain(scores[step])

        with patch.object(candidate, 'validate', return_value=(3, 248320)), patch.object(
                candidate, 'score_layout', side_effect=layout):
            records = candidate.execute(operations, mesh, anchor, base, predecessor, successor,
                owned, on_step_enqueued=observed.append)
        self.assertEqual(observed, [0, 1, 2])
        self.assertEqual(records, [dict(token=token, scores=score) for token, score in zip(tokens, scores)])
        for step, previous in enumerate([anchor, *feedback[:-1]]):
            self.assertIs(operations.embedding.call_args_list[step].args[0], previous)
            self.assertIs(operations.argmax.call_args_list[step].args[0], scores[step])
            self.assertIs(operations.matmul.call_args_list[step].args[1], successor)
        self.assertEqual(len(owned), 18)
        self.assertEqual(len({id(value) for value in owned}), 18)
        operations.slice.assert_not_called()
        operations.add.assert_not_called()
        operations.untilize.assert_not_called()
        self.assertEqual(operations.MatmulMultiCoreReuseMultiCast1DProgramConfig.call_args.kwargs[
            'compute_with_storage_grid_size'], (10, 10))
        self.assertFalse(operations.WormholeComputeKernelConfig.call_args.kwargs['math_approx_mode'])

    def test_invalid_mesh_fails_before_enqueue(self):
        operations = Mock()
        with patch.object(candidate, 'validate', return_value=(3, 64)), self.assertRaises(ValueError):
            candidate.execute(operations, SimpleNamespace(shape=[1, 1]), None, None, None, None, [])
        operations.embedding.assert_not_called()


if __name__ == '__main__':
    unittest.main()
