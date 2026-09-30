import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import dspark_t32_score_layout as candidate
from test_dspark_markov_device import operands, tensor


class T32ScoreLayoutTests(unittest.TestCase):
    def test_fused_segments_preserve_all_feedback_and_observer_indices(self):
        operations, values = operands(64, 31)
        mesh = SimpleNamespace(shape=[1, 2])
        slices, anchors, steps, owned = [], [], [], []
        tokens = [object() for _ in range(31)]

        def slice_rows(value, start, end):
            self.assertIs(value, values[1])
            slices.append((start[2], end[2]))
            return tensor((1, 1, end[2] - start[2], 64), 'float32', 'tile')

        def fused(ops, actual_mesh, anchor, logits, predecessor, successor, retained, *, on_step_enqueued):
            self.assertIs(actual_mesh, mesh)
            self.assertIs(retained, owned)
            anchors.append(anchor)
            start, end = slices[-1]
            for step in range(end - start):
                on_step_enqueued(step)
            return [dict(token=token) for token in tokens[start:end]]

        operations.slice = slice_rows
        operations.reshape = lambda value, shape: value
        with patch.dict('os.environ', {'QWEN_SIM_ONLY': '1'}, clear=True), patch.object(
                candidate, 'fused', side_effect=fused):
            result = candidate.execute(operations, *values, owned, mesh=mesh, on_step_enqueued=steps.append)
        self.assertEqual([record['token'] for record in result], tokens)
        self.assertEqual(anchors, [values[0], tokens[6], tokens[13], tokens[20], tokens[27]])
        self.assertEqual(slices, [(0, 7), (7, 14), (14, 21), (21, 28), (28, 31)])
        self.assertEqual(steps, list(range(31)))
        self.assertEqual(len(owned), 10)

    def test_serving_hardware_and_wrong_mesh_rejected_before_device_work(self):
        operations, values = operands(64, 31)
        operations.slice = Mock()
        for environment, shape in (({}, [1, 2]), ({'QWEN_SIM_ONLY': '1', 'QWEN_HARDWARE_TESTS': '1'}, [1, 2]),
                ({'QWEN_SIM_ONLY': '1', 'QWEN_CARDS_ALLOCATED': '1'}, [1, 2]),
                ({'QWEN_SIM_ONLY': '1'}, [1, 4])):
            with patch.dict('os.environ', environment, clear=True), self.assertRaises(ValueError):
                candidate.execute(operations, *values, [], mesh=SimpleNamespace(shape=shape))
        operations.slice.assert_not_called()

    def test_partial_segment_never_returns_incomplete_proposals(self):
        operations, values = operands(64, 31)
        operations.slice = Mock(return_value=tensor((1, 1, 7, 64), 'float32', 'tile'))
        with patch.dict('os.environ', {'QWEN_SIM_ONLY': '1'}, clear=True), patch.object(
                candidate, 'fused', return_value=[]), self.assertRaises(AssertionError):
            candidate.execute(operations, *values, [], mesh=SimpleNamespace(shape=[1, 2]))


if __name__ == '__main__':
    unittest.main()
