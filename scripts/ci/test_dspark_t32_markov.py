import unittest
from unittest.mock import Mock, patch

from dspark_t32_markov import execute
from test_dspark_markov_device import operands, tensor


class T32MarkovTests(unittest.TestCase):
    def test_segments_preserve_all_rows_and_feedback_boundaries(self):
        operations, values = operands(64, 31)
        tokens = [object() for _ in range(31)]
        slices, anchors, steps, owned = [], [], [], []

        def slice_rows(value, start, end):
            self.assertIs(value, values[1])
            slices.append((start[2], end[2]))
            return tensor((1, 1, end[2] - start[2], 64), 'float32', 'tile')

        def segment(ops, anchor, local, predecessor, successor, retained, *, on_step_enqueued):
            anchors.append(anchor)
            self.assertIs(retained, owned)
            start, end = slices[-1]
            for step in range(end - start):
                on_step_enqueued(step)
            return [dict(token=token) for token in tokens[start:end]]

        operations.slice = slice_rows
        operations.reshape = lambda value, shape: value
        with patch('dspark_t32_markov.native', side_effect=segment):
            result = execute(operations, *values, owned, on_step_enqueued=steps.append)
        self.assertEqual([entry['token'] for entry in result], tokens)
        self.assertEqual(anchors, [values[0], tokens[6], tokens[13], tokens[20], tokens[27]])
        self.assertEqual(slices, [(0, 7), (7, 14), (14, 21), (21, 28), (28, 31)])
        self.assertEqual(steps, list(range(31)))
        self.assertEqual(len(owned), 10)

    def test_reject_invalid_width_before_device_work(self):
        for width in (7, 15, 30, 32):
            operations, values = operands(64, width)
            operations.slice = Mock()
            with self.assertRaises(ValueError):
                execute(operations, *values, [])
            operations.slice.assert_not_called()


if __name__ == '__main__':
    unittest.main()
