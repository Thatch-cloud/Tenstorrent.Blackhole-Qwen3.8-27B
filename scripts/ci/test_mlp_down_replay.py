import importlib.util
from pathlib import Path
import unittest

import torch


spec = importlib.util.spec_from_file_location('mlp_down_replay', Path(__file__).with_name('mlp-down-replay.py'))
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class MlpDownReplayTests(unittest.TestCase):
    def fixture(self):
        activation = torch.ones(1, 1, 8, 32, dtype=torch.bfloat16)
        weight = torch.arange(128).reshape(32, 4).bfloat16()
        reference = activation.double() @ weight.double()
        actual = reference.clone()
        actual[0, 0, 3, 2] += 10
        return dict(activation=activation, actual=actual, reference=reference), weight

    def test_failed_columns_preserve_all_rows_and_input_terms(self):
        captured, weight = self.fixture()
        columns, selected, actual, reference = replay.select_failure_columns(captured, weight)
        self.assertEqual(columns.tolist(), [2])
        self.assertEqual(selected.shape, (32, 1))
        self.assertEqual(actual.shape, (1, 1, 8, 1))
        self.assertTrue(torch.equal(captured['activation'].double() @ selected.double(), reference))

    def test_no_failure_nonfinite_and_bad_shape_are_rejected(self):
        for change in ('exact', 'nonfinite', 'shape'):
            captured, weight = self.fixture()
            if change == 'exact':
                captured['actual'] = captured['reference'].clone()
            elif change == 'nonfinite':
                captured['actual'][0, 0, 0, 0] = float('nan')
            else:
                weight = weight[:-1]
            with self.assertRaises(ValueError):
                replay.select_failure_columns(captured, weight)
