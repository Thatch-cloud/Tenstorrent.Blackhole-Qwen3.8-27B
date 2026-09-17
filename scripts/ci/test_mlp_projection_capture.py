from pathlib import Path
import tempfile
import unittest

import torch

from draft_mlp_branch import validate_projection


class MlpProjectionCaptureTests(unittest.TestCase):
    def test_each_projection_failure_preserves_reproduction(self):
        inputs = torch.ones(1, 1, 8, 32, dtype=torch.bfloat16)
        weight = torch.ones(32, 4, dtype=torch.bfloat16)
        actual = torch.full((1, 1, 8, 4), 32.0)
        actual[..., 2] = 1e7
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'failure.pt'
            for stage in ('convolution', 'gate', 'up', 'down'):
                with self.assertRaisesRegex(AssertionError, f'Layer 2 chip 1 {stage}'):
                    validate_projection(actual, inputs, weight, stage=stage, layer=2, chip=1,
                        failure_capture=path, checkpoint={'revision': 'test'})
                saved = torch.load(path, weights_only=True)
                self.assertEqual(saved['stage'], stage)
                self.assertEqual(saved['mismatches'].shape, (8, 4))
                self.assertTrue(torch.equal(saved['weight'], weight))
                self.assertTrue(torch.equal(saved['activation'], inputs))
                self.assertTrue(torch.equal(saved['actual'], actual.double()))
                self.assertTrue(torch.equal(saved['reference'], saved['ordinary_reference']))
                self.assertEqual(saved['checkpoint'], {'revision': 'test'})

    def test_success_does_not_write_capture(self):
        inputs = torch.ones(1, 32, dtype=torch.bfloat16)
        weight = torch.ones(32, 2, dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'failure.pt'
            self.assertEqual(validate_projection(torch.full((1, 2), 32.), inputs, weight,
                stage='gate', failure_capture=path), 0)
            self.assertFalse(path.exists())

    def test_missing_capture_path_still_fails(self):
        with self.assertRaises(AssertionError):
            validate_projection(torch.zeros(1, 2), torch.ones(1, 32, dtype=torch.bfloat16),
                torch.ones(32, 2, dtype=torch.bfloat16), stage='up')


if __name__ == '__main__':
    unittest.main()
