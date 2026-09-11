import unittest

import torch

from dspark_native_reference import NativeMarkovReference
from dspark_t32_reference import T32MarkovReference


class T32ReferenceTests(unittest.TestCase):
    def test_long_trajectory_matches_original_prefix_and_keeps_feedback(self):
        generator = torch.Generator().manual_seed(38256)
        weights = [(torch.randn(64, 256, generator=generator) / 16).bfloat16() for _ in range(2)]
        base = torch.randn(1, 31, 64, generator=generator) / 8
        oracle = T32MarkovReference(*weights)
        steps = []
        actual = oracle.trajectory(base, 2, progress=steps.append)
        prefix = NativeMarkovReference(*weights).trajectory(base[:, :15], 2)
        for observed, expected in zip(actual[:15], prefix, strict=True):
            for key in expected:
                if isinstance(expected[key], torch.Tensor):
                    self.assertTrue(torch.equal(observed[key], expected[key]))
                else:
                    self.assertEqual(observed[key], expected[key])
        self.assertEqual(steps, list(range(31)))
        for step, record in enumerate(actual):
            self.assertEqual(record['previous'], actual[step - 1]['token'] if step else 2)
            self.assertEqual(record['fp32_previous'], actual[step - 1]['fp32_token'] if step else 2)
            expected = base[:, step] + oracle.bias(record['previous'], native=True)
            self.assertTrue(torch.equal(record['scores'], expected))
            expected_fp32 = base[:, step] + oracle.bias(record['fp32_previous'], native=False)
            self.assertTrue(torch.equal(record['fp32_scores'], expected_fp32))

    def test_existing_oracle_remains_bounded(self):
        weights = torch.zeros(64, 256, dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            NativeMarkovReference(weights, weights).trajectory(torch.zeros(1, 31, 64), 0)
        with self.assertRaises(ValueError):
            T32MarkovReference(weights, weights).trajectory(torch.zeros(1, 15, 64), 0)


if __name__ == '__main__':
    unittest.main()
