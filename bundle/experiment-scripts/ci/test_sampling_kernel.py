"""Host oracle coverage for the actual-vocabulary sampler gate."""

import importlib.util
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location("probe", Path(__file__).with_name("sampling-kernel.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class SamplerOracleTests(unittest.TestCase):
    def test_boundaries_and_ties(self):
        for kind, expected in (("boundaries", [0, 31, 32, 63, 64, 127]),
                               ("cross-shard-tie", [31] * 6), ("near-tie", [64] * 6),
                               ("all-equal", [0] * 6)):
            values = probe.logits_case(128, 6, kind)
            self.assertEqual(values.dtype, torch.bfloat16)
            self.assertEqual(values.argmax(-1).reshape(-1).tolist(), expected)

    def test_random_is_repeatable(self):
        self.assertTrue(torch.equal(probe.logits_case(128, 6, "random"), probe.logits_case(128, 6, "random")))

    def test_bad_cases_rejected(self):
        with self.assertRaises(ValueError):
            probe.logits_case(65, 6, "random")
        with self.assertRaises(ValueError):
            probe.logits_case(128, 6, "unknown")


if __name__ == "__main__":
    unittest.main()
