import importlib.util
from pathlib import Path
import unittest

import torch


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeFixedProbeTests(unittest.TestCase):
    def test_native_candidate_keeps_original_fixture_and_acceptance_policy(self):
        original = load('dspark-fixed-attention-probe')
        candidate = load('dspark-native-fixed-attention-probe')
        self.assertEqual(candidate.COUNTS, original.COUNTS)
        self.assertEqual(candidate.POSITIONS, original.POSITIONS)
        self.assertEqual(candidate.REPLAYS, original.REPLAYS)
        self.assertEqual(candidate.PROPOSALS, 15)
        self.assertEqual(candidate.CAPACITY, 4384)
        for actual, expected in zip(candidate.fixtures(), original.fixtures(), strict=True):
            self.assertEqual(set(actual), set(expected))
            for name in actual:
                self.assertTrue(torch.equal(actual[name], expected[name]), name)
        self.assertIn('dspark_native_full_attention.py', candidate.source_hashes())
        self.assertIn('native_draft_sdpa.py', candidate.source_hashes())
        self.assertIsNot(candidate.execute, original.execute)
