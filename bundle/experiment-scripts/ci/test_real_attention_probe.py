import hashlib
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch


SPEC = importlib.util.spec_from_file_location('real_attention_probe', Path(__file__).with_name('real-attention-probe.py'))
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class RealAttentionFixtureTests(unittest.TestCase):
    def fixture(self):
        return dict(query=torch.ones(1, 8, 12, 256, dtype=torch.bfloat16),
            actual=torch.ones(1, 8, 12, 256, dtype=torch.bfloat16),
            expected=torch.ones(1, 8, 12, 256, dtype=torch.bfloat16),
            keys=torch.ones(4, 2, 64, 256, dtype=torch.bfloat16),
            values=torch.ones(4, 2, 64, 256, dtype=torch.float32),
            pages=torch.arange(4, dtype=torch.int32).flip(0).reshape(1, 4),
            position=170, scale=0.0625, chip=1)

    def test_exact_file_digest_and_real_geometry_are_required(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'real-query.pt'
            torch.save(self.fixture(), path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            fixture, capacity = PROBE.load_fixture(path, digest)
            self.assertEqual(capacity, 256)
            self.assertEqual(fixture['position'], 170)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                PROBE.load_fixture(path, '0' * 64)
            for change in (dict(position=249), dict(chip=True), dict(scale=1),
                           dict(pages=torch.tensor([[0, 0, 1, 2]], dtype=torch.int32)),
                           dict(query=torch.zeros(1, 8, 12, 128, dtype=torch.bfloat16))):
                torch.save(dict(self.fixture(), **change), path)
                with self.assertRaises(ValueError):
                    PROBE.load_fixture(path, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_comparison_does_not_hide_small_differences(self):
        actual = torch.ones(3, dtype=torch.bfloat16)
        expected = actual.clone()
        actual[1] = 1.0078125
        result = PROBE.comparison(actual, expected)
        self.assertFalse(result['exact'])
        self.assertEqual(result['differing_elements'], 1)
        self.assertEqual(result['max_absolute_error'], 0.0078125)
