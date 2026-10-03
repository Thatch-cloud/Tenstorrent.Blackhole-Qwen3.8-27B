import importlib.util
from pathlib import Path
import unittest

import torch

from draft_dot import dot_geometry


SPEC = importlib.util.spec_from_file_location('fixed_attention_fixture', Path(__file__).with_name('dspark-fixed-attention-probe.py'))
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class FixedAttentionFixtureTests(unittest.TestCase):
    def test_all_semantic_controls_are_observable_at_full_context(self):
        patterns = PROBE.fixtures()
        expected = [[PROBE.reference(values, chip) for chip in range(2)] for values in patterns]
        checks = PROBE.controls(patterns, expected)
        self.assertEqual(len(checks), 8)
        self.assertTrue(all(value['detected'] for value in checks))
        self.assertTrue(all(not torch.equal(expected[0][chip], expected[1][chip]) for chip in range(2)))

    def test_query_rows_follow_capacity_and_every_invalid_storage_region_is_poisoned(self):
        for position, values in zip(PROBE.POSITIONS, PROBE.fixtures(), strict=True):
            for name in ('key', 'value'):
                joined = PROBE.joined(values, name)
                self.assertEqual(tuple(joined.shape), (2, 4, 4416, 128))
                self.assertTrue(torch.all(joined[:, :, position:4384] == 8192))
                self.assertTrue(torch.equal(joined[:, :, 4384:4399], values['query_' + name][:, :, :15]))
                self.assertTrue(torch.all(joined[:, :, 4399:] == 0))

    def test_new_final_chunk_is_supported_by_existing_native_dot_geometry(self):
        self.assertEqual(PROBE.geometry(4384, 15), ((0, 2048), (2048, 4096), (4096, 4416)))
        self.assertEqual(dot_geometry((1, 16, 32, 128), (1, 16, 320, 128)), (64, 10, 4))
        self.assertEqual(dot_geometry((1, 16, 32, 320), (1, 16, 128, 320)), (64, 4, 10))
        self.assertEqual(PROBE.COUNTS, dict(eager_checks=4, replay_checks=4, input_checks=48,
            layout_checks=16, fixture_controls=8, stale_controls=2))


if __name__ == '__main__':
    unittest.main()
