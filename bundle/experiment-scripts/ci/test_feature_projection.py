from types import SimpleNamespace
import unittest

import torch

from feature_projection import concatenate_local_features, projection_shards, sparse_input_permutation, require_projection_environment


class FeatureProjectionTests(unittest.TestCase):
    def test_projection_runtime_requires_explicit_allocation(self):
        hardware = dict(QWEN_HARDWARE_TESTS='1', QWEN_CARDS_ALLOCATED='1')
        require_projection_environment(hardware, True)
        require_projection_environment(dict(TT_METAL_SIMULATOR='sim.so'), False)
        for environment, selected in (({}, False), ({}, True), (hardware, False),
                (dict(QWEN_HARDWARE_TESTS='1'), True),
                (dict(hardware, TT_METAL_SIMULATOR='sim.so'), True),
                (dict(hardware, TT_METAL_MOCK_CLUSTER_DESC_PATH='mock.yaml'), True),
                (dict(hardware, TT_METAL_SLOW_DISPATCH_MODE='1'), True),
                (dict(TT_METAL_SIMULATOR='sim.so', TT_METAL_SLOW_DISPATCH_MODE='1'), False)):
            with self.assertRaises(RuntimeError):
                require_projection_environment(environment, selected)

    def test_sparse_controls_preserve_identical_operands(self):
        features = torch.arange(64, dtype=torch.float64).reshape(2, 32)
        features[:, 2:] = 0
        weight = torch.arange(96, dtype=torch.float64).reshape(32, 3) % 11
        for stride in (1, 4, 8, 16):
            permutation = torch.tensor(sparse_input_permutation(32, 2, stride))
            self.assertEqual(sorted(permutation.tolist()), list(range(32)))
            self.assertTrue(torch.equal(features[:, permutation] @ weight[permutation], features @ weight))
            self.assertTrue(torch.equal(features[:, permutation][:, [0, stride]], features[:, :2]))
        self.assertEqual(sparse_input_permutation(32, 32, 1), tuple(range(32)))

    def test_sparse_controls_reject_invalid_geometry(self):
        for args in ((32, 3, 16), (32, 33, 1), (0, 1, 1), (32, 0, 1), (32, 1, 0), (True, 1, 1)):
            with self.assertRaises(ValueError):
                sparse_input_permutation(*args)

    def test_tp2_preserves_tap_order_and_full_projection(self):
        taps, hidden, outputs, rows = 5, 64, 32, 8
        features = torch.arange(taps * rows * hidden, dtype=torch.float64).reshape(taps, 1, 1, rows, hidden) % 19
        weight = torch.arange(outputs * taps * hidden, dtype=torch.float64).reshape(outputs, taps * hidden) % 7
        operations = SimpleNamespace(DRAM_MEMORY_CONFIG=None,
            concat=lambda values, dim, memory_config: torch.cat(values, dim=dim))
        shards = projection_shards(weight, tap_count=taps, hidden_size=hidden)
        partials = []
        locals = []
        for chip, shard in enumerate(shards):
            values = [value[..., chip * (hidden // 2):(chip + 1) * (hidden // 2)] for value in features]
            local = concatenate_local_features(operations, values, tap_count=taps, hidden_size=hidden)
            locals.append(local)
            partials.append(local @ shard)
        reference = torch.cat(tuple(features), dim=-1) @ weight.T
        self.assertTrue(torch.equal(partials[0] + partials[1], reference))
        wrong = torch.cat(locals, dim=-1) @ weight.T
        self.assertFalse(torch.equal(wrong, reference))

    def test_real_hidden_geometry_reorders_every_tap_and_rank(self):
        weight = torch.arange(2 * 5 * 5120).reshape(2, 5 * 5120)
        shards = projection_shards(weight)
        for chip, shard in enumerate(shards):
            self.assertEqual(tuple(shard.shape), (12800, 2))
            for tap in range(5):
                self.assertTrue(torch.equal(shard[tap * 2560:(tap + 1) * 2560].T,
                    weight[:, tap * 5120 + chip * 2560:tap * 5120 + (chip + 1) * 2560]))

    def test_invalid_dimensions_and_missing_taps_fail(self):
        for options in (dict(hidden_size=3), dict(tap_count=True), dict(hidden_size=True), dict(tap_count=0)):
            with self.assertRaises(ValueError):
                projection_shards(torch.zeros(2, 25600), **options)
        with self.assertRaises(ValueError):
            projection_shards(torch.zeros(2, 12800))
        with self.assertRaises(ValueError):
            concatenate_local_features(None, [])
        with self.assertRaises(ValueError):
            concatenate_local_features(None, [torch.zeros(1, 1, 8, 5120)] * 5)
