from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from mlp_weight_staging import forward, stage_weights, weight_budget
from tiny_tile_matmul import PROJECTIONS


class WeightStagingTests(unittest.TestCase):
    def fixture(self):
        operations = SimpleNamespace(TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram', L1_MEMORY_CONFIG='l1',
            bfloat4_b='bf4', bfloat8_b='bf8', copy=Mock(side_effect=lambda source, target: target),
            get_device_tensors=lambda value: value.shards)
        sources, targets = {}, {}
        for index, (name, (inner, width, unused_cores, dtype, unused_activation)) in enumerate(PROJECTIONS.items()):
            for group, memory, offset in ((sources, 'dram', 0), (targets, 'l1', 1000)):
                group[name] = SimpleNamespace(shape=(1, 1, inner, width), dtype=getattr(operations, dtype),
                    layout='tile', memory_config=lambda memory=memory: memory,
                    shards=[SimpleNamespace(buffer_address=lambda address=index * 10 + offset + chip: address)
                        for chip in range(2)])
        return operations, sources, targets

    def test_budget_counts_compressed_pages_not_bf16_or_full_l1_capacity(self):
        budget = weight_budget()
        self.assertEqual(budget['total_bytes'], 97484800)
        self.assertEqual(budget['weight_bytes_per_bank_upper_bound'], 887040)
        self.assertEqual(budget['weights']['gate']['page_bytes'], 576)
        self.assertEqual(budget['weights']['down']['page_bytes'], 1088)
        with self.assertRaises(ValueError):
            weight_budget(120)

    def test_copy_uses_preallocated_storage_with_no_arithmetic(self):
        operations, sources, targets = self.fixture()
        stage_weights(operations, sources, targets)
        self.assertEqual(operations.copy.call_count, 3)
        for call, name in zip(operations.copy.call_args_list, PROJECTIONS, strict=True):
            self.assertIs(call.args[0], sources[name])
            self.assertIs(call.args[1], targets[name])

    def test_invalid_last_weight_fails_before_any_copy(self):
        for field, value in (('shape', (1, 1, 8704, 5119)), ('dtype', 'bf16'), ('layout', 'row'),
                ('memory_config', lambda: 'dram')):
            operations, sources, targets = self.fixture()
            setattr(targets['down'], field, value)
            with self.assertRaises(ValueError):
                stage_weights(operations, sources, targets)
            operations.copy.assert_not_called()

    def test_missing_weights_aliases_and_new_output_are_rejected(self):
        operations, sources, targets = self.fixture()
        with self.assertRaises(ValueError):
            stage_weights(operations, {key: value for key, value in sources.items() if key != 'down'}, targets)
        targets['up'].shards[0] = targets['gate'].shards[0]
        with self.assertRaises(ValueError):
            stage_weights(operations, sources, targets)
        operations.copy.assert_not_called()
        operations, sources, targets = self.fixture()
        operations.copy.side_effect = lambda source, target: source
        with self.assertRaises(ValueError):
            stage_weights(operations, sources, targets)

    @patch('mlp_weight_staging.execute')
    def test_cost_boundaries_are_explicit_and_math_is_unchanged(self, execute):
        for mode in ('dram', 'resident', 'staged'):
            operations, sources, targets = self.fixture()
            forward(operations, 'input', sources, targets, 'programs', 'compute', Mock(), mode=mode)
            self.assertEqual(operations.copy.call_count, 3 if mode == 'staged' else 0)
            self.assertIs(execute.call_args.args[2], sources if mode == 'dram' else targets)
            self.assertEqual(execute.call_args.kwargs, dict(tiny=False))
        with self.assertRaises(ValueError):
            forward(operations, 'input', sources, targets, 'programs', 'compute', Mock(), mode='overlapped')
