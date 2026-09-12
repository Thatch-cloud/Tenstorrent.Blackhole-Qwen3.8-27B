from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gdn_native_slot_recurrence as candidate


class NativeSlotRecurrenceTests(unittest.TestCase):
    def setUp(self):
        self.operations = SimpleNamespace(bfloat16='bf16', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram')
        self.mesh = SimpleNamespace(shape=(1, 2), compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
        shapes = ((1, 16, 5120), (1, 16, 24), (1, 16, 24), (8, 24, 128, 128), (1, 16, 3072), (1, 1, 128))
        self.inputs = [SimpleNamespace(shape=shape, dtype='bf16', layout='tile', memory_config=lambda: 'dram') for shape in shapes]
        binding = patch.object(candidate, 'addresses', side_effect=lambda operations, value: (id(value), id(value)))
        binding.start()
        self.addCleanup(binding.stop)

    def test_native_b8_geometry_admitted_without_copy(self):
        self.assertEqual(len(candidate.validate_inputs(self.operations, self.mesh, self.inputs)), 6)

    def test_compact_wrong_width_dtype_and_alias_rejected(self):
        state = self.inputs[3]
        for shape in ((1, 24, 128, 128), (8, 24, 64, 128)):
            state.shape = shape
            with self.assertRaises(ValueError):
                candidate.validate_inputs(self.operations, self.mesh, self.inputs)
        state.shape = (8, 24, 128, 128)
        self.inputs[0].dtype = 'bf8'
        with self.assertRaises(ValueError):
            candidate.validate_inputs(self.operations, self.mesh, self.inputs)
        self.inputs[0].dtype = 'bf16'
        self.inputs[2] = self.inputs[1]
        with self.assertRaisesRegex(ValueError, 'Independent'):
            candidate.validate_inputs(self.operations, self.mesh, self.inputs)

    def test_default_rejects_before_allocation(self):
        operations = Mock()
        with self.assertRaisesRegex(ValueError, 'explicit experiment'):
            candidate.execute(operations, self.mesh, *self.inputs[:4], z=self.inputs[4], norm_w=self.inputs[5], root='/unused')
        operations.empty.assert_not_called()
