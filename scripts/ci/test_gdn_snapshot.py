import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from gdn_snapshot import ActiveSnapshot


class SnapshotTests(unittest.TestCase):
    def fixture(self):
        layer = SimpleNamespace(B=8, _stable_state=True, rec_state="rec", conv_states=["conv0", "conv1"],
                                _slice_along=Mock(), _write_recurrent_state_prefix=Mock(), _write_index=Mock())
        operations = SimpleNamespace(clone=Mock(side_effect=lambda source, **kwargs: ("clone", source)),
                                     copy=Mock(), deallocate=Mock(), DRAM_MEMORY_CONFIG="DRAM")
        return layer, operations, ActiveSnapshot(layer, operations)

    def test_slices_correct_batch_axes(self):
        layer, operations, snapshots = self.fixture()
        self.assertEqual(len(snapshots.allocate()), 3)
        self.assertEqual([call.args for call in layer._slice_along.call_args_list],
                         [("rec", 0, 0, 1), ("conv0", 1, 0, 1), ("conv1", 1, 0, 1)])
        self.assertEqual(operations.deallocate.call_count, 3)

    def test_restore_clones_before_consuming_native_writes(self):
        layer, operations, snapshots = self.fixture()
        snapshots.restore(["saved-rec", "saved0", "saved1"])
        layer._write_recurrent_state_prefix.assert_called_once_with(("clone", "saved-rec"), 1)
        self.assertEqual([call.args for call in layer._write_index.call_args_list],
                         [("conv0", ("clone", "saved0"), 0, 1), ("conv1", ("clone", "saved1"), 0, 1)])

    def test_incomplete_snapshot_fails_before_device_work(self):
        layer, operations, snapshots = self.fixture()
        for operation in (snapshots.save, snapshots.restore):
            with self.assertRaises(ValueError):
                operation(["only-one"])
        operations.clone.assert_not_called()
        layer._slice_along.assert_not_called()

    def test_requires_stable_eight_slot_state(self):
        layer, operations, _ = self.fixture()
        layer._stable_state = False
        with self.assertRaises(ValueError):
            ActiveSnapshot(layer, operations)


if __name__ == "__main__":
    unittest.main(verbosity=2)
