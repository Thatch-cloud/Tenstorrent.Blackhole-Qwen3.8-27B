import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from gdn_snapshot import ActiveSnapshot
from gdn_state_copy import page_counts, transfer_counts


class SnapshotTests(unittest.TestCase):
    def test_transfer_modes_reject_unintended_shapes(self):
        compact = [(1, 24, 128, 128)] + [(1, 1, 5120)] * 4
        full = [(8, 24, 128, 128)] + [(1, 8, 5120)] * 4
        self.assertEqual(transfer_counts(compact, compact, True), [384, 160, 160, 160, 160])
        self.assertEqual(transfer_counts(compact, full), transfer_counts(full, compact))
        for source, destination, mode in ((full, full, False), (compact, compact, False),
                                          (full, compact, True), (compact, full, True),
                                          (compact[:4], compact, True), (full, compact[:4], False)):
            with self.assertRaises(ValueError):
                transfer_counts(source, destination, mode)

    def test_face_transfers_preserve_noc_address_alignment(self):
        source = Path(__file__).with_name("gdn_state_copy.cpp").read_text()
        self.assertIn("source.get_noc_addr(page, 512), scratch + 512, 32", source)
        self.assertIn("scratch + 512, destination.get_noc_addr(page, 512), 32", source)
        self.assertNotIn("scratch + 32", source)

    def test_direct_copy_accepts_only_frozen_page_geometry(self):
        shapes = [(1, 24, 128, 128)] + [(1, 1, 5120)] * 4
        self.assertEqual(page_counts(shapes), [384, 160, 160, 160, 160])
        for changed in (shapes[:4], [(8, 24, 128, 128)] + shapes[1:],
                        shapes[:4] + [(1, 1, 1024)]):
            with self.assertRaises(ValueError):
                page_counts(changed)

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

    def adoption_fixture(self, direct):
        # Slices are distinguishable per tensor and row, and every device call records
        # its order, so the copy sequence can be read back exactly.
        order = []

        def slice_along(tensor, dimension, start, count):
            order.append("slice")
            return ("slice", tensor, dimension, start, count)

        layer = SimpleNamespace(B=8, _stable_state=True, rec_state="rec", conv_states=["conv0", "conv1"],
                                _slice_along=Mock(side_effect=slice_along),
                                _write_recurrent_state_prefix=Mock(side_effect=lambda *args: order.append("write")),
                                _write_index=Mock(side_effect=lambda *args: order.append("write")))
        operations = SimpleNamespace(clone=Mock(side_effect=lambda source, **kwargs: ("clone", source)),
                                     copy=Mock(), deallocate=Mock(side_effect=lambda value: order.append("free")),
                                     DRAM_MEMORY_CONFIG="DRAM")
        return layer, operations, order, ActiveSnapshot(layer, operations, direct=direct)

    def test_adopt_slot_copies_the_prefill_row_into_slot_zero_over_the_slice_path(self):
        # The batched prefill wrote the user's state into its decode slot; serving builds
        # the helpers direct=True, and the DMA kernel copies slot 0 only, so adoption
        # takes the ttnn slice path whatever the mode: slice row k of every live tensor,
        # then the writes restore() makes, then release the slices.
        for direct in (False, True):
            layer, operations, order, snapshots = self.adoption_fixture(direct)
            with self.subTest(direct=direct), patch("gdn_state_copy.copy_active") as dma:
                snapshots.adopt_slot(1)
            dma.assert_not_called()
            slices = [("slice", "rec", 0, 1, 1), ("slice", "conv0", 1, 1, 1), ("slice", "conv1", 1, 1, 1)]
            self.assertEqual([call.args for call in layer._slice_along.call_args_list],
                             [sliced[1:] for sliced in slices])
            layer._write_recurrent_state_prefix.assert_called_once_with(("clone", slices[0]), 1)
            self.assertEqual([call.args for call in layer._write_index.call_args_list],
                             [("conv0", ("clone", slices[1]), 0, 1), ("conv1", ("clone", slices[2]), 0, 1)])
            self.assertEqual([call.args for call in operations.deallocate.call_args_list],
                             [(sliced,) for sliced in slices])
            self.assertEqual(order, ["slice"] * 3 + ["write"] * 3 + ["free"] * 3)
            operations.copy.assert_not_called()

    def test_adopt_slot_zero_touches_nothing(self):
        # The first user prefills into slot 0: the single-user path is unchanged.
        layer, operations, order, snapshots = self.adoption_fixture(True)
        snapshots.adopt_slot(0)
        self.assertEqual(order, [])
        operations.clone.assert_not_called()

    def test_adopt_slot_refuses_indices_outside_the_eight_slot_batch(self):
        layer, operations, order, snapshots = self.adoption_fixture(True)
        for index in (-1, 8, True, 1.0, None, "1"):
            with self.subTest(index=index), self.assertRaises(ValueError):
                snapshots.adopt_slot(index)
        self.assertEqual(order, [])

    def test_adopt_slot_releases_its_slices_when_a_write_fails(self):
        layer, operations, order, snapshots = self.adoption_fixture(True)
        layer._write_index.side_effect = RuntimeError("write failed")
        with self.assertRaises(RuntimeError):
            snapshots.adopt_slot(2)
        self.assertEqual([call.args[2] for call in layer._slice_along.call_args_list], [2, 2, 2])
        self.assertEqual(operations.deallocate.call_count, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
