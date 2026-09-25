import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

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

    def adoption_fixture(self, direct, *, wrong=None, chips=2):
        """A fake GDN layer whose live tensors are real per-chip torch shards.

        rec_state is (8, 24, 4, 4) per chip on dim 0 and each conv state (1, 8, 16) on
        dim 1, every row distinct, so a slice that returns any other row is caught.
        `wrong` = (tensor name, chip) makes that one device slice hand back row 0 -
        what a broken tile-internal slice would do - on that chip only. `hosts`
        records every shard read back to the host, by tensor name and chip.
        """
        order, slices, hosts, owner = [], [], [], {}

        def shards(name, shape, dimension):
            rows = [torch.full([size if axis != dimension else 1 for axis, size in enumerate(shape)],
                               float(100 * chip + 10 * row + 1), dtype=torch.bfloat16)
                    for chip in range(chips) for row in range(8)]
            value = SimpleNamespace(name=name, chips=[torch.cat(rows[chip * 8:(chip + 1) * 8], dim=dimension)
                                                      for chip in range(chips)])
            for chip, shard in enumerate(value.chips):
                owner[id(shard)] = (name, chip)
            return value

        live = {"rec": shards("rec", (8, 24, 4, 4), 0), "conv0": shards("conv0", (1, 8, 16), 1),
                "conv1": shards("conv1", (1, 8, 16), 1)}

        def slice_along(tensor, dimension, start, stop):
            order.append("slice")
            parts = []
            for chip, shard in enumerate(tensor.chips):
                row = 0 if wrong == (tensor.name, chip) else start
                parts.append(shard.narrow(dimension, row, stop - start).clone())
                owner[id(parts[-1])] = ("slice:" + tensor.name, chip)
            sliced = SimpleNamespace(name="slice:" + tensor.name, chips=parts)
            slices.append(sliced)
            return sliced

        def to_torch(shard):
            hosts.append(owner.get(id(shard), ("?", None)))
            return shard

        def device_tensors(value):
            order.append("read")
            return value.chips

        layer = SimpleNamespace(B=8, _stable_state=True, rec_state=live["rec"], conv_states=[live["conv0"], live["conv1"]],
                                _slice_along=Mock(side_effect=slice_along),
                                _write_recurrent_state_prefix=Mock(side_effect=lambda *args: order.append("write")),
                                _write_index=Mock(side_effect=lambda *args: order.append("write")))
        operations = SimpleNamespace(clone=Mock(side_effect=lambda source, **kwargs: ("clone", source)),
                                     copy=Mock(), deallocate=Mock(side_effect=lambda value: order.append("free")),
                                     get_device_tensors=device_tensors, to_torch=to_torch,
                                     DRAM_MEMORY_CONFIG="DRAM", hosts=hosts)
        return layer, operations, order, slices, ActiveSnapshot(layer, operations, direct=direct)

    CONV_READBACKS = [("conv0", 0), ("slice:conv0", 0), ("conv0", 1), ("slice:conv0", 1),
                      ("conv1", 0), ("slice:conv1", 0), ("conv1", 1), ("slice:conv1", 1)]

    def test_adopt_slot_copies_the_prefill_row_into_slot_zero_over_the_slice_path(self):
        # The batched prefill wrote the user's state into its decode slot; serving builds
        # the helpers direct=True, and the DMA kernel copies slot 0 only, so adoption
        # takes the ttnn slice path whatever the mode: slice row k of every live tensor,
        # read every conv slice and its live tensor back (rec_state by metadata only),
        # then the writes restore() makes, then release the slices.
        for direct in (False, True):
            layer, operations, order, slices, snapshots = self.adoption_fixture(direct)
            with self.subTest(direct=direct), patch("gdn_state_copy.copy_active") as dma:
                self.assertEqual(snapshots.adopt_slot(1, layer=3), 2, "verified on both chips")
            dma.assert_not_called()
            self.assertEqual(operations.hosts, self.CONV_READBACKS, "the conv states and their slices, nothing else")
            self.assertEqual([call.args for call in layer._slice_along.call_args_list],
                             [(layer.rec_state, 0, 1, 2), (layer.conv_states[0], 1, 1, 2), (layer.conv_states[1], 1, 1, 2)])
            self.assertEqual(len(slices), 3)
            layer._write_recurrent_state_prefix.assert_called_once_with(("clone", slices[0]), 1)
            self.assertEqual([call.args for call in layer._write_index.call_args_list],
                             [(layer.conv_states[0], ("clone", slices[1]), 0, 1), (layer.conv_states[1], ("clone", slices[2]), 0, 1)])
            self.assertEqual([call.args for call in operations.deallocate.call_args_list], [(sliced,) for sliced in slices])
            self.assertEqual(order, ["slice"] * 3 + ["read"] * 6 + ["write"] * 3 + ["free"] * 3)
            operations.copy.assert_not_called()

    def test_adopt_slot_verifies_every_row_it_takes(self):
        # Each slot is a distinct value per chip in the fixture, so every row 1..7 must
        # come back as exactly that row, on every chip.
        for index in range(1, 8):
            layer, operations, order, slices, snapshots = self.adoption_fixture(True)
            with self.subTest(index=index):
                self.assertEqual(snapshots.adopt_slot(index), 2)
                for sliced, live, dimension in zip(slices, [layer.rec_state, *layer.conv_states], [0, 1, 1]):
                    for chip in range(2):
                        self.assertTrue(torch.equal(sliced.chips[chip], live.chips[chip].narrow(dimension, index, 1)))

    def test_rec_state_is_checked_by_metadata_and_never_read_back(self):
        # rec_state's page-aligned slice was proven on device (run 35496290854) and the
        # eight-slot tensor is 6.3 MB per chip per layer, so it is no longer read back:
        # a wrong row there is not what this check is for, and it passes without a
        # single rec_state shard reaching the host. The conv proof is unchanged.
        layer, operations, order, slices, snapshots = self.adoption_fixture(True, wrong=("rec", 1))
        self.assertEqual(snapshots.adopt_slot(2, layer=7), 2)
        self.assertEqual(operations.hosts, self.CONV_READBACKS)
        self.assertEqual(order, ["slice"] * 3 + ["read"] * 6 + ["write"] * 3 + ["free"] * 3)

    def test_a_conv_slice_that_returns_the_wrong_row_is_refused_before_any_write(self):
        # A tile-internal conv-state slice at row k != 0 is what no other path exercises.
        # A fake whose slice hands back row 0 instead must fail the readback check, name
        # the layer, chip and difference, and reach no write; the slices are released.
        for name, chip, kind in (("conv1", 1, "Unaligned conv-state"), ("conv0", 0, "Unaligned conv-state")):
            layer, operations, order, slices, snapshots = self.adoption_fixture(True, wrong=(name, chip))
            with self.subTest(name=name, chip=chip), self.assertRaises(ValueError) as refused:
                snapshots.adopt_slot(2, layer=7)
            message = str(refused.exception)
            self.assertIn("%s slice at row 2 differs from the row" % kind, message)
            self.assertIn("layer 7", message)
            self.assertIn("chip %d" % chip, message)
            self.assertIn("max_abs=20", message)
            self.assertEqual(order[:3], ["slice"] * 3)
            self.assertNotIn("write", order, "the refusal must precede any write into slot 0")
            layer._write_recurrent_state_prefix.assert_not_called()
            layer._write_index.assert_not_called()
            self.assertEqual([call.args for call in operations.deallocate.call_args_list], [(sliced,) for sliced in slices])

    def test_a_slice_with_the_wrong_shape_dtype_or_missing_shards_is_refused(self):
        # rec_state is the first tensor checked, by metadata: two rows instead of one,
        # a float32 slice of a bf16 tensor, or one shard against two are each refused
        # before any write, and without reading rec_state back.
        cases = (("shape", lambda shard, dimension, start: shard.narrow(dimension, start, 2),
                  "Recurrent-state slice at row 1 differs from the row: layer [?] rec_state chip 0 shape"),
                 ("dtype", lambda shard, dimension, start: shard.narrow(dimension, start, 1).float(),
                  "rec_state chip 0 shape [(]1, 24, 4, 4[)] torch.float32, row [(]1, 24, 4, 4[)] torch.bfloat16"))
        for label, cut, message in cases:
            layer, operations, order, slices, snapshots = self.adoption_fixture(True)
            layer._slice_along.side_effect = lambda tensor, dimension, start, stop, cut=cut: SimpleNamespace(
                name="slice", chips=[cut(shard, dimension, start) for shard in tensor.chips])
            with self.subTest(label), self.assertRaisesRegex(ValueError, message):
                snapshots.adopt_slot(1)
            layer._write_recurrent_state_prefix.assert_not_called()
            self.assertEqual(operations.hosts, [])
        layer, operations, order, slices, snapshots = self.adoption_fixture(True)
        layer._slice_along.side_effect = lambda tensor, dimension, start, stop: SimpleNamespace(
            name="slice", chips=[tensor.chips[0].narrow(dimension, start, 1)])
        with self.assertRaisesRegex(ValueError, "Recurrent-state slice at row 1 has 1 shards against 2 live shards"):
            snapshots.adopt_slot(1)
        layer._write_recurrent_state_prefix.assert_not_called()
        self.assertEqual(operations.hosts, [])
        # The same geometry checks guard the conv states, through their readback.
        layer, operations, order, slices, snapshots = self.adoption_fixture(True)
        cut = layer._slice_along.side_effect
        layer._slice_along.side_effect = lambda tensor, dimension, start, stop: cut(
            tensor, dimension, start, stop) if dimension == 0 else SimpleNamespace(
            name="slice", chips=[shard.narrow(dimension, start, 2) for shard in tensor.chips])
        with self.assertRaisesRegex(ValueError, "Unaligned conv-state slice at row 1 differs from the row: layer [?] conv_states\\[0\\] chip 0 shape"):
            snapshots.adopt_slot(1)
        layer._write_recurrent_state_prefix.assert_not_called()

    def test_adopt_slot_zero_touches_nothing(self):
        # The first user prefills into slot 0: the single-user path is unchanged.
        layer, operations, order, slices, snapshots = self.adoption_fixture(True)
        self.assertEqual(snapshots.adopt_slot(0), 0)
        self.assertEqual(order, [])
        operations.clone.assert_not_called()

    def test_adopt_slot_refuses_indices_outside_the_eight_slot_batch(self):
        layer, operations, order, slices, snapshots = self.adoption_fixture(True)
        for index in (-1, 8, True, 1.0, None, "1"):
            with self.subTest(index=index), self.assertRaises(ValueError):
                snapshots.adopt_slot(index)
        self.assertEqual(order, [])

    def test_adopt_slot_releases_its_slices_when_a_write_fails(self):
        layer, operations, order, slices, snapshots = self.adoption_fixture(True)
        layer._write_index.side_effect = RuntimeError("write failed")
        with self.assertRaises(RuntimeError):
            snapshots.adopt_slot(2)
        self.assertEqual([call.args[2] for call in layer._slice_along.call_args_list], [2, 2, 2])
        self.assertEqual(operations.deallocate.call_count, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
