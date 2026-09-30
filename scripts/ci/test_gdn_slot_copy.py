import unittest
from unittest.mock import patch

import gdn_slot_copy


class SlotCopyTests(unittest.TestCase):
    compact = [(1, 24, 128, 128)] + [(1, 1, 5120)] * 4
    full = [(8, 24, 128, 128)] + [(1, 8, 5120)] * 4

    def test_all_slots_have_disjoint_recurrence_pages_and_conv_rows(self):
        recurrent, convolution = set(), set()
        for slot in range(8):
            load = gdn_slot_copy.transfer_plan(self.full, self.compact, slot)
            save = gdn_slot_copy.transfer_plan(self.compact, self.full, slot)
            self.assertEqual(load, [(384, slot * 384, 0)] + [(160, slot * 32, 0)] * 4)
            self.assertEqual(save, [(count, target, source) for count, source, target in load])
            pages = set(range(load[0][1], load[0][1] + load[0][0]))
            faces = set(range(slot * 32, slot * 32 + 32)) | set(range(512 + slot * 32, 544 + slot * 32))
            self.assertFalse(recurrent & pages)
            self.assertFalse(convolution & faces)
            recurrent.update(pages)
            convolution.update(faces)
        self.assertEqual(len(recurrent), 8 * 384)
        self.assertEqual(len(convolution), 8 * 64)

    def test_zero_slot_retains_original_transfer_geometry(self):
        self.assertEqual(gdn_slot_copy.transfer_plan(self.full, self.compact, 0),
            [(384, 0, 0)] + [(160, 0, 0)] * 4)

    def test_invalid_slots_and_unqualified_shapes_rejected(self):
        for slot in (-1, 8, True, 1.0, None):
            with self.subTest(slot=slot), self.assertRaises(ValueError):
                gdn_slot_copy.transfer_plan(self.full, self.compact, slot)
        for source, target in ((self.compact, self.compact), (self.full, self.full),
                ([(4, 24, 128, 128), *self.full[1:]], self.compact)):
            with self.assertRaises(ValueError):
                gdn_slot_copy.transfer_plan(source, target, 1)

    def test_hardware_and_serving_rejected_before_device_work(self):
        for environment in ({}, {'QWEN_SIM_ONLY': '1', 'QWEN_HARDWARE_TESTS': '1'},
                {'QWEN_SIM_ONLY': '1', 'QWEN_CARDS_ALLOCATED': '1'}):
            with patch.dict('os.environ', environment, clear=True), patch.object(
                    gdn_slot_copy, '_copy_state') as copy, self.assertRaises(ValueError):
                gdn_slot_copy.copy_slot([], [], 1)
            copy.assert_not_called()

    def test_simulator_dispatch_passes_explicit_slot(self):
        with patch.dict('os.environ', {'QWEN_SIM_ONLY': '1'}, clear=True), patch.object(
                gdn_slot_copy, '_copy_state') as copy:
            gdn_slot_copy.copy_slot(self.full, self.compact, 7)
        copy.assert_called_once_with(self.full, self.compact, 7)


if __name__ == '__main__':
    unittest.main()
