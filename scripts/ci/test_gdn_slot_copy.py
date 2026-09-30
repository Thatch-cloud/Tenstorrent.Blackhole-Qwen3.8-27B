import unittest

from gdn_slot_copy import page_segments, transfer_plan


COMPACT = [(1, 24, 128, 128)] + [(1, 1, 5120)] * 4
FULL = [(8, 24, 128, 128)] + [(1, 8, 5120)] * 4


class SlotCopyTests(unittest.TestCase):
    def test_slot_direction_and_geometry(self):
        for slot in range(8):
            self.assertEqual(transfer_plan(FULL, COMPACT, slot),
                [(count, slot, 0) for count in (384, 160, 160, 160, 160)])
            self.assertEqual(transfer_plan(COMPACT, FULL, slot),
                [(count, 0, slot) for count in (384, 160, 160, 160, 160)])

    def test_invalid_geometry_and_slots(self):
        for slot in (-1, 8, True, 0.0):
            with self.assertRaises(ValueError):
                transfer_plan(FULL, COMPACT, slot)
        for source, destination in ((FULL, FULL), (COMPACT, COMPACT), (FULL[:4], COMPACT)):
            with self.assertRaises(ValueError):
                transfer_plan(source, destination, 1)

    def test_all_slots_touch_only_selected_recurrent_pages_and_conv_rows(self):
        for slot in range(8):
            for saving in (True, False):
                source_slot, destination_slot = (slot, 0) if saving else (0, slot)
                for operand in range(5):
                    for page in range(384 if operand == 0 else 160):
                        segments = page_segments(operand, page, source_slot, destination_slot)
                        for source_page, source_offset, destination_page, destination_offset, scratch, length in segments:
                            self.assertEqual(source_offset % 64, scratch % 64)
                            self.assertEqual(scratch % 16, destination_offset % 16)
                            self.assertLessEqual(scratch + length, 2048)
                            if operand == 0:
                                self.assertEqual(source_page // 384, source_slot)
                                self.assertEqual(destination_page // 384, destination_slot)
                                self.assertEqual(length, 2048)
                            else:
                                self.assertEqual((source_page, destination_page), (page, page))
                                self.assertEqual((source_offset % 512) // 32, source_slot)
                                self.assertEqual((destination_offset % 512) // 32, destination_slot)
                                self.assertEqual(length, 32)

    def test_interleaved_restore_preserves_other_users_and_padding(self):
        pages = [bytearray([201] * 2048) for _ in range(160)]
        for slot, sentinel in ((0, 17), (1, 33), (0, 49), (7, 65)):
            before = [bytes(page) for page in pages]
            for page in range(160):
                writes = set()
                for _, _, destination_page, offset, _, length in page_segments(1, page, 0, slot):
                    pages[destination_page][offset:offset + length] = bytes([sentinel]) * length
                    writes.update(range(offset, offset + length))
                self.assertTrue(all(value == before[page][index]
                    for index, value in enumerate(pages[page]) if index not in writes))
                self.assertTrue(all(pages[page][index] == sentinel for index in writes))
