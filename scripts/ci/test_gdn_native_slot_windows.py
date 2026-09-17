import unittest

from gdn_native_slot_windows import validate_geometry


class NativeWindowGeometryTests(unittest.TestCase):
    def test_both_projection_widths_accept_full_native_history(self):
        for width in (8240, 8256):
            validate_geometry((1, 16, width), [(1, 8, 5120)] * 4)

    def test_compact_or_incomplete_history_rejected(self):
        for history in ([(1, 1, 5120)] * 4, [(1, 8, 5120)] * 3, [(1, 8, 5121)] * 4):
            with self.assertRaises(ValueError):
                validate_geometry((1, 16, 8256), history)

    def test_non_t16_projection_rejected(self):
        for shape in ((1, 8, 8256), (2, 16, 8256), (1, 16, 5120)):
            with self.assertRaises(ValueError):
                validate_geometry(shape, [(1, 8, 5120)] * 4)
