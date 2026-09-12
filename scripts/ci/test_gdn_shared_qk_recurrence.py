"""Host operand routing checks; complete recurrence still needs simulator admission."""

import unittest

import gdn_shared_qk_recurrence as candidate
from gdn_vsplit import stage_spec


class RecurrenceTests(unittest.TestCase):
    def test_only_query_and_key_accessor_bindings_change(self):
        native = stage_spec('recurrence', 16)
        transformed = candidate.recurrence_spec(native)
        self.assertEqual(transformed['reader_addresses'], [9, 10, 0, 1, 2, 3])
        self.assertEqual(transformed['reader_accessors'], [9, 10, 0, 1, 2, 3, 3, 3])
        for key in native.keys() - {'reader_addresses', 'reader_accessors'}:
            self.assertEqual(transformed[key], native[key])
        self.assertEqual(native['reader_addresses'], [0, 0, 0, 1, 2, 3])

    def test_wrong_operand_map_fails(self):
        native = stage_spec('recurrence', 16)
        for key, value in (('workers', 24), ('reader_addresses', [0]), ('reader_accessors', [])):
            with self.assertRaises(ValueError):
                candidate.recurrence_spec(dict(native, **{key: value}))

    def test_fp32_cache_and_rows_preserve_both_faces(self):
        cache = list(range(4096))
        for token in range(16):
            target = [0] * 4096
            offset = 512 * (token // 16) + 16 * (token % 16)
            for tile in range(4):
                for word in range(16):
                    target[tile * 1024 + word] = cache[tile * 1024 + offset + word]
                    target[tile * 1024 + 256 + word] = cache[tile * 1024 + offset + 256 + word]
                self.assertEqual(target[tile * 1024:tile * 1024 + 16],
                                 cache[tile * 1024 + offset:tile * 1024 + offset + 16])
                self.assertEqual(target[tile * 1024 + 16:tile * 1024 + 256], [0] * 240)


if __name__ == '__main__':
    unittest.main()
