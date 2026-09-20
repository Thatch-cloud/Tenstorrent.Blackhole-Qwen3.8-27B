from typing import NamedTuple
import unittest

from packed_shapes import (BLOCK_ROWS, M3_BLOCK_ROWS, M3_ROWS_PER_USER, M3_USERS, PackedShape,
                           commit_traces, m3_shape, segment_rows, validate_shape)


class EngineShape(NamedTuple):
    """Stands in for packed_verifier.PackedShape: same fields, a different class."""
    users: int
    rows_per_user: int
    block_rows: int
    page_width: int
    capacity: int


class M3ShapeTests(unittest.TestCase):
    def test_m3_is_four_t16_users_in_one_sixty_four_row_block(self):
        self.assertEqual((M3_USERS, M3_ROWS_PER_USER, M3_BLOCK_ROWS), (4, 16, 64))
        shape = m3_shape(1024)
        self.assertEqual(shape, PackedShape(4, 16, 64, 1024, 65536))
        self.assertEqual([segment_rows(shape, user) for user in range(4)],
                         [(0, 16), (16, 32), (32, 48), (48, 64)])
        self.assertEqual(commit_traces(shape), 64, 'sixteen nonzero prefixes per user')
        for segment in (4, -1, True, None):
            with self.assertRaises(ValueError):
                segment_rows(shape, segment)

    def test_the_narrower_blocks_still_validate_and_sixty_four_is_the_widest(self):
        self.assertEqual(BLOCK_ROWS, (2, 4, 8, 16, 32, 64))
        for shape in (PackedShape(2, 16, 32, 68, 68 * 64), PackedShape(4, 8, 32, 512, 512 * 64),
                      PackedShape(2, 1, 2, 68, 68 * 64), PackedShape(2, 32, 64, 68, 68 * 64)):
            self.assertIs(validate_shape(shape), shape)
        self.assertEqual(commit_traces(PackedShape(2, 16, 32, 68, 68 * 64)), 32)

    def test_the_engines_own_shape_class_passes_by_its_fields(self):
        engine_shape = EngineShape(4, 16, 64, 1024, 65536)
        self.assertIs(validate_shape(engine_shape), engine_shape)
        self.assertEqual(segment_rows(engine_shape, 3), (48, 64))
        from packed_verifier import PackedShape as BlockShape
        self.assertIs(validate_shape(BlockShape(4, 16, 64, 1024, 65536)).block_rows, 64)

    def test_broken_shapes_are_refused(self):
        for broken in (PackedShape(4, 16, 64, 1024, 65536 + 1), PackedShape(3, 16, 64, 1024, 65536),
                       PackedShape(4, 16, 48, 1024, 48 * 64), PackedShape(8, 16, 128, 1024, 65536),
                       PackedShape(4, 12, 48, 1024, 65536), PackedShape(4, 16, 64, 67, 67 * 64),
                       PackedShape(4, 16, 64, True, 64), PackedShape(0, 16, 64, 1024, 65536),
                       (4, 16, 64, 1024, 65536), None):
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                validate_shape(broken)
        for page_width in (67, 0, '1024', 1024.0):
            with self.subTest(page_width=page_width), self.assertRaises(ValueError):
                m3_shape(page_width)


if __name__ == '__main__':
    unittest.main()
