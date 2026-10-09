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

    def test_the_per_request_captures_keep_their_full_width_except_beside_the_four_user_block(self):
        from packed_shapes import m1_shape, m3_shape, sequential_capture_rows

        # no block, and the M1 block: the engines' full T16 set, as the two-user gate ran
        self.assertEqual(sequential_capture_rows(None), 16)
        self.assertEqual(sequential_capture_rows(m1_shape(68)), 16)
        # beside the 64-row block only the sequential widths (1, 2, 4) fit: run 35509307389
        self.assertEqual(sequential_capture_rows(m3_shape(68)), 4)
        self.assertEqual(sequential_capture_rows(m3_shape(1024)), 4)
        with self.assertRaises(ValueError):
            sequential_capture_rows(PackedShape(4, 16, 64, 68, 1))

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


class OctoShapeTests(unittest.TestCase):
    """Octo-T8 (QWEN_FAST_OCTO): eight users of eight rows in the SAME 64-row block geometry, never a serving count."""

    def test_octo_is_eight_t8_users_in_one_sixty_four_row_block(self):
        from packed_shapes import OCTO_BLOCK_ROWS, OCTO_ROWS_PER_USER, OCTO_USERS, octo_shape

        self.assertEqual((OCTO_USERS, OCTO_ROWS_PER_USER, OCTO_BLOCK_ROWS), (8, 8, 64))
        shape = octo_shape(1024)
        self.assertEqual(shape, PackedShape(8, 8, 64, 1024, 65536))
        self.assertEqual(validate_shape(shape), shape)
        self.assertEqual([segment_rows(shape, user) for user in range(8)],
                         [(0, 8), (8, 16), (16, 24), (24, 32), (32, 40), (40, 48), (48, 56), (56, 64)])
        self.assertEqual(commit_traces(shape), 64, 'eight nonzero prefixes per user')
        self.assertEqual(shape.block_rows, m3_shape(1024).block_rows, 'the rows per pass are M3 : the same matmul, norm and sampler programs')
        for segment in (8, -1, True, None):
            with self.subTest(segment=segment), self.assertRaises(ValueError):
                segment_rows(shape, segment)
        for page_width in (67, 0, '1024', 1024.0):
            with self.subTest(page_width=page_width), self.assertRaises(ValueError):
                octo_shape(page_width)

    def test_octo_is_not_a_serving_count_and_changes_no_other_shape(self):
        import packed_shapes
        from packed_shapes import m1_shape, sequential_capture_rows

        self.assertEqual(sorted(packed_shapes.SERVING_SHAPES), [2, 4])
        self.assertIsNone(packed_shapes.serving_shape(8, 68), 'eight requests build no block of their own from the count')
        self.assertEqual(packed_shapes.serving_shape(4, 68), m3_shape(68))
        self.assertEqual(packed_shapes.serving_shape(2, 68), m1_shape(68))
        # the per-request engines beside a 64-row block keep the sequential widths, whatever its geometry
        self.assertEqual(sequential_capture_rows(packed_shapes.octo_shape(68)), 4)

    def test_the_engines_own_shape_class_passes(self):
        from packed_shapes import octo_shape
        from packed_verifier import PackedShape as BlockShape

        self.assertIs(validate_shape(BlockShape(8, 8, 64, 1024, 65536)).rows_per_user, 8)
        self.assertEqual(tuple(octo_shape(1024)), tuple(BlockShape(8, 8, 64, 1024, 65536)))


if __name__ == '__main__':
    unittest.main()
