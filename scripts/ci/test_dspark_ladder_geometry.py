import unittest

from dspark_ladder_geometry import geometry, ladder


class LadderGeometryTests(unittest.TestCase):
    def test_long_context_chunk_candidate_preserves_exact_headroom(self):
        for context, capacity, storage, native in (
                (32768, 33792, 33856, 34304),
                (65536, 66560, 66624, 67072)):
            with self.subTest(context=context):
                row = geometry(context)
                self.assertEqual((row['capacity'], row['storage_keys'], row['native_keys']),
                    (capacity, storage, native))
                self.assertEqual(row['key_chunk'], 512)
                self.assertEqual(row['extra_masked_keys'], 448)
                self.assertEqual(row['probe_positions'], (context, capacity - 15))

    def test_preserves_existing_eight_k_comparison_geometry(self):
        value = geometry(8192, 256)
        self.assertEqual((value['capacity'], value['storage_keys'], value['native_keys']),
            (8448, 8512, 8704))
        self.assertEqual(value['probe_positions'], (8192, 8433))

    def test_every_ladder_row_retains_full_history_and_output_headroom(self):
        rows = ladder()
        self.assertEqual(tuple(row['context'] for row in rows), (128, 4096, 8192, 32768, 65536))
        for row in rows:
            with self.subTest(context=row['context']):
                self.assertGreaterEqual(row['capacity'], row['context'] + 1024)
                self.assertGreaterEqual(row['storage_keys'], row['capacity'] + 15)
                self.assertGreaterEqual(row['native_keys'], row['storage_keys'])
                self.assertEqual(row['native_keys'] % 256, 0)
                self.assertFalse(row['runtime_admitted'])
                self.assertFalse(row['numerical_qualified'])
                self.assertFalse(row['performance_qualified'])

    def test_implicit_or_unsupported_geometry_rejected(self):
        for context, output in ((True, 1024), (8192, True), (16384, 1024), (8192, 0), (8192, 257)):
            with self.subTest(context=context, output=output), self.assertRaises(ValueError):
                geometry(context, output)
