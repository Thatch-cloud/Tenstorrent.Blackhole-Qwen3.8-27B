import unittest

from dspark_splitk_chunk_oracle import compare_chunk_statistics


class ChunkOracleTests(unittest.TestCase):
    def test_partitioned_reference_handles_empty_masked_chunks(self):
        reports = compare_chunk_statistics()
        self.assertEqual(len(reports), 4)
        for report in reports:
            self.assertFalse(report['device_qualified'])
            self.assertEqual(report['failed_elements'], 0)
            self.assertLess(report['max_abs'], .35)
        self.assertGreater(reports[3]['max_abs'], reports[1]['max_abs'])


if __name__ == '__main__':
    unittest.main()
