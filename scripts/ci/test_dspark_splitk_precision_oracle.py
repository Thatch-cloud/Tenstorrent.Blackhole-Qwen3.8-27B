import unittest

from dspark_splitk_precision_oracle import compare_score_precision


class ScorePrecisionTests(unittest.TestCase):
    def test_bf16_score_storage_alone_breaks_retained_tolerance(self):
        reports = compare_score_precision()
        self.assertEqual(reports[0]['failed_elements'], 0)
        for report in reports[1:3]:
            self.assertGreater(report['failed_elements'], 7000)
            self.assertGreater(report['max_abs'], 1)
            self.assertFalse(report['device_qualified'])
        self.assertEqual(reports[3]['failed_elements'], 0)
        self.assertLess(reports[3]['max_abs'], .13)


if __name__ == '__main__':
    unittest.main()
