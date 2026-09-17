import unittest

import torch

from feature_rows import check_published_features, compare_feature_rows


class FeatureRowsTests(unittest.TestCase):
    def test_publication_checks_both_epochs_and_empty_abort(self):
        serial, captured = self.fixture(2)
        first = [[value[:, :, :1].clone() for value in parts] for parts in captured]
        second = [[value[:, :, 1:].clone() for value in parts] for parts in captured]
        published = [('abort', 0, 0, ()), ('first', 0, 1, first), ('second', 1, 1, second)]
        checks = check_published_features(serial, published, (5, 19), lambda value: value, stage='after-replay')
        self.assertEqual([len(check['checks']) for check in checks], [0, 4, 4])
        self.assertTrue(all(check['stage'] == 'after-replay' for check in checks))
        first[1][0].add_(1)
        with self.assertRaisesRegex(AssertionError, "after-correction.*phase='first'.*layer 19, chip 0"):
            check_published_features(serial, published, (5, 19), lambda value: value, stage='after-correction')

    def test_publication_rejects_missing_and_abort_taps(self):
        serial, captured = self.fixture(2)
        for published in ([('first', 0, 1, captured[:1])], [('abort', 0, 0, captured)]):
            with self.assertRaisesRegex(AssertionError, 'Abort or committed feature tap count'):
                check_published_features(serial, published, (5, 19), lambda value: value, stage='immediate')

    def fixture(self, rows):
        serial = [[[torch.full((1, 1, 1, 2560), row + layer + chip, dtype=torch.bfloat16)
                    for chip in (0, 1)] for layer in (5, 19)] for row in range(rows)]
        captured = [[torch.cat([step[index][chip] for step in serial], dim=2)
                     for chip in (0, 1)] for index in range(2)]
        return serial, captured

    def test_exact_and_shifted_feature_rows(self):
        for rows in (2, 16, 32):
            serial, captured = self.fixture(rows)
            self.assertEqual(len(compare_feature_rows(serial, captured, (5, 19))), 4)
            captured[1][1] = captured[1][1].roll(1, dims=2)
            with self.assertRaisesRegex(AssertionError, 'Stale or misaligned'):
                compare_feature_rows(serial, captured, (5, 19))

    def test_stale_features_missing_shards_and_wrong_axis_fail(self):
        serial, captured = self.fixture(16)
        changed = [[[value + 1 for value in parts] for parts in step] for step in serial]
        with self.assertRaises(AssertionError):
            compare_feature_rows(changed, captured, (5, 19))
        with self.assertRaises(ValueError):
            compare_feature_rows(serial, [captured[0][:1], captured[1]], (5, 19))
        captured[0][0] = captured[0][0].transpose(1, 2)
        with self.assertRaises(ValueError):
            compare_feature_rows(serial, captured, (5, 19))
