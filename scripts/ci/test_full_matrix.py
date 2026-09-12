from itertools import product
import unittest

from full_matrix import validate_static_matrix


class FullMatrixTests(unittest.TestCase):
    def fixture(self, maximum):
        lengths = (4095, 16383)
        widths = (1, 2, 4, 8, 16, 32) if maximum == 32 else (1, 2, 4, 8, 16)
        return dict(rows=maximum,
            mode_checks=[dict(length=length, trace=trace, logits_exact=True) for length, trace in product(lengths, (False, True))],
            batch_checks=[dict(length=length, rows=rows, trace=trace, logits_exact=True, all_gdn_states_exact=True, valid_kv_exact=True)
                          for length, rows, trace in product(lengths, widths, (False, True))],
            checks=[dict(length=length, prefix=prefix, trace=trace, logits_exact=True, all_gdn_states_exact=True, valid_kv_exact=True, correction_steps=2)
                    for length, prefix, trace in product(lengths, (0, 1, maximum // 2, maximum), (False, True))],
            negative_controls=[dict(length=length, trace=trace, stale_gdn_detected=True, wrong_page_detected=True)
                               for length, trace in product(lengths, (False, True))],
            timings=[dict(length=length, rows=rows, exact=True) for length, rows in product(lengths, widths)])

    def test_old_and_wider_matrices_are_complete(self):
        for maximum in (16, 32):
            validate_static_matrix(self.fixture(maximum), maximum)

    def test_count_alone_cannot_hide_a_missing_or_duplicate_fixture(self):
        for key in ('mode_checks', 'batch_checks', 'checks', 'negative_controls', 'timings'):
            for invalid in ('missing', 'duplicate'):
                report = self.fixture(32)
                report[key].pop()
                if invalid == 'duplicate':
                    report[key].append(report[key][0])
                with self.assertRaisesRegex(AssertionError, key):
                    validate_static_matrix(report, 32)

    def test_exactness_and_correction_guards_are_required(self):
        for key, field in (('batch_checks', 'valid_kv_exact'), ('checks', 'all_gdn_states_exact'),
                           ('checks', 'correction_steps'), ('negative_controls', 'wrong_page_detected'), ('timings', 'exact')):
            report = self.fixture(32)
            report[key][-1][field] = False
            with self.assertRaises(AssertionError):
                validate_static_matrix(report, 32)
