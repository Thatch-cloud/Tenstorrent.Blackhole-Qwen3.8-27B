import unittest

from dspark_splitk_row_diagnostic import transform


class RowDiagnosticTests(unittest.TestCase):
    def fixture(self):
        return '\n'.join(
            'TSLICE(' + buffer + ', 0,\n'
            '    (SliceRange{.h0=0, .h1=4, .hs=1, .w0=0, .w1=1, .ws=1}), true, true);'
            for buffer in ('cb_prev_sum', 'cb_out_accumulate_im', 'cb_prev_sum',
                'cb_out_accumulate_im', 'cb_out_accumulate_im'))

    def test_selects_row_and_value_column_without_changing_compute(self):
        source = 'compute_before();\n' + self.fixture() + '\ncompute_after();'
        result = transform(source)
        self.assertEqual(result.count('.h0=23, .h1=24'), 5)
        self.assertEqual(result.count('.w0=8, .w1=9'), 3)
        self.assertEqual(result.count('.w0=0, .w1=1'), 2)
        self.assertTrue(result.startswith('compute_before();'))
        self.assertTrue(result.endswith('compute_after();'))
        self.assertEqual(result.count('TSLICE'), 5)

    def test_rejects_missing_or_already_changed_snapshots(self):
        for source in ('', self.fixture().replace('cb_prev_sum', 'other'), transform(self.fixture())):
            with self.assertRaises(ValueError):
                transform(source)


if __name__ == '__main__':
    unittest.main()
