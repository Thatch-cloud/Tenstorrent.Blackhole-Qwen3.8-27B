import unittest

from winning_interval_attribution import coverage


def row(start, end, name='A'):
    return {'OP NAME': name, 'CORE COUNT': '1', 'DEVICE KERNEL START CYCLE': str(start),
        'DEVICE KERNEL END CYCLE': str(end), 'DEVICE KERNEL DURATION [ns]': str(end - start)}


class IntervalAttributionTests(unittest.TestCase):
    def test_same_group_overlap_is_not_double_counted(self):
        result = coverage([row(0, 10), row(5, 15)])
        self.assertAlmostEqual(result['groups'][('A', '1')]['inclusive_ms'], 15e-6)
        self.assertAlmostEqual(result['groups'][('A', '1')]['exclusive_ms'], 15e-6)
        self.assertEqual(result['overlapping_groups_ms'], 0)

    def test_cross_group_overlap_and_gaps_are_separate(self):
        result = coverage([row(0, 10), row(5, 15, 'B'), row(20, 25)])
        self.assertAlmostEqual(result['groups'][('A', '1')]['exclusive_ms'], 10e-6)
        self.assertAlmostEqual(result['groups'][('B', '1')]['exclusive_ms'], 5e-6)
        self.assertAlmostEqual(result['overlapping_groups_ms'], 5e-6)

    def test_invalid_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            coverage([row(10, 5)])
