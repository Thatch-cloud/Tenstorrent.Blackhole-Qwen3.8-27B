import unittest
from unittest.mock import Mock

from runner_io_gate import evaluate, observe


def pressure(total):
    return f'some avg10=50.00 total={total}\nfull avg10=40.00 total={total}\n'


class RunnerIoGateTests(unittest.TestCase):
    def test_bounded_wait_preserves_threshold_and_records_rejected_samples(self):
        readings = Mock(side_effect=[pressure(value) for value in (0, 9_000_000, 9_000_000, 9_001_000)])
        sleep, emit = Mock(), Mock()
        report = observe(4, readings, sleep, Mock(side_effect=(0, 15, 30, 45)), emit)
        self.assertTrue(report['passed'])
        self.assertEqual([value['passed'] for value in report['observations']], [False, True])
        self.assertEqual(report['maximum_fraction'], 0.01)
        self.assertEqual(sleep.call_count, 3)
        self.assertEqual(emit.call_count, 2)

    def test_busy_host_exhausts_bound_without_admission(self):
        readings = Mock(side_effect=[pressure(value * 9_000_000) for value in range(8)])
        sleep = Mock()
        report = observe(4, readings, sleep, Mock(side_effect=range(0, 120, 15)), Mock())
        self.assertFalse(report['passed'])
        self.assertEqual(len(report['observations']), 4)
        self.assertEqual(sleep.call_count * 15, 105)
        for attempts in (0, 5, True):
            with self.assertRaises(ValueError):
                observe(attempts, Mock(), Mock(), Mock(), Mock())

    def test_uses_interval_counter_not_stale_average(self):
        self.assertTrue(evaluate(pressure(1000), pressure(1100), 15)['passed'])
        self.assertFalse(evaluate(pressure(1000), pressure(5_001_000), 15)['passed'])

    def test_rejects_missing_reset_or_invalid_time(self):
        for before, after, duration in (
                ('some total=1', pressure(2), 15),
                (pressure(10), pressure(9), 15),
                (pressure(1), pressure(2), 0),
                (pressure(1), pressure(2), float('nan'))):
            with self.assertRaises(ValueError):
                evaluate(before, after, duration)


if __name__ == '__main__':
    unittest.main()
