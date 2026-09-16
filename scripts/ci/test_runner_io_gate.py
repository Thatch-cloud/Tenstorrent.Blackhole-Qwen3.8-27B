import unittest

from runner_io_gate import evaluate


def pressure(total):
    return f'some avg10=50.00 total={total}\nfull avg10=40.00 total={total}\n'


class RunnerIoGateTests(unittest.TestCase):
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
