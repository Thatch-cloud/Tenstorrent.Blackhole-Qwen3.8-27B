import unittest

from gdn_multitoken_timing import measure


class TimingTests(unittest.TestCase):
    def test_abba_order_counts_and_fences(self):
        events = []
        ticks = iter(range(0, 24_000_000, 1_000_000))
        result = measure(lambda arm: events.append(('replay', arm)),
                         lambda arm: events.append(('validate', arm)),
                         lambda: events.append(('sync', None)), clock=lambda: next(ticks))
        self.assertEqual(result['timed_replays'], 120)
        self.assertEqual(len([event for event in events if event[0] == 'replay']), 122)
        self.assertEqual(len([event for event in events if event[0] == 'validate']), 14)
        self.assertEqual([sample['arm'] for sample in result['samples']], ['serial', 'multitoken', 'multitoken', 'serial'] * 3)
        self.assertEqual(result['serial_median_ms'], 0.1)
        self.assertEqual(result['median_paired_speedup'], 1)
        for index, event in enumerate(events):
            if event[0] == 'validate':
                self.assertEqual(events[index - 1][0], 'sync')

    def test_failure_in_post_sample_validation_propagates(self):
        validations = []

        def validate(arm):
            validations.append(arm)
            if len(validations) == 3:
                raise AssertionError('State changed')

        ticks = iter((0, 1_000_000))
        with self.assertRaisesRegex(AssertionError, 'State changed'):
            measure(lambda arm: None, validate, lambda: None, clock=lambda: next(ticks))

    def test_invalid_counts_and_clock_fail(self):
        for counts in ({'repeats': 0}, {'blocks': 0}):
            with self.assertRaises(ValueError):
                measure(lambda arm: None, lambda arm: None, lambda: None, **counts)
        with self.assertRaises(ValueError):
            measure(lambda arm: None, lambda arm: None, lambda: None, clock=lambda: 0)


if __name__ == '__main__':
    unittest.main()
