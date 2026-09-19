import unittest

from winning_risc_attribution import summarize


class RiscAttributionTests(unittest.TestCase):
    def fixture(self):
        devices = [dict(trace_id=3, device='0', replays=[dict(replay_session=index, operations=1,
            rows=16, first_replay=index == 0) for index in range(3)])]
        rows = [dict({'METAL TRACE ID': '3', 'METAL TRACE REPLAY SESSION ID': str(index),
            'DEVICE ID': '0', 'OP NAME': 'generic', 'CORE COUNT': '99',
            'DEVICE BRISC KERNEL DURATION [ns]': str(value)}) for index, value in enumerate((900, 100, 200))]
        return devices, rows

    def test_first_replay_excluded_and_missing_measurements_not_zero(self):
        devices, rows = self.fixture()
        result = summarize(devices, rows)
        self.assertEqual(result[0]['replays'], 2)
        self.assertEqual(result[0]['riscs']['BRISC']['median_summed_ms'], .00015)
        self.assertIsNone(result[0]['riscs']['TRISC0']['median_summed_ms'])

    def test_incomplete_operations_and_invalid_durations_rejected(self):
        devices, rows = self.fixture()
        with self.assertRaises(ValueError):
            summarize(devices, rows[:-1])
        rows[-1]['DEVICE BRISC KERNEL DURATION [ns]'] = 'nan'
        with self.assertRaises(ValueError):
            summarize(devices, rows)

    def test_changed_risc_coverage_rejected(self):
        devices, rows = self.fixture()
        rows[-1]['DEVICE BRISC KERNEL DURATION [ns]'] = ''
        with self.assertRaises(ValueError):
            summarize(devices, rows)
