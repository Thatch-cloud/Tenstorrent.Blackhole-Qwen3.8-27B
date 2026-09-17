from pathlib import Path
import tempfile
import unittest

from request_host_health import snapshot, summarize


class HostHealthTests(unittest.TestCase):
    def test_reads_available_files_and_preserves_missing_as_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'cpu.stat').write_text('usage_usec 100\nnr_throttled 2\n')
            result = snapshot(directory)
        self.assertEqual(result['cgroup']['cpu.stat'], 'usage_usec 100\nnr_throttled 2')
        self.assertEqual(result['unavailable']['memory.events'], 'FileNotFoundError')

    def test_deltas_not_absolute_counters(self):
        before = dict(monotonic_ns=1, cgroup={'cpu.stat': 'nr_throttled 10\nthrottled_usec 100'})
        after = dict(monotonic_ns=1000001, cgroup={'cpu.stat': 'nr_throttled 12\nthrottled_usec 300'})
        result = summarize(before, after)
        self.assertEqual(result['counter_deltas'], {'cpu.stat': {'nr_throttled': 2, 'throttled_usec': 200}})
        self.assertEqual(result['elapsed_ms'], 1)

    def test_missing_counters_are_not_zero_pressure(self):
        result = summarize(dict(monotonic_ns=1, cgroup={}), dict(monotonic_ns=2, cgroup={}))
        self.assertEqual(result['counter_deltas'], {})

    def test_counter_reset_or_reversed_time_rejected(self):
        before = dict(monotonic_ns=2, cgroup={'cpu.stat': 'usage_usec 9'})
        for after in (dict(monotonic_ns=1, cgroup={}),
                dict(monotonic_ns=3, cgroup={'cpu.stat': 'usage_usec 8'})):
            with self.assertRaises(ValueError):
                summarize(before, after)
