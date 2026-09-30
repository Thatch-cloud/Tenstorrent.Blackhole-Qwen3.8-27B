from pathlib import Path
import unittest

from mlp_clock_samples import MAGIC, SCRATCH, ZONES, decode, instrument, remove, wrapper


class ClockSampleTests(unittest.TestCase):
    def test_reader_operations_round_trip_without_global_profiler(self):
        for role in ZONES:
            source = Path(__file__).with_name(f'fused_1d_{role}.cpp').read_text()
            candidate = instrument(source, role)
            self.assertEqual(remove(candidate, role), source)
            self.assertNotIn('DeviceZone', candidate)
            self.assertIn(f'get_write_ptr({SCRATCH[role]})', candidate)
            with self.assertRaises(ValueError):
                instrument(candidate, role)
            for index, (statement, name, predicate) in enumerate(ZONES[role]):
                self.assertEqual(wrapper(statement, predicate, index).count(statement), 2)

    def test_clock_low_word_rollover_and_missing_markers(self):
        words = [0xffffffff] * 32
        for index in (0, 1, 2, 3):
            words[index * 6:index * 6 + 6] = [0xfffffff0, 3 + index, 0x10, 4 + index, index, MAGIC ^ index]
        records = decode(words, 'weights', 0)
        self.assertEqual([entry['duration_cycles'] for entry in records], [32] * 4)
        words[5] = 0xffffffff
        with self.assertRaises(ValueError):
            decode(words, 'weights', 0)

    def test_invalid_pages_and_unbounded_intervals_rejected(self):
        for values, role, worker in (([], 'input', 0), ([0] * 32, 'input', 2), ([0] * 32, 'compute', 0)):
            with self.assertRaises(ValueError):
                decode(values, role, worker)
        values = [0] * 32
        values[:6] = [10, 1, 9, 1, 0, MAGIC]
        with self.assertRaises(ValueError):
            decode(values, 'weights', 0)

    def test_out_of_order_samples_rejected(self):
        values = [0xffffffff] * 32
        for index in range(4):
            values[index * 6:index * 6 + 6] = [100, 0, 110, 0, index, MAGIC ^ index]
        with self.assertRaisesRegex(ValueError, 'execution order'):
            decode(values, 'weights', 0)

    def test_scratch_is_distinct_and_does_not_alias_projection(self):
        self.assertEqual(len(set(SCRATCH.values())), 2)
        self.assertFalse(set(SCRATCH.values()) & {0, 1, 4, 5, 10, 30})
        self.assertTrue(all(0 <= index < 32 for index in SCRATCH.values()))
