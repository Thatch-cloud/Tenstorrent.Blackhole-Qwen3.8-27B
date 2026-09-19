from pathlib import Path
import unittest

from gdn_wait_clock import MAGIC, TAIL, WORDS, ZONES, decode, instrument, remove


class WaitClockTests(unittest.TestCase):
    def source(self):
        return ('#include <cstdint>\nvoid kernel_main() {\n'
            + '\n'.join(anchor for _, anchor in ZONES)
            + '\nif (it + 1 < n_inst) { copy_tiles(cb_snew, 30, kv); }\n'
            + '\n}\n')

    def page(self, processor=0, token=8):
        words = [0xffffffff] * WORDS
        for index in range(len(ZONES)):
            start = (1 << 32) - 5 + 100 * index
            end = start + 20
            words[index * 6:index * 6 + 6] = [start & 0xffffffff, start >> 32,
                end & 0xffffffff, end >> 32, index, MAGIC ^ index ^ (processor << 16) ^ (token << 8)]
        return words

    def test_source_roundtrip_and_each_phase_once(self):
        source = self.source()
        result = instrument(source)
        self.assertEqual(remove(result), source)
        for index in range(len(ZONES)):
            self.assertEqual(result.count(f'qwen_samples[{index * 6 + 5}] ='), 1)
        self.assertIn('get_arg_val<uint32_t>(3)', result)
        with self.assertRaises(ValueError):
            instrument(result)

    def test_missing_duplicate_reordered_or_changed_feedback_rejected(self):
        source = self.source()
        first, second = ZONES[0][1], ZONES[1][1]
        for changed in (source.replace(first, ''), source + first,
                source.replace(first, 'TEMP').replace(second, first).replace('TEMP', second),
                source.replace('copy_tiles(cb_snew, 30, kv)', 'copy_tiles(cb_snew, 31, kv)')):
            with self.assertRaises(ValueError):
                instrument(changed)

    def test_all_processors_first_middle_last_tokens_and_rollover(self):
        for processor in range(3):
            for token in (0, 8, 15):
                records = decode(self.page(processor, token), processor, token)
                self.assertEqual([row['duration_cycles'] for row in records], [20] * len(ZONES))
                self.assertEqual([row['zone'] for row in records], [name for name, _ in ZONES])

    def test_invalid_pages_identity_order_and_bounds_rejected(self):
        for words, processor, token in (([0xffffffff] * WORDS, 0, 8),
                (self.page(), 1, 8), (self.page(), 0, 9), (self.page(), 1, 9), (self.page()[:-1], 0, 8),
                (self.page(), True, 8), (self.page(), 0, 16)):
            with self.assertRaises(ValueError):
                decode(words, processor, token)
        words = self.page()
        words[6:10] = words[:4]
        with self.assertRaises(ValueError):
            decode(words, 0, 8)
        with self.assertRaises(ValueError):
            decode(self.page(), 0, 8, max_cycles=19)

    def test_retained_native_source_roundtrip(self):
        from gdn_shared_qk_recurrence import load_kernels

        root = Path('D:/qwen-evidence/35092212895/sources')
        if not root.exists():
            self.skipTest('Retained pinned native export unavailable')
        source = load_kernels(root)['recurrence']['compute']
        self.assertEqual(remove(instrument(source)), source)


if __name__ == '__main__':
    unittest.main()
