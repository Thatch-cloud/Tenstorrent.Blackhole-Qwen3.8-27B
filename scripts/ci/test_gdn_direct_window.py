import unittest

from gdn_direct_window import START, END, causal_source, reader


class DirectWindowTests(unittest.TestCase):
    def test_all_prefixes_match_native_shift_before_convolution(self):
        history = [('history', index) for index in range(4)]
        for token in range(16):
            history = history[1:] + [('projected', token)]
            self.assertEqual([causal_source(token, slot) for slot in range(4)], history)
        for token, slot in ((-1, 0), (16, 0), (0, 4), (True, 0)):
            with self.assertRaises(ValueError):
                causal_source(token, slot)

    def test_source_replacement_preserves_gate_and_tap_paths(self):
        before, after = 'native declarations\n', END + 'native taps and gates\n'
        result = reader(before + START + 'old window\n' + after)
        self.assertTrue(result.startswith(before))
        self.assertTrue(result.endswith(after))
        self.assertEqual(result.count('noc_async_read_tile('), 4)
        self.assertIn('get_write_ptr(14)', result)
        self.assertIn('zero_words(base, 4 * 512)', result)
        self.assertIn('B == 16', result)
        for source in ('', START + START + END, END + START):
            with self.assertRaises(ValueError):
                reader(source)
