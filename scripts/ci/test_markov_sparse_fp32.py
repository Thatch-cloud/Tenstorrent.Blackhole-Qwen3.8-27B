import os
from pathlib import Path
import unittest

from markov_sparse_fp32 import SOURCE, ANCHOR, INSERT, CONFIG, CONFIG_REPLACEMENT, transform


class MarkovSparseFp32Tests(unittest.TestCase):
    def test_unknown_source_rejected(self):
        with self.assertRaises(ValueError):
            transform(b'unknown runtime')

    @unittest.skipUnless(os.environ.get('TT_NATIVE_TEST_ROOT'), 'Pinned native source required')
    def test_exact_round_trip_and_modified_source_rejected(self):
        original = (Path(os.environ['TT_NATIVE_TEST_ROOT']) / SOURCE).read_bytes()
        candidate = transform(original)
        self.assertEqual(candidate.count(INSERT.encode()), 1)
        self.assertEqual(candidate.count(CONFIG_REPLACEMENT.encode()), 1)
        restored = candidate.replace(INSERT.encode(), b'').replace(CONFIG_REPLACEMENT.encode(), CONFIG.encode())
        self.assertEqual(restored, original)
        for invalid in (candidate, original + b'\n'):
            with self.assertRaises(ValueError):
                transform(invalid)

    def test_protocol_and_precision_remain_explicit(self):
        for guard in ('!nnz.has_value()', '!use_indices', '!packer_l1_acc_en',
                'is_input_a_sparse', 'operation_attributes.is_input_b_sparse',
                'ttnn::Shape{1, 1, 1, 256}', 'ttnn::Shape{1, 1, 256, 248320}'):
            self.assertIn(guard, INSERT)
        self.assertIn('qwen_unpack_mode[tt::CBIndex::c_5]', INSERT)
        self.assertTrue(CONFIG_REPLACEMENT.startswith(CONFIG))


if __name__ == '__main__':
    unittest.main()
