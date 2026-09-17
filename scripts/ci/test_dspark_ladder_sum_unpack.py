import os
from pathlib import Path
import unittest

from dspark_fp32_intermediates import SOURCE
from dspark_ladder_factory import transform as ladder_transform
from dspark_ladder_sum_unpack import INSERT, remove_unpack_transform, transform


class SumUnpackTests(unittest.TestCase):
    def test_only_sum_operands_are_explicitly_routed(self):
        self.assertEqual(INSERT.count('UnpackToDestFp32'), 2)
        self.assertIn('if (qwen_draft_fp32_intermediates)', INSERT)
        self.assertIn('.at(cb_ids.sum_A)', INSERT)
        self.assertIn('.at(cb_ids.sum_B)', INSERT)
        for operand in ('qk_im', 'out_im_A', 'out_im_B', 'max_A', 'max_B', 'exp_max_diff'):
            self.assertNotIn('cb_ids.' + operand, INSERT)
        with self.assertRaises(ValueError):
            remove_unpack_transform(b'unknown')

    @unittest.skipUnless(os.environ.get('TT_NATIVE_TEST_ROOT'), 'Pinned native source required')
    def test_pinned_transform_reverses_to_exact_ladder_candidate(self):
        source = (Path(os.environ['TT_NATIVE_TEST_ROOT']) / SOURCE).read_bytes()
        candidate = transform(source)
        self.assertEqual(remove_unpack_transform(candidate), ladder_transform(source))
        self.assertIn(b'!use_attention_sink && !is_windowed && !use_streaming_compute', candidate)
        with self.assertRaises(ValueError):
            transform(source + b'\n')
        with self.assertRaises(ValueError):
            remove_unpack_transform(candidate.replace(b'.at(cb_ids.sum_B)', b'.at(cb_ids.qk_im)'))
