import hashlib
import unittest
from unittest.mock import patch

import t32_attention_fp32_patch as candidate


class PatchTests(unittest.TestCase):
    def test_stats_only_preserves_bf16_partial_outputs(self):
        source = (candidate.BEFORE + '\n    tt::DataFormat stats_df = im_df;\n').encode()
        with patch.object(candidate, 'SOURCE_SHA256', hashlib.sha256(source).hexdigest()):
            result = candidate.patched_bytes(source, variant='stats-only')
        self.assertIn(b'tt::DataFormat stats_df = im_df;\n    im_df = tt::DataFormat::Float16_b;', result)
        self.assertIn(candidate.AFTER.encode(), result)

    def test_only_intermediate_format_changes(self):
        source = ('before\n' + candidate.BEFORE + '\n    tt::DataFormat stats_df = im_df;\nafter\n').encode()
        with patch.object(candidate, 'SOURCE_SHA256', hashlib.sha256(source).hexdigest()):
            result = candidate.patched_bytes(source)
        self.assertEqual(result.replace(candidate.AFTER.encode(), candidate.BEFORE.encode()), source)
        for guard in ('B == 1', 'NQH == 16', 'NKH == 4', 'Sq == 32', 'DH == 128',
                      '!is_causal', 'use_provided_mask', '!use_streaming_compute', 'fp32_dest_acc_en'):
            self.assertIn(guard, candidate.AFTER)

    def test_unpinned_or_duplicate_source_fails(self):
        with self.assertRaises(ValueError):
            candidate.patched_bytes(candidate.BEFORE.encode())
        source = (candidate.BEFORE * 2).encode()
        with patch.object(candidate, 'SOURCE_SHA256', hashlib.sha256(source).hexdigest()), self.assertRaises(ValueError):
            candidate.patched_bytes(source)

    def test_output_only_retains_bf16_statistics(self):
        source = (candidate.BEFORE + '\n    tt::DataFormat stats_df = im_df;\n').encode()
        with patch.object(candidate, 'SOURCE_SHA256', hashlib.sha256(source).hexdigest()):
            result = candidate.patched_bytes(source, variant='output-only')
        self.assertIn(candidate.AFTER.encode(), result)
        self.assertIn(b'tt::DataFormat stats_df = tt::DataFormat::Float16_b;', result)
