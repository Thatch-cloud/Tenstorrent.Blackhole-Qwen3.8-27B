import hashlib
import unittest
from unittest.mock import patch

import dspark_fp32_intermediates as candidate


class FactoryTests(unittest.TestCase):
    def test_only_format_block_changes(self):
        original = ('before\n' + candidate.ANCHOR + '\nafter').encode()
        with patch.object(candidate, 'SOURCE_SHA256', hashlib.sha256(original).hexdigest()):
            changed = candidate.transform(original)
        self.assertEqual(changed.replace(candidate.REPLACEMENT.encode(), candidate.ANCHOR.encode()), original)

    def test_unpinned_source_rejected(self):
        with self.assertRaises(ValueError):
            candidate.transform(candidate.ANCHOR.encode())

    def test_duplicate_anchor_rejected(self):
        original = (candidate.ANCHOR * 2).encode()
        with patch.object(candidate, 'SOURCE_SHA256', hashlib.sha256(original).hexdigest()):
            with self.assertRaises(ValueError):
                candidate.transform(original)

    def test_target_and_streaming_paths_excluded(self):
        for clause in ('NQH == 16', 'NKH == 4', 'DHt == 4', 'Skt == 266',
                       'Sk_chunk_t == 2', '!use_streaming_compute', 'fp32_dest_acc_en', '!is_causal'):
            self.assertIn(clause, candidate.REPLACEMENT)
