import os
from pathlib import Path
import unittest

import native_draft_sdpa
from dspark_ladder_factory import scoped_stats_pack
from dspark_ladder_stage_print import SNAPSHOT, stage_snapshots


class StagePrintTests(unittest.TestCase):
    def test_snapshot_does_not_consume_or_write_buffers(self):
        for forbidden in ('pop_front', 'push_back', 'reserve_back', 'pack_tile', 'copy_tile'):
            self.assertNotIn(forbidden, SNAPSHOT)
        self.assertIn('UNPACK((', SNAPSHOT)
        self.assertNotIn('TSLICE_INPUT_CB', SNAPSHOT)
        self.assertNotIn('TSLICE_RD_PTR', SNAPSHOT)
        self.assertEqual(SNAPSHOT.count('TSLICE('), 3)
        self.assertIn('!QWEN_DRAFT_EXP_APPROX', SNAPSHOT)
        original = native_draft_sdpa.replacements
        with self.assertRaises(RuntimeError):
            with stage_snapshots():
                raise RuntimeError('abort')
        self.assertIs(native_draft_sdpa.replacements, original)

    @unittest.skipUnless(os.environ.get('TT_NATIVE_TEST_ROOT'), 'Pinned native source required')
    def test_exact_source_composition(self):
        directory = Path(os.environ['TT_NATIVE_TEST_ROOT']) / native_draft_sdpa.KERNEL_DIRECTORY
        source = {name: (directory / name).read_bytes() for name in native_draft_sdpa.SOURCE_HASHES}
        with stage_snapshots(), scoped_stats_pack():
            patched = native_draft_sdpa.patched_sources(source)
        self.assertEqual(patched['compute_common.hpp'].count(b'QWEN_STAGE'), 1)
        self.assertIn(b'Ladder probe requires an explicit padded history geometry', patched['sdpa.cpp'])
