import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest

import native_draft_sdpa
from dspark_ladder_factory import scoped_stats_pack
from dspark_ladder_stage_print import PARTIAL_SNAPSHOT, RECIPROCAL_SNAPSHOT, SNAPSHOT, stage_snapshots


class StagePrintTests(unittest.TestCase):
    def test_64k_coordinates_and_bounds(self):
        with stage_snapshots(row=5, column=0):
            edits = native_draft_sdpa.replacements()['compute_common.hpp']
            snapshot = next(after for before, after in edits if 'QWEN_STAGE' in after)
            self.assertIn('.h0=5, .h1=6', snapshot)
            self.assertNotIn('.w0=5, .w1=6', snapshot)
        for row, column in ((15, 0), (5, 32), (-1, 0), (True, 0)):
            with self.assertRaises(ValueError):
                with stage_snapshots(row=row, column=column):
                    pass

    def test_snapshot_does_not_consume_or_write_buffers(self):
        for forbidden in ('pop_front', 'push_back', 'reserve_back', 'pack_tile', 'copy_tile'):
            self.assertNotIn(forbidden, SNAPSHOT)
            self.assertNotIn(forbidden, RECIPROCAL_SNAPSHOT)
            self.assertNotIn(forbidden, PARTIAL_SNAPSHOT)
        self.assertIn('COMPILE_FOR_TRISC == 0', SNAPSHOT)
        self.assertNotIn('TSLICE_INPUT_CB', SNAPSHOT)
        self.assertNotIn('TSLICE_RD_PTR', SNAPSHOT)
        self.assertEqual(SNAPSHOT.count('TSLICE('), 3)
        self.assertIn('!QWEN_DRAFT_EXP_APPROX', SNAPSHOT)
        original = native_draft_sdpa.replacements
        with self.assertRaises(RuntimeError):
            with stage_snapshots():
                raise RuntimeError('abort')
        self.assertIs(native_draft_sdpa.replacements, original)

    @unittest.skipUnless(shutil.which('g++'), 'Host C++ syntax compiler required')
    def test_snapshot_cpp_syntax_and_compute_slice_arity(self):
        source = '''
#include <cstdint>
#define QWEN_DRAFT_EXP_APPROX false
struct SliceRange { int h0, h1, hs, w0, w1, ws; };
struct CircularBuffer {
    explicit CircularBuffer(uint32_t) {}
    void wait_front(uint32_t) {}
};
int TSLICE(uint32_t, int, const SliceRange&, bool, bool) { return 0; }
template <typename... Arguments> void DEVICE_PRINT(const char*, Arguments...) {}
void snapshot() {
    uint32_t alias_prev_sum=0, alias_prev_max=1, alias_mm2_prev_out=2;
    uint32_t Sq_chunk_t=1, out_chunk_tiles=4, local_q_start=12, q_iter=0, iter_q_start=0;
''' + PARTIAL_SNAPSHOT + SNAPSHOT + RECIPROCAL_SNAPSHOT + '\n}\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'snapshot.cpp'
            path.write_text(source)
            for processor in (0, 1, 2):
                result = subprocess.run(['g++', '-std=c++17', '-fsyntax-only',
                    '-DCOMPILE_FOR_TRISC=' + str(processor), str(path)],
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(os.environ.get('TT_NATIVE_TEST_ROOT'), 'Pinned native source required')
    def test_exact_source_composition(self):
        directory = Path(os.environ['TT_NATIVE_TEST_ROOT']) / native_draft_sdpa.KERNEL_DIRECTORY
        source = {name: (directory / name).read_bytes() for name in native_draft_sdpa.SOURCE_HASHES}
        with stage_snapshots(), scoped_stats_pack():
            patched = native_draft_sdpa.patched_sources(source)
        self.assertEqual(patched['compute_common.hpp'].count(b'QWEN_STAGE'), 1)
        self.assertEqual(patched['compute_common.hpp'].count(b'QWEN_RECIP'), 1)
        self.assertEqual(patched['compute_common.hpp'].count(b'QWEN_PARTIAL_SUM'), 1)
        self.assertLess(patched['compute_common.hpp'].index(b'QWEN_PARTIAL_SUM'),
            patched['compute_common.hpp'].index(b'matmul_reduce<Sq_chunk_t>(cb_col_identity, alias_prev_sum);'))
        self.assertIn(b'Ladder probe requires an explicit padded history geometry', patched['sdpa.cpp'])
