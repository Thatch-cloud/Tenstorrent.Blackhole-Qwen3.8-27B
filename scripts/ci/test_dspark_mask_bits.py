from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

import dspark_ladder_score_center
from dspark_mask_bits import AFTER, BEFORE, mask_scope, transform


class MaskBitsTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('TT_NATIVE_TEST_ROOT'), 'Pinned native sources required')
    def test_full_candidate_source_composes(self):
        import native_draft_sdpa
        from dspark_ladder_factory import scoped_stats_pack
        from dspark_ladder_sum_update import scalar_sum_update
        from dspark_score_bitwise import bitwise_infinity_checks
        from dspark_score_sfpu import kernel_scope
        from dspark_score_smoke_geometry import small_score_fixture
        from dspark_sum_sfpu import sum_scope

        root = Path(os.environ['TT_NATIVE_TEST_ROOT']) / native_draft_sdpa.KERNEL_DIRECTORY
        sources = {name: (root / name).read_bytes() for name in native_draft_sdpa.SOURCE_HASHES}
        with mask_scope(), sum_scope(), small_score_fixture(), bitwise_infinity_checks(), kernel_scope():
            with scalar_sum_update(), dspark_ladder_score_center.scalar_score_center(key_tiles=40), scoped_stats_pack():
                kernel = native_draft_sdpa.patched_sources(sources)['compute_common.hpp']
        self.assertIn(AFTER.encode(), kernel)
        self.assertIn(b'qwen_sum_update_sfpu(alias_prev_sum', kernel)
        self.assertIn(b'sfpu_sub_bcast_col(j, 1)', kernel)

    def test_restores_original_and_keeps_arbitrary_bias_fallback(self):
        original = dspark_ladder_score_center.HELPER
        with mask_scope():
            changed = dspark_ladder_score_center.HELPER
            self.assertIn(AFTER, changed)
            self.assertIn('if ((mask_bits & 0x7fffffffU) == 0) continue;', changed)
        self.assertEqual(dspark_ladder_score_center.HELPER, original)
        with self.assertRaises(ValueError):
            transform('unrelated')

    @unittest.skipUnless(shutil.which('g++'), 'Host C++ compiler required')
    def test_generated_arithmetic_matches_original_edge_and_random_bits(self):
        template = '''#include <cstdint>
#include <cstdio>
uint32_t FUNCTION(uint32_t input, uint32_t mask_bits) {
    union { uint32_t bits; float value; } initial;
    initial.bits = input;
    volatile float scores[1] = {initial.value};
    const uint32_t offset = 0;
    if ((mask_bits & 0x7fffffffU) == 0) return input;
    union { uint32_t bits; float value; } converted;
    BODY
    initial.value = scores[0];
    return initial.bits;
}
'''
        source = template.replace('FUNCTION', 'original').replace('BODY', BEFORE)
        source += template.replace('FUNCTION', 'candidate').replace('BODY', AFTER)
        source += '''int main() {
    const uint32_t masks[] = {0, 0x80000000U, 0xff800000U, 0x7f800000U, 0x3f800000U, 0xbf000000U};
    const uint32_t edges[] = {0, 0x80000000U, 1, 0x80000001U, 0x7f7fffffU,
        0xff7fffffU, 0x7f800000U, 0xff800000U, 0x7fc00001U, 0x7f800001U};
    for (auto input : edges) for (auto mask : masks)
        if (original(input, mask) != candidate(input, mask)) return 1;
    uint32_t state = 12345;
    for (uint32_t count = 0; count < 30000; ++count) {
        state = state * 1664525U + 1013904223U;
        for (auto mask : masks) if (original(state, mask) != candidate(state, mask)) return 2;
    }
}
'''
        with tempfile.TemporaryDirectory() as directory:
            binary = str(Path(directory) / 'mask-check')
            subprocess.run(['g++', '-O2', '-std=c++17', '-x', 'c++', '-', '-o', binary],
                input=source, text=True, check=True, capture_output=True, timeout=15)
            subprocess.run([binary], check=True, timeout=5)


if __name__ == '__main__':
    unittest.main()
