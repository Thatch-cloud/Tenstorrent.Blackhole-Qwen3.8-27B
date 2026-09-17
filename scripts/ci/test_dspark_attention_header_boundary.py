from pathlib import Path
import os
import tempfile
import unittest

import native_draft_sdpa
from dspark_attention_header_boundary import PREFIX, SEPARATOR, SUFFIX, boundary_scope, original_header


class HeaderTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('TT_NATIVE_TEST_ROOT'), 'Pinned sources required')
    def test_draft_math_and_native_header_are_preserved_and_reversible(self):
        from dspark_ladder_factory import scoped_stats_pack
        from dspark_ladder_sum_update import scalar_sum_update
        import dspark_ladder_score_center
        from dspark_mask_bits import mask_scope
        from dspark_score_bitwise import bitwise_infinity_checks
        from dspark_score_sfpu import kernel_scope
        from dspark_score_smoke_geometry import small_score_fixture
        from dspark_sum_sfpu import sum_scope

        root = Path(os.environ['TT_NATIVE_TEST_ROOT'])
        sources = {name: (root / native_draft_sdpa.KERNEL_DIRECTORY / name).read_bytes()
            for name in native_draft_sdpa.SOURCE_HASHES}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / native_draft_sdpa.KERNEL_DIRECTORY
            directory.mkdir(parents=True)
            for name, source in sources.items():
                (directory / name).write_bytes(source)
            with mask_scope(), sum_scope(), small_score_fixture(), bitwise_infinity_checks(), kernel_scope(), \
                    scalar_sum_update(), dspark_ladder_score_center.scalar_score_center(key_tiles=40), scoped_stats_pack():
                unguarded = native_draft_sdpa.patched_sources(sources)
                with boundary_scope(temporary):
                    guarded = native_draft_sdpa.patched_sources(sources)
                    expected = PREFIX.encode() + unguarded['compute_common.hpp'] + SEPARATOR.encode()
                    expected += sources['compute_common.hpp'] + SUFFIX.encode()
                    self.assertEqual(guarded['compute_common.hpp'], expected)
                    self.assertEqual(guarded['sdpa.cpp'], unguarded['sdpa.cpp'])
                    self.assertEqual(original_header(expected.decode()).encode(), sources['compute_common.hpp'])
                    with native_draft_sdpa.precise_draft_kernel(temporary):
                        native_draft_sdpa.audit_active_kernel(temporary)
            for name, source in sources.items():
                self.assertEqual((directory / name).read_bytes(), source)

    def test_unpinned_native_fallback_rejected(self):
        with self.assertRaises(ValueError):
            original_header('unrelated')


if __name__ == '__main__':
    unittest.main()
