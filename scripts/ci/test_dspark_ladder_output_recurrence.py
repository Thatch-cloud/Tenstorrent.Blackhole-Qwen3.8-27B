import os
from pathlib import Path
import unittest

import native_draft_sdpa
from dspark_ladder_factory import scoped_stats_pack
from dspark_ladder_output_recurrence import AFTER, BEFORE, explicit_output_recurrence


class OutputRecurrenceTests(unittest.TestCase):
    def test_scope_restores_after_exception(self):
        original = native_draft_sdpa.replacements
        with self.assertRaises(RuntimeError):
            with explicit_output_recurrence(), scoped_stats_pack():
                substitutions = native_draft_sdpa.replacements()['compute_common.hpp']
                self.assertIn((BEFORE, AFTER), substitutions)
                self.assertIn('if constexpr (!QWEN_DRAFT_EXP_APPROX)', AFTER)
                self.assertEqual(AFTER.count(BEFORE), 1)
                raise RuntimeError('abort')
        self.assertIs(native_draft_sdpa.replacements, original)

    @unittest.skipUnless(os.environ.get('TT_NATIVE_TEST_ROOT'), 'Pinned native source required')
    def test_exact_native_transform_retains_guards_and_consumption(self):
        directory = Path(os.environ['TT_NATIVE_TEST_ROOT']) / native_draft_sdpa.KERNEL_DIRECTORY
        original = {name: (directory / name).read_bytes() for name in native_draft_sdpa.SOURCE_HASHES}
        with explicit_output_recurrence(), scoped_stats_pack():
            result = native_draft_sdpa.patched_sources(original)
            self.assertEqual(result['compute_common.hpp'].count(AFTER.encode()), 1)
            self.assertIn(b'add_block_inplace(alias_mm2_cur_out, alias_mm2_prev_out, out_chunk_tiles)',
                result['compute_common.hpp'])
            self.assertIn(b'Ladder probe requires an explicit padded history geometry', result['sdpa.cpp'])
            with self.assertRaises(ValueError):
                native_draft_sdpa.patched_sources({**original, 'compute_common.hpp': original['compute_common.hpp'] + b'\n'})
