import unittest

from gdn_copy_pairs import ORIGINAL
from gdn_dual_state_copy import FEEDBACK, transform
from gdn_vsplit import cb_plan


def fixture():
    return ('void copy_tiles(uint32_t in, uint32_t o, uint32_t n) {\n'
        '    cb_reserve_back(o, n);\n' + ORIGINAL + '\n    cb_push_back(o, n);\n}\n'
        'void kernel_main() {\n' + FEEDBACK + '\n}\n')


class DualStateCopyTests(unittest.TestCase):
    def test_one_unpack_two_packs_and_original_final_token(self):
        source = fixture()
        changed = transform(source)
        helper = changed[changed.index('void copy_state_twice'):changed.index('void kernel_main')]
        self.assertEqual(helper.count('copy_tile(in, i, 0);'), 1)
        self.assertEqual(helper.count('pack_tile(0,'), 2)
        self.assertEqual(helper.count('tile_regs_acquire();'), 1)
        self.assertLess(helper.index('pack_tile(0, feedback, i);'), helper.index('tile_regs_release();'))
        self.assertIn('if (it + 1 < n_inst)', changed)
        self.assertIn('} else {\n            copy_tiles(cb_snew, cb_sout, kv);', changed)
        self.assertIn(source[:source.index('void kernel_main')], changed)

    def test_both_consumers_have_same_bf16_page_geometry(self):
        io, fp32 = cb_plan('recurrence')
        self.assertEqual(io[18], 4)
        self.assertEqual(io[30], 4)
        self.assertNotIn(18, fp32)
        self.assertNotIn(30, fp32)

    def test_changed_or_already_transformed_sources_rejected(self):
        for source in (transform(fixture()), fixture().replace(FEEDBACK, ''),
                fixture().replace('pack_tile(0, o, i);', 'pack_tile(1, o, i);')):
            with self.assertRaises(ValueError):
                transform(source)
