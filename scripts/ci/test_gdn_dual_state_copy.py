import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import gdn_dual_state_copy as candidate

from gdn_copy_pairs import ORIGINAL
from gdn_dual_state_copy import FEEDBACK, transform
from gdn_vsplit import cb_plan
from gdn_dual_state_stage import adapt


def fixture():
    return ('void copy_tiles(uint32_t in, uint32_t o, uint32_t n) {\n'
        '    cb_reserve_back(o, n);\n' + ORIGINAL + '\n    cb_push_back(o, n);\n}\n'
        'void kernel_main() {\n' + FEEDBACK + '\n}\n')


class DualStateCopyTests(unittest.TestCase):
    def test_program_construction_changes_only_recurrence_compute_and_restores(self):
        for failed in (False, True):
            native = {'recurrence': {'compute': fixture(), 'reader': 'reader', 'writer': 'writer'},
                'norm_gate': {'compute': 'norm'}}
            original = lambda root: native
            pipeline = SimpleNamespace(load_kernels=original, cb_plan=cb_plan)
            def construct(*args, **kwargs):
                kernels = pipeline.load_kernels(kwargs['root'])
                self.assertEqual(kernels['norm_gate'], native['norm_gate'])
                self.assertEqual(kernels['recurrence']['reader'], 'reader')
                self.assertEqual(kernels['recurrence']['writer'], 'writer')
                self.assertIn('copy_state_twice', kernels['recurrence']['compute'])
                if failed:
                    raise RuntimeError('construction failed')
                return 'program'
            pipeline.build = construct
            with patch.dict('sys.modules', {'gdn_shared_qk_pipeline': pipeline}), \
                    patch.object(candidate, 'BUILD_RECORDS', []):
                if failed:
                    with self.assertRaises(RuntimeError):
                        candidate.build(None, None, [], root='root')
                else:
                    self.assertEqual(candidate.build(None, None, [], root='root'), 'program')
                self.assertEqual(len(candidate.BUILD_RECORDS), 1)
            self.assertIs(pipeline.load_kernels, original)
            self.assertEqual(native['recurrence']['compute'], fixture())

    def test_real_probe_keeps_complete_state_and_input_matrix(self):
        source = Path(__file__).with_name('gdn-shared-recurrence-probe.py').read_text()
        changed = adapt(source)
        self.assertIn("len(report['checks']) != 24", changed)
        self.assertIn("len(report['immutable_checks']) != 48", changed)
        self.assertIn('for seed in (1, 2, 0)', changed)
        self.assertIn('generated_kernels=BUILD_RECORDS', changed)
        with self.assertRaises(ValueError):
            adapt(changed)

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
