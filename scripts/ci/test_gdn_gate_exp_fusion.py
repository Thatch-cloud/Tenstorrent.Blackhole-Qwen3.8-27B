import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gdn_gate_exp_fusion as candidate
from gdn_gate_exp_stage import adapt
from gdn_vsplit import cb_plan


def fixture():
    return ('void kernel_main() {\n        WAIT(cb_g, 1);\n' + candidate.COPY
        + '        POP(cb_g, 1);\n        WAIT(cb_gf, 1);\n' + candidate.EXP
        + '        WAIT(cb_gexp, 1);\n        POP(cb_gf, 1);\n'
        + '        bcast_scalar_mul(cb_sf, cb_gexp, cb_sdec, kv);\n'
        + '        POP(cb_gexp, 1);\n}\n')


class GateExpFusionTests(unittest.TestCase):
    def test_exact_removal_preserves_consumer(self):
        changed = candidate.transform(fixture())
        self.assertNotIn('cb_gf', changed)
        self.assertEqual(changed.count('expc('), 1)
        self.assertIn('expc(cb_g, cb_gexp, 1);', changed)
        self.assertIn('bcast_scalar_mul(cb_sf, cb_gexp, cb_sdec, kv);', changed)

    def test_rejects_changed_duplicate_or_reordered_anchors(self):
        for source in (candidate.transform(fixture()), fixture() + candidate.COPY,
                       fixture().replace('WAIT(cb_g, 1);', 'WAIT(cb_g, 2);'),
                       fixture().replace('        WAIT(cb_g, 1);\n', '')
                       + '        WAIT(cb_g, 1);\n'):
            with self.assertRaises(ValueError):
                candidate.transform(source)

    def test_scoped_compute_only_replacement_and_restoration(self):
        for failed in (False, True):
            native = {'recurrence': dict(compute=fixture(), reader='reader', writer='writer'),
                      'norm_gate': dict(compute='norm')}
            original = lambda root: native
            pipeline = SimpleNamespace(load_kernels=original, cb_plan=cb_plan)

            def construct(*args, **kwargs):
                kernels = pipeline.load_kernels(kwargs['root'])
                self.assertEqual(kernels['norm_gate'], native['norm_gate'])
                for role in ('reader', 'writer'):
                    self.assertEqual(kernels['recurrence'][role], role)
                self.assertEqual(kernels['recurrence']['compute'], candidate.transform(fixture()))
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

    def test_probe_retains_full_correctness_matrix(self):
        original = Path(__file__).with_name('gdn-shared-recurrence-probe.py').read_text()
        changed = adapt(original)
        for retained in ("len(report['checks']) != 24", "len(report['immutable_checks']) != 48",
                         'for seed in (1, 2, 0)', 'generated_kernels=BUILD_RECORDS'):
            self.assertIn(retained, changed)
        with self.assertRaises(ValueError):
            adapt(changed)
