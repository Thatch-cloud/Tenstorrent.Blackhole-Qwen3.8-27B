from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gdn_input_overlap as candidate
from gdn_shared_qk_recurrence import INPUTS


class InputOverlapTests(unittest.TestCase):
    def test_only_gather_order_changes(self):
        source = 'before\n' + INPUTS + 'after\n'
        changed = candidate.reader(source)
        self.assertEqual(sorted(source.splitlines()), sorted(changed.splitlines()))
        self.assertLess(changed.index('gather_scalar(g_acc'), changed.index('gather_normalized(20'))
        self.assertLess(changed.index('gather_row(v_acc'), changed.index('gather_scalar(beta_acc'))
        self.assertLess(changed.index('gather_normalized(20'), changed.index('gather_normalized(21'))
        with self.assertRaises(ValueError):
            candidate.reader(changed)
        with self.assertRaises(ValueError):
            candidate.reader(source + INPUTS)

    def test_program_preserves_compute_writer_norm_and_restores_on_failure(self):
        for failed in (False, True):
            native = dict(recurrence=dict(reader='before\n' + INPUTS + 'after\n', compute='compute', writer='writer'),
                norm_gate=dict(reader='prefetch', compute='norm'))
            original = lambda root: native
            pipeline = SimpleNamespace(load_kernels=original)
            def construct(*args, **kwargs):
                kernels = pipeline.load_kernels(kwargs['root'])
                self.assertEqual(kernels['norm_gate'], native['norm_gate'])
                self.assertEqual(kernels['recurrence']['compute'], 'compute')
                self.assertEqual(kernels['recurrence']['writer'], 'writer')
                self.assertEqual(kernels['recurrence']['reader'], candidate.reader(native['recurrence']['reader']))
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
                self.assertEqual(candidate.BUILD_RECORDS[0]['extra_cb_bytes'], 0)
            self.assertIs(pipeline.load_kernels, original)
            self.assertIn(INPUTS, native['recurrence']['reader'])

    def test_actual_pinned_reader_changes_only_gather_order(self):
        from gdn_shared_qk_recurrence import load_kernels

        root = Path('D:/qwen-evidence/35092212895/sources')
        if not root.exists():
            self.skipTest('Retained native export unavailable')
        source = load_kernels(root)['recurrence']['reader']
        changed = candidate.reader(source)
        self.assertEqual(sorted(source.splitlines()), sorted(changed.splitlines()))


if __name__ == '__main__':
    unittest.main()
