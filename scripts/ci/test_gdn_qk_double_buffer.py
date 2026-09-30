from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gdn_qk_double_buffer as candidate
from gdn_vsplit import cb_plan


class DoubleBufferTests(unittest.TestCase):
    def test_two_rings_only(self):
        original_io, original_fp32 = cb_plan('recurrence')
        io, fp32 = candidate.buffers(cb_plan, 'recurrence')
        self.assertEqual(io, original_io)
        self.assertEqual({key: value for key, value in fp32.items() if key not in (10, 11)},
            {key: value for key, value in original_fp32.items() if key not in (10, 11)})
        self.assertEqual((fp32[10], fp32[11]), (8, 8))
        self.assertEqual((sum(fp32.values()) - sum(original_fp32.values())) * 4096, 32768)
        self.assertEqual(original_fp32[10], 4)
        for stage, prefetch in [('norm_gate', False), ('recurrence', True)]:
            with self.assertRaises(ValueError):
                candidate.buffers(cb_plan, stage, prefetch_inputs=prefetch)

    def test_initialize_both_slots_and_reject_reapplication(self):
        source = 'before\n' + candidate.ORIGINAL_ZERO + '\nafter'
        changed = candidate.reader(source)
        self.assertEqual(changed.replace(candidate.CANDIDATE_ZERO, candidate.ORIGINAL_ZERO), source)
        for invalid in (changed, source + source, ''):
            with self.assertRaises(ValueError):
                candidate.reader(invalid)

    def test_scoped_build_restores_and_preserves_math(self):
        for failed in (False, True):
            native = dict(recurrence=dict(reader=candidate.ORIGINAL_ZERO, compute='math', writer='writer'),
                norm_gate=dict(reader='norm reader'))
            original = lambda root: native
            pipeline = SimpleNamespace(load_kernels=original, cb_plan=cb_plan)
            def construct(*args, **kwargs):
                result = pipeline.load_kernels(kwargs['root'])
                self.assertEqual(result['recurrence']['compute'], 'math')
                self.assertEqual(result['recurrence']['writer'], 'writer')
                self.assertEqual(result['norm_gate'], native['norm_gate'])
                self.assertEqual(pipeline.cb_plan('recurrence')[1][10], 8)
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
                self.assertEqual(candidate.BUILD_RECORDS[0]['extra_cb_bytes'], 32768)
            self.assertIs(pipeline.cb_plan, cb_plan)
            self.assertIs(pipeline.load_kernels, original)
            self.assertEqual(native['recurrence']['reader'], candidate.ORIGINAL_ZERO)

    def test_retained_native_source(self):
        from gdn_shared_qk_recurrence import load_kernels

        root = Path('D:/qwen-evidence/35092212895/sources')
        if not root.exists():
            self.skipTest('Retained export unavailable')
        source = load_kernels(root)['recurrence']['reader']
        self.assertEqual(candidate.reader(source).replace(candidate.CANDIDATE_ZERO, candidate.ORIGINAL_ZERO), source)

    def test_probe_checks_are_retained(self):
        from gdn_qk_double_buffer_stage import adapt

        source = Path(__file__).with_name('gdn-shared-recurrence-probe.py').read_text(encoding='utf-8')
        changed = adapt(source)
        for statement in ("len(report['checks']) != 24", "len(report['immutable_checks']) != 48",
                'for seed in (1, 2, 0)', 'qk_double_buffer=True', 'generated_kernels=BUILD_RECORDS'):
            self.assertIn(statement, changed)
        with self.assertRaises(ValueError):
            adapt(changed)


if __name__ == '__main__':
    unittest.main()
