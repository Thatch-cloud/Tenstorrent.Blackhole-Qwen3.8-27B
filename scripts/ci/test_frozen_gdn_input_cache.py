import unittest
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

from frozen_gdn_input_cache import buffers, cache_source, reader, LOOP, build
from gdn_shared_qk_recurrence import INPUTS
from gdn_vsplit import cb_plan


class GdnInputCacheTests(unittest.TestCase):
    def test_staging_preserves_full_replay_matrix(self):
        from frozen_gdn_cache_stage import adapt_probe
        from frozen_recipe_context import REVISION
        source = subprocess.check_output(['git', 'show',
            f'{REVISION}:scripts/ci/gdn-shared-recurrence-probe.py'], text=True)
        result = adapt_probe(source)
        self.assertIn('from frozen_gdn_input_cache import build as build_pipeline', result)
        self.assertEqual(source.split('    mesh, inputs, prepared, traces =', 1)[1],
            result.split('    mesh, inputs, prepared, traces =', 1)[1])

    def test_only_reader_and_recurrence_buffer_plan_change(self):
        source = LOOP + '\n' + INPUTS + '}\n    query_cache.pop_front(4);\n'
        original = {'recurrence': {'reader': source, 'compute': 'math', 'writer': 'outputs'}}
        load = lambda root: original
        plan = lambda stage, prefetch_inputs=False: cb_plan(stage, prefetch_inputs=prefetch_inputs)
        pipeline = SimpleNamespace(load_kernels=load, cb_plan=plan)

        def execute(*args, root):
            kernels = pipeline.load_kernels(root)
            self.assertEqual(kernels['recurrence']['compute'], 'math')
            self.assertEqual(kernels['recurrence']['writer'], 'outputs')
            self.assertEqual(pipeline.cb_plan('recurrence')[0][31], 3)
            return kernels

        pipeline.build = execute
        with patch.dict('sys.modules', {'gdn_shared_qk_pipeline': pipeline}):
            result = build(None, None, [], root='pinned')
        self.assertNotEqual(result['recurrence']['reader'], source)
        self.assertEqual(original['recurrence']['reader'], source)
        self.assertIs(pipeline.load_kernels, load)
        self.assertIs(pipeline.cb_plan, plan)

    def test_three_bf16_pages_and_no_compute_buffer_change(self):
        io, fp32 = cb_plan('recurrence')
        changed_io, changed_fp32 = buffers(io, fp32)
        self.assertEqual(changed_fp32, fp32)
        self.assertNotIn(31, io)
        self.assertEqual(changed_io, io | {31: 3})
        with self.assertRaises(ValueError):
            buffers(changed_io, fp32)

    def test_prefetches_only_v_beta_gate_before_loop(self):
        cache = cache_source()
        self.assertEqual(cache.count('noc.async_read('), 3)
        self.assertNotIn('noc.async_read(q_acc', cache)
        self.assertNotIn('noc.async_read(k_acc', cache)
        source = LOOP + '\n' + INPUTS + '}\n    query_cache.pop_front(4);\n'
        changed = reader(source)
        before, loop = changed.split(LOOP)
        self.assertEqual(before.count('noc.async_read('), 3)
        self.assertNotIn('gather_scalar(', loop)
        self.assertIn('gather_normalized(20, 10, token)', loop)
        self.assertIn('cache.pop_front(3)', loop)
        with self.assertRaises(ValueError):
            reader(changed)


if __name__ == '__main__':
    unittest.main()
