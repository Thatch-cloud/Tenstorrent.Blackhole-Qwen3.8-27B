import unittest
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

from frozen_gdn_norm_prefetch import buffers, build, reader
from gdn_vsplit import cb_plan
from gdn_vsplit_norm_batch import READER_BODY


class NormPrefetchTests(unittest.TestCase):
    def test_staging_preserves_full_probe_matrix(self):
        from frozen_gdn_cache_stage import adapt_probe
        from frozen_recipe_context import REVISION
        source = subprocess.check_output(['git', 'show',
            f'{REVISION}:scripts/ci/gdn-shared-recurrence-probe.py'], text=True)
        candidate = adapt_probe(source, norm_prefetch=True)
        self.assertIn('from frozen_gdn_norm_prefetch import build as build_pipeline', candidate)
        self.assertEqual(source.split('    mesh, inputs, prepared, traces =', 1)[1],
            candidate.split('    mesh, inputs, prepared, traces =', 1)[1])

    def test_exact_bridge_mapping_all_heads(self):
        for head in range(24):
            source = [[page * 32 + word for word in range(32)] for page in range(16 * 96)]
            cached = [word for token in range(16) for partition in range(4)
                for word in source[token * 96 + head * 4 + partition]]
            self.assertEqual(len(cached) * 4, 8192)
            expected, actual = [None] * 4096, [None] * 4096
            for token in range(16):
                for partition in range(4):
                    for column in range(32):
                        target = partition * 1024 + token * 16 + (column // 16) * 256 + column % 16
                        expected[target] = source[token * 96 + head * 4 + partition][column]
                        actual[target] = cached[(token * 4 + partition) * 32 + column]
            self.assertEqual(actual, expected)

    def test_reader_batches_waits_and_preserves_copy(self):
        candidate = reader(READER_BODY)
        self.assertNotIn('stick.pop_front(1)', candidate)
        self.assertIn('sticks.reserve_back(2)', candidate)
        copy = '            for (uint32_t word = 0; word < 16; ++word) {'
        self.assertIn(copy + READER_BODY.split(copy)[1].split('            stick.pop_front')[0], candidate)
        with self.assertRaises(ValueError):
            reader(candidate)

    def test_only_norm_reader_and_staging_change(self):
        original = dict(norm_gate=dict(reader=READER_BODY, compute='math', writer='output'),
            recurrence=dict(reader='input', compute='recurrence', writer='state'))
        load = lambda root: original
        pipeline = SimpleNamespace(load_kernels=load)
        split = SimpleNamespace(cb_plan=cb_plan)

        def execute(*args, root):
            result = pipeline.load_kernels(root)
            self.assertEqual(result['recurrence'], original['recurrence'])
            self.assertEqual(result['norm_gate']['compute'], 'math')
            self.assertEqual(result['norm_gate']['writer'], 'output')
            self.assertEqual(split.cb_plan('norm_gate')[1][5], 2)
            self.assertEqual(split.cb_plan('recurrence'), cb_plan('recurrence'))
            raise RuntimeError('injected')

        pipeline.build = execute
        with patch.dict('sys.modules', dict(gdn_shared_qk_pipeline=pipeline, gdn_vsplit=split)):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                build(None, None, [], root='pinned')
        self.assertIs(pipeline.load_kernels, load)
        self.assertIs(split.cb_plan, cb_plan)
        self.assertEqual(original['norm_gate']['reader'], READER_BODY)
        io, fp32 = cb_plan('norm_gate')
        self.assertEqual(buffers(io, fp32), (io, fp32 | {5: 2}))
        self.assertEqual(fp32[5], 1)


if __name__ == '__main__':
    unittest.main()
