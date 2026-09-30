import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from frozen_recipe_context import REVISION
from gdn_norm_scatter import replace_reader
from gdn_vsplit_norm_batch import READER_BODY
from shared_qk_norm_scatter import build


class SharedScatterTests(unittest.TestCase):
    def test_scoped_composition_preserves_shared_recurrence(self):
        original = dict(norm_gate=dict(reader=READER_BODY, compute='norm', writer='output'),
            recurrence=dict(reader='shared qk', compute='recurrence', writer='states'))
        loader = lambda root: original
        pipeline = SimpleNamespace(load_kernels=loader)
        tensors = object()

        def execute(operations, mesh, actual, *, root):
            self.assertIs(actual, tensors)
            result = pipeline.load_kernels(root)
            self.assertEqual(result['recurrence'], original['recurrence'])
            self.assertEqual(result['norm_gate'], dict(reader=replace_reader(READER_BODY),
                compute='norm', writer='output'))
            raise RuntimeError('injected build failure')

        pipeline.build = execute
        with patch.dict('sys.modules', gdn_shared_qk_pipeline=pipeline):
            with self.assertRaisesRegex(RuntimeError, 'injected build failure'):
                build(None, None, tensors, root='pinned')
        self.assertIs(pipeline.load_kernels, loader)
        self.assertEqual(original['norm_gate']['reader'], READER_BODY)

    def test_simulator_keeps_full_eager_and_replay_matrix(self):
        from frozen_gdn_cache_stage import adapt_probe

        source = subprocess.check_output(['git', 'show',
            f'{REVISION}:scripts/ci/gdn-shared-recurrence-probe.py'], text=True)
        candidate = adapt_probe(source, norm_scatter=True)
        self.assertIn('from shared_qk_norm_scatter import build as build_pipeline', candidate)
        self.assertIn('norm_bridge_scatter=True', candidate)
        self.assertEqual(source.split('    mesh, inputs, prepared, traces =', 1)[1],
            candidate.split('    mesh, inputs, prepared, traces =', 1)[1])
        with self.assertRaises(ValueError):
            adapt_probe(source, norm_scatter=True, norm_prefetch=True)
        with self.assertRaises(ValueError):
            adapt_probe(source, norm_scatter=1)


if __name__ == '__main__':
    unittest.main()
