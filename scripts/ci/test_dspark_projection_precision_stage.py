import ast
from pathlib import Path
import subprocess
import unittest

from dspark_projection_precision_stage import BRANCH, MOUNT, PROJECTIONS, adapt
from frozen_recipe_context import REVISION


class PrecisionStageTests(unittest.TestCase):
    def test_every_projection_selects_one_literal_argument(self):
        original = [subprocess.check_output(['git', 'show', f'{REVISION}:scripts/ci/{name}'], text=True)
            for name in ('run-simulator.sh', 'simulator-suite.sh')]
        for projection in PROJECTIONS:
            runner, suite = adapt(*original, projection=projection)
            self.assertIn('--projection ' + projection + ' ', suite)
            self.assertEqual(suite.count('--projection '), original[1].count('--projection ') + 1)
        with self.assertRaises(ValueError):
            adapt(*original, projection='unknown; command')

    def test_frozen_runner_admits_only_explicit_new_case_and_mount(self):
        original = [subprocess.check_output(['git', 'show', f'{REVISION}:scripts/ci/{name}'], text=True)
            for name in ('run-simulator.sh', 'simulator-suite.sh')]
        runner, suite = adapt(*original)
        restored = runner.replace('in dspark-projection-hifi2|', 'in ').replace(MOUNT, '')
        self.assertEqual(restored, original[0])
        self.assertEqual(suite.replace(BRANCH, ''), original[1])
        self.assertIn('timeout -k 15 510', BRANCH)
        self.assertIn('--projection self_attn.q_proj.weight', BRANCH)
        with self.assertRaises(ValueError):
            adapt(runner, suite)

    def test_probe_is_import_safe_and_contains_same_policy_replay(self):
        path = Path(__file__).with_name('dspark-projection-precision-probe.py')
        source = path.read_text()
        ast.parse(source)
        self.assertIn('torch.equal(actual, eager[case][chip])', source)
        self.assertIn('target_correctness_qualified=False', source)
        self.assertIn('ttnn.release_trace(mesh, trace)', source)
