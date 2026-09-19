import ast
from pathlib import Path
import re
import unittest

from serving_canary_runner import REQUIRED_ENVIRONMENT, shutdown_evidence, validate_environment


class CanaryEnvironmentTests(unittest.TestCase):
    def test_gate_exp_admission_runs_before_model_process(self):
        source = Path(__file__).with_name('serving_canary_runner.py').read_text()
        self.assertLess(source.index("admission = qualify('/canary/gate-exp-evidence'"),
                        source.index('process = subprocess.Popen('))
        workflow = (Path(__file__).resolve().parents[2] / '.github/workflows/qwen-fast-serving-canary.yml').read_text()
        self.assertIn('gdn_gate_exp_scope.py', workflow)
        self.assertIn('dst=/experiment-scripts/ci/$helper,readonly', workflow)
        self.assertIn('--candidate gate_exp', workflow)

    def test_api_exit_does_not_prove_worker_cleanup(self):
        evidence = shutdown_evidence('Application shutdown complete. force killing remaining processes count=1')
        self.assertEqual(evidence, dict(worker_closed=False, devices_closed=False, engine_forced=True))
        clean = shutdown_evidence('QWEN_FAST_WORKER_CLOSED\nClosing devices in cluster completed')
        self.assertEqual(clean, dict(worker_closed=True, devices_closed=True, engine_forced=False))

    def test_complete_recipe_is_admitted_without_mutation(self):
        environment = dict(REQUIRED_ENVIRONMENT)
        validate_environment(environment)
        self.assertEqual(environment, REQUIRED_ENVIRONMENT)

    def test_each_required_flag_is_checked_before_loading(self):
        for name in REQUIRED_ENVIRONMENT:
            with self.subTest(name=name):
                environment = dict(REQUIRED_ENVIRONMENT)
                del environment[name]
                with self.assertRaisesRegex(ValueError, name):
                    validate_environment(environment)

    def test_simulator_and_profiling_are_rejected(self):
        for name in ('TT_METAL_SIMULATOR', 'TT_METAL_DEVICE_PROFILER',
                'TT_METAL_SLOW_DISPATCH_MODE', 'TT_METAL_MOCK_CLUSTER_DESC_PATH', 'QWEN_SIM_ONLY'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                validate_environment(dict(REQUIRED_ENVIRONMENT, **{name: '1'}))

    def test_hardware_workflow_supplies_the_complete_recipe(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / '.github/workflows/qwen-fast-serving-canary.yml').read_text()
        supplied = dict(re.findall(r'-e ([A-Z0-9_]+)=([^\s\\]+)', source))
        validate_environment(supplied)

    def test_generated_direct_window_guard_is_covered_before_loading(self):
        from gdn_direct_window_hardware_sources import HARDWARE_GUARD

        tree = ast.parse(HARDWARE_GUARD.strip() + '\n    pass\n')
        required = [value.value for node in ast.walk(tree) if isinstance(node, ast.Tuple)
            for value in node.elts if isinstance(value, ast.Constant) and isinstance(value.value, str)]
        self.assertIn('QWEN_GDN_DIRECT_WINDOW', required)
        for name in required:
            self.assertEqual(REQUIRED_ENVIRONMENT[name], '1')
