from pathlib import Path
import re
import unittest

from serving_canary_runner import REQUIRED_ENVIRONMENT, validate_environment


class CanaryEnvironmentTests(unittest.TestCase):
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
