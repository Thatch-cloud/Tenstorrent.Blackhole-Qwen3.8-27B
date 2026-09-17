import subprocess
import unittest

from frozen_recipe_context import REVISION, replace_once
from frozen_wait_combined_stage import adapt, SOURCE_FILES, drain_setup


class CombinedWaitStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = {name: subprocess.check_output(['git', 'show',
            f'{REVISION}:scripts/ci/{name}'], text=True) for name in SOURCE_FILES}
        cls.sources['dspark_8k_scope.py'] = replace_once(cls.sources['dspark_8k_scope.py'],
            '        yield evidence',
            '        from frozen_gdn_norm_scope import runtime_scope as incremental_scope\n'
            '        stack.enter_context(incremental_scope(directory))\n        yield evidence')

    def test_scope_order_raw_capture_and_route_validation(self):
        result = adapt(self.sources)
        scope = result['dspark_8k_scope.py']
        self.assertLess(scope.index('enter_context(incremental_scope'), scope.index('enter_context(marker_scope'))
        self.assertIn('from frozen_wait_zone_scope import validate_route', result['request_verifier_profile_report.py'])
        self.assertIn('profile_log_device.csv', result['dspark-combined-profile.sh'])
        self.assertIn('frozen_wait_combined_report.py', result['dspark-combined-profile.sh'])
        self.assertIn("schedule = (('publication', True),)", result['dspark_request_experiment.py'])
        self.assertIn('timeout -k 30 720', result['dspark-combined-profile.sh'])
        self.assertIn('--max-new-tokens 256', result['dspark-combined-profile.sh'])
        self.assertIn('cp -u "$output/.logs/$name"', result['dspark-combined-profile.sh'])

    def test_history_capacity_matches_existing_transaction_gate(self):
        from frozen_incremental_scope import validate_records
        result = adapt(self.sources)
        self.assertIn('history_capacity=((len(prompt) + max_new_tokens + 31) // 32) * 32',
            result['full_dspark_request.py'])
        capacity = ((32768 + 256 + 31) // 32) * 32
        validate_records([dict(initial_position=32768, capacity=capacity, failed=False,
            restored=True, committed=1, prepared=1, discarded=0, max_touched_rows=32)], True)
        with self.assertRaises(ValueError):
            validate_records([dict(initial_position=32768, capacity=((32768 + 64 + 31) // 32) * 32,
                failed=False, restored=True, committed=1, prepared=1, discarded=0, max_touched_rows=32)], True)

    def test_missing_base_scope_rejected(self):
        changed = dict(self.sources)
        changed['dspark_8k_scope.py'] = ''
        with self.assertRaises(ValueError):
            adapt(changed)

    def test_drains_are_outside_trace_operations(self):
        import ast
        result = drain_setup(self.sources)
        tree = ast.parse(result['verifier_engine.py'])
        engine = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'VerifierEngine')
        for method in engine.body:
            if isinstance(method, ast.FunctionDef) and method.name != '__init__':
                self.assertNotIn('ReadDeviceProfiler', ast.unparse(method))
        for node in ast.walk(tree):
            if isinstance(node, ast.Lambda):
                self.assertNotIn('ReadDeviceProfiler', ast.unparse(node))
        self.assertEqual(result['verifier_engine.py'].count('ttnn.ReadDeviceProfiler(self.mesh)'), 4)
        self.assertEqual(result['full_dspark_request.py'].count('operations.ReadDeviceProfiler(model.mesh_device)'), 4)
        with self.assertRaises(ValueError):
            drain_setup(result)

    def test_drain_guards_leave_nonprofiled_host_tests_untouched(self):
        import ast
        from types import SimpleNamespace
        from unittest.mock import Mock
        result = drain_setup(self.sources)
        for filename, guard in (('full_dspark_request.py', 'combined_profile'),
                ('verifier_engine.py', "os.environ.get('QWEN_COMBINED_TRACE_PROFILE') == '1'")):
            tree = ast.parse(result[filename])
            guards = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                and ast.unparse(node.test) == guard
                and any(isinstance(child, ast.Attribute) and child.attr == 'ReadDeviceProfiler'
                    for child in ast.walk(node))]
            self.assertEqual(len(guards), 4)
            for node in guards:
                code = compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), filename, 'exec')
                disabled = dict(combined_profile=False, os=SimpleNamespace(environ={}))
                exec(code, disabled)
                operations = SimpleNamespace(synchronize_device=Mock(), ReadDeviceProfiler=Mock())
                mesh = object()
                enabled = dict(combined_profile=True,
                    os=SimpleNamespace(environ={'QWEN_COMBINED_TRACE_PROFILE': '1'}),
                    operations=operations, model=SimpleNamespace(mesh_device=mesh),
                    ttnn=operations, self=SimpleNamespace(mesh=mesh))
                exec(code, enabled)
                operations.ReadDeviceProfiler.assert_called_once_with(mesh)


if __name__ == '__main__':
    unittest.main()
