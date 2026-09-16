import subprocess
import unittest

from frozen_recipe_context import REVISION, replace_once
from frozen_verifier_profile import FILES
from frozen_wait_combined_stage import adapt


class CombinedWaitStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = {name: subprocess.check_output(['git', 'show',
            f'{REVISION}:scripts/ci/{name}'], text=True) for name in (*FILES, 'dspark_8k_scope.py')}
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

    def test_missing_base_scope_rejected(self):
        changed = dict(self.sources)
        changed['dspark_8k_scope.py'] = ''
        with self.assertRaises(ValueError):
            adapt(changed)


if __name__ == '__main__':
    unittest.main()
