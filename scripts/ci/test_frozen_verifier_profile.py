import subprocess
import unittest

from frozen_recipe_context import REVISION
from frozen_verifier_profile import FILES, adapt_sources


class FrozenVerifierProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = {name: subprocess.check_output(['git', 'show',
            f'{REVISION}:scripts/ci/{name}'], text=True) for name in FILES}

    def test_adapts_exact_historical_profile_without_touching_kernel_sources(self):
        result = adapt_sources(self.sources)
        self.assertEqual(set(result), set(FILES))
        self.assertIn("shared_flag != '1'", result['dspark_request_experiment.py'])
        self.assertIn("schedule = (('publication', True),)", result['dspark_request_experiment.py'])
        self.assertIn('from gdn_shared_qk_variants import validate_route',
            result['request_verifier_profile_report.py'])
        self.assertIn("report.get('combined_runtime_profile') is not True",
            result['request_verifier_profile_report.py'])
        self.assertIn('pp=None, committed_tg=None', result['dspark_request_experiment.py'])
        self.assertIn('--max-new-tokens 256', result['dspark-combined-profile.sh'])
        self.assertIn('timeout -k 30 720', result['dspark-combined-profile.sh'])

    def test_unknown_source_rejected(self):
        changed = dict(self.sources, **{'full_dspark_request.py': ''})
        with self.assertRaises(ValueError):
            adapt_sources(changed)


if __name__ == '__main__':
    unittest.main()
