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

    def test_composes_with_runtime_admission_and_shells_parse(self):
        import sys
        from frozen_runtime_context import FILES as RUNTIME_FILES, adapt_runtime_sources
        from frozen_combined_adapters import FILES as COMBINED_FILES, adapt_combined_sources
        names = set(FILES + RUNTIME_FILES + COMBINED_FILES)
        sources = {name: subprocess.check_output(['git', 'show',
            f'{REVISION}:scripts/ci/{name}'], text=True) for name in names}
        result = adapt_sources(adapt_combined_sources(adapt_runtime_sources(sources)))
        bash = 'C:/Program Files/Git/bin/bash.exe' if sys.platform == 'win32' else 'bash'
        for name in ('run-dspark-hardware.sh', 'dspark-hardware-suite.sh', 'dspark-combined-profile.sh'):
            subprocess.run([bash, '-n'], input=result[name], text=True, check=True, timeout=10)
        self.assertIn('export QWEN_GDN_SHARED_QK_EXPERIMENT=1', result['dspark-hardware-suite.sh'])
        self.assertIn('export QWEN_COMBINED_TRACE_PROFILE=1', result['dspark-hardware-suite.sh'])
        self.assertIn('context=request[\'length\']', result['request_verifier_profile_report.py'])


if __name__ == '__main__':
    unittest.main()
