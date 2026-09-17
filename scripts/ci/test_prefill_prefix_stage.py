import ast
from pathlib import Path
import subprocess
import tempfile
import unittest

from prefill_prefix_stage import adapt_full_request, function_source, stage


ROOT = Path(__file__).resolve().parents[2]
REVISION = '8c102b20df22329106955b4006bf4d650bb94e40'


def original(name):
    return subprocess.check_output(['git', 'show', f'{REVISION}:scripts/ci/{name}'],
        cwd=ROOT, text=True, encoding='utf-8')


class PrefixStageTests(unittest.TestCase):
    def test_historical_request_imports_only_cache_hook_not_t32(self):
        source = original('full_dspark_request.py')
        tested = Path(__file__).with_name('full_dspark_request.py').read_text()
        result = adapt_full_request(source, tested)
        self.assertNotIn('t32', result)
        self.assertIn('cached_prefill_factory=None', result)
        for name in ('factory', 'gold_decode', 'validate_features'):
            self.assertEqual(function_source(source, name), function_source(result, name))
        self.assertEqual(function_source(result, 'captured_prefill'), function_source(tested, 'captured_prefill'))
        compile(result, 'historical-request', 'exec')
        with self.assertRaises(ValueError):
            adapt_full_request(result, tested)

    def test_stage_records_source_delta_and_requires_ladder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / 'scripts/ci'
            scripts.mkdir(parents=True)
            for name in ('full_dspark_request.py', 'dspark_request_experiment.py',
                    'dspark-target-hardware.py', 'run-dspark-hardware.sh'):
                (scripts / name).write_text(original(name))
            manifest = root / 'manifest.json'
            with self.assertRaisesRegex(ValueError, 'ladder'):
                stage(root, manifest)
            experiment = (scripts / 'dspark_request_experiment.py').read_text()
            boundary = experiment.index('    if profile_verifier or profile_drafter or combined_profile:',
                experiment.index("            progress(f'full_request_{ordinal}_complete')"))
            experiment = experiment[:boundary] + '    from frozen_ladder_requests import finish\n    finish(report, summarize)\n'
            (scripts / 'dspark_request_experiment.py').write_text(experiment)
            stage(root, manifest)
            import json

            record = json.loads(manifest.read_text())
            self.assertFalse(record['hardware_qualified'])
            self.assertFalse(record['serving_enabled'])
            self.assertEqual(record['context'], 4096)
            staged = (scripts / 'dspark_request_experiment.py').read_text()
            before = ast.dump(ast.parse(function_source(experiment, 'run_loaded_requests')))
            after = ast.dump(ast.parse(function_source(staged, 'run_loaded_requests')))
            self.assertEqual(before, after)
            self.assertIn('prefill_prefix_session.py', record['after'])
            self.assertIn('QWEN_PREFIX_CACHE_EXPERIMENT:-0', (scripts / 'run-dspark-hardware.sh').read_text())
            with self.assertRaisesRegex(ValueError, 'Fresh'):
                stage(root, manifest)


if __name__ == '__main__':
    unittest.main()
