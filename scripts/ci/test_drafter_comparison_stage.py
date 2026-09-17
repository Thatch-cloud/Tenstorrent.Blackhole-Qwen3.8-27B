import json
from pathlib import Path
import tempfile
import unittest

from drafter_comparison_stage import stage


class ComparisonStageTests(unittest.TestCase):
    def test_overlay_keeps_control_and_copies_only_pinned_cached_fixtures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / 'scripts/ci'
            scripts.mkdir(parents=True)
            for name in ('compact-score-evidence', 'mlp-down-grid-evidence', 'register-epilogue-evidence'):
                (scripts / name).mkdir()
            (scripts / 'dspark-target-hardware.py').write_text('from cumulative_t16_experiment import run_loaded_requests\n')
            (scripts / 'run-dspark-hardware.sh').write_text(
                '    -e "QWEN_DSPARK_MODE=$mode"\ndocker start -a "$test_id" | tee "$output/dspark-console.log"\n')
            manifest = root / 'manifest.json'
            result = stage(root, manifest)
            self.assertEqual(json.loads(manifest.read_text()), result)
            shell = (scripts / 'run-dspark-hardware.sh').read_text()
            self.assertIn('QWEN_DRAFTER_COMPARISON=1', shell)
            self.assertIn('copy_dflash_fixtures', shell)
            self.assertNotIn('prepare_dflash_fixtures', shell)
            self.assertLess(shell.index('copy_dflash_fixtures'), shell.index('docker start'))
            self.assertFalse(result['hardware_qualified'])
            with self.assertRaises(ValueError):
                stage(root, manifest)

    def test_missing_target_evidence_fails_before_overlay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'component evidence'):
                stage(root, root / 'manifest.json')
