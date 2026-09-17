from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cumulative_t16_stage as staging


class CumulativeStageTests(unittest.TestCase):
    def test_both_component_sets_stage_with_explicit_down_admission(self):
        source = Path(staging.__file__).parent
        for with_down in (False, True):
            with self.subTest(with_down=with_down), TemporaryDirectory() as temporary:
                root = Path(temporary)
                scripts = root / 'scripts/ci'
                scripts.mkdir(parents=True)
                for name in ('dspark_markov_device.py', 'dspark_markov_score_layout.py', 'dspark_score_layout.py',
                        'dspark_score_layout_io.cpp', 'dspark_score_layout_compute.cpp',
                        'attention_batch.py', 'gdn_multitoken_conv.py'):
                    shutil.copyfile(source / name, scripts / name)
                (scripts / 'dspark-target-hardware.py').write_text(
                    'def run():\n            from gdn_direct_window_experiment import run_loaded_requests\n')
                (scripts / 'run-dspark-hardware.sh').write_text('    -e "QWEN_DSPARK_MODE=$mode"\n')
                evidence = root / 'evidence'
                evidence.mkdir()
                (evidence / 'fixture.json').write_text('{}')
                with patch.object(staging, 'qualify', return_value={'compact': True}), \
                        patch.object(staging, 'qualify_down', return_value={'down': True}) as qualify_down:
                    result = staging.stage(root, evidence, root / 'manifest.json',
                        down_evidence=evidence if with_down else None,
                        native_root=root if with_down else None)
                self.assertEqual(qualify_down.call_count, 2 if with_down else 0)
                self.assertEqual('wider_mlp_down' in result['components'], with_down)
                self.assertFalse(result['hardware_qualified'])
                self.assertEqual((scripts / 'mlp-down-grid-evidence').exists(), with_down)
                self.assertIn('QWEN_CUMULATIVE_MLP_DOWN', (scripts / 'run-dspark-hardware.sh').read_text())
                self.assertIn('cumulative_t16_experiment', (scripts / 'dspark-target-hardware.py').read_text())
                self.assertEqual((scripts / 'mlp_down_grid.py').read_bytes(), (source / 'mlp_down_grid.py').read_bytes())

    def test_missing_native_admission_fails_before_writes(self):
        with TemporaryDirectory() as temporary, patch.object(staging, 'qualify', return_value={}):
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'supplied together'):
                staging.stage(root, root, root / 'manifest.json', down_evidence=root)
            self.assertFalse((root / 'scripts').exists())
