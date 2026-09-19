from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cumulative_t16_stage as staging


class CumulativeStageTests(unittest.TestCase):
    def test_both_component_sets_stage_with_explicit_down_admission(self):
        source = Path(staging.__file__).parent
        for with_down, with_norm, with_register in ((False, False, False), (True, False, False),
                (False, True, False), (True, True, False), (True, True, True)):
            with self.subTest(with_down=with_down, with_norm=with_norm, with_register=with_register), TemporaryDirectory() as temporary:
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
                (scripts / 'frozen_draft_tail_scope.py').write_text(
                    'from frozen_gdn_norm_scope import runtime_scope as norm_scope\n')
                evidence = root / 'evidence'
                evidence.mkdir()
                (evidence / 'fixture.json').write_text('{}')

                def register_payloads(directory):
                    for name in staging.REGISTER_PAYLOADS:
                        path = directory / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text('fixture')

                with patch.object(staging, 'qualify', return_value={'compact': True}), \
                        patch.object(staging, 'stage_register', side_effect=register_payloads) as register_stage, \
                        patch.object(staging, 'qualify_register', return_value={'register': True}) as register_gate, \
                        patch.object(staging, 'qualify_prefetch') as prefetch, \
                        patch.object(staging, 'qualify_scatter', return_value={'scatter': True}) as scatter, \
                        patch.object(staging, 'qualify_down', return_value={'down': True}) as qualify_down:
                    result = staging.stage(root, evidence, root / 'manifest.json',
                        down_evidence=evidence if with_down else None,
                        native_root=root if with_down or with_norm else None,
                        norm_report=evidence / 'fixture.json' if with_norm else None,
                        register_evidence=evidence if with_register else None)
                self.assertEqual(register_stage.call_count, int(with_register))
                self.assertEqual(register_gate.call_count, int(with_register))
                self.assertEqual('register_epilogue' in result['components'], with_register)
                self.assertEqual((scripts / 'register-epilogue-evidence').exists(), with_register)
                self.assertEqual(all(name in result['after'] for name in staging.REGISTER_PAYLOADS), with_register)
                self.assertEqual(qualify_down.call_count, 2 if with_down else 0)
                self.assertEqual(prefetch.call_count, 2 if with_norm else 0)
                self.assertEqual(scatter.call_count, 2 if with_norm else 0)
                self.assertEqual('norm_scatter' in result['components'], with_norm)
                self.assertEqual('selected_runtime_scope' in
                    (scripts / 'frozen_draft_tail_scope.py').read_text(), with_norm)
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

    def test_workflow_uses_the_downloaded_inventory_subdirectory(self):
        workflow = Path(staging.__file__).resolve().parents[2] / '.github/workflows/qwen-cumulative-t16.yml'
        source = workflow.read_text()
        native = '"$evidence/model-native/prefill-cache/sources"'
        self.assertIn('--native-root ' + native, source)
        self.assertIn('--destination ' + native, source)
        self.assertNotIn('"$evidence/model-native/sources"', source)
