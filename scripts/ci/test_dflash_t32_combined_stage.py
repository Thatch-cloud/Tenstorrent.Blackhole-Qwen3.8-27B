import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

from dflash_t32_combined_stage import stage


class T32CombinedStageTests(unittest.TestCase):
    def test_runtime_admission_does_not_import_simulator_stagers(self):
        command = (
            'import sys; '
            "sys.modules['gdn_direct_window_stage'] = None; "
            "sys.modules['mlp_down_grid_stage'] = None; "
            'import gdn_direct_window_t32_gate, mlp_down_grid_t32_gate'
        )
        subprocess.run([sys.executable, '-B', '-c', command],
            cwd=Path(__file__).parent, check=True, timeout=10)

    def test_stage_generates_all_components_and_stops_before_weights(self):
        self.check_stage(preflight_only=True)

    def test_full_comparison_keeps_preload_then_routes_loaded_model(self):
        self.check_stage(preflight_only=False)

    def check_stage(self, *, preflight_only):
        source = Path(__file__).parent
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary) / 'checkout'
            scripts = checkout / 'scripts/ci'
            scripts.mkdir(parents=True)
            names = ('dflash_device.py', 'draft_attention_branch.py', 'dflash_proposal_trace.py',
                'dflash-t16-native-attention-probe.py', 'dflash_t16_native_attention_gate.py',
                'gdn_shared_qk_program.py', 'gdn_shared_qk_pipeline.py', 'shared_qk_norm_scatter.py',
                'gdn_direct_window.py', 'gdn_direct_window_device.py', 'gdn_batched_conv.py')
            for name in names:
                (scripts / name).write_bytes((source / name).read_bytes())
            (scripts / 'simulator-suite.sh').write_text('unchanged simulator launcher')
            (scripts / 'dspark-target-hardware.py').write_text(
                'from mlp_block_stream_experiment import run_loaded_requests\n')
            (scripts / 'frozen_context_geometry.py').write_text('pinned context geometry')
            (scripts / 'run-dspark-hardware.sh').write_text('    -e "QWEN_MLP_BLOCK_STREAM_EXPERIMENT=1"\n')
            (scripts / 'dspark-hardware-suite.sh').write_text(
                '> /experiment/results/block-stream-preload-admission.json\nset +e\nload-model\n')
            evidence = Path(temporary) / 'evidence'
            for name in ('attention', 'cache', 'gdn', 'windows', 'down', 'stream'):
                (evidence / name).mkdir(parents=True)
            manifest = checkout / 'manifest.json'
            stage(checkout, evidence, manifest, preflight_only=preflight_only)
            report = json.loads(manifest.read_text())
            self.assertIs(report['preflight_only'], preflight_only)
            self.assertFalse(report['hardware_qualified'])
            self.assertEqual((scripts / 'fused_t16_scope.py').read_bytes(), (source / 'fused_t16_scope.py').read_bytes())
            self.assertEqual((scripts / 'frozen_context_geometry.py').read_text(), 'pinned context geometry')
            for name, expected in report['sources'].items():
                self.assertEqual(hashlib.sha256((scripts / name).read_bytes()).hexdigest(), expected)
            suite = (scripts / 'dspark-hardware-suite.sh').read_text()
            self.assertLess(suite.index('dflash_t32_preload.py'), suite.index('load-model'))
            if preflight_only:
                self.assertLess(suite.index('dflash_t32_preload.py'), suite.index('exit 0'))
                self.assertLess(suite.index('exit 0'), suite.index('load-model'))
            else:
                self.assertNotIn('exit 0', suite)
                self.assertIn('from dflash_t32_comparison_experiment import run_loaded_requests',
                    (scripts / 'dspark-target-hardware.py').read_text())
            self.assertIn('QWEN_T32_COMBINED_EXPERIMENT=1', (scripts / 'run-dspark-hardware.sh').read_text())
            self.assertIn('rows == 32', (scripts / 'gdn_direct_window_t32_hardware_batch.py').read_text())
            self.assertIn('(1, 32, 5120)', (scripts / 'gdn_shared_qk_t32_pipeline.py').read_text())
            self.assertIn('block_rows not in (8, 16, 32)', (scripts / 'dflash_device.py').read_text())
            with self.assertRaises(ValueError):
                stage(checkout, evidence, manifest)

    def test_workflow_pins_successful_retry_artifact(self):
        workflow = Path(__file__).resolve().parents[2] / '.github/workflows/qwen-cumulative-t16.yml'
        text = workflow.read_text()
        self.assertIn('actions/artifacts/10531081390/zip', text)
        self.assertIn('python3 -B -m zipfile -e "$evidence/t32-stream.zip" "$evidence/t32/stream"', text)
        self.assertNotIn('unzip ', text)
        self.assertIn('t32-preflight-block-stream-dflash-native', text)


if __name__ == '__main__':
    unittest.main()
