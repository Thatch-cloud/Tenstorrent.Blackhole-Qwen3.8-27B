import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlp_block_stream_hardware_stage import stage


class HardwareStageTests(unittest.TestCase):
    def test_explicit_flag_preload_gate_and_complete_driver(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / 'scripts/ci'
            scripts.mkdir(parents=True)
            (scripts / 'dspark-target-hardware.py').write_text('from dflash_native_comparison_experiment import run_loaded_requests\n')
            (scripts / 'run-dspark-hardware.sh').write_text('docker create \\\n    -e "QWEN_DFLASH_NATIVE_COMPARISON=1" image\n')
            (scripts / 'dspark-hardware-suite.sh').write_text('native-check > /experiment/results/dflash-native-preload-admission.json\nset +e\nload-weights\n')
            evidence = root / 'evidence'
            evidence.mkdir()
            (evidence / 'fused-batch.json').write_text('{}')
            manifest = root / 'manifest.json'
            with patch('mlp_block_stream_hardware_stage.qualify_register', return_value={'register': True}), \
                    patch('mlp_block_stream_hardware_stage.qualify', return_value={'passed': True}):
                stage(root, evidence, root, manifest)
                self.assertIn('from mlp_block_stream_experiment', (scripts / 'dspark-target-hardware.py').read_text())
                shell = (scripts / 'run-dspark-hardware.sh').read_text()
                self.assertIn('\n    -e "QWEN_MLP_BLOCK_STREAM_EXPERIMENT=1"', shell)
                self.assertNotIn('\n+', shell)
                suite = (scripts / 'dspark-hardware-suite.sh').read_text()
                self.assertLess(suite.index('mlp_block_stream_preload.py'), suite.index('load-weights'))
                self.assertFalse(json.loads(manifest.read_text())['hardware_qualified'])
                with self.assertRaises(ValueError):
                    stage(root, evidence, root, manifest)


if __name__ == '__main__':
    unittest.main()
