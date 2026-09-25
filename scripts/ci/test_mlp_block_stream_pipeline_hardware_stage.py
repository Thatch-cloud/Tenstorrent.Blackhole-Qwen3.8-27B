import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlp_block_stream_pipeline_hardware_stage import stage


class PipelineHardwareStageTests(unittest.TestCase):
    def test_admission_before_weights_and_serial_sources_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / 'scripts/ci'
            scripts.mkdir(parents=True)
            (scripts / 'mlp_block_stream.py').write_text('serial control unchanged')
            (scripts / 'run-dspark-hardware.sh').write_text('    -e "QWEN_MLP_BLOCK_STREAM_EXPERIMENT=1"\n')
            (scripts / 'dspark-hardware-suite.sh').write_text(
                '> /experiment/results/block-stream-preload-admission.json\nset +e\nload-weights\n')
            evidence = root / 'evidence'
            evidence.mkdir()
            manifest = root / 'manifest.json'
            with patch('mlp_block_stream_pipeline_hardware_stage.preload', return_value={'passed': True}) as admit:
                stage(root, evidence, root, manifest)
                admit.assert_called_once_with(scripts, root)
                self.assertEqual((scripts / 'mlp_block_stream.py').read_text(), 'serial control unchanged')
                self.assertIn('QWEN_BULK_PIPELINE_EXPERIMENT=1', (scripts / 'run-dspark-hardware.sh').read_text())
                suite = (scripts / 'dspark-hardware-suite.sh').read_text()
                self.assertLess(suite.index('mlp_block_stream_pipeline_preload.py'), suite.index('load-weights'))
                self.assertFalse(json.loads(manifest.read_text())['hardware_qualified'])
                with self.assertRaises(ValueError):
                    stage(root, evidence, root, manifest)
