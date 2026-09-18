import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from draft_kv_slide_hardware_stage import stage


class PublicationStageTests(unittest.TestCase):
    def test_direct_dma_stages_only_explicit_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / 'scripts/ci'
            scripts.mkdir(parents=True)
            (scripts / 'run-dspark-hardware.sh').write_text('    -e "QWEN_MLP_BLOCK_STREAM_EXPERIMENT=1"\n')
            (scripts / 'dspark-hardware-suite.sh').write_text(
                '> /experiment/results/block-stream-preload-admission.json\nset +e\nload-weights\n')
            evidence = root / 'evidence'
            evidence.mkdir()
            manifest = root / 'manifest.json'
            with patch('draft_kv_slide_hardware_stage.qualify', return_value={'passed': True}):
                stage(root, evidence, manifest, direct_dma=True)
            self.assertEqual((scripts / 'draft_kv_slide.cpp').read_bytes(),
                Path(__file__).with_name('draft_kv_slide_direct.cpp').read_bytes())
            self.assertTrue(json.loads(manifest.read_text())['direct_dma'])

    def test_preload_precedes_model_loading_and_native_source_stays_intact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / 'scripts/ci'
            scripts.mkdir(parents=True)
            (scripts / 'draft_kv_history.py').write_text('native history unchanged')
            (scripts / 'mlp_block_stream.py').write_text('serial weight reader unchanged')
            (scripts / 'run-dspark-hardware.sh').write_text('    -e "QWEN_MLP_BLOCK_STREAM_EXPERIMENT=1"\n')
            (scripts / 'dspark-hardware-suite.sh').write_text(
                '> /experiment/results/block-stream-preload-admission.json\nset +e\nload-weights\n')
            evidence = root / 'evidence'
            evidence.mkdir()
            manifest = root / 'manifest.json'
            with patch('draft_kv_slide_hardware_stage.qualify', return_value={'passed': True}) as gate:
                stage(root, evidence, manifest)
                gate.assert_called_once_with(scripts, scripts / 'draft-kv-slide-evidence')
                self.assertEqual((scripts / 'draft_kv_history.py').read_text(), 'native history unchanged')
                self.assertEqual((scripts / 'mlp_block_stream.py').read_text(), 'serial weight reader unchanged')
                suite = (scripts / 'dspark-hardware-suite.sh').read_text()
                self.assertLess(suite.index('draft_kv_slide_gate.py'), suite.index('load-weights'))
                self.assertIn('QWEN_DRAFT_KV_SLIDE_EXPERIMENT=1', (scripts / 'run-dspark-hardware.sh').read_text())
                self.assertFalse(json.loads(manifest.read_text())['hardware_qualified'])
                with self.assertRaises(ValueError):
                    stage(root, evidence, manifest)
