import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlp_progressive_input_hardware_stage import FILES, stage


class ProgressiveStageTests(unittest.TestCase):
    def test_explicit_overlay_preloads_before_model_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / 'scripts/ci'
            scripts.mkdir(parents=True)
            (scripts / 'draft-kv-slide-evidence').mkdir()
            (scripts / 'run-dspark-hardware.sh').write_text('    -e "QWEN_DRAFT_KV_SLIDE_EXPERIMENT=1"\n')
            (scripts / 'dspark-hardware-suite.sh').write_text(
                '> /experiment/results/draft-kv-slide-preload-admission.json\nset +e\n')
            evidence = root / 'evidence'
            evidence.mkdir()
            (evidence / 'retained.json').write_text('{}')
            manifest = root / 'stage.json'
            with patch('mlp_progressive_input_hardware_stage.preload', return_value={'passed': True}) as admission:
                stage(root, evidence, 'native', manifest)
            admission.assert_called_once_with(scripts, 'native')
            record = json.loads(manifest.read_text())
            self.assertFalse(record['hardware_qualified'])
            self.assertTrue(set(FILES).issubset(record['sources']))
            self.assertTrue((scripts / 'progressive-input-evidence/retained.json').is_file())
            self.assertIn('QWEN_PROGRESSIVE_INPUT_EXPERIMENT=1', (scripts / 'run-dspark-hardware.sh').read_text())
            suite = (scripts / 'dspark-hardware-suite.sh').read_text()
            self.assertLess(suite.index('mlp_progressive_input_preload.py'), suite.index('set +e'))
            with self.assertRaises(ValueError):
                stage(root, evidence, 'native', manifest)

    def test_missing_publication_recipe_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'direct KV'):
                stage(root, root, root, root / 'stage.json')


if __name__ == '__main__':
    unittest.main()
