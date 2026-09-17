from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class CombinedCliTests(unittest.TestCase):
    def test_preflight_validates_installation_without_loading_weights(self):
        path = Path(__file__).with_name('t32-combined-hardware.py')
        spec = importlib.util.spec_from_file_location('combined_cli_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        metadata_spec = importlib.util.spec_from_file_location('target_cli_metadata', path.with_name('dspark-target-hardware.py'))
        metadata = importlib.util.module_from_spec(metadata_spec)
        metadata_spec.loader.exec_module(metadata)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'preflight.json'
            target = Path(temporary) / '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
            arguments = ['probe', '--preflight', '--checkpoint', 'weights', '--config', 'config.json',
                '--target', str(target), '--output', str(output), '--proposal-evidence', 'proposal.json',
                '--score-evidence', 'score.json', '--attention-evidence', 'attention.json', '--commit-evidence', 'commit']
            def digest(value):
                if str(value) == 'config.json':
                    return module.FILES['config.json'][1]
                return metadata.TARGET_SOURCES[Path(value).relative_to('/runtime').as_posix()]
            with patch('sys.argv', arguments), patch.dict('os.environ', TT_METAL_HOME='/runtime'), \
                    patch.object(module, 'environment'), patch.object(module, 'require_projection_environment'), \
                    patch.object(module, 'digest', side_effect=digest), \
                    patch.object(module, 'composition_audit', return_value={'component': True}), \
                    patch.object(module, 'qualify_request', return_value={'attention': True}), \
                    patch('fused_t16_admission.qualify_simulator', return_value={'mlp': True}), \
                    patch('t32_commit_gate.qualify', return_value={'prefixes': 33}), \
                    patch('t32_hardware_kernel.installed', return_value=nullcontext({'runtime': True})) as installed, \
                    patch.object(module, 'run_request') as loaded:
                module.main()
                self.assertTrue(json.loads(output.read_text())['runtime_preflight']['runtime'])
                installed.assert_called_once()
                loaded.assert_not_called()
                with self.assertRaisesRegex(ValueError, 'Fresh output'):
                    module.main()
