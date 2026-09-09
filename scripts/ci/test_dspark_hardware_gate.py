import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dspark_hardware_gate import REPORTS, simulator_preflight, require_compatible_native
from test_dspark_intake import configuration


ROOT = Path(__file__).parent


def dependency(name):
    spec = importlib.util.spec_from_file_location(name.replace('-','_'),ROOT/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DSparkHardwareGateTests(unittest.TestCase):
    def test_current_complete_component_evidence_does_not_claim_pipeline_or_numerical_qualification(self):
        result = simulator_preflight(ROOT)
        self.assertEqual(result['reports'],REPORTS)
        self.assertFalse(result['complete_pipeline_simulated'])
        self.assertFalse(result['retained_cpu_numerical_gate_passed'])
        self.assertIn('draft_dot_compute.cpp',result['arithmetic'])
        self.assertIn('dspark_layer.py',result['arithmetic'])

    def test_modified_arithmetic_rejects_even_with_passing_reports(self):
        from dspark_projection import digest

        with patch('dspark_hardware_gate.digest',side_effect=lambda path:'changed' if path.name=='dspark_layer.py' else digest(path)):
            with self.assertRaisesRegex(ValueError,'Changed arithmetic'):
                simulator_preflight(ROOT)

    def test_rebuilt_library_does_not_allow_unsimulated_kernel_changes(self):
        native = {'build_Release/lib/_ttnncpp.so':'new','build_Release/ttnn/_ttnncpp.so':'new','kernel.cpp':'same'}
        simulator = {**native,'build_Release/lib/_ttnncpp.so':'old','build_Release/ttnn/_ttnncpp.so':'old','simulator/library':'sim'}
        require_compatible_native(native,simulator)
        with self.assertRaises(ValueError):
            require_compatible_native({**native,'kernel.cpp':'changed'},simulator)
        with self.assertRaises(ValueError):
            require_compatible_native({**native,'build_Release/lib/_ttnncpp.so':'mismatched'},simulator)

    def test_complete_fixture_uses_pinned_patterns_hidden_shards_and_zero_padding(self):
        import torch

        module = dependency('dspark-pipeline-hardware')
        config = configuration()
        config['max_position_embeddings'] = 262144
        cases = module.fixtures(config)
        self.assertEqual(len(cases),3)
        self.assertEqual(len(module.PARAMETERS),58)
        for case in cases:
            self.assertEqual(len(case['values']),12)
            self.assertEqual(tuple(case['values']['feature_5'].shape),(2,1,32,2560))
            self.assertEqual(tuple(case['values']['noise'].shape),(1,1,32,5120))
            self.assertEqual(int(torch.count_nonzero(case['values']['noise'][:,:,7:])),0)
        self.assertTrue(torch.equal(cases[0]['values']['noise'],cases[1]['values']['noise']))
        self.assertFalse(torch.equal(cases[0]['values']['q_cos'],cases[1]['values']['q_cos']))
        self.assertFalse(torch.equal(cases[0]['values']['noise'],cases[2]['values']['noise']))

    def test_cached_config_is_verified_before_reuse_without_network(self):
        module = dependency('dspark-hardware-fixtures')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'model.safetensors').write_bytes(b'existing')
            (root/'config.json').write_text('{}')
            with patch.object(module,'verify',return_value={}),patch.object(module.urllib.request,'urlopen') as network:
                with self.assertRaisesRegex(ValueError,'Cached DSpark config'):
                    module.prepare(root)
                network.assert_not_called()


if __name__=='__main__':
    unittest.main()
