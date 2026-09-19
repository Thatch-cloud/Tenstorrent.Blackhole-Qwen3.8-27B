import json
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch

from mlp_register_epilogue_gate import COMPUTE, HARDWARE_PACKER, HELPERS, SIM_PACKER, expected_kernel, qualify, stage_candidate


class RegisterGateTests(unittest.TestCase):
    def test_simulator_to_physical_mapping_only_changes_known_packer_hash(self):
        baseline = dict(token_rows=16, fused_compute_sha256='control', reader_sha256={'input': 'reader'})
        simulated = expected_kernel(baseline)
        physical = expected_kernel(baseline, hardware=True)
        self.assertEqual(simulated['fused_compute_sha256'], COMPUTE)
        self.assertEqual(simulated['rounding_runtime']['packer_header_sha256'], SIM_PACKER)
        self.assertEqual(physical['rounding_runtime']['packer_header_sha256'], HARDWARE_PACKER)
        physical['rounding_runtime']['packer_header_sha256'] = SIM_PACKER
        self.assertEqual(simulated, physical)
        self.assertEqual(baseline['fused_compute_sha256'], 'control')
        self.assertFalse(simulated['rounding_runtime']['hardware_qualified'])

    def test_wrong_report_rejected_before_runtime_construction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'fused-batch.json').write_text(json.dumps(dict(passed=True)))
            with patch('mlp_register_epilogue_gate.runtime') as runtime:
                with self.assertRaisesRegex(ValueError, 'Exact reviewed'):
                    qualify(root, root, runtime_root=root)
                runtime.assert_not_called()

    def test_current_helpers_match_reviewed_simulator(self):
        from mlp_register_epilogue_gate import digest
        for name, expected in HELPERS.items():
            self.assertEqual(digest(Path(__file__).with_name(name)), expected)

    def test_retained_artifact_and_source_changes(self):
        from frozen_recipe_context import REVISION
        evidence = Path('D:/qwen-evidence/35238822290')
        if not evidence.exists():
            self.skipTest('Retained simulator artifact unavailable')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('fused_1d.py', 'fused_1d_input.cpp', 'fused_1d_weights.cpp', 'fusion_trace.py',
                    'fused-t16-target-simulator.json', 'fused-t16-target-simulator.exit-status'):
                payload = subprocess.check_output(['git', 'show', f'{REVISION}:scripts/ci/{name}'])
                (root / name).write_bytes(payload)
            for name in HELPERS:
                (root / name).write_bytes(Path(__file__).with_name(name).read_bytes())
            candidate = stage_candidate(root)
            admitted = qualify(root, evidence)
            self.assertTrue(admitted['passed'])
            self.assertFalse(admitted['hardware_qualified'])
            hardware = dict(admitted['kernels'][0]['rounding_runtime'], packer_header_sha256=HARDWARE_PACKER)
            with patch('mlp_register_epilogue_gate.runtime', return_value=hardware):
                self.assertEqual(qualify(root, evidence, runtime_root='pinned')['kernels'][0]['rounding_runtime'], hardware)
            with patch('mlp_register_epilogue_gate.runtime', return_value=dict(hardware, packer_header_sha256=SIM_PACKER)):
                with self.assertRaisesRegex(ValueError, 'Pinned physical'):
                    qualify(root, evidence, runtime_root='pinned')
            path = candidate / 'fused_1d_weights.cpp'
            path.write_bytes(path.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'Unchanged candidate reader'):
                qualify(root, evidence)


if __name__ == '__main__':
    unittest.main()
