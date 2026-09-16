import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from frozen_mlp_wait_zones import instrument
from frozen_wait_zone_gate import qualify


@unittest.skipUnless(os.environ.get('QWEN_WAIT_GATE_EVIDENCE'), 'retained marker evidence optional')
class WaitGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        scripts = Path(__file__).parent
        for name in ('fused-t16-target-simulator.json', 'fused-t16-target-simulator.exit-status',
                'fusion_trace.py', 'fused_1d_input.cpp', 'fused_1d_weights.cpp'):
            shutil.copyfile(scripts / name, self.root / name)
        evidence = Path(os.environ['QWEN_WAIT_GATE_EVIDENCE'])
        shutil.copyfile(evidence / 'simulator/fused-batch.json', self.root / 'frozen-wait-zones-simulator.json')
        shutil.copyfile(evidence / 'marker-qualification.json', self.root / 'frozen-wait-zones-hardware.json')
        candidate = self.root / 'frozen-wait-zone-candidate'
        candidate.mkdir()
        (candidate / 'fused_1d.py').write_bytes((scripts / 'fused_1d.py').read_text().encode())
        for role in ('input', 'weights'):
            name = f'fused_1d_{role}.cpp'
            (candidate / name).write_bytes(instrument((scripts / name).read_text(), role).encode())

    def test_actual_simulator_and_hardware_admission(self):
        result = qualify(self.root)
        self.assertTrue(result['passed'])
        self.assertTrue(result['diagnostic_only'])
        self.assertFalse(result['performance_qualified'])

    def test_candidate_source_drift_rejected(self):
        path = self.root / 'frozen-wait-zone-candidate/fused_1d_weights.cpp'
        path.write_bytes(path.read_bytes() + b'\n')
        with self.assertRaises(ValueError):
            qualify(self.root)

    def test_incomplete_marker_report_rejected(self):
        path = self.root / 'frozen-wait-zones-hardware.json'
        report = json.loads(path.read_text())
        report['samples'].pop()
        path.write_text(json.dumps(report))
        with self.assertRaises(ValueError):
            qualify(self.root)


if __name__ == '__main__':
    unittest.main()
