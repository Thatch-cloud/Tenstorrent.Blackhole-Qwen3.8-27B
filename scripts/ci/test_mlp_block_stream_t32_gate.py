import hashlib
from pathlib import Path
import tempfile
import unittest

from mlp_block_stream_gate import validate_report
from mlp_block_stream_t32_gate import CANDIDATE_SHA256, candidate_sources, qualify
from mlp_register_epilogue import adapt_projection
from mlp_block_stream_runtime import t32_hardware_projection
from test_mlp_block_stream_gate import BlockStreamGateTests


class T32StreamAdmissionTests(unittest.TestCase):
    def test_t16_and_t32_reports_are_not_interchangeable(self):
        report = BlockStreamGateTests().fixture()
        validate_report(report)
        with self.assertRaises(ValueError):
            validate_report(report, rows=32)
        for check in report['checks']:
            check['rows'] = 32
        report['trace_replays'][0]['rows'] = 32
        validate_report(report, rows=32)
        with self.assertRaises(ValueError):
            validate_report(report)

    def test_regenerated_projection_matches_executed_t32_hash(self):
        source = Path(__file__).parent
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            candidate = directory / 'mlp-register-epilogue-candidate'
            candidate.mkdir()
            (candidate / 'fused_1d.py').write_text(adapt_projection(
                (source / 'fused_1d.py').read_text(), nearest_away=True))
            for name in ('fused-batch-probe.py', 'mlp_block_stream_projection.py'):
                (directory / name).write_bytes((source / name).read_bytes())
            generated = candidate_sources(directory)
            self.assertEqual(hashlib.sha256(generated['fused_1d.py'].encode()).hexdigest(), CANDIDATE_SHA256)
            self.assertIn('token_rows != 32', generated['fused_1d.py'])
            self.assertIn('require_simulator()', generated['fused_1d.py'])
            hardware = t32_hardware_projection(generated['fused_1d.py'])
            restored = hardware.replace('from mlp_block_stream_projection import validate_binding',
                'from mlp_block_stream_t32_projection import validate_binding').replace(
                '        require_hardware(os.environ)\n',
                '        from mlp_block_stream_t32_stage import require_simulator\n        require_simulator()\n')
            self.assertEqual(restored, generated['fused_1d.py'])
            with self.assertRaises(ValueError):
                t32_hardware_projection(hardware)

    def test_report_pin_precedes_runtime_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / 'fused-batch.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Exact retained T32'):
                qualify(temporary, temporary, {})


if __name__ == '__main__':
    unittest.main()
