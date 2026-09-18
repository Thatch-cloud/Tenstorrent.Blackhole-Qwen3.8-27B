from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mlp_down_grid_t32_scope import scoped_down_grid_t32
from mlp_down_grid_t32_gate import REPORT_SHA256
from test_mlp_down_grid import DownGridTests
from test_mlp_down_grid_gate import fixture
from mlp_down_grid_gate import validate_report


class T32DownScopeTests(unittest.TestCase):
    def test_width_admission_is_explicit_and_t16_default_unchanged(self):
        report = fixture()
        validate_report(report)
        with self.assertRaises(ValueError):
            validate_report(report, rows=32)
        report['rows'] = 32
        validate_report(report, rows=32)
        with self.assertRaises(ValueError):
            validate_report(report)
        for rows in (True, 32.0, 8):
            with self.assertRaises(ValueError):
                validate_report(report, rows=rows)

    def test_only_t32_changes_and_restores_on_failure(self):
        for failed in (False, True):
            program = DownGridTests().original()
            arguments = SimpleNamespace(mlp_w2_decode_1d_progcfg=program)
            mlp = SimpleNamespace(args=arguments, weights=SimpleNamespace(w2=SimpleNamespace(dtype='BF8')),
                compute_kernel_config_decode=SimpleNamespace(math_fidelity='LoFi', fp32_dest_acc_en=True, packer_l1_acc=True))
            operations = SimpleNamespace(bfloat8_b='BF8', MathFidelity=SimpleNamespace(LoFi='LoFi'),
                get_device_tensors=lambda weight: [SimpleNamespace(shape=(1, 1, 8704, 5120))] * 2,
                MatmulMultiCoreReuseMultiCast1DProgramConfig=SimpleNamespace)

            class Base:
                token_rows = 16

                def forward(self, index, value):
                    selected = self.originals[index][0].args.mlp_w2_decode_1d_progcfg
                    if value.shape[2] != 32:
                        if selected is not program:
                            raise AssertionError('Native tail changed')
                        return 'tail'
                    if selected.compute_with_storage_grid_size != (11, 8):
                        raise AssertionError('T32 grid missing')
                    if failed:
                        raise RuntimeError('construction failed')
                    return 'result'

            class Arm(Base):
                token_rows = 32

            arm = Arm()
            arm.operations, arm.originals = operations, [(mlp, None)] * 64
            original = Base.forward
            admission = dict(report_sha256=REPORT_SHA256, rows=32, simulator_qualified=True)
            with patch.dict('sys.modules', {'fused_t16_scope': SimpleNamespace(FusedT16Arm=Base, FusedT32Arm=Arm)}):
                try:
                    with scoped_down_grid_t32(admission) as audit:
                        self.assertIs(Base.forward, original)
                        self.assertEqual(arm.forward(0, SimpleNamespace(shape=(1, 1, 16, 5120))), 'tail')
                        self.assertEqual(arm.forward(0, SimpleNamespace(shape=(1, 1, 32, 5120))), 'result')
                except RuntimeError:
                    self.assertTrue(failed)
            self.assertNotIn('forward', Arm.__dict__)
            self.assertIs(Base.forward, original)
            self.assertIs(arguments.mlp_w2_decode_1d_progcfg, program)
            self.assertTrue(audit['restored'])
            self.assertEqual(audit['hits'][0], 0 if failed else 1)
            self.assertEqual(audit['fallbacks'], 1)


if __name__ == '__main__':
    unittest.main()
