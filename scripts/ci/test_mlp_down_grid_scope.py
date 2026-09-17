from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mlp_down_grid_scope import scoped_down_grid
from mlp_down_grid_gate import REPORT_SHA256
from test_mlp_down_grid import DownGridTests


class DownScopeTests(unittest.TestCase):
    def test_only_t16_forward_changes_and_restores_even_on_failure(self):
        for failed in (False, True):
            program = DownGridTests().original()
            arguments = SimpleNamespace(mlp_w2_decode_1d_progcfg=program)
            weights = SimpleNamespace(w2=SimpleNamespace(dtype='BF8'))
            compute = SimpleNamespace(math_fidelity='LoFi', fp32_dest_acc_en=True, packer_l1_acc=True)
            mlp = SimpleNamespace(args=arguments, weights=weights, compute_kernel_config_decode=compute)
            operations = SimpleNamespace(bfloat8_b='BF8', MathFidelity=SimpleNamespace(LoFi='LoFi'),
                get_device_tensors=lambda weight: [SimpleNamespace(shape=(1, 1, 8704, 5120))] * 2,
                MatmulMultiCoreReuseMultiCast1DProgramConfig=SimpleNamespace)
            class Arm:
                def forward(self, index, value):
                    selected = self.originals[index][0].args.mlp_w2_decode_1d_progcfg
                    if value.shape[2] != 16:
                        if selected is not program:
                            raise AssertionError('Tail/native path changed')
                        return 'fallback'
                    if selected.per_core_N != 2 or selected.compute_with_storage_grid_size != (11, 8):
                        raise AssertionError('Candidate grid missing')
                    if failed:
                        raise RuntimeError('construction failed')
                    return 'result'
            arm = Arm()
            arm.operations, arm.originals = operations, [(mlp, None)] * 64
            original = Arm.forward
            with patch.dict('sys.modules', {'fused_t16_scope': SimpleNamespace(FusedT16Arm=Arm)}):
                try:
                    with scoped_down_grid(dict(report_sha256=REPORT_SHA256)) as audit:
                        self.assertEqual(arm.forward(0, SimpleNamespace(shape=(1, 1, 8, 5120))), 'fallback')
                        self.assertEqual(arm.forward(0, SimpleNamespace(shape=(1, 1, 16, 5120))), 'result')
                except RuntimeError:
                    self.assertTrue(failed)
            self.assertIs(Arm.forward, original)
            self.assertIs(arguments.mlp_w2_decode_1d_progcfg, program)
            self.assertTrue(audit['restored'])
            self.assertEqual(audit['hits'][0], 0 if failed else 1)

    def test_no_admission_no_override(self):
        module = SimpleNamespace(FusedT16Arm=SimpleNamespace(forward=lambda: None))
        with patch.dict('sys.modules', {'fused_t16_scope': module}):
            with self.assertRaises(ValueError):
                with scoped_down_grid({}):
                    self.fail('Unqualified scope entered')
