from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from draft_convolution_fused import fused_convolution, checked_convolution


class FusedConvolutionTests(unittest.TestCase):
    def test_checked_runtime_audits_composed_values_without_changing_output_ownership(self):
        for audit, wrong in ((False, False), (True, False), (True, True)):
            output = torch.ones((1, 1, 32, 32), dtype=torch.bfloat16)
            control = output.clone() + int(wrong)
            operations = SimpleNamespace(get_device_tensors=lambda value: [value, value], to_torch=Mock(side_effect=lambda value: value),
                deallocate=Mock())
            owned, checks = [], []
            with patch('draft_convolution_fused.fused_convolution', return_value=output), \
                    patch('draft_convolution.grouped_causal_convolution', return_value=control) as composed:
                arguments = dict(fp32_intermediates=True, retain_temporaries=owned.append, audit=audit,
                    checks=checks, context=dict(position=170, layer=4))
                if wrong:
                    with self.assertRaisesRegex(AssertionError, 'composed'):
                        checked_convolution(operations, None, output, [], [], **arguments)
                else:
                    self.assertIs(checked_convolution(operations, None, output, [], [], **arguments), output)
                self.assertEqual(len(owned), 1)
                self.assertIs(owned[0], output)
                if audit:
                    self.assertEqual(len(checks), 1 if wrong else 2)
                    self.assertIs(checks[0]['exact'], not wrong)
                    operations.deallocate.assert_called_once_with(control)
                else:
                    composed.assert_not_called()
                    operations.to_torch.assert_not_called()
                    operations.deallocate.assert_not_called()

    def test_checked_runtime_rejects_unowned_or_inexact_policy_before_dispatch(self):
        for arguments in ({}, dict(fp32_intermediates=True), dict(fp32_intermediates=1, retain_temporaries=lambda value: None),
                dict(fp32_intermediates=True, retain_temporaries=lambda value: None, audit=True)):
            with patch('draft_convolution_fused.fused_convolution') as operation, self.assertRaises(ValueError):
                checked_convolution(None, None, None, [], [], **arguments)
            operation.assert_not_called()

    def fixture(self, rows=32):
        shapes = [(1, 1, rows, 5120)] + [(1, 1, rows, 320)] * 2 + [(1, 1, 1, 5120)] * 2

        def tensor(shape, index):
            return SimpleNamespace(shape=shape, dtype='bf16', layout='tile', memory_config=lambda: 'dram',
                shards=[SimpleNamespace(buffer_address=lambda chip=chip: 4096 * index + 1048576 * chip) for chip in range(2)])

        values = [tensor(shape, index + 1) for index, shape in enumerate(shapes)]
        output = tensor(shapes[0], 6)
        operations = SimpleNamespace(bfloat16='bf16', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram',
            get_device_tensors=lambda value: value.shards, empty=Mock(return_value=output), deallocate=Mock(),
            CoreCoord=lambda *args: args, CoreRange=lambda *args: args, CoreRangeSet=lambda args: args,
            Tile=lambda args: args, TileDescriptor=lambda args: args, CBDescriptor=lambda **kwargs: kwargs,
            CBFormatDescriptor=lambda **kwargs: kwargs, KernelDescriptor=lambda **kwargs: kwargs,
            ComputeConfigDescriptor=lambda **kwargs: kwargs, DataMovementConfigDescriptor=lambda **kwargs: kwargs,
            MathFidelity=SimpleNamespace(HiFi4='hifi4'), MeshProgramDescriptor=dict, ProgramDescriptor=lambda **kwargs: kwargs,
            RuntimeArgs=lambda: [[None] * 10 for index in range(8)], MeshCoordinate=lambda *args: args,
            MeshCoordinateRange=lambda *args: args, DataMovementProcessor=SimpleNamespace(RISCV_0=0),
            NOC=SimpleNamespace(RISCV_0_default=0), generic_op=Mock(),
            TensorAccessorArgs=lambda value: SimpleNamespace(get_compile_time_args=lambda: [0]))
        return operations, SimpleNamespace(shape=[1, 2]), values, output

    def test_single_dispatch_two_chips_eighty_workers_and_exact_math(self):
        for rows in (1, 8, 32):
            operations, mesh, values, output = self.fixture(rows)
            self.assertIs(fused_convolution(operations, mesh, values[0], values[1:3], values[3:]), output)
            operations.generic_op.assert_called_once()
            tensors, program = operations.generic_op.call_args.args
            self.assertEqual(tensors, [*values, output])
            self.assertEqual(len(program), 2)
            for chip, descriptor in enumerate(program.values()):
                reader, compute = descriptor['kernels']
                self.assertEqual(compute['compile_time_args'], [2])
                self.assertEqual(compute['config'], dict(math_fidelity='hifi4', fp32_dest_acc_en=True, math_approx_mode=False))
                for worker in range(80):
                    self.assertEqual(reader['runtime_args'][worker % 8][worker // 8],
                        [value.shards[chip].buffer_address() for value in [*values, output]] + [rows, worker])
            operations.deallocate.assert_not_called()

    def test_invalid_geometry_dtype_layout_or_placement_precedes_allocation(self):
        for attribute, invalid in (('shape', (1, 1, 4, 5120)), ('dtype', 'fp32'), ('layout', 'row-major'),
                ('memory_config', lambda: 'l1'), ('shards', [])):
            operations, mesh, values, output = self.fixture()
            setattr(values[0], attribute, invalid)
            with self.subTest(attribute=attribute), self.assertRaises(ValueError):
                fused_convolution(operations, mesh, values[0], values[1:3], values[3:])
            operations.empty.assert_not_called()

    def test_failed_submission_or_alias_releases_only_owned_output(self):
        for alias in (False, True):
            operations, mesh, values, output = self.fixture()
            if alias:
                output.shards = values[0].shards
            else:
                operations.generic_op.side_effect = RuntimeError('submission')
            with self.subTest(alias=alias), self.assertRaises((ValueError, RuntimeError)):
                fused_convolution(operations, mesh, values[0], values[1:3], values[3:])
            operations.deallocate.assert_called_once_with(output)


if __name__ == '__main__':
    unittest.main()
