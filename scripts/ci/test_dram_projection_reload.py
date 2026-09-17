from types import SimpleNamespace
import unittest

from dram_projection_reload import COMPUTE, matmul_inputs, preserve_fp32_partials


class DramProjectionReloadTests(unittest.TestCase):
    def test_no_bias_is_an_explicit_optional_slot(self):
        operations = SimpleNamespace(MatmulInputs=SimpleNamespace)
        inputs = matmul_inputs(operations, 'source', 'weight')
        self.assertEqual(inputs.input_tensors, ['source', 'weight'])
        self.assertEqual(inputs.optional_input_tensors, [None])

    def fixture(self):
        operations = SimpleNamespace(float32='float32',
            CBFormatDescriptor=lambda **kwargs: SimpleNamespace(data_format_as_uint8=0),
            UnpackToDestMode=SimpleNamespace(
            Default='default', UnpackToDestFp32='fp32'))
        config = SimpleNamespace(fp32_dest_acc_en=True, unpack_to_dest_mode=[])
        kernel = SimpleNamespace(kernel_source=COMPUTE, config=config)
        descriptor = SimpleNamespace(kernels=[kernel], cbs=[SimpleNamespace(format_descriptors=[
            SimpleNamespace(buffer_index=5, data_format_as_uint8=0)])])
        return operations, descriptor

    def test_only_partial_reload_mode_changes(self):
        operations, descriptor = self.fixture()
        self.assertIs(preserve_fp32_partials(operations, descriptor), descriptor)
        self.assertEqual(descriptor.kernels[0].config.unpack_to_dest_mode,
            ['default'] * 5 + ['fp32'] + ['default'] * 58)

    def test_unknown_source_rejected(self):
        operations, descriptor = self.fixture()
        descriptor.kernels[0].kernel_source = 'other'
        with self.assertRaises(ValueError):
            preserve_fp32_partials(operations, descriptor)

    def test_wrong_partial_format_rejected(self):
        operations, descriptor = self.fixture()
        descriptor.cbs[0].format_descriptors[0].data_format_as_uint8 = 1
        with self.assertRaises(ValueError):
            preserve_fp32_partials(operations, descriptor)
    def test_non_fp32_accumulation_rejected(self):
        operations, descriptor = self.fixture()
        descriptor.kernels[0].config.fp32_dest_acc_en = False
        with self.assertRaises(ValueError):
            preserve_fp32_partials(operations, descriptor)
