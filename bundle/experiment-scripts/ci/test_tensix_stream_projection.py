from collections import defaultdict
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from tensix_projection_raw import raw_shape
from tensix_stream_projection import COMPUTE, compute_arguments, prepare_projection, validate_inputs
from tensix_weight_stream import stream_geometry


class TensixStreamProjectionTests(unittest.TestCase):
    def test_more_producers_keep_identical_native_compute_arguments(self):
        for name, blocks in (('gate', 20), ('up', 20), ('down', 34)):
            self.assertEqual(compute_arguments(stream_geometry(name, blocks, 16)),
                compute_arguments(stream_geometry(name, blocks, 8)))
        geometry = stream_geometry('down', 34, 16)
        geometry['mapping'][0][1].append(79)
        with self.assertRaises(ValueError):
            compute_arguments(geometry)

    def test_compute_arguments_keep_native_single_tile_k_loop(self):
        for projection, blocks, columns in (('gate', 5, 4), ('gate', 20, 4), ('up', 20, 4), ('down', 34, 2)):
            arguments = compute_arguments(stream_geometry(projection, blocks))
            self.assertEqual(arguments, [8, 1, 8, 8, 1, 8 * columns, columns, blocks, 1, 1,
                1, columns, columns, 1, columns, 0, 0, 0])
        geometry = stream_geometry('gate', 20)
        geometry['per_receiver'] = 7
        with self.assertRaises(ValueError):
            compute_arguments(geometry)

    def fixture(self):
        geometry = stream_geometry('gate', 20)
        activation = SimpleNamespace(shape=(1, 1, 8, 5120), dtype='bf16', layout='tile',
            memory_config=lambda: 'l1', buffer_address=lambda: 1024)
        weight = SimpleNamespace(shape=(1, 1, 5120, 8704), dtype='bf4', layout='tile',
            memory_config=lambda: 'dram', buffer_address=lambda: 2048)
        output = SimpleNamespace(shape=(1, 1, 8, 8704), dtype='bf16', layout='tile',
            memory_config=lambda: 'l1', buffer_address=lambda: 4096)
        mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
        operations = SimpleNamespace(bfloat16='bf16', bfloat4_b='bf4', L1_MEMORY_CONFIG='l1',
            DRAM_MEMORY_CONFIG='dram', TILE_LAYOUT='tile',
            get_device_tensors=Mock(side_effect=lambda value: [value, value]))
        return operations, mesh, activation, weight, output, geometry

    def test_native_t8_disjoint_inputs_validate_without_dispatch(self):
        operations, mesh, activation, weight, output, geometry = self.fixture()
        self.assertEqual(validate_inputs(operations, mesh, activation, weight, output, geometry),
            [[activation, activation], [weight, weight], [output, output]])

    def test_program_combines_disjoint_producers_and_zero_copy_native_consumers(self):
        operations, mesh, activation, weight, output, geometry = self.fixture()
        class Buffer(SimpleNamespace):
            def set_global_circular_buffer(self, gcb):
                self.gcb = gcb
        gcb = SimpleNamespace(sender_core_type=lambda: 'worker')
        operations.__dict__.update(CoreCoord=lambda column, row: SimpleNamespace(x=column, y=row),
            CoreRange=lambda begin, end: (begin.x, begin.y, end.x, end.y), CoreRangeSet=tuple,
            create_global_circular_buffer=Mock(return_value=gcb), CBDescriptor=Buffer,
            CBFormatDescriptor=lambda **values: SimpleNamespace(**values), Tile=tuple, TileDescriptor=lambda value: value,
            MeshProgramDescriptor=dict, RuntimeArgs=lambda: defaultdict(dict), float32='fp32',
            TensorAccessorArgs=lambda unused: SimpleNamespace(get_compile_time_args=lambda: [999]),
            DataMovementProcessor=SimpleNamespace(RISCV_0='risc0', RISCV_1='risc1'),
            NOC=SimpleNamespace(RISCV_0_default='noc0', RISCV_1_default='noc1'),
            KernelDescriptor=lambda **values: SimpleNamespace(**values),
            DataMovementConfigDescriptor=lambda **values: SimpleNamespace(**values),
            ComputeConfigDescriptor=lambda **values: SimpleNamespace(unpack_to_dest_mode=[], **values),
            MathFidelity=SimpleNamespace(LoFi='lofi'),
            UnpackToDestMode=SimpleNamespace(Default='default', UnpackToDestFp32='fp32'),
            MeshCoordinate=lambda row, column: (row, column), MeshCoordinateRange=lambda begin, end: (begin, end),
            ProgramDescriptor=lambda **values: SimpleNamespace(**values),
            SemaphoreDescriptor=lambda **values: SimpleNamespace(**values))
        activation.device = lambda: SimpleNamespace(worker_core_from_logical_core=lambda coordinate: coordinate)
        with tempfile.TemporaryDirectory() as directory:
            native = Path(directory) / COMPUTE
            native.parent.mkdir(parents=True)
            native.write_text('fixture')
            prepared = prepare_projection(operations, mesh, activation, weight, output, geometry, Path(directory))
        self.assertIs(prepared.gcb, gcb)
        self.assertEqual(len(prepared.program), 2)
        self.assertEqual(operations.create_global_circular_buffer.call_args.args[2], 36864)
        for program in prepared.program.values():
            self.assertEqual(len(program.kernels), 5)
            compute = program.kernels[-1]
            self.assertTrue(Path(compute.kernel_source).as_posix().endswith(COMPUTE))
            self.assertEqual(compute.config.math_fidelity, 'lofi')
            self.assertIs(compute.config.math_approx_mode, True)
            self.assertIs(compute.config.fp32_dest_acc_en, True)
            self.assertEqual(compute.config.unpack_to_dest_mode[5], 'fp32')
            self.assertEqual(dict(compute.named_compile_time_args)['activation_type'], 4)
            self.assertEqual(dict(compute.defines)['PACKER_L1_ACC'], '1')
            aliases = [buffer for buffer in program.cbs
                if hasattr(buffer, 'gcb') and buffer.format_descriptors]
            self.assertEqual(len(aliases), 1)
            self.assertEqual(aliases[0].format_descriptors[0].buffer_index, 1)
            self.assertEqual(aliases[0].remote_format_descriptors[0].buffer_index, 31)
            self.assertEqual(aliases[0].remote_format_descriptors[0].page_size, 18432)
            producer = set(program.kernels[1].core_ranges)
            consumer = set(compute.core_ranges)
            self.assertEqual((len(producer), len(consumer)), (8, 68))
            self.assertFalse(producer & consumer)

    def test_shape_dtype_memory_grid_and_alias_drift_fail_closed(self):
        for mutation in ('activation_shape', 'activation_dtype', 'activation_memory', 'weight_shape',
                'weight_dtype', 'weight_memory', 'output_shape', 'output_dtype', 'output_memory', 'grid', 'shards', 'alias'):
            operations, mesh, activation, weight, output, geometry = self.fixture()
            if mutation == 'grid':
                mesh.compute_with_storage_grid_size = lambda: SimpleNamespace(x=12, y=10)
            elif mutation == 'shards':
                operations.get_device_tensors.side_effect = lambda value: [value]
            elif mutation == 'alias':
                output.buffer_address = activation.buffer_address
            else:
                name, field = mutation.split('_')
                value = dict(activation=activation, weight=weight, output=output)[name]
                if field == 'memory':
                    value.memory_config = lambda: 'invalid'
                else:
                    setattr(value, field, (1, 1, 16, 5120) if field == 'shape' else 'invalid')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_inputs(operations, mesh, activation, weight, output, geometry)

    def test_raw_readback_includes_all_physical_rows_and_compressed_metadata(self):
        self.assertEqual(raw_shape((1, 1, 32, 8704), 2048), (1, 1, 272, 512))
        self.assertEqual(raw_shape((1, 1, 5120, 8704), 576), (1, 1, 43520, 144))
        self.assertEqual(raw_shape((1, 1, 8704, 5120), 1088), (1, 1, 43520, 272))
        for shape, size in (((1, 1, 8, 8704), 2048), ((1, 2, 32, 5120), 2048),
                ((1, 1, 32, 5120), 512), ((1, 1, 32, 5120), 576.0)):
            with self.assertRaises(ValueError):
                raw_shape(shape, size)


if __name__ == '__main__':
    unittest.main()
