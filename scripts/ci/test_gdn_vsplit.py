"""Host-only mapping, source contract, CB and mocked orchestration tests."""

from collections import defaultdict
import hashlib
import os
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gdn_multitoken as native
import gdn_vsplit as split


ROOT = Path(os.environ.get('GDN_VSPLIT_SOURCE_ROOT', str(
    Path(__file__).resolve().parents[2] / 'hardware-evidence.local/34009341359'
    / 'qwen-hardware-inventory-34009341359/gdn-source')))


class MappingTests(unittest.TestCase):
    def test_state_mapping_covers_every_physical_tile_once(self):
        for token in range(32):
            pages = [split.state_page(token, worker, key_tile)
                     for worker in range(96) for key_tile in range(4)]
            self.assertEqual(sorted(pages), list(range(token * 384, (token + 1) * 384)))
            for head in range(24):
                for partition in range(4):
                    for key_tile in range(4):
                        self.assertEqual(split.state_page(token, head * 4 + partition, key_tile),
                                         token * 384 + head * 16 + key_tile * 4 + partition)

    def test_packed_qk_repeat_twelve_times_and_scalar_repeats_four(self):
        for worker in range(96):
            pages = split.packed_pages(worker)
            head, partition = divmod(worker, 4)
            self.assertEqual(pages['q'], (head // 3) * 4)
            self.assertEqual(pages['k'], 32 + (head // 3) * 4)
            self.assertEqual(pages['v'], 64 + head * 4 + partition)
            self.assertEqual(pages['scalar'], head)
            self.assertLess(pages['q'] + 3, 32)
            self.assertLess(pages['k'] + 3, 64)
            self.assertLess(pages['v'], 160)

    def test_fp32_bridge_gather_preserves_all_bits_and_partition_order(self):
        for token in (0, 1, 15, 16, 31):
            for head in range(24):
                reconstructed = []
                for partition in range(4):
                    page = split.bridge_page(token, head, partition)
                    self.assertEqual(page, token * 96 + head * 4 + partition)
                    words = [0x3f800001 + page * 32 + column for column in range(32)]
                    stick = struct.pack('<32I', *words)
                    self.assertEqual(len(stick), 128)
                    tile = bytearray(4096)
                    tile[:64] = stick[:64]
                    tile[1024:1088] = stick[64:]
                    self.assertEqual(tile[64:1024], bytes(960))
                    self.assertEqual(tile[1088:], bytes(3008))
                    reconstructed += list(struct.unpack('<16I', tile[:64]))
                    reconstructed += list(struct.unpack('<16I', tile[1024:1088]))
                self.assertEqual(reconstructed,
                    [0x3f800001 + (token * 96 + head * 4) * 32 + column for column in range(128)])
                self.assertTrue(any(word & 0xffff for word in reconstructed))

    def test_output_assembly_face_boundaries_and_padding(self):
        for rows in (1, 2, 4, 8, 16, 32):
            tiles = [[0] * 1024 for tile in range(96)]
            seen = set()
            for token in range(rows):
                for head in range(24):
                    for column in range(128):
                        page, element = split.output_element(token, head, column)
                        self.assertNotIn((page, element), seen)
                        seen.add((page, element))
                        tiles[page][element] = 1 + token * 3072 + head * 128 + column
            self.assertEqual(len(seen), rows * 3072)
            for token in range(32):
                for head in range(24):
                    for column in range(128):
                        page = head * 4 + column // 32
                        face = 2 * (token // 16) + (column % 32) // 16
                        element = face * 256 + (token % 16) * 16 + column % 16
                        expected = 1 + token * 3072 + head * 128 + column if token < rows else 0
                        self.assertEqual(tiles[page][element], expected)

    def test_bounds_and_grid(self):
        for args in ((32, 0, 0), (-1, 0, 0), (0, 96, 0), (0, 0, 4)):
            with self.assertRaises(ValueError):
                split.state_page(*args)
        for args in ((32, 0, 0), (0, 24, 0), (0, 0, 4)):
            with self.assertRaises(ValueError):
                split.bridge_page(*args)
        for args in ((32, 0, 0), (0, 24, 0), (0, 0, 128)):
            with self.assertRaises(ValueError):
                split.output_element(*args)
        with self.assertRaises(ValueError):
            split.packed_pages(96)
        self.assertEqual(len(set(split.core_coordinates(12, 10, 96))), 96)
        for geometry in ((8, 8, 96), (0, 10, 96)):
            with self.assertRaises(ValueError):
                split.core_coordinates(*geometry)


class BufferAndAbiTests(unittest.TestCase):
    def test_recurrence_cb_formats_capacity_and_budget(self):
        io, fp32 = split.cb_plan('recurrence')
        self.assertFalse(io.keys() & fp32.keys())
        for index in (0, 1):
            self.assertEqual(io[index], 4)
        for index in (7, 8, 10, 11, 12, 20, 21):
            self.assertEqual(fp32[index], 4)
        for index in (5, 18, 30):
            self.assertEqual(io[index], 4)
        for index in (14, 17, 25, 26):
            self.assertEqual(fp32[index], 4)
        self.assertEqual(io[2], 1)
        self.assertEqual(fp32[15], 1)
        self.assertEqual(fp32[16], 2)
        self.assertEqual(fp32[22], 1)
        self.assertNotIn(19, io)
        self.assertEqual(fp32[19], 2)
        self.assertEqual(sum(io.values()) * 2048 + sum(fp32.values()) * 4096, 282624)

    def test_norm_cb_formats_capacity_and_disjoint_ownership(self):
        io, fp32 = split.cb_plan('norm_gate')
        self.assertFalse(io.keys() & fp32.keys())
        self.assertEqual(io, {2: 8, 19: 8, 27: 4, 30: 8})
        self.assertEqual(fp32[15], 4)
        self.assertEqual(fp32[31], 4)
        self.assertEqual(fp32[5], 1)
        self.assertEqual(fp32[6], 1)
        for index in (7, 10, 11):
            self.assertEqual(fp32[index], 4)
        self.assertNotIn(30, fp32)
        self.assertNotIn(5, io)
        self.assertEqual(sum(io.values()) * 2048 + sum(fp32.values()) * 4096, 204800)

    def test_compile_and_runtime_abis(self):
        addresses = list(range(100, 109))
        for rows in (1, 2, 4, 8, 16, 32):
            first = split.stage_spec('recurrence', rows)
            second = split.stage_spec('norm_gate', rows)
            self.assertEqual(first['compute'][:3], [4, 1, 1])
            self.assertEqual(first['compute'][5:], [0, 0, 0])
            self.assertEqual(first['reader'][5:], [96, 1, 160, 0, 32, 64, 12, 1, 0, 0, 0])
            self.assertEqual(first['writer'], [4, 1, 0, 1, 96, rows])
            self.assertEqual(second['compute'][1], 4)
            self.assertEqual(second['compute'][5:], [1, split.bits(128e-6), split.bits(128 ** 0.5)])
            self.assertEqual(second['reader'][13:], [1, 96, 0])
            self.assertEqual(split.runtime_args(first, 'writer', 95, rows, addresses),
                             [95, rows, 104, 128, 105])
            self.assertEqual(split.runtime_args(second, 'reader', 23, rows, addresses),
                             [23, rows, 104, 106, 106, 106, 106, 106, 106, 107])
            self.assertEqual(split.runtime_args(second, 'writer', 23, rows, addresses),
                             [23, rows, 108, 2048, 108])
        for rows in (0, 3, 17, 31, 33, 64):
            with self.assertRaises(ValueError):
                split.stage_spec('recurrence', rows)
        for operation in (split.cb_plan, lambda stage: split.stage_spec(stage, 1)):
            with self.assertRaises(ValueError):
                operation('unknown')


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT / native.KERNEL_ROOT).is_dir():
            raise unittest.SkipTest('Source audit unavailable: set GDN_VSPLIT_SOURCE_ROOT to pinned native kernels')
        cls.original = native.load_kernels(ROOT, False)
        cls.kernels = split.load_kernels(ROOT)

    def test_native_recurrence_and_helpers_are_unchanged(self):
        self.assertEqual(self.kernels['recurrence']['compute'], self.original['compute'])
        self.assertEqual(self.kernels['norm_gate']['compute'].split('void kernel_main()')[0],
                         self.original['compute'].split('void kernel_main()')[0])
        source = self.kernels['recurrence']['compute']
        self.assertIn('mm(cb_qn, cb_snew, cb_out, 1, Kt, Vt, false);', source)
        self.assertIn('copy_tiles(it == 0 ? cb_state : 30, cb_sf, kv);', source)
        self.assertIn('if (it + 1 < n_inst) { copy_tiles(cb_snew, 30, kv); }', source)

    def test_native_fng_block_is_verbatim_and_full_width(self):
        expected = split.checked_section(self.original['compute'], split.FNG_START, split.FNG_END)
        source = self.kernels['norm_gate']['compute']
        self.assertEqual(source.count(expected), 1)
        body = source[source.index(split.LOOP):]
        self.assertNotIn('mm(', body)
        self.assertNotIn('cb_state', body)
        self.assertNotIn('cb_snew', body)
        self.assertNotIn('cb_sout', body)
        self.assertIn('rowsum_k(cb_qsq, cb_sc, Vt);', body)
        self.assertIn('copy_tiles(cb_kn, cb_w, Vt);', body)
        self.assertIn('copy_tiles(cb_w, cb_kn, Vt);', body)
        self.assertIn('copy_tiles(cb_delta, cb_w, Vt);', body)
        self.assertIn('copy_tiles(cb_w, cb_delta, Vt);', body)
        self.assertIn('WAIT(cb_wf, Vt);', source)
        self.assertIn('POP(cb_w, 2 * Vt);', source)
        self.assertEqual(source.count('rowsum_k(cb_qsq, cb_sc, Vt);'), 1)

    def test_state_remap_and_independent_bf16_tile_size(self):
        reader = self.kernels['recurrence']['reader']
        writer = self.kernels['recurrence']['writer']
        for source in (reader, writer):
            self.assertIn(f'const uint32_t base_page = {split.STATE_BASE};', source)
            self.assertIn('.page_id = base_page + t * 4', source)
        self.assertIn('if (token == 0)', reader)
        for scalar in ('beta_acc, cb_beta', 'g_acc, cb_g'):
            self.assertIn(f'gather_scalar({scalar}, (b / 32) * NVT + (h / 4) / 32, r, (h / 4) % 32);', reader)
        self.assertIn('const uint32_t tb_state = get_tile_size(cb_sout);', writer)
        self.assertIn('TensorAccessor(s_a, s_addr, tb_state)', writer)
        self.assertIn('s_acc, tb_state, {.offset_bytes = t * tb_state}', writer)
        self.assertNotIn('s_acc, tb_io', writer)
        self.assertIn('const uint32_t tb_io = get_tile_size(cb_out);', writer)
        self.assertIn('scb.push_back(1);\n            scb.wait_front(1);', writer)

    def test_norm_reader_full_sticks_and_writer_full_tile_assembly(self):
        reader = self.kernels['norm_gate']['reader']
        body = reader[reader.index(split.READER_LOOP):]
        self.assertIn('TensorAccessor(q_a, q_addr, 128)', reader)
        self.assertIn('noc.async_read(pre_acc, stick, 128,', body)
        self.assertIn('token * 96 + bh_start * 4 + partition', body)
        self.assertIn('output[256 + word] = input[16 + word];', body)
        self.assertIn('zero(destination, 4 * 4096 / 4);', body)
        self.assertIn('pre.push_back(4);', body)
        self.assertIn('gather_row(z_acc, cb_v, Vt,', body)
        self.assertNotIn('gather_row(q_acc', body)
        self.assertNotIn('cbs.reserve_back', body)
        writer = self.kernels['norm_gate']['writer']
        self.assertNotIn('noc_semaphore', writer)
        self.assertNotIn('cb.wait_front(kv)', writer)
        self.assertNotIn('noc.async_write(src, s_acc', writer)
        self.assertIn('zero(asm_base, Vt * tb_io / 4)', writer)
        self.assertIn('const uint32_t row = token;', writer)
        self.assertIn('.page_id = bh_start * Vt + tile', writer)

    def test_native_hash_drift_fails_before_transform(self):
        original_read = Path.read_bytes
        for relative in native.HASHES:
            target = ROOT / native.KERNEL_ROOT / relative

            def changed(path, target=target):
                data = original_read(path)
                return data + b'\n' if path == target else data

            with patch.object(Path, 'read_bytes', changed):
                with self.assertRaisesRegex(ValueError, 'Native kernel hash changed'):
                    split.load_kernels(ROOT)

    def test_helper_and_runtime_drift_fail_closed_without_cache(self):
        with patch.object(Path, 'read_text', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'helper changed'):
                split.validate_helper()
        with patch.object(Path, 'read_bytes', return_value=b'changed'):
            with self.assertRaisesRegex(ValueError, 'CB handoff runtime changed'):
                split.validate_runtime(ROOT)

    def test_missing_duplicated_and_reordered_anchors_fail(self):
        for source in ('missing', split.READER_LOOP * 2 + '    }\n}\n'):
            with self.assertRaises(ValueError):
                split.norm_reader(source)
        for source in ('missing', split.LOOP * 2):
            with self.assertRaises(ValueError):
                split.norm_compute(source)
        for source in ('STARTSTARTEND', 'ENDSTART', 'STARTmissing'):
            with self.assertRaises(ValueError):
                split.checked_section(source, 'START', 'END')
        for role, transform, anchor in (
            ('reader', split.split_reader, 'const uint32_t base_page = bh * kv;'),
            ('writer', split.split_writer, 'const auto s_acc = TensorAccessor(s_a, s_addr, tb_io);'),
            ('writer', split.norm_writer, '        // new state:'),
        ):
            for source in (self.original[role].replace(anchor, 'missing'), self.original[role] + anchor):
                with self.assertRaises(ValueError):
                    transform(source)

    def test_audit_is_host_only_and_bounded(self):
        result = split.audit(ROOT)
        self.assertFalse(result['runtime_headers_checked'])
        self.assertIn('no compilation', result['status'])
        for stage, expected in (('recurrence', 394112), ('norm_gate', 316288)):
            entry = result['stages'][stage]
            self.assertEqual(entry['static_end_estimate_with_111488_reserved'], expected)
            for role, source in self.kernels[stage].items():
                self.assertEqual(entry['generated_sha256'][role], hashlib.sha256(source.encode()).hexdigest())


class FakeTensor:
    def __init__(self, shape, address, dtype='bf16', layout='tile', memory='dram'):
        self.shape = shape
        self.address = address
        self.dtype = dtype
        self.layout = layout
        self.memory = memory

    def buffer_address(self):
        return self.address

    def memory_config(self):
        return self.memory


class FakeTTNN:
    bfloat16 = 'bf16'
    float32 = 'fp32'
    TILE_LAYOUT = 'tile'
    ROW_MAJOR_LAYOUT = 'row_major'
    DRAM_MEMORY_CONFIG = 'dram'
    L1_MEMORY_CONFIG = 'l1'
    DataMovementProcessor = SimpleNamespace(RISCV_0=0, RISCV_1=1)
    NOC = SimpleNamespace(RISCV_0_default=0, RISCV_1_default=1)
    MathFidelity = SimpleNamespace(HiFi4='HiFi4')

    def __init__(self):
        self.events = []
        self.allocations = []

    class KernelDescriptor(SimpleNamespace):
        SourceType = SimpleNamespace(SOURCE_CODE='source')

    CBDescriptor = CBFormatDescriptor = ProgramDescriptor = SimpleNamespace
    ComputeConfigDescriptor = DataMovementConfigDescriptor = SimpleNamespace
    MeshProgramDescriptor = dict

    @staticmethod
    def RuntimeArgs():
        return defaultdict(dict)

    @staticmethod
    def CoreCoord(*point):
        return point

    CoreRange = MeshCoordinate = MeshCoordinateRange = CoreCoord
    CoreRangeSet = TileDescriptor = Tile = staticmethod(lambda value: value)

    @staticmethod
    def TensorAccessorArgs(tensor):
        return SimpleNamespace(get_compile_time_args=lambda: [tensor.address])

    @staticmethod
    def get_device_tensors(tensor):
        return [tensor, tensor]

    def empty(self, shape, *, device, dtype, layout, memory_config):
        tensor = FakeTensor(shape, 200 + len(self.allocations), dtype, layout, memory_config)
        self.allocations.append(tensor)
        return tensor

    def generic_op(self, tensors, program):
        self.events.append(('launch', tensors, program))

    def synchronize_device(self, mesh):
        self.events.append(('fence', mesh))


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.operations = FakeTTNN()
        self.mesh = SimpleNamespace(shape=(1, 2),
            compute_with_storage_grid_size=lambda: SimpleNamespace(x=12, y=10))
        self.inputs = [FakeTensor(shape, 100 + index) for index, shape in enumerate(
            ((1, 32, 5120), (1, 32, 24), (1, 32, 24), (1, 24, 128, 128),
             (1, 32, 3072), (1, 1, 128)))]

    def execute(self, experimental=True):
        with patch.dict(sys.modules, {'ttnn': self.operations}), patch.object(split, 'validate_runtime'):
            return split.execute(self.mesh, *self.inputs[:4], z=self.inputs[4], norm_w=self.inputs[5],
                                 root=ROOT, experimental=experimental)

    def test_two_fenced_programs_and_descriptor_contracts(self):
        output, states, pre_norm = self.execute()
        self.assertEqual([event[0] for event in self.operations.events], ['launch', 'fence', 'launch', 'fence'])
        self.assertEqual((output.shape, output.dtype, output.layout), ((1, 32, 3072), 'bf16', 'tile'))
        self.assertEqual((states.shape, states.dtype, states.layout), ((32, 24, 128, 128), 'bf16', 'tile'))
        self.assertEqual((pre_norm.shape, pre_norm.dtype, pre_norm.layout), ((32, 1, 96, 32), 'fp32', 'row_major'))
        self.assertEqual(self.inputs[3].address, 103)
        for event, stage in zip(self.operations.events[::2], ('recurrence', 'norm_gate')):
            tensors, program = event[1:]
            self.assertIs(tensors[4], pre_norm)
            self.assertEqual(len(program), 2)
            spec = split.stage_spec(stage, 32)
            for descriptor in program.values():
                self.assertEqual(len(descriptor.kernels), 3)
                formats = {buffer.format_descriptors[0].buffer_index: buffer for buffer in descriptor.cbs}
                io, fp32 = split.cb_plan(stage)
                for counts, dtype, page in ((io, 'bf16', 2048), (fp32, 'fp32', 4096)):
                    for index, count in counts.items():
                        self.assertEqual(formats[index].total_size, count * page)
                        self.assertEqual(formats[index].format_descriptors[0].data_format, dtype)
                        self.assertEqual(formats[index].format_descriptors[0].page_size, page)
                for role, kernel in zip(('reader', 'writer', 'compute'), descriptor.kernels):
                    self.assertEqual(len(kernel.core_ranges), spec['workers'])
                    self.assertEqual(kernel.compile_time_args[:len(spec[role])], spec[role])
                    expected_accessors = [tensors[index].address for index in spec.get(role + '_accessors', [])]
                    self.assertEqual(kernel.compile_time_args[len(spec[role]):], expected_accessors)
                    for worker, (horizontal, vertical) in enumerate(split.core_coordinates(12, 10, spec['workers'])):
                        self.assertEqual(kernel.runtime_args[horizontal][vertical],
                            split.runtime_args(spec, role, worker, 32, [tensor.address for tensor in tensors]))
                    if role == 'compute':
                        self.assertEqual(kernel.config.math_fidelity, 'HiFi4')
                        self.assertTrue(kernel.config.fp32_dest_acc_en)
                        self.assertFalse(kernel.config.math_approx_mode)

    def test_opt_in_required_before_import_or_allocation(self):
        with self.assertRaisesRegex(ValueError, 'experimental=True'):
            self.execute(False)
        self.assertEqual(self.operations.allocations, [])
        self.assertEqual(self.operations.events, [])

    def test_norm_batch_factory_enqueues_without_internal_fences(self):
        import gdn_vsplit_norm_batch as batch
        with patch.dict(sys.modules, {'ttnn': self.operations}), patch.object(split, 'validate_runtime'), \
                patch.object(batch, 'validate_runtime'):
            output, states, bridge = split.execute(self.mesh, *self.inputs[:4], z=self.inputs[4], norm_w=self.inputs[5],
                root=ROOT, experimental=True, batch_norm=True, synchronize=False, output_memory='l1')
        self.assertEqual([event[0] for event in self.operations.events], ['launch', 'launch'])
        self.assertEqual(output.memory_config(), 'l1')
        self.assertEqual(states.memory_config(), 'dram')
        self.assertEqual(bridge.dtype, 'fp32')
        expected = batch.load_kernels(ROOT)
        for event, stage in zip(self.operations.events, ('recurrence', 'norm_gate')):
            for descriptor in event[2].values():
                for kernel, role in zip(descriptor.kernels, ('reader', 'writer', 'compute')):
                    self.assertEqual(kernel.kernel_source, expected[stage][role])

    def test_active_runtime_is_checked_independently_of_source_root(self):
        with patch.dict(sys.modules, {'ttnn': self.operations}), patch.dict(os.environ,
                {'TT_METAL_HOME': '/active/tt-metal'}), patch.object(split, 'validate_runtime') as validate:
            split.execute(self.mesh, *self.inputs[:4], z=self.inputs[4], norm_w=self.inputs[5],
                          root=ROOT, experimental=True)
            validate.assert_called_once_with(Path('/active/tt-metal'))

    def test_runtime_drift_prevents_allocation_and_launch(self):
        with patch.dict(sys.modules, {'ttnn': self.operations}), patch.object(split, 'validate_runtime',
                side_effect=ValueError('CB handoff runtime changed')):
            with self.assertRaisesRegex(ValueError, 'CB handoff runtime changed'):
                split.execute(self.mesh, *self.inputs[:4], z=self.inputs[4], norm_w=self.inputs[5],
                              root=ROOT, experimental=True)
        self.assertEqual(self.operations.allocations, [])
        self.assertEqual(self.operations.events, [])

    def test_all_supported_token_lengths(self):
        for rows in (1, 2, 4, 8, 16, 32):
            self.operations = FakeTTNN()
            for index in (0, 1, 2, 4):
                self.inputs[index].shape = (1, rows, self.inputs[index].shape[-1])
            output, states, pre_norm = self.execute()
            self.assertEqual(output.shape, (1, rows, 3072))
            self.assertEqual(states.shape, (rows, 24, 128, 128))
            self.assertEqual(pre_norm.shape, (rows, 1, 96, 32))
            self.assertEqual([event[0] for event in self.operations.events], ['launch', 'fence', 'launch', 'fence'])

    def test_reject_bad_geometry_dtype_memory_mesh_and_alias_before_launch(self):
        for attribute, value in (('shape', (1, 33, 5120)), ('dtype', 'fp32'),
                                 ('layout', 'row_major'), ('memory', 'l1')):
            original = getattr(self.inputs[0], attribute)
            setattr(self.inputs[0], attribute, value)
            with self.assertRaises(ValueError):
                self.execute()
            setattr(self.inputs[0], attribute, original)
        self.mesh.shape = (2, 1)
        with self.assertRaisesRegex(ValueError, '1x2 mesh'):
            self.execute()
        self.mesh.shape = (1, 2)
        self.inputs[1].address = self.inputs[0].address
        with self.assertRaisesRegex(ValueError, 'must not alias'):
            self.execute()
        self.assertEqual(self.operations.events, [])


if __name__ == '__main__':
    unittest.main()
