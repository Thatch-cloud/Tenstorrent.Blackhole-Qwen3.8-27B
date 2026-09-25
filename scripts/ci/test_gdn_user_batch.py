"""One launch for four packed users must build four copies of the native fused program.

The whole safety argument for `gdn_user_batch` is that it transforms nothing: the kernel
sources are `gdn_multitoken`'s fused ones verbatim, and every user's descriptor triple is
exactly the descriptor triple the single-user `gdn_multitoken.execute` would build, moved
onto its own disjoint core share. So the central test here is not "does it look right" but
"is one user's batched program identical, descriptor for descriptor, to the native one" -
held against the real builder over a recording fake, never against a copy of its literals.
"""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gdn_multitoken as native
import gdn_user_batch as batch


class FakeTensor:
    def __init__(self, name, shape, address, memory='dram', dtype='bf16', layout='tile'):
        self.name, self.shape, self.address = name, tuple(shape), address
        self.dtype, self.layout, self._memory = dtype, layout, memory

    def memory_config(self):
        return self._memory

    def buffer_address(self):
        return self.address


class FakeRuntimeArgs(dict):
    def __getitem__(self, key):
        if key not in self:
            dict.__setitem__(self, key, {})
        return dict.__getitem__(self, key)

    def flattened(self):
        return sorted((horizontal, vertical, tuple(args))
                      for horizontal, column in self.items() for vertical, args in column.items())


class FakeKernelDescriptor:
    class SourceType:
        SOURCE_CODE = 'source-code'

    def __init__(self, kernel_source, source_type, core_ranges, compile_time_args, config):
        self.kernel_source, self.source_type = kernel_source, source_type
        self.core_ranges, self.compile_time_args, self.config = core_ranges, list(compile_time_args), config
        self.runtime_args = None

    def signature(self):
        return (self.kernel_source, self.source_type, tuple(self.core_ranges), tuple(self.compile_time_args),
                self.config, tuple(self.runtime_args.flattened()))


class FakeTTNN:
    """Enough ttnn to record what a program descriptor would contain."""

    bfloat16, float32 = 'bf16', 'fp32'
    TILE_LAYOUT, ROW_MAJOR_LAYOUT = 'tile', 'row-major'
    DRAM_MEMORY_CONFIG, L1_MEMORY_CONFIG = 'dram', 'l1'
    KernelDescriptor = FakeKernelDescriptor
    MathFidelity = SimpleNamespace(HiFi4='hifi4')
    DataMovementProcessor = SimpleNamespace(RISCV_0='riscv0', RISCV_1='riscv1')
    NOC = SimpleNamespace(RISCV_0_default='noc0', RISCV_1_default='noc1')

    def __init__(self, first_address=9000):
        self.next_address = first_address
        self.allocated, self.freed, self.launches = [], [], []

    def empty(self, shape, device=None, dtype=None, layout=None, memory_config=None):
        value = FakeTensor('empty%d' % len(self.allocated), shape, self.next_address,
                           memory=memory_config, dtype=dtype, layout=layout)
        self.next_address += 16
        self.allocated.append(value)
        return value

    def deallocate(self, value):
        self.freed.append(value.name)

    def get_device_tensors(self, value):
        return [FakeTensor(value.name + ':' + str(chip), value.shape, value.address + chip,
                           memory=value._memory, dtype=value.dtype, layout=value.layout) for chip in range(2)]

    @staticmethod
    def CoreCoord(horizontal, vertical):
        return (horizontal, vertical)

    @staticmethod
    def CoreRange(first, second):
        return (first, second)

    @staticmethod
    def CoreRangeSet(ranges):
        return tuple(ranges)

    @staticmethod
    def Tile(dimensions):
        return tuple(dimensions)

    @staticmethod
    def TileDescriptor(tile):
        return ('tile', tile)

    @staticmethod
    def CBFormatDescriptor(buffer_index, data_format, page_size, tile):
        return ('format', buffer_index, data_format, page_size, tile)

    @staticmethod
    def CBDescriptor(total_size, core_ranges, format_descriptors):
        return ('cb', total_size, tuple(core_ranges), tuple(format_descriptors))

    @staticmethod
    def TensorAccessorArgs(value):
        return SimpleNamespace(get_compile_time_args=lambda: [7, value.address])

    @staticmethod
    def DataMovementConfigDescriptor(processor, noc):
        return ('movement', processor, noc)

    @staticmethod
    def ComputeConfigDescriptor(math_fidelity, fp32_dest_acc_en, math_approx_mode):
        return ('compute', math_fidelity, fp32_dest_acc_en, math_approx_mode)

    @staticmethod
    def RuntimeArgs():
        return FakeRuntimeArgs()

    @staticmethod
    def MeshProgramDescriptor():
        return {}

    @staticmethod
    def MeshCoordinate(row, chip):
        return (row, chip)

    @staticmethod
    def MeshCoordinateRange(first, second):
        return (first, second)

    @staticmethod
    def ProgramDescriptor(kernels, cbs):
        return SimpleNamespace(kernels=list(kernels), cbs=list(cbs))

    def generic_op(self, tensors, program):
        self.launches.append((tuple(value.name for value in tensors), program))


KERNELS = dict(reader='READER-SOURCE', writer='WRITER-SOURCE', compute='COMPUTE-SOURCE')


def mesh(horizontal=11, vertical=10):
    return SimpleNamespace(shape=(1, 2),
                           compute_with_storage_grid_size=lambda: SimpleNamespace(x=horizontal, y=vertical))


def user_inputs(index, rows=16, base=1000):
    start = base + 100 * index
    return (FakeTensor('qkv%d' % index, (1, rows, 5120), start),
            FakeTensor('beta%d' % index, (1, rows, 24), start + 10),
            FakeTensor('gate%d' % index, (1, rows, 24), start + 20),
            FakeTensor('initial%d' % index, (1, 24, 128, 128), start + 30),
            FakeTensor('z%d' % index, (1, rows, 3072), start + 40),
            FakeTensor('norm_w', (1, 1, 128), 500))


class GeometryTests(unittest.TestCase):
    def test_core_shares_are_disjoint_contiguous_and_start_where_the_native_launch_does(self):
        shares = batch.core_shares(11, 10, 4)
        self.assertEqual(len(shares), 4)
        self.assertTrue(all(len(share) == 24 for share in shares))
        native_points = [(head // 10, head % 10) for head in range(24)]
        self.assertEqual(shares[0], native_points)
        flat = [point for share in shares for point in share]
        self.assertEqual(len(set(flat)), 96)

    def test_core_shares_reject_a_grid_that_cannot_hold_every_user(self):
        for horizontal, vertical, users in ((8, 8, 4), (11, 10, 5), (3, 8, 2), (11, 10, 0)):
            with self.assertRaises(ValueError):
                batch.core_shares(horizontal, vertical, users)
        # The audited 8x6 window grid holds two users but not four.
        self.assertEqual(len(batch.core_shares(8, 6, 2)), 2)
        with self.assertRaises(ValueError):
            batch.core_shares(8, 6, 4)

    def test_validate_users_takes_ragged_widths_and_rejects_bad_geometry(self):
        good = [[tuple(value.shape) for value in user_inputs(index, rows=rows)]
                for index, rows in enumerate((16, 8, 32, 2))]
        self.assertEqual(batch.validate_users(good), [16, 8, 32, 2])
        with self.assertRaises(ValueError):
            batch.validate_users(good + [good[0]])
        with self.assertRaises(ValueError):
            batch.validate_users([])
        broken = [list(good[0])]
        broken[0][4] = (1, 15, 3072)
        with self.assertRaises(ValueError):
            batch.validate_users(broken)
        broken = [list(good[0])]
        broken[0][5] = (1, 1, 64)
        with self.assertRaises(ValueError):
            batch.validate_users(broken)

    def test_runtime_args_place_every_address_where_the_fused_kernel_reads_it(self):
        addresses = list(range(10, 90, 10))
        self.assertEqual(batch.runtime_args('reader', 5, 16, addresses),
                         [5, 16, 10, 10, 10, 20, 30, 40, 70, 80])
        self.assertEqual(batch.runtime_args('writer', 5, 16, addresses), [5, 16, 50, 2048, 60])
        self.assertEqual(batch.runtime_args('compute', 5, 16, addresses), [16])
        for role in ('reader', 'writer', 'compute'):
            with self.assertRaises(ValueError):
                batch.runtime_args(role, 24, 16, addresses)
            with self.assertRaises(ValueError):
                batch.runtime_args(role, 0, 16, addresses[:7])
        with self.assertRaises(ValueError):
            batch.runtime_args('unknown', 0, 16, addresses)

    def test_compile_args_carry_the_fused_flag_and_the_users_own_row_count(self):
        self.assertEqual(batch.compile_args('reader', 16)[13:16], [1, 96, 0])
        self.assertEqual(batch.compile_args('writer', 16), [4, 4, 1, 1, 24, 16])
        self.assertEqual(batch.compile_args('writer', 8)[-1], 8)
        self.assertEqual(batch.compile_args('compute', 16)[5], 1)
        with self.assertRaises(ValueError):
            batch.compile_args('unknown', 16)


class FlagTests(unittest.TestCase):
    def test_the_flag_is_off_by_default_and_rejects_anything_but_zero_or_one(self):
        self.assertFalse(batch.enabled({}))
        self.assertFalse(batch.enabled({batch.FLAG: '0'}))
        self.assertTrue(batch.enabled({batch.FLAG: '1'}))
        for value in ('true', 'yes', '', '2', 'on'):
            with self.assertRaises(ValueError):
                batch.enabled({batch.FLAG: value})

    def test_the_threshold_defaults_to_one_and_only_takes_a_decimal_integer(self):
        self.assertEqual(batch.min_users({}), 1)
        self.assertEqual(batch.min_users({batch.MIN_USERS_FLAG: '0'}), 0)
        self.assertEqual(batch.min_users({batch.MIN_USERS_FLAG: '4'}), 4)
        # Above the batched path's own cap the flag can never engage: the bisect off switch.
        self.assertEqual(batch.min_users({batch.MIN_USERS_FLAG: '5'}), 5)
        self.assertGreater(batch.min_users({batch.MIN_USERS_FLAG: '5'}), batch.MAX_USERS)
        for value in ('', '-1', '2.0', 'four', ' 2', '02', '+2', 'true'):
            with self.assertRaisesRegex(ValueError, 'non-negative decimal integer'):
                batch.min_users({batch.MIN_USERS_FLAG: value})

    def test_load_kernels_asks_for_the_fused_sources_and_nothing_else(self):
        with patch('gdn_multitoken.load_kernels', return_value=KERNELS) as load:
            self.assertIs(batch.load_kernels('/audited'), KERNELS)
        self.assertEqual(load.call_args.args, (Path('/audited'), True))


class ProgramTests(unittest.TestCase):
    """The batched program against the real native builder, over the same fake."""

    def run_native(self, rows=16):
        fake = FakeTTNN()
        qkv, beta, gate, initial, z, norm_w = user_inputs(0, rows)
        with patch.dict(sys.modules, {'ttnn': fake}), patch('gdn_multitoken.validate_handoff_runtime'):
            native.execute(mesh(), qkv, beta, gate, initial, KERNELS, z=z, norm_w=norm_w)
        return fake

    def run_batched(self, users, rows=16):
        fake = FakeTTNN()
        groups = [user_inputs(index, rows) for index in range(users)]
        with patch('gdn_multitoken.validate_handoff_runtime'):
            produced = batch.execute(mesh(), groups, KERNELS, fake, output_memory=fake.L1_MEMORY_CONFIG)
        return fake, produced

    def signatures(self, fake):
        chips = []
        for key in sorted(fake.launches[-1][1]):
            chips.append([descriptor.signature() for descriptor in fake.launches[-1][1][key].kernels])
        return chips

    def test_one_batched_user_is_the_native_fused_program_descriptor_for_descriptor(self):
        control, (candidate, produced) = self.run_native(), self.run_batched(1)
        self.assertEqual(len(produced), 1)
        self.assertEqual(self.signatures(candidate), self.signatures(control))
        for key in sorted(control.launches[-1][1]):
            self.assertEqual(candidate.launches[-1][1][key].cbs, control.launches[-1][1][key].cbs)

    def test_the_native_output_placement_and_prefix_geometry_are_reproduced(self):
        control = self.run_native()
        candidate, produced = self.run_batched(1)
        self.assertEqual([value.shape for value in control.allocated],
                         [value.shape for value in candidate.allocated])
        self.assertEqual(produced[0][0].shape, (1, 16, 3072))
        self.assertEqual(produced[0][1].shape, (16, 24, 128, 128))

    def test_four_users_are_four_native_triples_on_four_disjoint_shares(self):
        fake, produced = self.run_batched(4)
        self.assertEqual(len(produced), 4)
        chip = fake.launches[-1][1][((0, 0), (0, 0))]
        self.assertEqual(len(chip.kernels), 12)
        seen = set()
        for user in range(4):
            triple = chip.kernels[3 * user:3 * user + 3]
            self.assertEqual([descriptor.kernel_source for descriptor in triple],
                             ['READER-SOURCE', 'WRITER-SOURCE', 'COMPUTE-SOURCE'])
            points = {point for first, second in triple[0].core_ranges for point in (first, second)}
            self.assertEqual(len(points), 24)
            self.assertFalse(points & seen)
            seen |= points
            for descriptor in triple:
                heads = sorted(args[0] for unused, unused_too, args in descriptor.runtime_args.flattened()
                               if len(args) > 1)
                self.assertEqual(heads, list(range(24)) if len(heads) == 24 else [])
        self.assertEqual(len(seen), 96)

    def test_every_users_reader_points_at_that_users_own_buffers(self):
        fake, produced = self.run_batched(4)
        chip = fake.launches[-1][1][((0, 0), (0, 0))]
        for user in range(4):
            expected = user_inputs(user)[0].address
            for unused, unused_too, args in chip.kernels[3 * user].runtime_args.flattened():
                self.assertEqual(args[2:5], (expected, expected, expected))

    def test_the_shared_norm_weight_is_the_one_buffer_four_users_may_hold(self):
        fake, produced = self.run_batched(4)
        chip = fake.launches[-1][1][((0, 0), (0, 0))]
        weights = {args[9] for user in range(4)
                   for unused, unused_too, args in chip.kernels[3 * user].runtime_args.flattened()}
        self.assertEqual(len(weights), 1)

    def test_two_users_sharing_a_state_buffer_are_rejected_before_any_launch(self):
        fake = FakeTTNN()
        groups = [list(user_inputs(index)) for index in range(2)]
        groups[1][3] = groups[0][3]
        with patch('gdn_multitoken.validate_handoff_runtime'):
            with self.assertRaisesRegex(ValueError, 'must not share'):
                batch.execute(mesh(), groups, KERNELS, fake)
        self.assertEqual(fake.launches, [])
        self.assertEqual(len(fake.freed), len(fake.allocated))

    def test_a_one_chip_mesh_builds_one_chip_program_for_the_single_card_rig(self):
        fake = FakeTTNN()
        single = SimpleNamespace(shape=(1, 1),
                                 compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
        fake.get_device_tensors = lambda value: [FakeTensor(value.name + ':0', value.shape, value.address)]
        with patch('gdn_multitoken.validate_handoff_runtime'):
            batch.execute(single, [user_inputs(index) for index in range(4)], KERNELS, fake)
        self.assertEqual(len(fake.launches[-1][1]), 1)
        self.assertEqual(len(fake.launches[-1][1][((0, 0), (0, 0))].kernels), 12)
        for shape in ((2, 2), (1, 4), (1, 0)):
            with self.assertRaisesRegex(ValueError, '1x2 mesh'):
                batch.mesh_chips(SimpleNamespace(shape=shape))

    def test_a_grid_too_small_for_every_user_is_rejected_before_any_allocation(self):
        fake = FakeTTNN()
        with patch('gdn_multitoken.validate_handoff_runtime'):
            with self.assertRaisesRegex(ValueError, 'worker cores required'):
                batch.execute(mesh(8, 8), [user_inputs(index) for index in range(4)], KERNELS, fake)
        self.assertEqual(fake.allocated, [])

    def test_non_dram_or_non_tile_inputs_are_rejected(self):
        for field, value in ((3, 'l1'), (0, 'l1')):
            fake = FakeTTNN()
            groups = [list(user_inputs(0))]
            groups[0][field] = FakeTensor('bad', groups[0][field].shape, 777, memory=value)
            with patch('gdn_multitoken.validate_handoff_runtime'):
                with self.assertRaisesRegex(ValueError, 'interleaved DRAM BF16 TILE'):
                    batch.execute(mesh(), groups, KERNELS, fake)

    def test_every_allocation_is_released_when_the_launch_fails(self):
        fake = FakeTTNN()
        fake.generic_op = lambda tensors, program: (_ for _ in ()).throw(RuntimeError('device'))
        with patch('gdn_multitoken.validate_handoff_runtime'):
            with self.assertRaises(RuntimeError):
                batch.execute(mesh(), [user_inputs(index) for index in range(4)], KERNELS, fake)
        self.assertEqual(len(fake.freed), 8)


class AuditTests(unittest.TestCase):
    def test_audit_reports_no_transform_and_the_launch_reduction(self):
        with patch('gdn_multitoken.load_kernels', return_value=KERNELS):
            report = batch.audit('/audited', 4, 16)
        self.assertEqual(report['launches_per_layer'], dict(per_user_recurrence_and_norm=8, batched=1))
        self.assertEqual(report['workers'], 96)
        self.assertIn('verbatim', report['transforms'])
        self.assertEqual(report['native_sha256'], native.HASHES)
        self.assertEqual(sorted(report['generated_sha256']), ['compute', 'reader', 'writer'])


if __name__ == '__main__':
    unittest.main()
