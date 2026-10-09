"""The four-card packed GDN launch (gdn_user_batch_tp) and gdn_seq_block's width switch.

At the pair gdn_seq_block's `batch` is the pinned gdn_user_batch itself and every derived argument is the literal it always
was; at QWEN_FAST_TP=4 the same arguments come out of the width's geometry (12 heads, 80 conv pages, 16 / 32 tile offsets)
and the launch builds four users x 12 heads on 48 cores over a 1x4 mesh."""

import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gdn_seq_block as seq
import gdn_user_batch as pair
import gdn_user_batch_tp as quad
import tp_shapes
from test_gdn_user_batch import FakeTensor, FakeTTNN, KERNELS


class FourChipTTNN(FakeTTNN):
    def get_device_tensors(self, value):
        return [FakeTensor(value.name + ':' + str(chip), value.shape, value.address + chip,
                           memory=value._memory, dtype=value.dtype, layout=value.layout) for chip in range(4)]


def four_mesh(horizontal=11, vertical=10):
    return SimpleNamespace(shape=(1, 4),
                           compute_with_storage_grid_size=lambda: SimpleNamespace(x=horizontal, y=vertical))


def user_inputs(index, rows=16, base=1000):
    start = base + 100 * index
    return (FakeTensor('qkv%d' % index, (1, rows, 2560), start),
            FakeTensor('beta%d' % index, (1, rows, 12), start + 10),
            FakeTensor('gate%d' % index, (1, rows, 12), start + 20),
            FakeTensor('initial%d' % index, (1, 12, 128, 128), start + 30),
            FakeTensor('z%d' % index, (1, rows, 1536), start + 40),
            FakeTensor('norm_w', (1, 1, 128), 500))


class Four(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {'QWEN_FAST_TP': '4'})
        patcher.start()
        self.addCleanup(patcher.stop)


class PairIsUntouchedTests(unittest.TestCase):
    def test_the_width_switch_hands_the_pair_the_pinned_modules_own_objects(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIs(seq.batch.core_shares, pair.core_shares)
            self.assertIs(seq.batch.execute, pair.execute)
            self.assertIs(seq.batch.coalesced_descriptors, pair.coalesced_descriptors)
            self.assertEqual(seq.batch.heads(), 24)
            self.assertEqual(seq.batch.geometry().gdn_value, 3072)

    def test_the_pairs_k5_arguments_are_the_literals_they_were(self):
        build = SimpleNamespace(tag=lambda role: 99)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(seq.compile_args('reader', build), [4, 4, 24, 3, 160, 0, 32, 64, 96, 0, 99])
            self.assertEqual(seq.compile_args('writer', build), [4, 4, 24, 99])
            with self.assertRaises(ValueError):
                seq.runtime_args('reader', 24, list(range(8)))

    def test_the_pairs_served_launch_arguments_are_still_the_pinned_ones(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(pair.compile_args('reader', 16)[13:16], [1, 96, 0])
            self.assertEqual(pair.compile_args('writer', 16), [4, 4, 1, 1, 24, 16])


class QuadArgumentTests(Four):
    def test_the_width_is_read_from_the_launch_environment(self):
        self.assertIs(seq.batch.module(), quad)
        self.assertEqual((quad.heads(), seq.batch.heads()), (12, 12))
        self.assertIs(seq.batch.core_shares, quad.core_shares)

    def test_forty_eight_cores_four_disjoint_shares_from_the_native_origin(self):
        shares = quad.core_shares(11, 10, 4)
        self.assertEqual([len(share) for share in shares], [12] * 4)
        flat = [point for share in shares for point in share]
        self.assertEqual(len(set(flat)), 48)
        self.assertEqual(shares[0], [(head // 10, head % 10) for head in range(12)])
        with self.assertRaises(ValueError):
            quad.core_shares(4, 10, 4)
        # the four-card limit is the octo block's eight users (96 cores, one wave); the pair's pinned module still stops at four
        self.assertEqual((quad.MAX_USERS, quad.PAIR_MAX_USERS, pair.MAX_USERS), (8, 4, 4))
        eight = quad.core_shares(11, 10, 8)
        flat = [point for share in eight for point in share]
        self.assertEqual((len(eight), len(flat), len(set(flat))), (8, 96, 96))
        self.assertEqual(eight[:4], shares, 'the first four users sit on the very cores a four-user launch uses')
        self.assertTrue(all(0 <= x < 11 and 0 <= y < 10 for x, y in flat))
        for users in (0, 9):
            with self.assertRaisesRegex(ValueError, 'One to 8 packed users per batched GDN launch'):
                quad.core_shares(11, 10, users)
        with self.assertRaisesRegex(ValueError, 'One to 4 packed users per batched GDN launch'):
            pair.core_shares(11, 10, 8)

    def test_compile_args_carry_the_four_card_geometry(self):
        reader = quad.compile_args('reader', 16)
        self.assertEqual(reader[5], 12)
        self.assertEqual(reader[7], 80)
        self.assertEqual(reader[9:11], [16, 32])
        self.assertEqual(reader[13:16], [1, 48, 0])
        self.assertEqual(quad.compile_args('writer', 16), [4, 4, 1, 1, 12, 16])
        self.assertEqual(quad.compile_args('compute', 16), pair.compile_args('compute', 16))
        with self.assertRaises(ValueError):
            quad.compile_args('unknown', 16)

    def test_the_k5_reader_arguments_follow_the_same_geometry(self):
        build = SimpleNamespace(tag=lambda role: 99)
        self.assertEqual(seq.compile_args('reader', build), [4, 4, 12, 3, 80, 0, 16, 32, 48, 0, 99])
        self.assertEqual(seq.compile_args('writer', build), [4, 4, 12, 99])
        # the fused launch reads the same tile offsets in its own argument order
        fused = quad.compile_args('reader', 16)
        k5 = seq.compile_args('reader', build)
        self.assertEqual((fused[5], fused[7], fused[9], fused[10], fused[14]), (k5[2], k5[4], k5[6], k5[7], k5[8]))

    def test_the_head_bound_is_twelve(self):
        addresses = list(range(10, 90, 10))
        self.assertEqual(quad.runtime_args('reader', 11, 16, addresses), [11, 16, 10, 10, 10, 20, 30, 40, 70, 80])
        for role in ('reader', 'writer', 'compute'):
            with self.assertRaises(ValueError):
                quad.runtime_args(role, 12, 16, addresses)
        with self.assertRaises(ValueError):
            seq.runtime_args('reader', 12, addresses)
        self.assertEqual(seq.runtime_args('compute', 11, addresses), [16])

    def test_four_card_shapes_validate_and_the_pairs_do_not(self):
        good = [[tuple(value.shape) for value in user_inputs(index, rows=rows)]
                for index, rows in enumerate((16, 8, 32, 2))]
        self.assertEqual(quad.validate_users(good), [16, 8, 32, 2])
        self.assertEqual(seq.batch.validate_users(good), [16, 8, 32, 2])
        pair_shaped = [[(1, 16, 5120), (1, 16, 24), (1, 16, 24), (1, 24, 128, 128), (1, 16, 3072), (1, 1, 128)]]
        with self.assertRaises(ValueError):
            quad.validate_users(pair_shaped)
        for index, shape in ((4, (1, 15, 1536)), (5, (1, 1, 64))):
            broken = [list(good[0])]
            broken[0][index] = shape
            with self.assertRaises(ValueError):
                quad.validate_users(broken)
        # eight users of eight rows (the octo block) validate; a ninth is one past the 96-core launch
        octo = [[tuple(value.shape) for value in user_inputs(index, rows=8)] for index in range(8)]
        self.assertEqual(quad.validate_users(octo), [8] * 8)
        self.assertEqual(seq.batch.validate_users(octo), [8] * 8)
        with self.assertRaisesRegex(ValueError, 'One to 8 packed users'):
            quad.validate_users(octo + [octo[0]])

    def test_the_mesh_is_one_by_four_or_a_single_card_rig(self):
        self.assertEqual(quad.mesh_chips(four_mesh()), 4)
        self.assertEqual(quad.mesh_chips(SimpleNamespace(shape=(1, 1))), 1)
        for shape in ((1, 2), (2, 2), (4, 1)):
            with self.assertRaises(ValueError):
                quad.mesh_chips(SimpleNamespace(shape=shape))


class QuadLaunchTests(Four):
    def run_launch(self, users=4):
        fake = FourChipTTNN()
        groups = [user_inputs(index) for index in range(users)]
        with patch.dict(sys.modules, {'ttnn': fake}), patch('gdn_multitoken.validate_handoff_runtime'):
            produced = quad.execute(four_mesh(), groups, KERNELS, fake, output_memory='dram')
        return fake, produced

    def test_one_launch_builds_every_chip_users_times_twelve_cores(self):
        fake, produced = self.run_launch()
        self.assertEqual(len(fake.launches), 1)
        program = fake.launches[0][1]
        self.assertEqual(sorted(program), [((0, chip), (0, chip)) for chip in range(4)])
        for descriptor in program.values():
            for role in ('READER-SOURCE', 'WRITER-SOURCE', 'COMPUTE-SOURCE'):
                kernels = [kernel for kernel in descriptor.kernels if kernel.kernel_source == role]
                cores = [(horizontal, vertical) for kernel in kernels
                         for horizontal, column in kernel.runtime_args.items() for vertical in column]
                self.assertEqual(len(cores), 48, role)
                self.assertEqual(len(set(cores)), 48, role)
        self.assertEqual(len(produced), 4)

    def test_outputs_and_prefix_states_have_the_four_card_shapes(self):
        fake, produced = self.run_launch(users=2)
        for output, states in produced:
            self.assertEqual(output.shape, (1, 16, 1536))
            self.assertEqual(states.shape, (16, 12, 128, 128))

    def test_a_pair_shaped_input_is_refused_before_any_allocation(self):
        fake = FourChipTTNN()
        pair_user = (FakeTensor('qkv', (1, 16, 5120), 1), FakeTensor('beta', (1, 16, 24), 2),
                     FakeTensor('gate', (1, 16, 24), 3), FakeTensor('initial', (1, 24, 128, 128), 4),
                     FakeTensor('z', (1, 16, 3072), 5), FakeTensor('norm_w', (1, 1, 128), 6))
        with patch.dict(sys.modules, {'ttnn': fake}), patch('gdn_multitoken.validate_handoff_runtime'):
            with self.assertRaises(ValueError):
                quad.execute(four_mesh(), [pair_user], KERNELS, fake)
        self.assertEqual(fake.allocated, [])

    def test_a_chip_missing_from_a_tensor_is_refused(self):
        fake = FourChipTTNN()
        fake.get_device_tensors = lambda value: FourChipTTNN.get_device_tensors(fake, value)[:3]
        with patch.dict(sys.modules, {'ttnn': fake}), patch('gdn_multitoken.validate_handoff_runtime'):
            with self.assertRaisesRegex(ValueError, 'Every chip of the mesh'):
                quad.execute(four_mesh(), [user_inputs(0)], KERNELS, fake)
        self.assertEqual(len(fake.freed), len(fake.allocated))

    def test_the_audit_describes_ninety_six_workers_for_eight_users_and_forty_eight_for_four(self):
        with patch('gdn_user_batch_tp.load_kernels', return_value=KERNELS):
            found = quad.audit('/root')
        self.assertEqual((found['users'], found['workers_per_user'], found['workers'], found['tp']), (8, 12, 96, 4))
        with patch('gdn_user_batch_tp.load_kernels', return_value=KERNELS):
            four = quad.audit('/root', users=4)
        self.assertEqual((four['users'], four['workers']), (4, 48), 'the M3 block\'s launch is the one it always was')


class KernelsAreTheServedOnesTests(unittest.TestCase):
    def test_the_sibling_shares_the_pinned_loading_and_packing(self):
        self.assertIs(quad.load_kernels, pair.load_kernels)
        self.assertIs(quad.bits, pair.bits)
        self.assertIs(quad.coalesced_descriptors, pair.coalesced_descriptors)
        self.assertIs(quad.enabled, pair.enabled)
        self.assertEqual(tp_shapes.k5_reader_arguments(2)[2], pair.HEADS)


if __name__ == '__main__':
    unittest.main()
