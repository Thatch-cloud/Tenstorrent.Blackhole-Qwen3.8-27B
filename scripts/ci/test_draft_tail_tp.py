"""draft_tail_tp (QWEN_FAST_DRAFT_TAIL, F-F3a): the drafter MLP branch's SwiGLU and residual tail as one launch each.

Held here, on the CPU (nothing runs on a card):

  - the launches, EXECUTED by the transliterations of draft_tail_tp_io.cpp / _swiglu_compute.cpp / _residual_compute.cpp / draft_fuse_out.cpp that read
    the launch's own runtime arguments, circular buffers and compile-time arguments: the SwiGLU equals draft_mlp.swiglu_device (nine ops) and the
    residual equals the four-op tail bit for bit on every chip, at 32 and 64 rows and at the drafter's per-chip width of 4,352, on data with every
    rounding point in play (bf16 ties, overflow to infinity, zeros of both signs, denormals, silu saturation both ways, infinities); the same
    SwiGLU read in place from a fused gate|up projection (one tensor, the up half a column offset away) equals it too;
  - the transliterations keep the rounded values in 'registers' (fp32 lanes) and never convert a dtype; the served side converts through bf16 tensors;
    a missing rounding after the silu, after the product or on the up half, a swapped gate and up, and an unrounded residual each DIFFER on the same
    data (the test can fail);
  - every output tile is written exactly once, on a 11 x 10 and a 13 x 10 grid;
  - the fall-backs (shape, dtype, layout, a launch that raises) run the served ops and say why once; the audit is exact on a good launch and a
    flipped output bit is a logged mismatch and an AssertionError;
  - the .cpp sources read exactly the runtime arguments the builder writes.

The silu is the one primitive a CPU cannot reproduce bit for bit against the SFPU, so both sides share torch's float32 silu here; the card
audit (QWEN_FAST_DRAFT_TAIL_AUDIT=1) is what holds it.

    py -3.11 -B -m unittest test_draft_tail_tp      (from scripts/ci)
"""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import draft_fusion_tp as fusion  # noqa: E402
import draft_mlp  # noqa: E402
import draft_tail_tp as tail  # noqa: E402
import wp6_fake_device as fake  # noqa: E402
from test_draft_reduce_tp import EnvTestCase, SourceTests, fresh, mesh_for  # noqa: E402

TAIL_ON = {'QWEN_FAST_TP': '4', 'QWEN_FAST_DRAFT_TAIL': '1'}
AUDIT_ON = {'QWEN_FAST_TP': '4', 'QWEN_FAST_DRAFT_TAIL': '1', 'QWEN_FAST_DRAFT_TAIL_AUDIT': '1'}
WIDTH = 4352                                  # the drafter MLP's per-chip width at four cards (136 tiles)


def bits16(tensor):
    return [shard.data.contiguous().view(torch.int16) for shard in tensor.shards]


def same16(left, right):
    return all(a.shape == b.shape and torch.equal(a, b) for a, b in zip(bits16(left), bits16(right), strict=True))


def hard_projection(generator, shape, chip, up=False):
    """fp32 projections that put every rounding point in play."""
    value = torch.randn(*shape, generator=generator) * torch.exp(torch.randn(*shape, generator=generator) * 1.5)
    flat = value.reshape(-1)
    flat[::89] = 0.0
    flat[1::89] = -0.0
    flat[2::97] = 1.0 + 2.0 ** -8                  # exactly half way between two bf16 values: ties to even
    flat[3::97] = 1.0 + 3 * 2.0 ** -9              # just past half way
    flat[4::101] = -(1.0 + 2.0 ** -8)
    flat[5::103] = 1e-40                           # denormal
    flat[6::107] = -1e-40
    flat[7::211] = 80.0                            # silu saturates to x
    flat[8::223] = -80.0                           # silu saturates to -0
    flat[9::307] = 3.0e38                          # the product / activation overflow to infinity when rounded
    flat[10::409] = float('inf') if chip % 2 == 0 else float('-inf')
    if up:
        flat[11::113] = 3.0e38
    return value


def projections(operations, rows, width=WIDTH, seed=0):
    generator = torch.Generator().manual_seed(seed)
    gate = operations.from_chips([hard_projection(generator, (1, 1, rows, width), chip) for chip in range(operations.chips)], 'fp32')
    up = operations.from_chips([hard_projection(generator, (1, 1, rows, width), chip, up=True) for chip in range(operations.chips)], 'fp32')
    return gate, up


def hard_bf16(generator, shape, chip):
    value = torch.randn(*shape, generator=generator) * torch.exp(torch.randn(*shape, generator=generator) * 2)
    flat = value.reshape(-1)
    flat[::83] = 0.0
    flat[1::83] = -0.0
    flat[2::89] = 3.0e38
    flat[3::97] = -3.0e38                          # finished + hidden overflows when rounded
    flat[4::101] = 1e-38
    return value.to(torch.bfloat16)


def blocks(operations, rows, seed=0):
    generator = torch.Generator().manual_seed(seed)
    finished = operations.from_chips([hard_bf16(generator, (1, 1, rows, 5120), chip) for chip in range(operations.chips)], 'bf16')
    hidden = operations.from_chips([hard_bf16(generator, (1, 1, rows, 5120), chip) for chip in range(operations.chips)], 'bf16')
    return finished, hidden


def keep(tensor):
    return tensor


class ArithmeticTests(EnvTestCase):
    environment = TAIL_ON

    def test_swiglu_is_the_nine_served_ops_bit_for_bit(self):
        for rows in (32, 64):
            for width in (WIDTH, 64):
                for grid in ((11, 10), (13, 10)):
                    with self.subTest(rows=rows, width=width, grid=grid):
                        operations = fresh(grid=grid)
                        gate, up = projections(operations, rows, width, seed=rows + width)
                        mine = tail.swiglu_launch(operations, mesh_for(*grid), gate, up, rows, width)
                        reference = draft_mlp.swiglu_device(operations, gate, up, keep)
                        self.assertEqual(mine.shape, (1, 1, rows, width))
                        self.assertEqual(mine.dtype, 'bf16')
                        self.assertTrue(same16(mine, reference))
                        self.assertEqual(fake.emulate_tail.touched, 4 * (rows // 32) * (width // 32))

    def test_the_served_swiglu_is_nine_launches_and_the_fused_one_is_one(self):
        operations = fresh()
        gate, up = projections(operations, 64, 64)
        before = len(operations.log)
        draft_mlp.swiglu_device(operations, gate, up, keep)
        served = [entry[0] for entry in operations.log[before:]]
        self.assertEqual(served, ['typecast'] * 2 + ['typecast', 'silu', 'typecast', 'typecast', 'typecast', 'multiply', 'typecast'])
        before = len(operations.log)
        tail.swiglu_launch(operations, mesh_for(), gate, up, 64, 64)
        self.assertEqual([entry[0] for entry in operations.log[before:]], ['empty', 'generic_op'])

    def test_the_fused_gate_up_projection_is_read_in_place(self):
        for rows in (32, 64):
            with self.subTest(rows=rows):
                operations = fresh()
                gate, up = projections(operations, rows, WIDTH, seed=7)
                fused = operations.from_chips([torch.cat([g.data, u.data], dim=3) for g, u in zip(gate.shards, up.shards, strict=True)], 'fp32')
                mine = tail.swiglu_fused_launch(operations, mesh_for(), fused, rows, WIDTH)
                reference = draft_mlp.swiglu_device(operations, gate, up, keep)
                self.assertTrue(same16(mine, reference))
                _, program = operations.launches[-1]
                reader = fake.kernels_by_name(next(iter(program.values())))['draft_tail_tp_io.cpp']
                words = reader.runtime_args[0][0]
                columns = WIDTH // 32
                self.assertEqual(words[0], words[1], 'both operands are the one fused tensor')
                self.assertEqual(words[4:], [2 * columns, 2 * columns, columns, columns], 'strides 2C, up half C columns in')

    def test_the_residual_is_the_four_served_ops_bit_for_bit(self):
        for rows in (32, 64):
            for grid in ((11, 10), (13, 10)):
                with self.subTest(rows=rows, grid=grid):
                    operations = fresh(grid=grid)
                    finished, hidden = blocks(operations, rows, seed=rows)
                    mine = tail.residual_launch(operations, mesh_for(*grid), finished, hidden, rows)
                    reference = tail.served_residual(operations, finished, hidden, keep)
                    self.assertEqual(mine.shape, (1, 1, rows, 5120))
                    self.assertTrue(same16(mine, reference))
                    self.assertEqual(fake.emulate_tail.touched, 4 * (rows // 32) * 160)

    def test_the_data_really_exercises_every_rounding_point(self):
        operations = fresh()
        gate, up = projections(operations, 64, WIDTH, seed=1)
        out = tail.swiglu_launch(operations, mesh_for(), gate, up, 64, WIDTH).shards[0].data
        self.assertTrue(torch.isinf(out).any(), 'overflow to infinity must occur')
        self.assertTrue(torch.isnan(tail.swiglu_launch(operations, mesh_for(), gate, up, 64, WIDTH).shards[1].data).any(), 'silu(-inf) is NaN')
        self.assertTrue(((out == 0) & (out.view(torch.int16) < 0)).any(), 'a negative zero must occur')
        finished, hidden = blocks(operations, 64, seed=1)
        total = tail.residual_launch(operations, mesh_for(), finished, hidden, 64).shards[1].data
        self.assertTrue(torch.isinf(total).any())

    def test_a_missing_rounding_point_or_a_swap_changes_the_bits(self):
        operations = fresh()
        gate, up = projections(operations, 64, WIDTH, seed=5)
        reference = draft_mlp.swiglu_device(operations, gate, up, keep)
        wide = lambda shard: shard.data.float()                                            # noqa: E731
        silu = torch.nn.functional.silu
        round16 = fake.round_bf16

        def variant(name):
            out = []
            for g, u in zip(gate.shards, up.shards, strict=True):
                a, b = wide(g), wide(u)
                if name == 'no_gate_round':
                    r = round16(round16(silu(a)) * round16(b))
                elif name == 'no_activation_round':
                    r = round16(silu(round16(a)) * round16(b))
                elif name == 'no_up_round':
                    r = round16(round16(silu(round16(a))) * b)
                elif name == 'truncating_pack':
                    r = round16(silu(round16(a))) * round16(b)
                    r = fake._int_to_float((r.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF) & 0xFFFF0000)    # truncate, not round
                elif name == 'swapped':
                    r = round16(round16(silu(round16(b))) * round16(a))
                else:
                    r = round16(round16(silu(round16(a))) * round16(b))
                out.append(r.to(torch.bfloat16))
            return operations.from_chips(out, 'bf16')
        self.assertTrue(same16(variant('exact'), reference), 'the control variant is the pipeline')
        for name in ('no_gate_round', 'no_activation_round', 'no_up_round', 'truncating_pack', 'swapped'):
            self.assertFalse(same16(variant(name), reference), name)

    def test_the_residual_sum_has_bits_below_bf16_so_a_missing_final_rounding_would_show(self):
        operations = fresh()
        finished, hidden = blocks(operations, 64, seed=9)
        reference = tail.served_residual(operations, finished, hidden, keep)
        unrounded = [a.data.float() + b.data.float() for a, b in zip(finished.shards, hidden.shards, strict=True)]
        differing = sum(int((total.contiguous().view(torch.int32) != kept.data.float().contiguous().view(torch.int32)).sum())
                        for total, kept in zip(unrounded, reference.shards, strict=True))
        self.assertGreater(differing, 1000)


class LaunchTests(EnvTestCase):
    environment = TAIL_ON

    def test_the_entry_points_return_the_launch_and_retain_only_the_result(self):
        operations = fresh()
        mesh = mesh_for()
        gate, up = projections(operations, 64, 64)
        kept = []
        result = tail.swiglu(operations, mesh, gate, up, lambda tensor: kept.append(tensor) or tensor, served=draft_mlp.swiglu_device)
        self.assertEqual(kept, [result])
        self.assertTrue(same16(result, draft_mlp.swiglu_device(operations, gate, up, keep)))
        kept.clear()
        finished, hidden = blocks(operations, 64)
        result = tail.residual(operations, mesh, finished, hidden, lambda tensor: kept.append(tensor) or tensor)
        self.assertEqual(kept, [result])
        kept.clear()
        fused = operations.from_chips([torch.cat([g.data, u.data], dim=3) for g, u in zip(gate.shards, up.shards, strict=True)], 'fp32')
        result = tail.swiglu_fused(operations, mesh, fused, lambda tensor: kept.append(tensor) or tensor, served=draft_mlp.swiglu_device)
        self.assertEqual(kept, [result])

    def test_one_engaged_line_per_kind_and_site(self):
        operations = fresh()
        mesh = mesh_for()
        gate, up = projections(operations, 64, 64)
        finished, hidden = blocks(operations, 64)
        for _ in range(3):
            tail.swiglu(operations, mesh, gate, up, keep, served=draft_mlp.swiglu_device)
            tail.residual(operations, mesh, finished, hidden, keep)
        engaged = [line for line in self.lines if line.startswith(fusion.TAIL_ENGAGED)]
        self.assertEqual(len(engaged), 2)
        self.assertIn('kind=swiglu', engaged[0])
        self.assertIn('kind=residual', engaged[1])
        self.assertIn('tasks=4 workers=4', engaged[0], '2 tile rows x 2 columns at width 64')
        self.assertIn('tasks=320 workers=110', engaged[1], '2 tile rows x 160 columns')

    def test_what_it_cannot_take_goes_to_the_served_ops_and_says_why_once(self):
        operations = fresh()
        mesh = mesh_for()
        gate, up = projections(operations, 64, 64)
        served_calls = []

        def served(*args, **options):
            served_calls.append(1)
            return 'served'
        wrong = [operations.from_chips([torch.zeros(1, 1, 64, 64)] * 4, 'bf16')] * 2
        self.assertEqual(tail.swiglu(operations, mesh, wrong[0], wrong[1], keep, served=served), 'served')
        narrow = operations.from_chips([torch.zeros(1, 1, 64, 48)] * 4, 'fp32')
        self.assertEqual(tail.swiglu(operations, mesh, narrow, narrow, keep, served=served), 'served')
        odd = operations.from_chips([torch.zeros(1, 1, 64, 32)] * 4, 'fp32')
        self.assertEqual(tail.swiglu(operations, mesh, gate, odd, keep, served=served), 'served')
        self.assertEqual(tail.swiglu(operations, mesh, gate, odd, keep, served=served), 'served')
        self.assertEqual(len(served_calls), 4)
        self.assertEqual(operations.launches, [])
        lines = [line for line in self.lines if line.startswith(fusion.TAIL_FALLBACK)]
        self.assertEqual(len(lines), 3, 'the same reason prints once')
        finished, hidden = blocks(operations, 64)
        wide = operations.from_chips([torch.zeros(1, 1, 64, 4096)] * 4, 'bf16')
        result = tail.residual(operations, mesh, wide, wide, keep)
        self.assertEqual(result.shape, (1, 1, 64, 4096))
        self.assertEqual(operations.launches, [], 'the served four ops ran')

    def test_a_failing_launch_is_a_served_run_with_the_output_freed(self):
        operations = fresh()
        operations.emulators.insert(0, lambda ops, tensors, program: (_ for _ in ()).throw(RuntimeError('compile failed')))
        mesh = mesh_for()
        gate, up = projections(operations, 64, 64)
        result = tail.swiglu(operations, mesh, gate, up, keep, served=draft_mlp.swiglu_device)
        reference = draft_mlp.swiglu_device(operations, gate, up, keep)
        self.assertTrue(same16(result, reference))
        self.assertTrue(any('the launch failed' in line for line in self.lines))
        self.assertIn(('deallocate', 'empty'), operations.log)

    def test_a_grid_that_cannot_be_read_is_a_fall_back(self):
        def broken():
            raise RuntimeError('no grid')
        mesh = type(mesh_for())(shape=[1, 4], compute_with_storage_grid_size=broken)
        operations = fresh()
        finished, hidden = blocks(operations, 32)
        tail.residual(operations, mesh, finished, hidden, keep)
        self.assertEqual(operations.launches, [])
        self.assertTrue(any('the compute grid cannot be read' in line for line in self.lines))


class AuditTests(EnvTestCase):
    environment = AUDIT_ON

    def test_exact_launches_log_audit_lines_and_free_their_references(self):
        operations = fresh()
        mesh = mesh_for()
        gate, up = projections(operations, 64, 64)
        finished, hidden = blocks(operations, 64)
        before = len(operations.tensors)
        tail.swiglu(operations, mesh, gate, up, keep, served=draft_mlp.swiglu_device)
        tail.residual(operations, mesh, finished, hidden, keep)
        fused = operations.from_chips([torch.cat([g.data, u.data], dim=3) for g, u in zip(gate.shards, up.shards, strict=True)], 'fp32')
        tail.swiglu_fused(operations, mesh, fused, keep, served=draft_mlp.swiglu_device)
        audits = [line for line in self.lines if line.startswith(fusion.TAIL_AUDIT_LINE)]
        self.assertEqual(len(audits), 3)
        self.assertTrue(all('exact=True' in line for line in audits))
        self.assertIn('kind=swiglu', audits[0])
        self.assertIn('kind=residual', audits[1])
        self.assertIn('fused=1', audits[2])
        references = [tensor for tensor in operations.tensors[before:] if tensor.name in ('typecast', 'silu', 'multiply', 'add', 'slice')]
        # the served references the audits ran: nine SwiGLU ops, four residual ops, two slices and nine SwiGLU ops for the fused call
        self.assertEqual(len(references), 9 + 4 + 2 + 9)
        self.assertTrue(all(tensor.freed for tensor in references), 'the audit frees every reference it made')

    def test_a_flipped_bit_is_a_logged_mismatch_and_an_assertion(self):
        operations = fresh()
        genuine = operations.emulators[1]

        def corrupt(ops, tensors, program):
            taken = genuine(ops, tensors, program)
            output = tensors[-1]
            output.shards[3].data.view(torch.int16).reshape(-1)[77] ^= 0x0100
            return taken
        operations.emulators[1] = corrupt
        mesh = mesh_for()
        finished, hidden = blocks(operations, 64)
        with self.assertRaises(AssertionError):
            tail.residual(operations, mesh, finished, hidden, keep)
        mismatch = [line for line in self.lines if line.startswith(fusion.TAIL_MISMATCH)]
        self.assertEqual(len(mismatch), 1)
        self.assertIn('chips=[3]', mismatch[0])

    def test_the_audit_needs_the_lever(self):
        with patch.dict(os.environ, {'QWEN_FAST_TP': '4', 'QWEN_FAST_DRAFT_TAIL_AUDIT': '1'}, clear=True):
            operations = fresh()
            finished, hidden = blocks(operations, 32)
            with self.assertRaisesRegex(ValueError, 'compare nothing'):
                tail.residual(operations, mesh_for(), finished, hidden, keep)


class SourceCrossChecks(unittest.TestCase):
    def test_each_kernel_reads_exactly_the_runtime_arguments_the_builder_writes(self):
        with patch.dict(os.environ, TAIL_ON, clear=True):
            operations = fresh()
            mesh = mesh_for()
            gate, up = projections(operations, 64, 64)
            finished, hidden = blocks(operations, 64)
            tail.swiglu_launch(operations, mesh, gate, up, 64, 64)
            tail.residual_launch(operations, mesh, finished, hidden, 64)
        for _, program in operations.launches:
            for name, kernel in fake.kernels_by_name(next(iter(program.values()))).items():
                words = kernel.runtime_args[0][0]
                self.assertEqual(SourceTests.indices(name), list(range(len(words))), name)

    def test_the_swiglu_kernel_runs_the_served_primitives_in_the_served_order(self):
        text = (HERE / 'draft_tail_tp_swiglu_compute.cpp').read_text()
        order = ['copy_tile(0, 0, 0);', 'copy_tile(1, 0, 1);', 'typecast_tile<', 'silu_tile(0);', 'typecast_tile<', 'typecast_tile<',
                 'mul_binary_tile(0, 1, 0);', 'typecast_tile<', 'pack_tile(0, 16);']
        position = 0
        for needle in order:
            position = text.index(needle, position) + 1
        self.assertEqual(text.count('        typecast_tile<'), 4, 'four fp32 -> bf16 roundings: gate, activation, up, product')
        self.assertEqual(text.count('typecast_tile_init<'), 3)
        self.assertIn('silu_tile_init();', text)
        self.assertIn('mul_binary_tile_init();', text)
        self.assertNotIn('approx', text.lower().replace('math_approx', ''))
        residual = (HERE / 'draft_tail_tp_residual_compute.cpp').read_text()
        for needle in ('add_binary_tile(0, 1, 0);', 'typecast_tile<', 'pack_tile(0, 16);'):
            self.assertIn(needle, residual)

    def test_the_compute_configs_unpack_fp32_inputs_to_the_destination_and_leave_bf16_ones_alone(self):
        with patch.dict(os.environ, TAIL_ON, clear=True):
            operations = fresh()
            mesh = mesh_for()
            gate, up = projections(operations, 32, 64)
            finished, hidden = blocks(operations, 32)
            tail.swiglu_launch(operations, mesh, gate, up, 32, 64)
            tail.residual_launch(operations, mesh, finished, hidden, 32)
        swiglu_config = fake.kernels_by_name(next(iter(operations.launches[0][1].values())))['draft_tail_tp_swiglu_compute.cpp'].config
        residual_config = fake.kernels_by_name(next(iter(operations.launches[1][1].values())))['draft_tail_tp_residual_compute.cpp'].config
        self.assertEqual([i for i, mode in enumerate(swiglu_config.unpack_to_dest_mode) if mode == 'fp32'], [0, 1])
        self.assertEqual([i for i, mode in enumerate(residual_config.unpack_to_dest_mode) if mode == 'fp32'], [])
        for config in (swiglu_config, residual_config):
            self.assertTrue(config.fp32_dest_acc_en)
            self.assertFalse(config.math_approx_mode)

    def test_every_runtime_file_exists(self):
        for name in tail.RUNTIME_FILES:
            self.assertTrue((HERE / name).is_file(), name)


if __name__ == '__main__':
    unittest.main()
