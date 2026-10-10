"""The four drafter levers of tp4/fx-wp6 inside draft_mlp_branch, on the functional fake device: QWEN_FAST_DRAFT_REDUCE, _TAIL, _GATEUP1, _MM_GRID.

execute_mlp_branch runs for real on wp6_fake_device (torch values, the K-blocked matmul, the launches EXECUTED by the transliterations of the
.cpp kernels), as the single-user pass (32 rows, the feature_collective dispatcher) and as the quad pass (64 rows, quad.gather_add_projection,
trace-owned), with random bf16 weights split over four chips. Held:

  - every combination of the four flags returns the very bits the flags-off branch returns (the branch's `output`, and the intermediates each
    lever replaces), on both passes;
  - flags off: the branch is the served op sequence and none of the new modules is imported (prepare and execute);
  - what each lever removes from the op sequence: REDUCE the four slices and three adds of the quad's chain (the single-user pass takes the
    feature dispatcher's launch), TAIL the seven typecasts, silu and multiply of the SwiGLU and the residual's two typecasts, add and typecast,
    GATEUP1 one matmul and one weight upload; MM_GRID changes the grid field of the program configs and nothing else;
  - GATEUP1's weight: [gate | up] per chip, its halves equal the separate weights, its matmul output's halves equal the separate matmuls';
  - MM_GRID: the core counts, per-core columns, in0_block_w, output subblocks and per_core_M of every program config are unchanged, the grid is
    as wide as the device's (11, 13 or anything) and holds the cores, and the audit catches a grid that changes a value;
  - the audits: every audit flag on, every audit line exact=True, the references freed, and a corrupted launch is a logged mismatch.

    py -3.11 -B -m unittest test_draft_wp6_branch      (from scripts/ci)
"""

import itertools
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import draft_fusion_tp as fusion  # noqa: E402
import wp6_fake_device as fake  # noqa: E402
from tp_test_support import four_cards  # noqa: E402

FLAGS = (fusion.REDUCE, fusion.TAIL, fusion.GATEUP1, fusion.MM_GRID)
COLLECTIVES = SimpleNamespace(get_and_cycle_ag_semaphore_handles=lambda: 'ag', get_and_cycle_barrier_semaphore_handle=lambda: 'barrier')
WIDTH = 64                                   # per-chip MLP width of the test model (two tile columns); the real one is 4,352
NEW_MODULES = ('draft_reduce_tp', 'draft_tail_tp', 'draft_gateup_tp', 'draft_mmgrid_tp')


def mesh_for(x=13, y=10):
    return SimpleNamespace(shape=[1, 4], compute_with_storage_grid_size=lambda: SimpleNamespace(x=x, y=y))


def model(seed=0):
    generator = torch.Generator().manual_seed(seed)
    draw = lambda *shape: (torch.randn(*shape, generator=generator) * 0.05).to(torch.bfloat16)      # noqa: E731
    weights = {'layers.0.mlp.gate_proj.weight': draw(4 * WIDTH, 5120), 'layers.0.mlp.up_proj.weight': draw(4 * WIDTH, 5120),
               'layers.0.mlp.down_proj.weight': draw(5120, 4 * WIDTH)}
    convolution = {'layers.0.post_attention_layernorm.weight': torch.ones(5120),
                   'layers.0.mlp_conv.kernel_projection.weight': draw(1280, 5120),
                   'layers.0.mlp_conv.base_kernel': draw(2, 2, 5120)}
    return weights, convolution


def convolve(operations, mesh, hidden, dynamic, base, **options):
    """A stand-in for the learned convolution: a fixed function of its input (bf16 out)."""
    return operations._make([(shard.data.float() * 0.75).to(torch.bfloat16) for shard in hidden.shards], 'bf16', 'tile', 'conv')


class Run:
    """One execute_mlp_branch call under a set of flags."""

    def __init__(self, flags=(), audits=(), rows=32, quad=False, grid=(13, 10), seed=0, hook=None):
        self.environment = {'QWEN_FAST_TP': '4', **{name: '1' for name in flags}, **{name + '_AUDIT': '1' for name in audits}}
        self.rows, self.quad, self.grid, self.seed, self.hook = rows, quad, grid, seed, hook
        self.lines = []

    def __enter__(self):
        for name in FLAGS:
            os.environ.pop(name, None)
            os.environ.pop(name + '_AUDIT', None)
        self.patches = [patch.dict(os.environ, self.environment), patch('mesh_link_policy.projection_links', return_value=1),
                        patch('feature_collective_tp.projection_links', return_value=1),
                        patch.object(fusion, 'log_line', side_effect=self.lines.append)]
        for item in self.patches:
            item.start()
        fusion.reset()
        return self

    def __exit__(self, *exc):
        for item in reversed(self.patches):
            item.stop()

    def execute(self):
        import draft_mlp_branch
        import quad_draft_tp

        with four_cards():
            operations = fake.install_emulators(fake.FakeOperations(4), self.grid)
            if self.hook:
                self.hook(operations)
            mesh = mesh_for(*self.grid)
            weights, convolution = model(self.seed)
            kept = []
            retain = lambda value: kept.append(value) or value          # noqa: E731
            parameters = draft_mlp_branch.prepare_mlp_branch(operations, mesh, weights, convolution, lambda value: value)
            self.uploads = len([entry for entry in operations.tensors if entry.name == 'from_torch'])
            generator = torch.Generator().manual_seed(self.seed + 1)
            hidden = operations.from_chips([(torch.randn(1, 1, self.rows, 5120, generator=generator)).to(torch.bfloat16)] * 4, 'bf16')
            extra = {}
            if self.quad:
                extra = dict(quad=quad_draft_tp.QuadPass(), boundaries=tuple((start, start + 16) for start in range(0, self.rows, 16)))
            mark = len(operations.log)
            state = draft_mlp_branch.execute_mlp_branch(operations, mesh, COLLECTIVES, hidden, weights, convolution, retain,
                parameters=parameters, trace_safe=True, convolution_operation=convolve, **extra)
            self.operations, self.parameters, self.state, self.kept = operations, parameters, state, kept
            self.log = operations.log[mark:]
            self.programs = operations.programs
            return self


def output_bits(run):
    return [shard.data.contiguous().view(torch.int16) for shard in run.state['output'].shards]


def same(left, right):
    return all(a.shape == b.shape and torch.equal(a, b) for a, b in zip(left, right, strict=True))


def count(run, name):
    return len([entry for entry in run.log if entry[0] == name])


class EqualityTests(unittest.TestCase):
    def test_every_combination_of_the_four_flags_returns_the_flags_off_bits_on_both_passes(self):
        for rows, quad in ((32, False), (64, True)):
            with Run(rows=rows, quad=quad) as run:
                control = run.execute()
            expected = output_bits(control)
            combinations = [subset for size in range(len(FLAGS) + 1) for subset in itertools.combinations(FLAGS, size)]
            for flags in combinations:
                with self.subTest(rows=rows, flags=[name[len('QWEN_FAST_DRAFT_'):] for name in flags]):
                    with Run(flags, rows=rows, quad=quad) as run:
                        mine = run.execute()
                    self.assertTrue(same(output_bits(mine), expected))
                    self.assertEqual([line for line in run.lines if 'fell back' in line], [], 'no lever may fall back on these shapes')

    def test_the_replaced_intermediates_are_the_served_ones(self):
        with Run(rows=64, quad=True) as run:
            control = run.execute()
        with Run(FLAGS, rows=64, quad=True) as run:
            mine = run.execute()
        for name in ('activation', 'reduced', 'finished', 'conv_projection'):
            left = [shard.data.contiguous().view(torch.int16 if shard.data.element_size() == 2 else torch.int32)
                    for shard in control.state[name].shards]
            right = [shard.data.contiguous().view(torch.int16 if shard.data.element_size() == 2 else torch.int32)
                     for shard in mine.state[name].shards]
            self.assertTrue(same(left, right), name)

    def test_other_data_and_other_grids_do_not_change_the_answer(self):
        for seed, grid in ((1, (11, 10)), (2, (13, 10)), (3, (8, 10))):
            with Run(rows=64, quad=True, seed=seed, grid=grid) as run:
                control = run.execute()
            with Run(FLAGS, rows=64, quad=True, seed=seed, grid=grid) as run:
                mine = run.execute()
            self.assertTrue(same(output_bits(mine), output_bits(control)), (seed, grid))


class OffTests(unittest.TestCase):
    def test_flags_off_runs_the_served_ops_and_imports_none_of_the_new_modules(self):
        saved = {name: sys.modules.pop(name, None) for name in NEW_MODULES}
        try:
            with Run(rows=64, quad=True) as run:
                run.execute()
            self.assertEqual([name for name in NEW_MODULES if name in sys.modules], [])
            self.assertEqual(count(run, 'generic_op'), 0)
            self.assertEqual(count(run, 'matmul'), 4)
            self.assertEqual(count(run, 'silu'), 1)
            self.assertEqual(count(run, 'multiply'), 1)
            self.assertEqual(count(run, 'typecast'), 1 + 1 + 7 + 3, 'conv rounding, gather rounding, SwiGLU, residual')
            self.assertEqual(count(run, 'slice'), 4 + 4, 'dynamic kernels and the gather\'s slices')
            self.assertEqual(count(run, 'add'), 3 + 1)
            self.assertEqual(run.uploads, 9)
            self.assertEqual(len(run.parameters['device_projections']), 3)
            self.assertNotIn('gateup1', run.parameters)
            self.assertEqual([program.compute_with_storage_grid_size for program in run.programs], [(8, 5), (8, 10), (8, 10), (8, 10)])
            self.assertEqual(run.lines, [])
        finally:
            for name, module in saved.items():
                if module is not None:
                    sys.modules[name] = module

    def test_each_flag_is_strict(self):
        for name in FLAGS:
            with Run() as run:
                with patch.dict(os.environ, {name: 'yes'}):
                    with self.assertRaises(ValueError):
                        run.execute()


class LeverTests(unittest.TestCase):
    def run_with(self, *flags, rows=64, quad=True, **options):
        with Run(flags, rows=rows, quad=quad, **options) as run:
            return run.execute()

    def test_reduce_replaces_the_quads_four_slices_and_three_adds_with_one_launch(self):
        run = self.run_with(fusion.REDUCE)
        self.assertEqual(count(run, 'generic_op'), 1)
        self.assertEqual(count(run, 'slice'), 4, 'the four dynamic kernel slices remain')
        self.assertEqual(count(run, 'add'), 1, 'the residual add remains')
        self.assertEqual(count(run, 'all_gather'), 1)

    def test_reduce_on_the_single_user_pass_goes_through_the_feature_dispatcher(self):
        run = self.run_with(fusion.REDUCE, rows=32, quad=False)
        self.assertEqual(count(run, 'generic_op'), 1)
        self.assertEqual(count(run, 'add'), 1)
        self.assertTrue(any('site=feature' in line for line in run.lines if 'reduce engaged' in line))

    def test_tail_replaces_the_swiglu_and_the_residual(self):
        run = self.run_with(fusion.TAIL)
        self.assertEqual(count(run, 'generic_op'), 2)
        self.assertEqual((count(run, 'silu'), count(run, 'multiply')), (0, 0))
        self.assertEqual(count(run, 'typecast'), 1 + 1, 'conv rounding and the gather rounding remain')
        self.assertEqual(count(run, 'add'), 3, 'the gather\'s adds remain')

    def test_gateup1_runs_one_matmul_and_uploads_one_weight_fewer(self):
        run = self.run_with(fusion.GATEUP1)
        self.assertEqual(count(run, 'matmul'), 3)
        self.assertEqual(run.uploads, 8)
        self.assertEqual(len(run.parameters['device_projections']), 2)
        self.assertIs(run.parameters['gateup1'], True)
        self.assertTrue(run.state['gateup1'])
        self.assertEqual(count(run, 'slice'), 4 + 4 + 2, 'without the tail the halves are cut with two tile-aligned slices')
        fused = [entry for entry in run.log if entry[0] == 'matmul' and entry[2] == (5120, 2 * WIDTH)]
        self.assertEqual(fused, [('matmul', (1, 1, 64, 5120), (5120, 2 * WIDTH), 2)], 'one matmul over [gate | up] at twice the served columns')

    def test_gateup1_with_the_tail_reads_the_halves_in_place(self):
        run = self.run_with(fusion.GATEUP1, fusion.TAIL)
        self.assertEqual(count(run, 'slice'), 4 + 4, 'no half-slices')
        self.assertEqual(count(run, 'matmul'), 3)
        self.assertEqual(count(run, 'generic_op'), 2)

    def test_the_fused_weight_is_gate_then_up_per_chip_and_its_output_halves_are_the_separate_matmuls(self):
        import draft_mlp_branch

        with Run((fusion.GATEUP1,), rows=64, quad=True) as run:
            fused = run.execute()
        with Run(rows=64, quad=True) as run:
            separate = run.execute()
        gate_up = fused.parameters['device_projections'][0]
        gate, up = separate.parameters['device_projections'][0], separate.parameters['device_projections'][1]
        for chip in range(4):
            self.assertEqual(tuple(gate_up.shards[chip].data.shape), (5120, 2 * WIDTH))
            self.assertTrue(torch.equal(gate_up.shards[chip].data[:, :WIDTH], gate.shards[chip].data))
            self.assertTrue(torch.equal(gate_up.shards[chip].data[:, WIDTH:], up.shards[chip].data))
        self.assertTrue(torch.equal(fused.parameters['device_projections'][1].shards[2].data, separate.parameters['device_projections'][2].shards[2].data))
        halves = fused.state['projections'][0]
        for chip in range(4):
            self.assertTrue(torch.equal(halves.shards[chip].data[..., :WIDTH], separate.state['projections'][0].shards[chip].data))
            self.assertTrue(torch.equal(halves.shards[chip].data[..., WIDTH:], separate.state['projections'][1].shards[chip].data))
        self.assertEqual(draft_mlp_branch.gate_up_columns_of(fused.parameters), 1)

    def test_a_shard_the_fused_program_cannot_take_keeps_the_separate_weights_and_says_why(self):
        import draft_gateup_tp

        shards = ((torch.zeros(5120, 32 * 81), torch.zeros(5120, 32 * 81), torch.zeros(1, 1)),) * 4
        self.assertIn('whole per-core columns', draft_gateup_tp.shard_problem(shards))
        self.assertIn('differ', draft_gateup_tp.shard_problem(((torch.zeros(5120, 64), torch.zeros(5120, 96), None),)))
        self.assertIsNone(draft_gateup_tp.shard_problem(((torch.zeros(5120, 4352), torch.zeros(5120, 4352), None),)))
        found = draft_gateup_tp.plan(136)
        self.assertEqual((found['columns'], found['cores'], found['tiles']), (4, 68, 272))
        pair = draft_gateup_tp.plan(272)
        self.assertEqual((pair['columns'], pair['cores']), (8, 68))
        with self.assertRaises(ValueError):
            draft_gateup_tp.plan(0)

    def test_the_weight_the_flag_changes_is_not_extra_dram(self):
        with Run((fusion.GATEUP1,)) as run:
            fused = run.execute()
        with Run() as run:
            control = run.execute()
        bytes_of = lambda r: sum(t.shards[0].data.numel() for t in r.parameters['device_projections'])       # noqa: E731
        self.assertEqual(bytes_of(fused), bytes_of(control))


class GridTests(unittest.TestCase):
    CONV, GATE_UP, DOWN = 'conv', 'gate_up', 'down'

    def programs(self, flags, grid=(13, 10), **options):
        with Run(flags, grid=grid, rows=64, quad=True, **options) as run:
            return run.execute().programs

    def test_only_the_grid_field_changes_and_the_cores_are_the_served_cores(self):
        served = self.programs(())
        wide = self.programs((fusion.MM_GRID,))
        self.assertEqual(len(served), len(wide))
        for old, new in zip(served, wide, strict=True):
            for field in ('in0_block_w', 'out_subblock_h', 'out_subblock_w', 'per_core_M', 'per_core_N', 'fuse_batch', 'fused_activation', 'mcast_in0'):
                self.assertEqual(getattr(old, field), getattr(new, field), field)
            self.assertEqual(new.in0_block_w, 4)
            self.assertNotEqual(old.compute_with_storage_grid_size, new.compute_with_storage_grid_size)

    def test_the_grids_follow_the_device_width_and_hold_the_cores(self):
        # (conv: 40 cores at 1 column, gate and up: 2 tiles at 1 column = 2 cores each, down: 160 tiles at 2 = 80 cores)
        for grid, expected in (((13, 10), [(13, 4), (2, 1), (2, 1), (13, 7)]), ((11, 10), [(11, 4), (2, 1), (2, 1), (11, 8)]),
                               ((8, 10), [(8, 5), (2, 1), (2, 1), (8, 10)])):
            wide = self.programs((fusion.MM_GRID,), grid=grid)
            self.assertEqual([program.compute_with_storage_grid_size for program in wide], expected, grid)
            for program, cores in zip(wide, (40, 2, 2, 80), strict=True):
                x, y = program.compute_with_storage_grid_size
                self.assertGreaterEqual(x * y, cores)
                self.assertLess((y - 1) * x, cores, 'no empty row')

    def test_the_real_shapes_of_the_quad_at_four_cards(self):
        import draft_mmgrid_tp

        mesh13, mesh11 = mesh_for(13, 10), mesh_for(11, 10)
        # the drafter's per-chip shards: conv projection (5120, 1280) at 1 column, gate and up (5120, 4352) at 2, fused (5120, 8704) at 4, down (4352, 5120) at 2
        self.assertEqual([draft_mmgrid_tp.wide_grid(mesh13, draft_mmgrid_tp.cores_for(shape, columns))
                          for shape, columns in (((5120, 1280), 1), ((5120, 4352), 2), ((5120, 8704), 4), ((4352, 5120), 2))],
                         [(13, 4), (13, 6), (13, 6), (13, 7)])
        self.assertEqual([draft_mmgrid_tp.wide_grid(mesh11, draft_mmgrid_tp.cores_for(shape, columns))
                          for shape, columns in (((5120, 1280), 1), ((5120, 4352), 2), ((5120, 8704), 4), ((4352, 5120), 2))],
                         [(11, 4), (11, 7), (11, 7), (11, 8)])
        self.assertEqual([draft_mmgrid_tp.cores_for(shape, columns) for shape, columns in
                          (((5120, 1280), 1), ((5120, 4352), 2), ((5120, 8704), 4), ((4352, 5120), 2))], [40, 68, 68, 80])
        with self.assertRaises(ValueError):
            draft_mmgrid_tp.wide_grid(mesh_for(2, 2), 80)
        with self.assertRaises(ValueError):
            draft_mmgrid_tp.cores_for((5120, 40), 1)

    def test_a_weight_or_grid_it_cannot_place_keeps_the_served_grid_and_says_why_once(self):
        import draft_mmgrid_tp

        operations = fake.FakeOperations()
        fusion.reset()
        lines = []
        with patch.dict(os.environ, {'QWEN_FAST_TP': '4', fusion.MM_GRID: '1'}), patch.object(fusion, 'log_line', side_effect=lines.append):
            small = mesh_for(2, 2)
            weight = SimpleNamespace(shape=(4352, 5120))
            for _ in range(2):
                self.assertEqual(draft_mmgrid_tp.grid_for(operations, small, None, weight, (8, 10), 2, 64, None), (8, 10))
            self.assertEqual(draft_mmgrid_tp.grid_for(operations, mesh_for(), None, SimpleNamespace(shape=(4352, 5100)), (8, 10), 2, 64, None), (8, 10))
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(line.startswith(fusion.MM_GRID_FALLBACK) for line in lines))

    def test_the_engaged_line_names_the_grids_and_the_unchanged_k_loop(self):
        with Run((fusion.MM_GRID,), rows=64, quad=True) as run:
            run.execute()
        engaged = [line for line in run.lines if line.startswith(fusion.MM_GRID_ENGAGED)]
        self.assertEqual(len(engaged), 3, 'conv, gate/up and down')
        self.assertTrue(all('in0_block_w=4' in line for line in engaged))
        self.assertIn('served=8x10 wide=13x7 cores=80 per_core_N=2', ' '.join(engaged))


class AuditTests(unittest.TestCase):
    def test_every_audit_on_passes_exact_and_leaves_the_answer_alone(self):
        with Run(rows=64, quad=True) as run:
            control = run.execute()
        with Run(FLAGS, FLAGS, rows=64, quad=True) as run:
            mine = run.execute()
        self.assertTrue(same(output_bits(mine), output_bits(control)))
        for _, audit, _, _, marker, mismatch, what in fusion.TABLE:
            lines = [line for line in run.lines if line.startswith(marker) and mismatch not in line]
            self.assertTrue(lines, what)
            self.assertTrue(all('exact=True' in line for line in lines), what)
        self.assertEqual([line for line in run.lines if 'mismatch' in line or 'fell back' in line], [])
        import draft_wp6_smoke

        env = {name: '1' for name in FLAGS}
        env.update({name + '_AUDIT': '1' for name in FLAGS})
        self.assertEqual(draft_wp6_smoke.problems(env, '\n'.join(run.lines)), [])

    def test_the_gateup_audit_frees_the_separate_weights_it_uploaded(self):
        with Run((fusion.GATEUP1,), (fusion.GATEUP1,), rows=64, quad=True) as run:
            executed = run.execute()
        audited = [tensor for tensor in executed.operations.tensors if tensor.name == 'from_torch'][8:]
        self.assertEqual(len(audited), 2, 'separate gate and up weights, uploaded for the audit')
        self.assertTrue(all(tensor.freed for tensor in audited))

    def test_a_grid_that_changes_a_value_is_caught_by_the_grid_audit(self):
        def noise(grid, data):
            if grid is not None and grid[0] != 8:
                data = data.clone()
                data.view(torch.int32).reshape(-1)[5] ^= 1
            return data

        with Run((fusion.MM_GRID,), (fusion.MM_GRID,), rows=64, quad=True, hook=lambda ops: setattr(ops, 'grid_noise', noise)) as run:
            with self.assertRaises(AssertionError):
                run.execute()
        mismatch = [line for line in run.lines if line.startswith(fusion.MM_GRID_MISMATCH)]
        self.assertEqual(len(mismatch), 1)
        self.assertIn('served=8x5', mismatch[0])

    def test_the_audits_need_their_levers(self):
        for name in FLAGS:
            with Run((), (name,)) as run:
                with self.assertRaisesRegex(ValueError, 'compare nothing'):
                    run.execute()


if __name__ == '__main__':
    unittest.main()
