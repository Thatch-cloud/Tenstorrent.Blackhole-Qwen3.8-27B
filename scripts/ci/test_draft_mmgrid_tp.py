"""draft_mmgrid_tp (QWEN_FAST_DRAFT_MM_GRID, R2): the drafter's MLP-branch matmuls on program grids sized from the device.

v678 (the unlocked 13 x 10 cards) lost 17-36 % on matmuls whose program grid is a fixed 8-wide rectangle; this lever changes ONE field of those
programs, compute_with_storage_grid_size, and nothing that defines the arithmetic. Held here, on the CPU:

  - the grid: (min(device width, cores), ceil(cores / width)) - 13-, 11- or 8-wide devices alike, no empty row, never more rows than the device has,
    the same number of cores as the served program (ceil(N tiles / per_core_N)), refused (with the served grid kept and one logged line) when the
    device grid cannot be read or is too short;
  - the program: every field but the grid equals the served program's (in0_block_w 4, the subblocks, per_core_M, per_core_N, fuse_batch, the
    activation, mcast_in0) - compared against the program draft_mlp_branch builds with the lever off;
  - the audit: both grids run on the same operands and every chip's bytes are compared; a grid that changes one bit is a logged mismatch and an
    AssertionError; only the first AUDIT_CALLS per site and row count are audited, and the audit's temporaries are freed;
  - what the lever does NOT touch: the flag off leaves the served grids, and the module is not imported.

The whole MLP branch's equality with the lever on is test_draft_wp6_branch; the wo projection (draft_attention_branch.py) and the fused commit's 80-core
projection (fused_commit_tp.py) are other packages' files and call wide_grid() / program_config() the same way.

    py -3.11 -B -m unittest test_draft_mmgrid_tp      (from scripts/ci)
"""

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
import draft_mmgrid_tp as mmgrid  # noqa: E402
import wp6_fake_device as fake  # noqa: E402
from test_draft_wp6_branch import Run, mesh_for  # noqa: E402

SHAPES = (((5120, 1280), 1, 40), ((5120, 4352), 2, 68), ((5120, 8704), 4, 68), ((4352, 5120), 2, 80), ((1024, 5120), 2, 80))


class GridTests(unittest.TestCase):
    def test_cores_follow_the_weight_and_the_columns(self):
        for shape, columns, cores in SHAPES:
            self.assertEqual(mmgrid.cores_for(shape, columns), cores, shape)
        self.assertEqual(mmgrid.cores_for((5120, 96), 3), 1)
        for bad in (((5120,), 1), ((5120, 40), 1), ((5120, 64), 0), ((5120, 64), 1.5)):
            with self.assertRaises(ValueError):
                mmgrid.cores_for(*bad)

    def test_the_grid_is_as_wide_as_the_device_and_holds_exactly_the_cores_in_whole_rows(self):
        for device in ((13, 10), (11, 10), (8, 10), (13, 12), (16, 9)):
            mesh = mesh_for(*device)
            for _, _, cores in SHAPES + (((0, 0), 1, 1), ((0, 0), 1, 7), ((0, 0), 1, 13), ((0, 0), 1, 14), ((0, 0), 1, 130)):
                if cores > device[0] * device[1]:
                    with self.assertRaises(ValueError):
                        mmgrid.wide_grid(mesh, cores)
                    continue
                width, rows = mmgrid.wide_grid(mesh, cores)
                self.assertEqual(width, min(device[0], cores))
                self.assertGreaterEqual(width * rows, cores)
                self.assertLess(width * (rows - 1), cores, 'no empty row')
                self.assertLessEqual(rows, device[1])

    def test_a_device_that_is_too_short_or_unreadable_is_refused(self):
        with self.assertRaises(ValueError):
            mmgrid.wide_grid(mesh_for(2, 2), 80)

        def broken():
            raise RuntimeError('no grid')
        with self.assertRaisesRegex(ValueError, 'cannot be read'):
            mmgrid.wide_grid(SimpleNamespace(compute_with_storage_grid_size=broken), 8)
        with self.assertRaises(ValueError):
            mmgrid.wide_grid(mesh_for(), 0)

    def test_the_wide_grid_is_not_the_8_wide_one_on_the_unlocked_device(self):
        self.assertEqual(mmgrid.wide_grid(mesh_for(13, 10), 80), (13, 7))
        self.assertEqual(mmgrid.wide_grid(mesh_for(13, 10), 68), (13, 6))


class ProgramTests(unittest.TestCase):
    def test_only_the_grid_differs_from_the_program_the_branch_builds(self):
        with Run(rows=64, quad=True) as run:
            served = run.execute().programs
        operations = fake.FakeOperations()
        self.assertEqual(len(served), 4)
        for program in served:
            mine = mmgrid.program_config(operations, program.compute_with_storage_grid_size, 64, program.per_core_N)
            for field in vars(program):
                self.assertEqual(getattr(mine, field), getattr(program, field), field)
            wide = mmgrid.program_config(operations, (13, 7), 64, program.per_core_N)
            self.assertEqual({field for field in vars(wide) if getattr(wide, field) != getattr(program, field)}, {'compute_with_storage_grid_size'})
        self.assertEqual(mmgrid.IN0_BLOCK_W, 4)


class FlagOffTests(unittest.TestCase):
    def test_flag_off_keeps_the_served_grids_and_never_imports_the_module(self):
        saved = sys.modules.pop('draft_mmgrid_tp', None)
        try:
            with Run(rows=64, quad=True) as run:
                executed = run.execute()
            self.assertNotIn('draft_mmgrid_tp', sys.modules)
            self.assertEqual([program.compute_with_storage_grid_size for program in executed.programs], [(8, 5), (8, 10), (8, 10), (8, 10)])
        finally:
            if saved is not None:
                sys.modules['draft_mmgrid_tp'] = saved

    def test_the_lever_is_refused_at_the_pair(self):
        with patch.dict(os.environ, {fusion.MM_GRID: '1'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'TP4 levers'):
                mmgrid.enabled()


class AuditTests(unittest.TestCase):
    environment = {'QWEN_FAST_TP': '4', fusion.MM_GRID: '1', fusion.MM_GRID_AUDIT: '1'}

    def setUp(self):
        patcher = patch.dict(os.environ, self.environment, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.lines = []
        quiet = patch.object(fusion, 'log_line', side_effect=self.lines.append)
        quiet.start()
        self.addCleanup(quiet.stop)
        fusion.reset()
        self.operations = fake.FakeOperations(4)
        generator = torch.Generator().manual_seed(1)
        self.value = self.operations.from_chips([torch.randn(1, 1, 64, 5120, generator=generator).to(torch.bfloat16) for _ in range(4)], 'bf16')
        self.weight = self.operations.from_chips([(torch.randn(5120, 256, generator=generator) * 0.05).to(torch.bfloat16) for _ in range(4)], 'bf16')

    def call(self, grid=(8, 10)):
        return mmgrid.grid_for(self.operations, mesh_for(), self.value, self.weight, grid, 2, 64, SimpleNamespace(), site='test')

    def test_an_exact_pair_of_grids_logs_the_audit_line_and_frees_both_results(self):
        before = len(self.operations.tensors)
        self.assertEqual(self.call(), (8, 2) if False else mmgrid.wide_grid(mesh_for(), mmgrid.cores_for((5120, 256), 2)))
        made = [tensor for tensor in self.operations.tensors[before:] if tensor.name == 'matmul']
        self.assertEqual(len(made), 2)
        self.assertTrue(all(tensor.freed for tensor in made))
        audit = [line for line in self.lines if line.startswith(fusion.MM_GRID_AUDIT_LINE)]
        self.assertEqual(len(audit), 1)
        self.assertIn('audit 1 exact=True site=test rows=64 served=8x10 wide=', audit[0])
        programs = [program.compute_with_storage_grid_size for program in self.operations.programs]
        self.assertEqual(programs, [(8, 10), (4, 1)], 'the served grid first, the wide one second (4 cores: one row of 4)')

    def test_a_grid_that_changes_a_bit_is_a_mismatch_and_an_assertion(self):
        def noise(grid, data):
            if grid[0] != 8:
                data = data.clone()
                data.view(torch.int32).reshape(-1)[3] ^= 1
            return data
        self.operations.grid_noise = noise
        with self.assertRaises(AssertionError):
            self.call()
        self.assertEqual(len([line for line in self.lines if line.startswith(fusion.MM_GRID_MISMATCH)]), 1)

    def test_only_the_first_calls_per_site_and_row_count_are_audited(self):
        for _ in range(fusion.AUDIT_CALLS + 4):
            self.call()
        self.assertEqual(len([line for line in self.lines if line.startswith(fusion.MM_GRID_AUDIT_LINE)]), fusion.AUDIT_CALLS)
        self.assertEqual(len([line for line in self.lines if line.startswith(fusion.MM_GRID_ENGAGED)]), 1, 'one engaged line per distinct text')

    def test_a_weight_it_cannot_place_returns_the_served_grid_and_audits_nothing(self):
        odd = SimpleNamespace(shape=(5120, 100))
        self.assertEqual(mmgrid.grid_for(self.operations, mesh_for(), self.value, odd, (8, 10), 2, 64, None, site='odd'), (8, 10))
        self.assertEqual(self.operations.programs, [])
        self.assertTrue(any(line.startswith(fusion.MM_GRID_FALLBACK) for line in self.lines))


if __name__ == '__main__':
    unittest.main()
