"""draft_gateup_tp (QWEN_FAST_DRAFT_GATEUP1, the gate|up half of F-F3c): the drafter MLP's gate and up projections as one matmul over a load-time
concatenated weight.

Held here, on the CPU: the fused weight is [gate | up] per chip with no byte moved; its program is the pair's own gate/up shape (4 columns a core, 68
cores) and is refused when the shard cannot split into whole per-core columns; and the fused matmul's output columns equal the separate matmuls'
column for column at the drafter's real widths (5,120 -> 2 x 4,352, 32 and 64 rows) on a fake that reduces K the way the program does (in0_block_w = 4
tiles, each block summed exactly and accumulated into fp32 in order), including a negative control - a fake that reduces a column over another
column's K order does NOT pass. The audit is exact on the fused launch and a flipped bit in either half is a logged mismatch.

The branch-level equality of the whole MLP with the flag on and off is test_draft_wp6_branch.

    py -3.11 -B -m unittest test_draft_gateup_tp      (from scripts/ci)
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
import draft_gateup_tp as gateup  # noqa: E402
import wp6_fake_device as fake  # noqa: E402

WIDTH = 4352


def shards_for(width=WIDTH, seed=0, hidden=5120):
    generator = torch.Generator().manual_seed(seed)
    draw = lambda *shape: (torch.randn(*shape, generator=generator) * 0.05).to(torch.bfloat16)      # noqa: E731
    return tuple((draw(hidden, width), draw(hidden, width), draw(width, hidden)) for _ in range(4))


class PlanTests(unittest.TestCase):
    def test_the_fused_program_is_the_pairs_gate_up_shape_at_four_cards(self):
        found = gateup.plan(136)
        self.assertEqual((found['columns'], found['cores'], found['tiles'], found['half_tiles']), (4, 68, 272, 136))
        self.assertEqual(gateup.served_columns(136), 2)
        self.assertEqual(found['columns'], 2 * gateup.served_columns(136), 'twice the served per-core columns')
        self.assertLessEqual(found['cores'], gateup.GRID_CORES)

    def test_the_geometry_it_refuses(self):
        for tiles in (0, -1, 2.0, None):
            with self.assertRaises(ValueError):
                gateup.plan(tiles)
        with self.assertRaisesRegex(ValueError, 'whole per-core columns'):
            gateup.plan(81)                               # 162 tiles do not split into columns of 4
        self.assertIsNotNone(gateup.shard_problem(()))
        self.assertIn('differ', gateup.shard_problem(((torch.zeros(5120, 64), torch.zeros(5120, 96), None),)))
        self.assertIn('whole tiles', gateup.shard_problem(((torch.zeros(5120, 50), torch.zeros(5120, 50), None),)))
        self.assertIsNone(gateup.shard_problem(shards_for(64)))


class WeightTests(unittest.TestCase):
    def test_the_fused_host_weight_is_gate_then_up_per_rank_with_no_value_changed(self):
        shards = shards_for(256)
        fused = gateup.fused_host_weight(shards)
        self.assertEqual(tuple(fused.shape), (4 * 5120, 512))
        for rank in range(4):
            block = fused[rank * 5120:(rank + 1) * 5120]
            self.assertTrue(torch.equal(block[:, :256], shards[rank][0]))
            self.assertTrue(torch.equal(block[:, 256:], shards[rank][1]))
        self.assertTrue(torch.equal(gateup.separate_host_weight(shards, 0), torch.cat([rank[0] for rank in shards], dim=0)))
        self.assertTrue(torch.equal(gateup.separate_host_weight(shards, 2), torch.cat([rank[2] for rank in shards], dim=0)))
        low, high = gateup.split_halves(fused)
        self.assertEqual((tuple(low.shape), tuple(high.shape)), ((4 * 5120, 256), (4 * 5120, 256)))

    def test_the_fused_weight_has_the_same_element_count_as_the_two_it_replaces(self):
        shards = shards_for(256)
        separate = sum(gateup.separate_host_weight(shards, index).numel() for index in (0, 1))
        self.assertEqual(gateup.fused_host_weight(shards).numel(), separate)


class ColumnEqualityTests(unittest.TestCase):
    """The fused matmul's columns against the separate matmuls', on the K-blocked fake."""

    def matmuls(self, rows, width, noisy=False):
        operations = fake.FakeOperations(4)
        generator = torch.Generator().manual_seed(rows + width)
        shards = shards_for(width, seed=rows)
        activation = operations.from_chips([torch.randn(1, 1, rows, 5120, generator=generator) for _ in range(4)], 'bf16')
        mesh = None
        gate = operations.from_torch(gateup.separate_host_weight(shards, 0), mesh, 'bf16', 'tile', 'dram', operations.ShardTensorToMesh(mesh, 0))
        up = operations.from_torch(gateup.separate_host_weight(shards, 1), mesh, 'bf16', 'tile', 'dram', operations.ShardTensorToMesh(mesh, 0))
        fused = operations.from_torch(gateup.fused_host_weight(shards), mesh, 'bf16', 'tile', 'dram', operations.ShardTensorToMesh(mesh, 0))
        columns = gateup.served_columns(width // 32)

        def program(per_core):
            return operations.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(8, 10), in0_block_w=4,
                out_subblock_h=1, out_subblock_w=1, per_core_M=rows // 32, per_core_N=per_core, fuse_batch=True, fused_activation=None,
                mcast_in0=True)
        run = lambda weight, per_core: operations.matmul(activation, weight, dtype='fp32', program_config=program(per_core))    # noqa: E731
        return operations, run(gate, columns), run(up, columns), run(fused, 2 * columns), (activation, gate, up, fused)

    def test_every_column_of_the_fused_output_is_the_separate_matmuls_at_the_real_widths(self):
        for rows in (32, 64):
            with self.subTest(rows=rows):
                operations, gate, up, fused, _ = self.matmuls(rows, WIDTH)
                for chip in range(4):
                    low, high = gateup.split_halves(fused.shards[chip].data)
                    self.assertTrue(torch.equal(low.contiguous().view(torch.int32), gate.shards[chip].data.contiguous().view(torch.int32)))
                    self.assertTrue(torch.equal(high.contiguous().view(torch.int32), up.shards[chip].data.contiguous().view(torch.int32)))
                self.assertEqual(tuple(fused.shape), (1, 1, rows, 2 * WIDTH))

    def test_a_column_reduced_in_another_ones_k_order_would_not_pass(self):
        operations, gate, up, fused, (activation, _, _, fused_weight) = self.matmuls(32, 256)
        # a deliberately wrong 'fused' matmul: the up half reduced over K in reverse block order
        wrong = []
        for a, w in zip(activation.shards, fused_weight.shards, strict=True):
            x, weight = a.data.reshape(-1, 5120), w.data
            acc = torch.zeros(x.shape[0], weight.shape[1])
            for start in range(5120 - 128, -1, -128):
                acc = acc + (x[:, start:start + 128].double() @ weight[start:start + 128].double()).float()
            wrong.append(acc)
        bad = [chip for chip in range(4) if not torch.equal(gateup.split_halves(wrong[chip])[1].contiguous().view(torch.int32),
                                                           up.shards[chip].data.reshape(32, -1).contiguous().view(torch.int32))]
        self.assertTrue(bad, 'a reversed K order must change some fp32 sums')


class AuditTests(unittest.TestCase):
    environment = {'QWEN_FAST_TP': '4', 'QWEN_FAST_DRAFT_GATEUP1': '1', 'QWEN_FAST_DRAFT_GATEUP1_AUDIT': '1'}

    def setUp(self):
        patcher = patch.dict(os.environ, self.environment, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.lines = []
        quiet = patch.object(fusion, 'log_line', side_effect=self.lines.append)
        quiet.start()
        self.addCleanup(quiet.stop)
        fusion.reset()

    def scene(self, corrupt=None):
        operations = fake.FakeOperations(4)
        mesh = SimpleNamespace(shape=[1, 4], compute_with_storage_grid_size=lambda: SimpleNamespace(x=13, y=10))
        shards = shards_for(64)
        parameters = dict(shards=shards, kernel=SimpleNamespace(), operations=operations)
        weight = operations.from_torch(gateup.fused_host_weight(shards), mesh, 'bf16', 'tile', 'dram', operations.ShardTensorToMesh(mesh, 0))
        parameters['device_projections'] = [weight]
        generator = torch.Generator().manual_seed(3)
        prepared = operations.from_chips([torch.randn(1, 1, 64, 5120, generator=generator) for _ in range(4)], 'bf16')

        def project(value, weight_, grid, columns):
            program = operations.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=grid, in0_block_w=4,
                out_subblock_h=1, out_subblock_w=1, per_core_M=2, per_core_N=columns, fuse_batch=True, fused_activation=None, mcast_in0=True)
            return operations.matmul(value, weight_, dtype='fp32', program_config=program)
        return operations, mesh, parameters, prepared, project

    def test_project_fused_runs_one_matmul_at_twice_the_columns_and_audits_it(self):
        operations, mesh, parameters, prepared, project = self.scene()
        fused = gateup.project_fused(operations, mesh, parameters, prepared, project, 64)
        self.assertEqual(tuple(fused.shape), (1, 1, 64, 128))
        served = [entry for entry in operations.log if entry[0] == 'matmul']
        self.assertEqual(served[0], ('matmul', (1, 1, 64, 5120), (5120, 128), 2), 'the fused matmul: per_core_N = 2 x the served 1')
        self.assertEqual([entry[3] for entry in served[1:]], [1, 1], 'the audit ran the two served matmuls')
        audit = [line for line in self.lines if line.startswith(fusion.GATEUP1_AUDIT_LINE)]
        self.assertEqual(len(audit), 1)
        self.assertIn('exact=True', audit[0])
        uploads = [tensor for tensor in operations.tensors if tensor.name == 'from_torch']
        self.assertEqual(len(uploads), 3)
        self.assertTrue(all(tensor.freed for tensor in uploads[1:]), 'the audit frees the separate weights it uploaded')

    def test_a_flipped_bit_in_either_half_is_a_mismatch(self):
        for half in (0, 1):
            with self.subTest(half=half):
                fusion.reset()
                self.lines[:] = []
                operations, mesh, parameters, prepared, project = self.scene()
                genuine = operations.matmul
                state = {'count': 0}

                def corrupting(*args, **options):
                    result = genuine(*args, **options)
                    state['count'] += 1
                    if state['count'] == 1:                    # the fused launch
                        result.shards[1].data.view(torch.int32).reshape(1, 1, 64, -1)[0, 0, 5, 7 + 64 * half] ^= 1
                    return result
                operations.matmul = corrupting
                with self.assertRaises(AssertionError):
                    gateup.project_fused(operations, mesh, parameters, prepared, project, 64)
                mismatch = [line for line in self.lines if line.startswith(fusion.GATEUP1_MISMATCH)]
                self.assertEqual(len(mismatch), 1)
                self.assertIn('chips=[1]', mismatch[0])


if __name__ == '__main__':
    unittest.main()
