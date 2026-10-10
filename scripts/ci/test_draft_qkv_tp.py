"""QWEN_FAST_DRAFT_QKV1 (draft_qkv_tp, F-F3c): the fused q|k|v projection and its head split.

Held on the CPU: the per-chip [q | k | v] weight is the three separate weights column for column (so one matmul at per_core_N = 1 gives each output tile column the K loop the separate
matmuls gave it - shown here on exactly representable data, where a column's value cannot depend on its neighbours); the head split is the tile permutation nlp_create_qkv_heads makes of the
three projections, run through draft_permute_tp's kernel transliteration on raw face-ordered tiles; the entry point engages with one matmul, one typecast and one launch, retains what it makes,
falls back with a marker when it cannot take a call, audits in the eager warm pass only; the attention branch hands the fused heads to the served norm / rotary tail and skips the q projection;
flag off, the branch and the prepared parameters are the served ones.

Run: `python -m unittest test_draft_qkv_tp` from scripts/ci (py 3.11).
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import draft_attention_branch
import draft_permute_tp as perm
import draft_qkv_tp as qkv
import tp4_sampdraft
from test_draft_permute_tp import ExecutingOperations, Mesh, bits, run_kernel, tensor_pages, pages_tensor, same
from test_pair_row_exact import keep
from tp_test_support import four_cards

HEADS, KV, DIM = 8, 2, 128


class Quiet(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {'QWEN_FAST_TP': '4', qkv.FLAG: '1'})
        patcher.start()
        self.addCleanup(patcher.stop)
        perm._LOGGED.clear()
        perm._CACHE.clear()
        perm._PASS['eager'] = None
        perm._SERVED['depth'] = 0
        self.lines = []
        logger = patch.object(tp4_sampdraft, 'log_line', side_effect=self.lines.append)
        logger.start()
        self.addCleanup(logger.stop)


class WeightTests(unittest.TestCase):
    def weights(self, hidden=64, seed=1):
        generator = torch.Generator().manual_seed(seed)
        return {'layers.0.self_attn.%s_proj.weight' % name: torch.randint(-4, 5, (rows, hidden), generator=generator).float()
                for name, rows in (('q', 32 * 128), ('k', 8 * 128), ('v', 8 * 128))}

    def test_each_chips_fused_columns_are_its_q_then_k_then_v_columns(self):
        weights = self.weights()
        fused = qkv.fused_host_weight(torch, weights, 4)
        self.assertEqual(tuple(fused.shape), (4 * 64, 1024 + 256 + 256))
        for chip in range(4):
            shard = fused[64 * chip:64 * (chip + 1)]
            for name, low, high in (('q', 0, 1024), ('k', 1024, 1280), ('v', 1280, 1536)):
                separate = weights['layers.0.self_attn.%s_proj.weight' % name].chunk(4, dim=0)[chip].T
                self.assertTrue(torch.equal(shard[:, low:high], separate), (chip, name))

    def test_a_column_of_the_fused_matmul_is_the_separate_matmuls_column_on_exact_data(self):
        weights = self.weights(hidden=96, seed=2)
        fused = qkv.fused_host_weight(torch, weights, 4)
        x = torch.randint(-3, 4, (64, 96)).float()
        for chip in range(4):
            together = x @ fused[96 * chip:96 * (chip + 1)]
            alone = torch.cat([x @ weights['layers.0.self_attn.%s_proj.weight' % name].chunk(4, dim=0)[chip].T for name in 'qkv'], dim=1)
            self.assertTrue(torch.equal(together, alone), chip)


class SplitTests(Quiet):
    def run_split(self, rows, width=None):
        projection = bits(torch.Generator().manual_seed(rows), (1, rows, 1536))
        flat_source = tensor_pages(projection)
        records = qkv.split_records(HEADS, KV, rows)
        per_lane, size, lane_rows = perm.plan_lanes(records, (13, 10), 1, 3, 2)
        arguments = perm.runtime_arguments(per_lane, [100], [900, 901, 902])
        destinations = {900: {}, 901: {}, 902: {}}
        run_kernel(arguments, size, 1, 3, {100: flat_source}, destinations)
        return projection, [pages_tensor(destinations[address], count, rows) for address, count in ((900, HEADS), (901, KV), (902, KV))]

    def test_the_split_is_nlp_create_qkv_heads_of_the_three_projections(self):
        for rows in (32, 64, 96):
            with self.subTest(rows=rows):
                projection, (q, k, v) = self.run_split(rows)
                flat = projection[0]
                for got, low, high, count in ((q, 0, 1024, HEADS), (k, 1024, 1280, KV), (v, 1280, 1536, KV)):
                    want = flat[:, low:high].reshape(rows, count, DIM).permute(1, 0, 2)
                    self.assertTrue(torch.equal(got, want), (rows, low))

    def test_every_destination_tile_is_written_once_and_all_of_them_are_raw(self):
        records = qkv.split_records(HEADS, KV, 64)
        self.assertTrue(all(record[0] == 'run' and record[6] is False for record in records))
        for destination, count in ((0, HEADS), (1, KV), (2, KV)):
            pages = sorted(page for record in records if record[1] == destination for page in range(record[2], record[2] + record[5]))
            self.assertEqual(pages, list(range(count * 2 * 4)))
        with self.assertRaises(perm.Unsupported):
            qkv.split_records(HEADS, KV, 48)


def fused_ops(rows=64, mesh=None):
    ops = ExecutingOperations(mesh=mesh)
    ops.typecast = lambda value, dtype: ops.from_logical(ops.to_logical(value))
    ops.dealloc = []
    return ops


class ProjectTests(Quiet):
    def setUp(self):
        super().setUp()
        self.ops = fused_ops()
        self.owned = []
        self.projection = bits(torch.Generator().manual_seed(5), (1, 1, 64, 1536))
        self.prepared = self.ops.from_logical(bits(torch.Generator().manual_seed(6), (1, 1, 64, 5120)))
        self.matmuls = []

        def project(value, weight, grid, rows, columns):
            self.matmuls.append((value, weight, grid, rows, columns))
            return self.ops.from_logical(self.projection)

        self.project = project
        self.parameters = dict(projections=dict(qkv='fused-weight'))

    def call(self, served=None, **options):
        return qkv.project(self.ops, self.prepared, keep(self.owned), parameters=self.parameters, project=self.project, rows=64,
                           served=served or Mock(side_effect=AssertionError('the served projections must not run')), site='quad', **options)

    def test_one_matmul_on_the_fused_weight_one_typecast_and_one_launch_make_the_heads(self):
        with four_cards():
            made = self.call()
        self.assertEqual(len(self.matmuls), 1)
        self.assertEqual(self.matmuls[0][1:], ('fused-weight', (8, 8), 64, 1))
        self.assertEqual(len(self.ops.generic), 1)
        flat = self.projection[0, 0]
        for name, low, high, count in (('q', 0, 1024, HEADS), ('k', 1024, 1280, KV), ('v', 1280, 1536, KV)):
            got = self.ops.to_logical(made[name])
            self.assertEqual(tuple(got.shape), (1, count, 64, DIM))
            self.assertTrue(torch.equal(got[0], flat[:, low:high].reshape(64, count, DIM).permute(1, 0, 2)), name)
        self.assertTrue(all(tensor in self.owned for tensor in made.values()))
        engaged = [line for line in self.lines if line.startswith(qkv.ENGAGED)]
        self.assertEqual(len(engaged), 1)
        self.assertIn('site=quad rows=64 columns=48 heads=8/2', engaged[0])

    def test_without_the_prepared_weight_or_with_another_layout_it_falls_back_and_says_why(self):
        with four_cards():
            self.parameters = dict(projections={})
            self.assertIsNone(self.call())
            self.assertIn('no fused q|k|v weight', self.lines[-1])
            self.parameters = dict(projections=dict(qkv='w'))
            self.prepared.dtype = 'bf8'
            self.assertIsNone(self.call())
            self.assertIn('not bfloat16 TILE', self.lines[-1])
        self.assertEqual(self.matmuls, [])
        self.assertEqual(self.ops.generic, [])
        self.assertTrue(all(line.startswith(qkv.FALLBACK) for line in self.lines))

    def test_a_wrong_block_shape_falls_back(self):
        with four_cards():
            self.prepared = self.ops.from_logical(torch.zeros(1, 1, 32, 5120, dtype=torch.int16))
            self.assertIsNone(self.call())
        self.assertIn('is not (1, 1, 64, 5120)', self.lines[-1])

    def test_a_launch_the_builder_refuses_falls_back_after_the_matmul_and_frees_its_outputs(self):
        with four_cards():
            made = self.ops.empty

            def empty(*arguments, **keywords):
                tensor = made(*arguments, **keywords)
                if len(self.ops.empties) == 2:
                    for shard in tensor.shards:
                        shard.odd = True
                return tensor

            self.ops.empty = empty
            self.ops.TensorAccessorArgs = staticmethod(lambda value: SimpleNamespace(get_compile_time_args=lambda: [2 if getattr(value, 'odd', False) else 1]))
            result = self.call()
        self.assertIsNone(result)
        self.assertEqual(self.ops.freed, self.ops.empties[-3:], 'the three head tensors it allocated are freed')
        self.assertIn('share an accessor layout', self.lines[-1])
        self.assertTrue(self.lines[-1].startswith(qkv.FALLBACK))

    def test_the_audit_runs_the_served_projections_beside_the_launch_in_the_eager_pass_only(self):
        with four_cards(), patch.dict(os.environ, {qkv.AUDIT_FLAG: '1'}):
            reference = Mock()
            flat = self.projection[0, 0]
            heads = {name: self.ops.from_logical(flat[:, low:high].reshape(64, count, DIM).permute(1, 0, 2).reshape(1, count, 64, DIM).contiguous())
                     for name, low, high, count in (('q', 0, 1024, HEADS), ('k', 1024, 1280, KV), ('v', 1280, 1536, KV))}
            reference.return_value = heads
            previous = perm.set_pass(True)
            try:
                self.call(served=reference)
            finally:
                perm.set_pass(previous)
            reference.assert_called_once()
            self.assertIn('%s exact=True site=heads shape=quad tensors=3' % qkv.AUDIT, self.lines)
            heads['v'] = self.ops.from_logical(torch.ones(1, KV, 64, DIM, dtype=torch.int16))
            previous = perm.set_pass(True)
            try:
                with self.assertRaises(AssertionError):
                    self.call(served=reference)
            finally:
                perm.set_pass(previous)
            self.assertTrue(self.lines[-1].startswith(qkv.MISMATCH))
            reference.reset_mock()
            previous = perm.set_pass(False)
            try:
                self.call(served=Mock(side_effect=AssertionError('no served run in a capture')))
            finally:
                perm.set_pass(previous)

    def test_the_audit_flag_without_the_lever_is_refused(self):
        with patch.dict(os.environ, {qkv.FLAG: '0', qkv.AUDIT_FLAG: '1'}), self.assertRaisesRegex(ValueError, 'needs'):
            qkv.audit_enabled()

    def test_the_lever_is_refused_at_the_pair(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ[qkv.FLAG] = '1'
            with self.assertRaisesRegex(ValueError, 'TP4 lever'):
                qkv.enabled()


class BranchTests(Quiet):
    def run_branch(self, **flags):
        from test_quad_draft_tp4 import attention_run4

        with four_cards(), patch.dict(os.environ, flags):
            return attention_run4(quad=True)

    def test_flag_off_the_branch_is_the_served_one(self):
        os.environ.pop(qkv.FLAG, None)
        with patch.object(qkv, 'project', side_effect=AssertionError('flag off')):
            events = self.run_branch()
        self.assertGreater(len(events), 40)

    def test_flag_on_the_branch_skips_the_q_projection_and_gives_the_fused_heads_to_the_served_tail(self):
        from test_quad_draft_tp4 import ShapeOps4
        import test_quad_draft as base

        os.environ.pop(qkv.FLAG, None)
        served = self.run_branch()
        seen = {}

        def fake(operations, prepared, retain, *, parameters, project, rows, served, site, **options):
            seen.update(site=site, rows=rows, parameters=parameters)
            return dict(q=base.Tensor((1, 8, 64, 128)), k=base.Tensor((1, 2, 64, 128)), v=base.Tensor((1, 2, 64, 128)))

        with patch.object(qkv, 'project', side_effect=fake):
            events = self.run_branch(**{qkv.FLAG: '1'})
        self.assertEqual((seen['site'], seen['rows']), ('quad', 64))
        grids = lambda log: [dict(event[1])['compute_with_storage_grid_size'] for event in log if event[0] == 'program']
        self.assertEqual(grids(served), [(8, 5), (8, 8), (8, 8), (8, 10)])
        self.assertEqual(grids(events), [(8, 5), (8, 10)], 'no q, k or v matmul program in the branch: the fused call is the lever\'s')
        self.assertEqual([event for event in events if event[0] == 'create_heads'], [], 'no served head split')
        self.assertEqual(len([event for event in events if event[0] == 'rms_norm']), len([event for event in served if event[0] == 'rms_norm']), 'the k and q norms stay')

    def test_flag_on_and_the_lever_declines_the_branch_runs_the_served_projections(self):
        os.environ.pop(qkv.FLAG, None)
        served = self.run_branch()
        with patch.object(qkv, 'project', return_value=None):
            events = self.run_branch(**{qkv.FLAG: '1'})
        self.assertEqual(events, served)

    def test_the_prepared_parameters_hold_the_fused_weight_only_when_the_flag_is_on_and_upload_it_last(self):
        uploads = []

        class Ops(object):
            bfloat16, TILE_LAYOUT, ROW_MAJOR_LAYOUT, DRAM_MEMORY_CONFIG = 'bf16', 'tile', 'row', 'dram'
            MathFidelity = SimpleNamespace(HiFi4='hifi4')

            def WormholeComputeKernelConfig(self, **options):
                return 'kernel'

            def from_torch(self, value, **options):
                uploads.append(tuple(value.shape))
                return 'tensor%d' % len(uploads)

            def ShardTensorToMesh(self, mesh, dim):
                return 'shard'

            def ReplicateTensorToMesh(self, mesh):
                return 'replicate'

        hidden = 64
        weights = {'layers.0.self_attn.%s_proj.weight' % name: torch.zeros(rows, hidden) for name, rows in (('q', 4096), ('k', 1024), ('v', 1024), ('o', hidden))}
        weights['layers.0.self_attn.o_proj.weight'] = torch.zeros(hidden, 4096)
        weights['layers.0.self_attn.q_norm.weight'] = torch.zeros(128)
        weights['layers.0.self_attn.k_norm.weight'] = torch.zeros(128)
        weights['layers.0.input_layernorm.weight'] = torch.zeros(5120)
        convolution = {'layers.0.input_layernorm.weight': torch.zeros(5120), 'layers.0.attention_conv.kernel_projection.weight': torch.zeros(1280, 5120),
                       'layers.0.attention_conv.base_kernel': {(p, o): torch.zeros(5120) for p in range(2) for o in range(2)}}
        results = []
        for flag in ('0', '1'):
            del uploads[:]
            with four_cards(), patch.dict(os.environ, {qkv.FLAG: flag}), patch('dflash_t16_native_scope.require_active', return_value=None):
                results.append((draft_attention_branch.prepare_attention_branch(Ops(), 'mesh', weights, convolution, lambda value: value, native_head_layout=True,
                                                                                 block_rows=16, native_proposal_attention=True), list(uploads)))
        off, on = results
        self.assertNotIn('qkv', off[0]['projections'])
        self.assertIn('qkv', on[0]['projections'])
        self.assertEqual(on[1][:-1], off[1], 'every other upload is the same, in the same order')
        self.assertEqual(on[1][-1], (4 * hidden, 1536))


if __name__ == '__main__':
    unittest.main()
