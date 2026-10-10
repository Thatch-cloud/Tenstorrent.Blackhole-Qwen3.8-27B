"""CPU tests of the WP4 card-M harness: its candidates at the real shapes on both grids, its rule, its host data, its timing arithmetic, its run script, and its whole
run on a fake ttnn (matmuls by torch, a cost model per program config, traces as recorded costs).

Run: python -B -m unittest discover -s optimisation/ttnn-op/mlp_gateup -p 'test_*.py'
"""

import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

HERE = Path(__file__).resolve().parent
CI = HERE.parent.parent.parent / 'scripts' / 'ci'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CI))
import tp4_mlp_gateup as lever  # noqa: E402
import mlp_gateup_card_m as harness  # noqa: E402

SHAPES = lever.shapes(4)
GATE, UP, DOWN = SHAPES['mlp_w1'], SHAPES['mlp_w3'], SHAPES['mlp_w2']


# ---------------------------------------------------------------------------------------------------------------------------
# The candidates at the real shapes.
# ---------------------------------------------------------------------------------------------------------------------------

class CandidateTests(unittest.TestCase):
    def test_widths_start_at_the_device_and_never_exceed_it(self):
        self.assertEqual(harness.widths_for(13), [13, 12, 11, 10, 8])
        self.assertEqual(harness.widths_for(11), [11, 10, 8])

    def test_stage_one_covers_the_core_counts_up_to_the_grid_on_both_grids(self):
        for grid_x, grid_y, limit in ((13, 10, 130), (11, 10, 110)):
            for shape in (GATE, UP, DOWN):
                with self.subTest(grid=(grid_x, grid_y), shape=shape.key):
                    found = harness.stage_one(lever, shape, grid_x, grid_y)
                    cores = [lever.active_cores(item['config'], shape) for item in found]
                    self.assertEqual(len(set(cores)), len(cores), 'one candidate per core count')
                    self.assertTrue(all(lever.MIN_ACTIVE_CORES <= count <= limit for count in cores))
                    self.assertTrue(all(item['config'].grid[1] <= grid_y and item['config'].grid[0] <= grid_x for item in found))
                    self.assertEqual(min(item['pcn'] for item in found), math.ceil(lever.tiles(shape.n) / limit) if limit < lever.tiles(shape.n) else 1)

    def test_the_gate_and_up_reach_the_cores_the_profile_has_and_more(self):
        found = dict((item['pcn'], item) for item in harness.stage_one(lever, GATE, 13, 10))
        self.assertEqual(lever.active_cores(found[2]['config'], GATE), 68, 'the 13x10 gate partition of the profile (88 requested, 68 active)')
        self.assertEqual(lever.active_cores(found[3]['config'], GATE), 46)
        self.assertEqual(lever.active_cores(found[4]['config'], GATE), 34, 'the 11x10 up partition, now a candidate for the gate')
        self.assertNotIn(1, found, '136 cores do not fit 13x10 (130)')

    def test_every_candidate_keeps_the_served_k_loop(self):
        for grid_x in (13, 11):
            for shape in (GATE, UP, DOWN):
                served = lever.builder_config(shape, grid_x)
                for item in harness.stage_one(lever, shape, grid_x, 10):
                    with self.subTest(grid=grid_x, shape=shape.key, name=item['name']):
                        self.assertEqual(lever.same_k_loop(types.SimpleNamespace(in0_block_w=served.in0_block_w, per_core_M=served.per_core_M,
                                                                                fuse_batch=True, mcast_in0=True),
                                                           types.SimpleNamespace(in0_block_w=item['config'].in0_block_w,
                                                                                per_core_M=item['config'].per_core_M, fuse_batch=True,
                                                                                mcast_in0=True)), [])

    def test_the_names_are_the_levers_tokens(self):
        names = [item['name'] for item in harness.stage_one(lever, GATE, 13, 10)]
        self.assertIn('g2', names)
        self.assertTrue(all(re.fullmatch(r'g[1-9][0-9]*', name) for name in names))
        self.assertTrue(all(re.fullmatch(r'u[1-9][0-9]*', item['name']) for item in harness.stage_one(lever, UP, 13, 10)))
        self.assertTrue(all(re.fullmatch(r'd[1-9][0-9]*', item['name']) for item in harness.stage_one(lever, DOWN, 13, 10)))
        # and each is a name the lever parses
        for name in names:
            self.assertEqual(lever.parse_cfg(name).g, int(name[1:]))

    def test_stage_two_moves_the_rectangle_not_the_partition_and_never_repeats(self):
        seen = set((tuple(item['config'].grid), item['pcn']) for item in harness.stage_one(lever, GATE, 13, 10))
        found = harness.stage_two(lever, GATE, 13, 10, [2, 3], seen)
        self.assertTrue(found)
        self.assertTrue(all(item['width'] < 13 and '@w%d' % item['width'] in item['name'] for item in found))
        for item in found:
            self.assertEqual(lever.active_cores(item['config'], GATE), math.ceil(136 / item['pcn']))
        again = harness.stage_two(lever, GATE, 13, 10, [2, 3], seen)
        self.assertEqual(again, [], 'the seen set is shared: no candidate twice')

    def test_the_served_row_is_the_builders_and_honours_t1(self):
        on = harness.served_row(lever, GATE, 13, 88)
        off = harness.served_row(lever, GATE, 13, 44)
        self.assertEqual((lever.active_cores(on['config'], GATE), on['config'].grid), (68, (13, 7)))
        self.assertEqual(lever.active_cores(off['config'], GATE), 46)
        self.assertEqual(harness.served_row(lever, UP, 13)['config'], lever.builder_config(UP, 13))
        self.assertEqual(harness.served_row(lever, DOWN, 11)['config'].per_core_N, 5)


# ---------------------------------------------------------------------------------------------------------------------------
# The rule.
# ---------------------------------------------------------------------------------------------------------------------------

def row(shape, name, us, exact=True, stage=1, pcn=2):
    return dict(shape=shape.key, name=name, stage=stage, pcn=pcn, width=13, grid=[13, 6], active_cores=68, in0_block_w=8, subblock=[2, 2], us=us,
                gbps=round(lever.gbps(shape, us), 2), pct_peak=0.0, exact=exact, differing=0 if exact else 1)


def summaries_of(gate, up, down):
    out = {}
    for shape, rows in ((GATE, gate), (UP, up), (DOWN, down)):
        out[shape.key] = harness.shape_summary(lever, shape, rows)
    return out


class RuleTests(unittest.TestCase):
    def test_best_exact_ignores_inexact_and_errored_rows(self):
        rows = [row(GATE, 'served', 53.0), row(GATE, 'g3', 40.0, exact=False, pcn=3), dict(shape='mlp_w1', name='g4', error='boom', pcn=4, stage=1),
                row(GATE, 'g5', 48.0, pcn=5)]
        self.assertEqual(harness.best_exact(rows)['name'], 'g5')
        summary = harness.shape_summary(lever, GATE, rows)
        self.assertEqual(summary['best_any']['name'], 'g3')
        self.assertEqual(summary['inexact'], ['g3'])
        self.assertAlmostEqual(summary['gain_us'], 5.0)
        self.assertAlmostEqual(summary['gain_ms_pass'], 0.32)

    def test_config_enough(self):
        # gate 53 -> 36 and up 44 -> 34: 27 us x 64 = 1.75 ms; at the down's 358 GB/s the pair streams in 70 us: nothing left for a fused op
        summaries = summaries_of([row(GATE, 'served', 53.14), row(GATE, 'g3', 36.0, pcn=3)], [row(UP, 'served', 44.14), row(UP, 'u3', 34.0, pcn=3)],
                                 [row(DOWN, 'served', 66.27), row(DOWN, 'd5', 66.0, pcn=5)])
        decision = harness.decide(summaries, lever, multiply_us=0.0)
        self.assertEqual(decision['verdict'], 'CONFIG-ENOUGH', decision)
        self.assertGreaterEqual(decision['config_gain_ms_pass'], 0.35)
        self.assertLess(decision['fused_potential_ms_pass'], 0.35)
        self.assertEqual(decision['env'], {'QWEN_FAST_MLP_CFG': 'g3u3d5'})

    def test_build_fused_when_the_config_leaves_the_plans_floor_on_the_table(self):
        # the config does nothing (served is best) and the multiply is 4.1 us: the pair at the ceiling would stream 25 MB at the down's rate
        summaries = summaries_of([row(GATE, 'served', 53.14)], [row(UP, 'served', 44.14)], [row(DOWN, 'served', 66.27)])
        decision = harness.decide(summaries, lever, multiply_us=4.08)
        self.assertEqual(decision['verdict'], 'BUILD-FUSED', decision)
        self.assertEqual(decision['config_gain_ms_pass'], 0.0)
        self.assertEqual(decision['env'], {'QWEN_FAST_MLP_CFG': 'l1'})
        self.assertGreater(decision['fused_potential_ms_pass'], 1.9, 'the plan puts the whole prize at about 2.1 ms a pass')
        self.assertAlmostEqual(decision['ceiling_gbps'], 357.3, delta=0.5)

    def test_neither_when_streaming_is_already_at_the_ceiling(self):
        summaries = summaries_of([row(GATE, 'served', 35.0)], [row(UP, 'served', 35.0)], [row(DOWN, 'served', 66.27)])
        decision = harness.decide(summaries, lever, multiply_us=0.0)
        self.assertEqual(decision['verdict'], 'NEITHER', decision)

    def test_no_result_without_a_gate_or_an_exact_row(self):
        summaries = summaries_of([row(GATE, 'served', 53.0)], [row(UP, 'served', 44.0, exact=False)], [row(DOWN, 'served', 66.0)])
        self.assertEqual(harness.decide(summaries, lever)['verdict'], 'NO-RESULT')
        self.assertEqual(harness.decide({}, lever)['verdict'], 'NO-RESULT')

    def test_the_combined_name_is_the_best_default_width_rows(self):
        summaries = summaries_of([row(GATE, 'served', 53.0), row(GATE, 'g3', 44.0, pcn=3)], [row(UP, 'served', 44.0)],
                                 [row(DOWN, 'served', 66.0), row(DOWN, 'd4@w12', 60.0, stage=2, pcn=4), row(DOWN, 'd6', 63.0, pcn=6)])
        self.assertEqual(harness.combined_name(summaries), 'g3d6', 'the stage-2 row is not a lever name; the up kept the served partition')
        self.assertEqual(lever.parse_cfg(harness.combined_name(summaries)).name, 'g3d6')
        self.assertEqual(harness.combined_name({}), 'l1')

    def test_the_threshold_is_the_plans_low_end(self):
        self.assertEqual(harness.THRESHOLD_MS, 0.35)
        self.assertEqual(harness.LAYERS, 64)
        self.assertIn('0.35 ms', harness.__doc__)


# ---------------------------------------------------------------------------------------------------------------------------
# Timing arithmetic and host data.
# ---------------------------------------------------------------------------------------------------------------------------

class TimingTests(unittest.TestCase):
    def test_the_slope_removes_the_replays_fixed_cost(self):
        short = [100.0 + 8 * 50.0] * 5
        long = [100.0 + 40 * 50.0] * 5
        self.assertAlmostEqual(harness.slope_us(short, long, 8, 40), 50.0)
        with self.assertRaises(ValueError):
            harness.slope_us(short, long, 40, 8)

    def test_summarize(self):
        summary = harness.summarize([5.0, 1.0, 3.0, 2.0, 4.0])
        self.assertEqual((summary['n'], summary['median_us'], summary['min_us']), (5, 3.0, 1.0))


class HostDataTests(unittest.TestCase):
    def test_shapes_dtype_and_determinism(self):
        x = harness.host_x(torch, 64, 5120, 'random', 1)
        self.assertEqual((tuple(x.shape), x.dtype), ((1, 1, 64, 5120), torch.bfloat16))
        self.assertTrue(torch.equal(x, harness.host_x(torch, 64, 5120, 'random', 1)))
        self.assertFalse(torch.equal(x, harness.host_x(torch, 64, 5120, 'random', 2)))
        w = harness.host_w(torch, 5120, 4352, 3)
        self.assertEqual((tuple(w.shape), w.dtype), ((1, 1, 5120, 4352), torch.bfloat16))
        self.assertEqual(tuple(harness.host_w(torch, 4352, 5120, 3).shape), (1, 1, 4352, 5120))
        self.assertEqual(tuple(harness.host_w(torch, 100, 70, 3).shape), (1, 1, 128, 96), 'padded to tiles')

    def test_the_edge_regime_carries_zeros_and_extreme_magnitudes_and_no_infinity(self):
        x = harness.host_x(torch, 64, 512, 'edge', 4)
        self.assertGreater(int((x == 0).sum()), 64 * 512 // 5)
        self.assertTrue(bool(torch.isfinite(x.float()).all()))
        magnitudes = x.float().abs().amax(dim=-1).flatten()
        self.assertGreater(float(magnitudes.max()), 100.0)
        self.assertLess(float(magnitudes.min()), 1e-10)
        w = harness.host_w(torch, 256, 512, 5, 'edge')
        self.assertGreater(int((w == 0).all(dim=-2).sum()), 40, 'a fifth of the weight columns are zero')


# ---------------------------------------------------------------------------------------------------------------------------
# A fake ttnn: torch matmuls, a cost model per program config, traces as recorded costs.
# ---------------------------------------------------------------------------------------------------------------------------

class Tensor(object):
    def __init__(self, data):
        self.data = data
        self.shape = tuple(data.shape)


class FakeTTNN(object):
    bfloat16, bfloat4_b, bfloat8_b, float32 = 'bf16', 'bfp4', 'bfp8', 'f32'
    TILE_LAYOUT, DRAM_MEMORY_CONFIG, L1_MEMORY_CONFIG = 'tile', 'dram', 'l1'
    MathFidelity = types.SimpleNamespace(LoFi='lofi')
    UnaryOpType = types.SimpleNamespace(SILU='silu')

    def __init__(self, grid=(13, 10), cost=None, bad_pcn=(), replay_overhead_us=37.0):
        self.grid, self.bad_pcn, self.replay_overhead = grid, set(bad_pcn), replay_overhead_us
        self.cost = cost or (lambda config, shape: 40.0)
        self.now_us = 0.0
        self.capturing, self.traces, self.released, self.closed = None, {}, [], False
        self.linears, self.kernel_configs = [], []

    def clock(self):
        return self.now_us / 1e6

    def WormholeComputeKernelConfig(self, **options):
        config = types.SimpleNamespace(math_approx_mode=True, **options)
        self.kernel_configs.append(config)
        return config

    def MatmulMultiCoreReuseMultiCast1DProgramConfig(self, **options):
        return types.SimpleNamespace(**options)

    def MeshShape(self, *shape):
        return shape

    def open_mesh_device(self, shape, **options):
        self.open_options = options
        return types.SimpleNamespace(compute_with_storage_grid_size=lambda: types.SimpleNamespace(x=self.grid[0], y=self.grid[1]),
                                     dram_grid_size=lambda: types.SimpleNamespace(x=8, y=1))

    def close_mesh_device(self, mesh):
        self.closed = True

    def ReplicateTensorToMesh(self, mesh):
        return 'replicate'

    def from_torch(self, tensor, **options):
        return Tensor(tensor.clone())

    def get_device_tensors(self, tensor):
        return [tensor]

    def to_torch(self, tensor):
        return tensor.data

    def deallocate(self, tensor):
        pass

    def synchronize_device(self, mesh):
        pass

    def spend(self, microseconds):
        if self.capturing is not None:
            self.capturing.append(microseconds)

    def linear(self, x, weight, compute_kernel_config=None, program_config=None, memory_config=None, dtype=None):
        self.linears.append((program_config, memory_config))
        out = x.data.float() @ weight.data.float()
        if program_config.fused_activation == 'silu':
            out = out * torch.sigmoid(out)
        out = out if dtype == 'f32' else out.to(torch.bfloat16)
        if program_config.per_core_N in self.bad_pcn:
            out.view(torch.int16)[0, 0, 0, 0] ^= 1
        shape = types.SimpleNamespace(k=weight.data.shape[-2], n=weight.data.shape[-1])
        self.spend(self.cost(program_config, shape))
        return Tensor(out)

    def silu(self, t, memory_config=None):
        self.spend(5.0)
        return Tensor((t.data.float() * torch.sigmoid(t.data.float())).to(t.data.dtype))

    def typecast(self, t, dtype, memory_config=None):
        self.spend(2.0)
        return Tensor(t.data.to(torch.bfloat16))

    def multiply(self, a, b, input_tensor_a_activations=None, dtype=None, memory_config=None):
        self.spend(7.0)
        left = a.data.float()
        if input_tensor_a_activations:
            left = left * torch.sigmoid(left)
        return Tensor((left * b.data.float()).to(torch.bfloat16))

    def mul(self, a, b, memory_config=None):
        self.spend(4.0 if memory_config == 'dram' else 3.0)
        return Tensor((a.data.float() * b.data.float()).to(torch.bfloat16))

    def begin_trace_capture(self, mesh, cq_id=0):
        self.capturing = []
        return len(self.traces)

    def end_trace_capture(self, mesh, handle, cq_id=0):
        self.traces[handle] = self.capturing
        self.capturing = None

    def execute_trace(self, mesh, handle, cq_id=0, blocking=True):
        self.now_us += sum(self.traces[handle]) + self.replay_overhead

    def release_trace(self, mesh, handle):
        self.released.append(handle)


def small_shapes():
    """The real shapes' structure at a size torch can multiply in a unit test: K 256 (8 K tiles), N 1,088 (34 tiles)."""
    keys = lever.shapes(4)
    small = dict((key, shape._replace(k=256 if key != 'mlp_w2' else 1088, n=1088 if key != 'mlp_w2' else 256)) for key, shape in keys.items()
                 if key in ('mlp_w1', 'mlp_w3', 'mlp_w2'))
    return small


def cost_model(best):
    """µs per launch: 60 minus 4 per step toward the best per_core_N of the shape (smaller is faster), plus the multiply elsewhere."""
    def cost(config, shape):
        key = 'down' if shape.k > shape.n else 'gateup'
        return 40.0 + 3.0 * abs(config.per_core_N - best[key])
    return cost


class RunTests(unittest.TestCase):
    def run_harness(self, ttnn, extra=(), budget_clock=None, env=None):
        out = Path(tempfile.mkdtemp()) / 'report.json'
        shapes = small_shapes()
        environment = dict(QWEN_FAST_TP='4')
        environment.update(env or {})
        with mock.patch.dict(os.environ, environment, clear=False), mock.patch.object(lever, 'shapes', lambda tp=4: shapes), \
                mock.patch.object(lever, 'MIN_ACTIVE_CORES', 2), mock.patch.object(harness, 'WATCHDOG_S', 3600):
            status = harness.main(['--out', str(out), '--rounds', '3', '--budget-s', '100000'] + list(extra), torch=torch, ttnn=ttnn, lever=lever,
                                  clock=ttnn.clock, budget_clock=budget_clock or (lambda: 0.0))
        return status, json.loads(out.read_text())

    def test_the_whole_run_picks_the_cheapest_exact_config_and_composes_it(self):
        ttnn = FakeTTNN(cost=cost_model(dict(gateup=3, down=4)))
        status, report = self.run_harness(ttnn)
        self.assertEqual((status, report['verdict']), (0, 'PASS'), report.get('error'))
        self.assertEqual(report['grid'], [13, 10])
        summary = report['summary']
        self.assertEqual([summary[key]['best_exact']['name'] for key in ('mlp_w1', 'mlp_w3', 'mlp_w2')], ['g3', 'u3', 'd4'])
        self.assertEqual(report['decision']['env'], {'QWEN_FAST_MLP_CFG': 'g3u3d4'})
        self.assertGreater(summary['mlp_w1']['gain_us'], 0.0)
        # the served row is measured against its own second copy: the control's delta is the measurement floor
        served = summary['mlp_w1']['served']
        self.assertAlmostEqual(served['delta_us'], 0.0, delta=0.01)
        # the slope removes the 37 us the fake adds to every replay: the served gate costs exactly its cost-model time
        served_config = lever.builder_config(small_shapes()['mlp_w1'], 13)
        self.assertAlmostEqual(served['us'], 40.0 + 3.0 * abs(served_config.per_core_N - 3), delta=0.01)
        compose = report['compose']
        self.assertTrue(all(compose['exact'].values()), compose)
        self.assertEqual(compose['tuned_cfg'], 'g3u3d4')
        self.assertLess(compose['chain_us']['l1_multiply'], compose['chain_us']['served'], 'F-D2: the L1 multiply is one microsecond cheaper in the model')
        self.assertLess(compose['chain_us']['tuned_l1'], compose['chain_us']['l1_multiply'])
        self.assertGreater(compose['gain_ms_pass']['tuned_l1'], compose['gain_ms_pass']['l1_multiply'])
        self.assertEqual(report['multiply_us'], dict(dram=4.0, l1=3.0))
        self.assertTrue(ttnn.closed)

    def test_the_model_s_compute_config_is_the_decode_one(self):
        ttnn = FakeTTNN(cost=cost_model(dict(gateup=3, down=4)))
        self.run_harness(ttnn)
        config = ttnn.kernel_configs[0]
        self.assertEqual((config.math_fidelity, config.fp32_dest_acc_en, config.packer_l1_acc), ('lofi', True, True))
        self.assertTrue(config.math_approx_mode, 'left at the config class default, as the model leaves it')

    def test_a_config_that_changes_the_bytes_is_never_the_answer_and_raises_an_alert(self):
        ttnn = FakeTTNN(cost=cost_model(dict(gateup=3, down=4)), bad_pcn=(3,))
        status, report = self.run_harness(ttnn)
        summary = report['summary']
        self.assertNotIn(summary['mlp_w1']['best_exact']['name'], ('g3',))
        self.assertIn('g3', summary['mlp_w1']['inexact'])
        self.assertIn('mlp_w1', report['inexact'])
        self.assertEqual(summary['mlp_w1']['best_any']['name'], 'g3')
        self.assertEqual(status, 0, 'the served arm reproduced itself and every shape has an exact row')

    def test_the_stage_two_rows_are_timed_and_named_with_their_width(self):
        ttnn = FakeTTNN(cost=cost_model(dict(gateup=3, down=4)))
        status, report = self.run_harness(ttnn)
        names = [item['name'] for item in report['shapes']['mlp_w1']['rows']]
        self.assertTrue(any('@w' in name for name in names), names)
        self.assertEqual(names[0], 'served')

    def test_a_spent_budget_stops_adding_candidates_and_still_reports(self):
        ttnn = FakeTTNN(cost=cost_model(dict(gateup=3, down=4)))
        ticks = iter(range(10 ** 6))
        status, report = self.run_harness(ttnn, extra=['--budget-s', '5'], budget_clock=lambda: next(ticks) * 0.5)
        self.assertIn('truncated', report)
        self.assertEqual(status, 0)
        self.assertTrue(all(report['summary'][key]['served'] for key in ('mlp_w1', 'mlp_w3', 'mlp_w2')))

    def test_a_config_the_program_refuses_is_a_row_not_the_end(self):
        ttnn = FakeTTNN(cost=cost_model(dict(gateup=3, down=4)))
        original = ttnn.linear

        def refusing(x, weight, compute_kernel_config=None, program_config=None, memory_config=None):
            if program_config.per_core_N == 2 and program_config.out_subblock_w == 2 and weight.data.shape[-2] == 256:
                raise RuntimeError('TT_FATAL: out of L1')
            return original(x, weight, compute_kernel_config, program_config, memory_config)

        ttnn.linear = refusing
        status, report = self.run_harness(ttnn, extra=['--shapes', 'mlp_w1', '--arms', 'sweep'])
        errors = [item for item in report['shapes']['mlp_w1']['rows'] if 'error' in item]
        self.assertTrue(errors and 'out of L1' in errors[0]['error'], report['shapes']['mlp_w1']['rows'][:3])

    def test_the_four_card_geometry_is_required(self):
        ttnn = FakeTTNN()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('QWEN_FAST_TP', None)
            out = Path(tempfile.mkdtemp()) / 'report.json'
            status = harness.main(['--out', str(out)], torch=torch, ttnn=ttnn, lever=lever, clock=ttnn.clock)
        self.assertEqual(status, 4)
        self.assertEqual(json.loads(out.read_text())['verdict'], 'NOT-RUN')

    def test_the_grid_is_read_from_the_device(self):
        for grid in ((11, 10), (13, 10)):
            ttnn = FakeTTNN(grid=grid, cost=cost_model(dict(gateup=3, down=4)))
            status, report = self.run_harness(ttnn, extra=['--shapes', 'mlp_w1', '--arms', 'sweep'])
            self.assertEqual(report['grid'], list(grid))
            self.assertEqual(report['workers'], grid[0] * grid[1])
            self.assertTrue(all(item['grid'][0] <= grid[0] for item in report['shapes']['mlp_w1']['rows']))

    def test_unknown_arms_and_shapes_are_refused(self):
        self.assertEqual(harness.main(['--out', '/dev/null', '--arms', 'sweep,bogus'], torch=torch, ttnn=FakeTTNN(), lever=lever), 2)
        ttnn = FakeTTNN()
        status, report = self.run_harness(ttnn, extra=['--shapes', 'mlp_w9'])
        self.assertEqual((status, report['verdict']), (4, 'NOT-RUN'))
        self.assertIn('unknown shapes', report['error'])


class FakeFused(object):
    """The fused module's FusedGateUp stand-in: the unfused arithmetic in one call, a cost of 30 us less one per pair, and (when the SFPU approximation
    mode is not the served one) one flipped bit, as a card might show."""

    built = []

    def __init__(self, operations, mesh, w1, w3, pairs_per_worker=3, grid=(13, 10), width=None, math_approx_mode=None, **options):
        self.ttnn, self.w1, self.w3, self.pairs, self.approx = operations, w1, w3, pairs_per_worker, math_approx_mode
        self.plan = dict(workers=-(-136 // pairs_per_worker))
        FakeFused.built.append((pairs_per_worker, math_approx_mode, grid))

    def __call__(self, x):
        ttnn = self.ttnn
        gate = x.data.float() @ self.w1.data.float()
        gate = (gate * torch.sigmoid(gate)).to(torch.bfloat16)
        up = (x.data.float() @ self.w3.data.float()).to(torch.bfloat16)
        out = (gate.float() * up.float()).to(torch.bfloat16)
        if self.approx is not True:
            out.view(torch.int16)[0, 0, 0, 0] ^= 1
        ttnn.spend(30.0 - self.pairs)
        return Tensor(out)


class FusedArmTests(unittest.TestCase):
    def run_fused(self, ttnn):
        FakeFused.built.clear()
        out = Path(tempfile.mkdtemp()) / 'report.json'
        module = types.SimpleNamespace(FusedGateUp=FakeFused)
        shapes = small_shapes()
        with mock.patch.dict(os.environ, dict(QWEN_FAST_TP='4')), mock.patch.object(lever, 'shapes', lambda tp=4: shapes), \
                mock.patch.object(harness, 'WATCHDOG_S', 3600):
            status = harness.main(['--out', str(out), '--rounds', '3', '--arms', 'fused'], torch=torch, ttnn=ttnn, lever=lever,
                                  fused_module=module, clock=ttnn.clock)
        return status, json.loads(out.read_text())

    def test_every_pairs_per_worker_is_built_timed_and_compared_and_the_other_math_mode_is_the_diagnostic(self):
        ttnn = FakeTTNN(cost=lambda config, shape: 40.0)
        status, report = self.run_fused(ttnn)
        self.assertEqual(status, 0, report.get('error'))
        section = report['fused']
        self.assertTrue(section['served_math_approx_mode'])
        rows = section['rows']
        self.assertEqual([row['pairs'] for row in rows], [2, 3, 4, 5, 7, 3])
        self.assertEqual([row['diagnostic'] for row in rows], [False] * 5 + [True])
        self.assertEqual([row['math_approx_mode'] for row in rows], [True] * 5 + [False])
        self.assertTrue(all(row['exact'] for row in rows[:5]), rows)
        self.assertFalse(rows[5]['exact'])
        self.assertEqual(rows[5]['differing'], 1)
        # the served pair costs 40 + 40 + the multiply 4; the fake fused costs 30 - pairs: more pairs, faster
        self.assertAlmostEqual(section['served_pair_us'], 84.0, delta=0.01)
        self.assertAlmostEqual(rows[0]['us'], 28.0, delta=0.01)
        self.assertAlmostEqual(rows[4]['us'], 23.0, delta=0.01)
        self.assertGreater(rows[4]['gain_ms_pass'], rows[0]['gain_ms_pass'])
        self.assertAlmostEqual(rows[0]['gain_ms_pass'], 64 * (84.0 - 28.0) / 1000.0, delta=0.01)
        self.assertEqual(rows[0]['workers'], 68)
        self.assertEqual(FakeFused.built[0], (2, True, (13, 10)))

    def test_a_build_that_fails_is_a_row_and_the_rest_still_run(self):
        ttnn = FakeTTNN(cost=lambda config, shape: 40.0)
        original = FakeFused.__init__

        def refusing(self, operations, mesh, w1, w3, pairs_per_worker=3, **options):
            if pairs_per_worker == 4:
                raise RuntimeError('TT_FATAL: kernel compile failed')
            original(self, operations, mesh, w1, w3, pairs_per_worker=pairs_per_worker, **options)

        with mock.patch.object(FakeFused, '__init__', refusing):
            status, report = self.run_fused(ttnn)
        rows = dict((row['pairs'], row) for row in report['fused']['rows'] if not row['diagnostic'])
        self.assertIn('kernel compile failed', rows[4]['error'])
        self.assertNotIn('us', rows[4])
        self.assertTrue(all('us' in rows[pairs] for pairs in (2, 3, 5, 7)))


class EpilogueTests(unittest.TestCase):
    # the sweep's C1 rows (13x10, microseconds): gate g<n> and up u<n> at the same per_core_N
    GATE = {2: 52.18, 3: 51.76, 4: 53.66, 5: 55.91, 6: 61.55, 7: 69.31, 8: 77.56}
    UP = {2: 47.48, 3: 44.76, 4: 44.13, 5: 45.41, 6: 48.03, 7: 54.43, 8: 57.20}

    def rows(self, table, key, stage=1, exact=True):
        return [dict(pcn=pcn, us=us, exact=exact, stage=stage, name=key + str(pcn)) for pcn, us in table.items()]

    def test_the_gap_between_gate_and_up_is_a_cost_per_output_column(self):
        fit = harness.epilogue_fit(self.rows(self.GATE, 'g'), self.rows(self.UP, 'u'))
        self.assertEqual(fit['points'], 7)
        self.assertAlmostEqual(fit['us_per_column'], 2.3, delta=0.2)
        self.assertAlmostEqual(fit['us_per_tile'], 1.15, delta=0.1)
        self.assertLess(fit['max_residual_us'], 3.5)
        self.assertEqual(fit['gaps'][2], 4.7)

    def test_too_few_points_or_inexact_or_wider_rows_are_not_used(self):
        self.assertIsNone(harness.epilogue_fit(self.rows({2: 52.0, 3: 51.0}, 'g'), self.rows({2: 47.0, 3: 44.0}, 'u')))
        self.assertIsNone(harness.epilogue_fit(self.rows(self.GATE, 'g', exact=False), self.rows(self.UP, 'u')))
        self.assertIsNone(harness.epilogue_fit(self.rows(self.GATE, 'g', stage=2), self.rows(self.UP, 'u')))
        self.assertIsNone(harness.epilogue_fit([], []))


class DiagnosticShapeTests(unittest.TestCase):
    def test_the_same_shapes_with_bfloat8_weights(self):
        table = harness.shape_table(lever)
        self.assertEqual((table['gate_bf8'].k, table['gate_bf8'].n, table['gate_bf8'].dtype, table['gate_bf8'].silu), (5120, 4352, 'bfp8', True))
        self.assertEqual((table['up_bf8'].dtype, table['up_bf8'].silu), ('bfp8', False))
        self.assertEqual(lever.weight_bytes(table['gate_bf8']), 160 * 136 * 1088)
        self.assertEqual(sorted(set(table) - set(lever.shapes(4))), ['gate_bf8', 'up_bf8'])

    def test_the_t1_gate_cores_apply_to_the_diagnostic_gate_too(self):
        table = harness.shape_table(lever)
        self.assertEqual(lever.active_cores(harness.served_row(lever, table['gate_bf8'], 13, 88)['config'], table['gate_bf8']), 68)
        self.assertEqual(lever.active_cores(harness.served_row(lever, table['gate_bf8'], 13, 44)['config'], table['gate_bf8']), 46)

    def test_a_run_over_the_diagnostic_shapes(self):
        shapes = small_shapes()
        ttnn = FakeTTNN(cost=cost_model(dict(gateup=3, down=4)))
        out = Path(tempfile.mkdtemp()) / 'report.json'
        with mock.patch.dict(os.environ, dict(QWEN_FAST_TP='4')), mock.patch.object(lever, 'MIN_ACTIVE_CORES', 2), mock.patch.object(harness, 'WATCHDOG_S', 3600), \
                mock.patch.object(lever, 'shapes', lambda tp=4: shapes):
            status = harness.main(['--out', str(out), '--rounds', '3', '--shapes', 'gate_bf8', '--arms', 'sweep', '--budget-s', '100000'], torch=torch, ttnn=ttnn,
                                  lever=lever, clock=ttnn.clock, budget_clock=lambda: 0.0)
        report = json.loads(out.read_text())
        self.assertEqual((status, report['verdict']), (0, 'PASS'), report.get('error'))
        self.assertEqual(report['summary']['gate_bf8']['dtype'], 'bfp8')
        self.assertNotIn('decision', report, 'the rule needs the real gate and up')


class FakeProbeModule(object):
    """readprobe's stand-in: a launch costs 100 us at one tile a request and less as the chunk grows; write_back copies the source (or, for the broken point, not quite)."""

    BANK_POINTS = ((2, 1), (3, 3), (4, 4))
    STOCK_POINTS = ((2, 1), (3, 1), (4, 1))
    BROKEN = set()
    built = []

    class ReadProbe(object):
        def __init__(self, operations, mesh, source, destination, dtype, tile_rows, tile_columns, banks, mode, run, chunk, grid, write_back=False, width=None):
            if mode == 'bank' and tile_columns % banks and FakeProbeModule.strict:
                raise ValueError('not a bank multiple')
            self.ttnn, self.source, self.destination, self.write_back = operations, source, destination, write_back
            self.mode, self.run, self.chunk, self.dtype = mode, run, chunk, dtype
            self.plan = [None] * (-(-tile_columns // run))
            self.largest, self.mean, self.requests_per_row = chunk * 576, chunk * 576.0, 100
            FakeProbeModule.built.append((mode, run, chunk, dtype, write_back, banks))

        def __call__(self):
            self.ttnn.spend(100.0 / (1.0 + 0.3 * (self.chunk - 1)) * (0.5 if self.dtype == 'bfp8' else 1.0))
            if self.write_back:
                data = self.source.data.clone()
                if (self.mode, self.run, self.chunk) in FakeProbeModule.BROKEN:
                    data.view(torch.int16)[0, 0, 0, 0] ^= 1
                self.destination.data = data

    strict = False


class ProbeSectionTests(unittest.TestCase):
    def run_probe(self, ttnn, **overrides):
        FakeProbeModule.built[:] = []
        out = Path(tempfile.mkdtemp()) / 'report.json'
        small = (('gate_bf4', 'bfp4', 256, 1088), ('gate_bf8', 'bfp8', 256, 1088), ('down_bf8', 'bfp8', 1088, 256))
        with mock.patch.dict(os.environ, dict(QWEN_FAST_TP='4')), mock.patch.object(harness, 'PROBE_SHAPES', small), mock.patch.object(harness, 'WATCHDOG_S', 3600):
            status = harness.main(['--out', str(out), '--rounds', '3', '--arms', 'probe'], torch=torch, ttnn=ttnn, lever=lever, probe_module=FakeProbeModule,
                                  clock=ttnn.clock)
        return status, json.loads(out.read_text())

    def test_every_point_is_checked_then_timed_and_the_verdict_is_read_from_the_gate_shape(self):
        FakeProbeModule.BROKEN = set()
        status, report = self.run_probe(FakeTTNN())
        self.assertEqual((status, report['verdict']), (0, 'PASS'), report.get('error'))
        section = report['probe']
        self.assertEqual(section['banks'], 8)
        rows = section['rows']
        self.assertEqual(len(rows), 3 * 6)
        self.assertEqual([(row['mode'], row['run'], row['chunk']) for row in rows[:6]],
                         [('stock', 2, 1), ('stock', 3, 1), ('stock', 4, 1), ('bank', 2, 1), ('bank', 3, 3), ('bank', 4, 4)])
        self.assertTrue(all(row['exact'] for row in rows), rows)
        gate = dict(((row['mode'], row['run']), row) for row in rows if row['shape'] == 'gate_bf4')
        total_bytes = 8 * 34 * 576
        self.assertAlmostEqual(gate[('stock', 3)]['us'], 100.0, delta=0.01)
        self.assertAlmostEqual(gate[('stock', 3)]['gbps'], total_bytes / 100.0 / 1e3, delta=0.5)
        self.assertAlmostEqual(gate[('bank', 3)]['us'], 100.0 / 1.6, delta=0.05)
        self.assertAlmostEqual(gate[('bank', 4)]['delta_vs_stock_r3_us'], 100.0 / 1.9 - 100.0, delta=0.05)
        decision = section['decision']
        self.assertEqual(decision['verdict'], 'READ-GRANULARITY')
        self.assertAlmostEqual(decision['ratio'], 1.9, delta=0.05)
        self.assertEqual(decision['bank_point'], [4, 4])
        self.assertEqual(decision['stock_point'][1], 1)

    def test_the_correctness_launch_writes_back_and_the_timing_launch_does_not(self):
        FakeProbeModule.BROKEN = set()
        self.run_probe(FakeTTNN())
        flags = [(mode, run, chunk, dtype, write_back) for mode, run, chunk, dtype, write_back, banks in FakeProbeModule.built if dtype == 'bfp4']
        for point in (('bank', 3, 3), ('stock', 3, 1)):
            self.assertIn(point + ('bfp4', True), flags)
            self.assertIn(point + ('bfp4', False), flags)
        self.assertTrue(all(banks == 8 for unused, unused2, unused3, unused4, unused5, banks in FakeProbeModule.built))

    def test_a_point_whose_copy_differs_is_inexact_and_never_the_answer(self):
        FakeProbeModule.BROKEN = {('bank', 4, 4)}
        status, report = self.run_probe(FakeTTNN())
        rows = dict(((row['shape'], row['mode'], row['run']), row) for row in report['probe']['rows'])
        broken = rows[('gate_bf4', 'bank', 4)]
        self.assertFalse(broken['exact'])
        self.assertEqual(broken['differing'], 1)
        self.assertEqual(report['probe']['decision']['bank_point'], [3, 3], 'the fastest exact point wins, not the fastest')
        FakeProbeModule.BROKEN = set()

    def test_a_construction_that_fails_is_a_row_not_the_end(self):
        FakeProbeModule.BROKEN = set()
        FakeProbeModule.strict = True
        try:
            status, report = self.run_probe(FakeTTNN())
        finally:
            FakeProbeModule.strict = False
        self.assertEqual(status, 0)
        errors = [row for row in report['probe']['rows'] if 'error' in row]
        self.assertTrue(errors and all('bank multiple' in row['error'] for row in errors))

    def test_the_verdict_words(self):
        def row(mode, gbps, exact=True):
            return dict(shape='gate_bf4', mode=mode, run=3, chunk=3 if mode == 'bank' else 1, gbps=gbps, pct_peak=1.0, exact=exact)
        self.assertEqual(harness.probe_verdict([row('stock', 240.0), row('bank', 310.0)])['verdict'], 'READ-GRANULARITY')
        self.assertEqual(harness.probe_verdict([row('stock', 240.0), row('bank', 270.0)])['verdict'], 'MARGINAL')
        self.assertEqual(harness.probe_verdict([row('stock', 240.0), row('bank', 245.0)])['verdict'], 'NO-EFFECT')
        self.assertEqual(harness.probe_verdict([row('stock', 240.0), row('bank', 400.0, exact=False)])['verdict'], 'NO-RESULT')
        self.assertEqual(harness.probe_verdict([])['verdict'], 'NO-RESULT')


class SplitSectionTests(unittest.TestCase):
    def test_each_variant_is_compared_bit_for_bit_and_raced(self):
        ttnn = FakeTTNN(cost=cost_model(dict(gateup=3, down=4)))
        out = Path(tempfile.mkdtemp()) / 'report.json'
        shapes = small_shapes()
        with mock.patch.dict(os.environ, dict(QWEN_FAST_TP='4')), mock.patch.object(lever, 'shapes', lambda tp=4: shapes), mock.patch.object(harness, 'WATCHDOG_S', 3600):
            status = harness.main(['--out', str(out), '--rounds', '3', '--arms', 'split'], torch=torch, ttnn=ttnn, lever=lever, clock=ttnn.clock)
        report = json.loads(out.read_text())
        self.assertEqual((status, report['verdict']), (0, 'PASS'), report.get('error'))
        rows = dict((row['variant'], row) for row in report['split']['rows'])
        self.assertEqual(sorted(rows), ['a_silu_typecast', 'b_fused_in_multiply', 'c_bf16_then_fused', 'served'])
        self.assertTrue(rows['served']['exact'] and rows['a_silu_typecast']['exact'], 'silu on the float32 sum, rounded to bfloat16, then the product: the served arithmetic')
        self.assertFalse(rows['b_fused_in_multiply']['exact'], 'no bfloat16 rounding between the SiLU and the product')
        self.assertFalse(rows['c_bf16_then_fused']['exact'], 'the gate rounded before the SiLU')
        self.assertGreater(rows['b_fused_in_multiply']['differing'], 0)
        for row in rows.values():
            self.assertIn('us', row)
        # served = gate(40+3|3-3|=40) + up 40+ + multiply 3; variant a adds a silu 5 and a typecast 2
        self.assertAlmostEqual(rows['a_silu_typecast']['delta_us'], 5.0 + 2.0, delta=0.5)
        self.assertAlmostEqual(rows['a_silu_typecast']['gain_ms_pass'], -64 * rows['a_silu_typecast']['delta_us'] / 1000.0, delta=0.002)

    def test_a_variant_the_build_refuses_is_a_row(self):
        ttnn = FakeTTNN(cost=cost_model(dict(gateup=3, down=4)))

        def refusing(*args, **options):
            raise TypeError("multiply() got an unexpected keyword argument 'input_tensor_a_activations'")

        ttnn.multiply = refusing
        out = Path(tempfile.mkdtemp()) / 'report.json'
        shapes = small_shapes()
        with mock.patch.dict(os.environ, dict(QWEN_FAST_TP='4')), mock.patch.object(lever, 'shapes', lambda tp=4: shapes), mock.patch.object(harness, 'WATCHDOG_S', 3600):
            harness.main(['--out', str(out), '--rounds', '3', '--arms', 'split'], torch=torch, ttnn=ttnn, lever=lever, clock=ttnn.clock)
        rows = dict((row['variant'], row) for row in json.loads(out.read_text())['split']['rows'])
        self.assertIn('unexpected keyword', rows['b_fused_in_multiply']['error'])
        self.assertIn('us', rows['served'])
        self.assertIn('us', rows['a_silu_typecast'])


# ---------------------------------------------------------------------------------------------------------------------------
# The run script.
# ---------------------------------------------------------------------------------------------------------------------------

class FakeBank(object):
    """The bank module's BankGateUp stand-in: the unfused arithmetic in one call, a cost of 30 us less one per chunk, and one flipped bit when the SFPU approximation mode is not the
    served one (or for the chunks in `inexact`), as a card might show."""

    built = []
    inexact = ()
    cost = staticmethod(lambda chunk: 30.0 - chunk)

    def __init__(self, operations, mesh, w1, w3, chunk=4, grid=(13, 10), width=None, math_approx_mode=None, banks=None, **options):
        self.ttnn, self.w1, self.w3, self.chunk, self.approx = operations, w1, w3, chunk, math_approx_mode
        self.plan = dict(workers=-(-136 // 8 // chunk) * 8, request_bytes=chunk * 576)
        FakeBank.built.append((chunk, math_approx_mode, grid, banks))

    def __call__(self, x):
        gate = x.data.float() @ self.w1.data.float()
        gate = (gate * torch.sigmoid(gate)).to(torch.bfloat16)
        up = (x.data.float() @ self.w3.data.float()).to(torch.bfloat16)
        out = (gate.float() * up.float()).to(torch.bfloat16)
        if self.approx is not True or self.chunk in FakeBank.inexact:
            out.view(torch.int16)[0, 0, 0, 0] ^= 1
        self.ttnn.spend(FakeBank.cost(self.chunk))
        return Tensor(out)


def fake_bank_module():
    return types.SimpleNamespace(BankGateUp=FakeBank, CHUNKS=(2, 3, 4), DEFAULT_CHUNK=4)


class BankArmTests(unittest.TestCase):
    def setUp(self):
        FakeBank.built.clear()
        FakeBank.inexact = ()

    def run_bank(self, ttnn, extra=()):
        out = Path(tempfile.mkdtemp()) / 'report.json'
        shapes = small_shapes()
        with mock.patch.dict(os.environ, dict(QWEN_FAST_TP='4')), mock.patch.object(lever, 'shapes', lambda tp=4: shapes), \
                mock.patch.object(harness, 'WATCHDOG_S', 3600):
            status = harness.main(['--out', str(out), '--rounds', '3', '--arms', 'bank'] + list(extra), torch=torch, ttnn=ttnn, lever=lever,
                                  bank_module=fake_bank_module(), clock=ttnn.clock)
        return status, json.loads(out.read_text())

    @staticmethod
    def served_cost(config, shape):
        return 40.0 if config.per_core_N == 1 else 20.0           # the small shapes' served partition is per_core_N 1; the tuned g3u4d3 is wider and "faster"

    def test_every_chunk_is_built_compared_on_both_regimes_timed_and_chained(self):
        ttnn = FakeTTNN(cost=self.served_cost)
        status, report = self.run_bank(ttnn)
        self.assertEqual(status, 0, report.get('error'))
        section = report['bank']
        rows = section['rows']
        self.assertEqual([(row['regime'], row['chunk'], row['diagnostic']) for row in rows],
                         [('random', 2, False), ('random', 3, False), ('random', 4, False), ('random', 4, True),
                          ('edge', 2, False), ('edge', 3, False), ('edge', 4, False)])
        self.assertTrue(all(row['exact'] for row in rows if not row['diagnostic']), rows)
        diagnostic = [row for row in rows if row['diagnostic']][0]
        self.assertEqual((diagnostic['math_approx_mode'], diagnostic['exact'], diagnostic['differing']), (False, False, 1))
        self.assertEqual(section['banks'], 8)
        self.assertEqual(FakeBank.built[0], (2, True, (13, 10), 8))
        # the served pair is two projections of 40 and the multiply of 4; the fake bank op costs 30 - chunk
        self.assertAlmostEqual(section['pair_us']['served'], 84.0, delta=0.01)
        self.assertAlmostEqual(section['pair_us']['bank_c4'], 26.0, delta=0.01)
        self.assertAlmostEqual(section['pair_us']['bank_c2'], 28.0, delta=0.01)
        timed = dict((row['chunk'], row) for row in rows if row['regime'] == 'random' and not row['diagnostic'])
        self.assertAlmostEqual(timed[4]['gain_ms_pass'], 64 * (84.0 - 26.0) / 1000.0, delta=0.01)
        self.assertAlmostEqual(timed[4]['gbps'], section['pair_bytes'] / 26.0 / 1e3, delta=0.1)
        self.assertNotIn('us', [row for row in rows if row['regime'] == 'edge'][0], 'the edge regime is a comparison, not a timing')
        chain = section['chain']
        self.assertEqual((chain['chunk'], chain['tuned_down']), (4, 3))
        self.assertTrue(all(chain['exact'].values()), chain)
        self.assertAlmostEqual(chain['chain_us']['served'], 124.0, delta=0.05)
        self.assertAlmostEqual(chain['chain_us']['bank_served_down'], 26.0 + 40.0, delta=0.05)
        self.assertAlmostEqual(chain['chain_us']['bank_tuned_down'], 26.0 + 20.0, delta=0.05)
        decision = section['decision']
        self.assertEqual(decision['verdict'], 'BANK-GO')
        self.assertEqual(decision['env'], {'QWEN_FAST_MLP_GATEUP_BANK': '1', 'QWEN_FAST_MLP_GATEUP_BANK_CHUNK': '4', 'QWEN_FAST_MLP_CFG': 'd3'})
        self.assertAlmostEqual(decision['gain_vs_tuned_ms_pass'], 64 * ((20.0 + 20.0 + 3.0 + 20.0) - (26.0 + 20.0)) / 1000.0, delta=0.01)
        self.assertEqual(report['verdict'], 'PASS')

    def test_a_chunk_whose_product_differs_is_never_chained_and_the_verdict_is_inexact(self):
        ttnn = FakeTTNN(cost=self.served_cost)
        FakeBank.inexact = (4,)
        status, report = self.run_bank(ttnn)
        section = report['bank']
        self.assertEqual(section['decision']['verdict'], 'BANK-INEXACT')
        self.assertEqual(sorted(set(row['chunk'] for row in section['rows'] if row['exact'] is False and not row['diagnostic'])), [4])
        self.assertEqual(section['chain']['chunk'], 3, 'the fastest EXACT chunk is chained, never the inexact 4')
        self.assertNotIn('bank_c4', section['pair_us'])
        self.assertEqual((status, report['verdict']), (1, 'FAIL'))

    def test_a_build_that_fails_is_a_row_and_the_rest_still_run(self):
        ttnn = FakeTTNN(cost=self.served_cost)
        original = FakeBank.__init__

        def refusing(self, operations, mesh, w1, w3, chunk=4, **options):
            if chunk == 3:
                raise RuntimeError('TT_FATAL: kernel compile failed')
            original(self, operations, mesh, w1, w3, chunk=chunk, **options)

        with mock.patch.object(FakeBank, '__init__', refusing):
            status, report = self.run_bank(ttnn)
        section = report['bank']
        failed = [row for row in section['rows'] if row['chunk'] == 3]
        self.assertTrue(failed and all('kernel compile failed' in row['error'] for row in failed))
        self.assertNotIn('bank_c3', section['pair_us'])
        self.assertIn('bank_c4', section['pair_us'])
        self.assertEqual(section['decision']['verdict'], 'BANK-GO')
        self.assertEqual(status, 0)

    def test_the_verdict_words(self):
        base = dict(rows=[dict(regime='random', chunk=4, diagnostic=False, exact=True, differing=0, us=26.0)],
                    chain=dict(chunk=4, tuned_down=3, exact={'served': True, 'bank_tuned_down': True},
                               chain_us={'served': 124.0, 'tuned': 100.0, 'bank_tuned_down': 90.0}))
        self.assertEqual(harness.bank_decision(base)['verdict'], 'BANK-GO')                      # 10 us x 64 = 0.64 ms over the tuned chain
        marginal = dict(base, chain=dict(base['chain'], chain_us={'served': 124.0, 'tuned': 91.0, 'bank_tuned_down': 90.0}))
        self.assertEqual(harness.bank_decision(marginal)['verdict'], 'BANK-MARGINAL')            # 34 us over the served chain, 1 us over the tuned
        flat = dict(base, chain=dict(base['chain'], chain_us={'served': 91.0, 'tuned': 91.0, 'bank_tuned_down': 90.0}))
        self.assertEqual(harness.bank_decision(flat)['verdict'], 'BANK-NO-GAIN')
        wrong = dict(base, chain=dict(base['chain'], exact={'served': True, 'bank_tuned_down': False}))
        self.assertEqual(harness.bank_decision(wrong)['verdict'], 'BANK-INEXACT')
        bad_row = dict(base, rows=base['rows'] + [dict(regime='edge', chunk=3, diagnostic=False, exact=False, differing=7)])
        self.assertEqual(harness.bank_decision(bad_row)['rows'], [('edge', 3, 7)])
        diagnostic_only = dict(base, rows=base['rows'] + [dict(regime='random', chunk=4, diagnostic=True, exact=False, differing=1)])
        self.assertEqual(harness.bank_decision(diagnostic_only)['verdict'], 'BANK-GO', 'the opposite math mode is a diagnostic, not a verdict')
        self.assertEqual(harness.bank_decision(dict(rows=[], chain={}))['verdict'], 'NO-RESULT')
        self.assertEqual(harness.bank_decision(dict(base, chain={}))['verdict'], 'NO-RESULT')
        self.assertEqual(harness.bank_decision(base)['env']['QWEN_FAST_MLP_CFG'], 'd3')
        self.assertAlmostEqual(harness.bank_decision(base)['gain_vs_tuned_ms_pass'], 0.64, delta=0.001)

    def test_the_threshold_is_the_plans_low_end_and_the_rule_is_documented(self):
        self.assertEqual(harness.THRESHOLD_MS, 0.35)
        for word in ('BANK-GO', 'BANK-MARGINAL', 'BANK-NO-GAIN', 'BANK-INEXACT', 'NO-RESULT'):
            self.assertIn(word, harness.__doc__)

    def test_the_tuned_partition_must_name_g_u_and_d(self):
        self.assertEqual(harness.tuned_partition(lever, 'g3u4d3'), (3, 4, 3))
        for bad in ('g3', 'd3', 'g3u4', 'l1', 'x'):
            with self.assertRaises(ValueError):
                harness.tuned_partition(lever, bad)
        status, report = self.run_bank(FakeTTNN(), ['--tuned', 'g3u4'])
        self.assertEqual((status, report['verdict']), (4, 'NOT-RUN'))
        self.assertIn('--tuned', report['error'])

    def test_a_different_tuned_partition_changes_the_down_the_verdict_prints(self):
        ttnn = FakeTTNN(cost=self.served_cost)
        status, report = self.run_bank(ttnn, ['--tuned', 'g2u2d2'])
        self.assertEqual(report['bank']['chain']['tuned_down'], 2)
        self.assertEqual(report['bank']['decision']['env']['QWEN_FAST_MLP_CFG'], 'd2')

    def test_the_arm_is_registered_and_the_other_arms_do_not_run_it(self):
        self.assertIn('bank', harness.ARMS)
        ttnn = FakeTTNN()
        out = Path(tempfile.mkdtemp()) / 'report.json'
        shapes = small_shapes()
        with mock.patch.dict(os.environ, dict(QWEN_FAST_TP='4')), mock.patch.object(lever, 'shapes', lambda tp=4: shapes), mock.patch.object(harness, 'WATCHDOG_S', 3600):
            status = harness.main(['--out', str(out), '--rounds', '3', '--arms', 'compose'], torch=torch, ttnn=ttnn, lever=lever, bank_module=fake_bank_module(),
                                  clock=ttnn.clock)
        self.assertNotIn('bank', json.loads(out.read_text()))
        self.assertEqual(FakeBank.built, [])


class RunScriptTests(unittest.TestCase):
    script = HERE / 'run_card_m.sh'
    text = script.read_text(encoding='utf-8')

    def test_it_embeds_the_canonical_qual_card_block_and_selects_the_board_with_it(self):
        import c2_serving_job
        library = (CI / 'qual_card.sh').read_text(encoding='utf-8')
        self.assertTrue(c2_serving_job.embeds_qual_card(self.text, library))

    def test_it_forwards_its_own_grid_clamp_not_the_steps(self):
        self.assertIn('${MLP_GRID_CLAMP:+-e "TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=$MLP_GRID_CLAMP"}', self.text)
        self.assertNotIn('QUAL_TT_GRID', self.text, 'test_c2_tt_grid keeps a registry of the harnesses that forward the step\'s variable; this is not one of them')

    @unittest.skipUnless(shutil.which('bash'), 'needs bash')
    def test_it_refuses_a_clamp_that_is_none_of_the_jobs_values_before_it_looks_at_a_card(self):
        env = dict((key, value) for key, value in os.environ.items() if not key.startswith('QUAL_'))
        for bad in ('9,9', '10,10', 'x'):
            result = subprocess.run(['bash', str(self.script)], capture_output=True, text=True, env=dict(env, MLP_GRID_CLAMP=bad))
            self.assertEqual(result.returncode, 1)
            self.assertIn('MLP_GRID_CLAMP=%s is none of 10,9 11,9 12,9' % bad, result.stderr)
        for good in ('10,9', '11,9', '12,9', ''):
            result = subprocess.run(['bash', str(self.script)], capture_output=True, text=True, env=dict(env, MLP_GRID_CLAMP=good))
            self.assertIn('QUAL_CARD is not set', result.stderr, 'a good value passes the check and meets the missing card')

    def test_it_mounts_only_files_that_exist_and_names_the_harness(self):
        found = re.search(r'for file in ([^;]+); do', self.text)
        names = found.group(1).split()
        self.assertIn('tp4_mlp_gateup.py', names)
        for name in names:
            self.assertTrue((CI / name).is_file(), name)
        self.assertIn('mlp_gateup_card_m.py readprobe.py readprobe_reader.cpp', self.text)
        for name in ('tp4_mlp_bank.py', 'tp4_mlp_bank_weights.cpp'):
            self.assertIn(name, names)
        for name in ('mlp_gateup_card_m.py', 'readprobe.py', 'readprobe_reader.cpp'):
            self.assertTrue((HERE / name).is_file(), name)

    def test_it_runs_one_card_with_the_geometry_and_no_network(self):
        for needle in ('--network none', '-e QWEN_FAST_TP=4', '-e QWEN_C2_SERVING=0', 'timeout -k 30 "$timeout_s" docker run', 'qual_refuse_holders',
                       'qual_card_recheck', 'HANG SUSPECTED'):
            self.assertIn(needle, self.text)
        self.assertNotRegex(self.text, r'\bsudo\b.*tt-smi -r')                       # it never resets: it prints the hint
        self.assertGreater(int(re.search(r'timeout_s=(\d+)', self.text).group(1)), harness.WATCHDOG_S, 'the container outlives the watchdog')

    @unittest.skipUnless(shutil.which('bash'), 'needs bash')
    def test_it_parses(self):
        self.assertEqual(subprocess.run(['bash', '-n', str(self.script)], capture_output=True).returncode, 0)

    def test_it_refuses_without_a_card(self):
        if not shutil.which('bash'):
            self.skipTest('needs bash')
        env = dict((key, value) for key, value in os.environ.items() if not key.startswith('QUAL_'))
        env.pop('ALLOW_SERVING_CARD', None)
        result = subprocess.run(['bash', str(self.script)], capture_output=True, text=True, env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('QUAL_CARD is not set', result.stderr)


class PublicTextTests(unittest.TestCase):
    BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/home/|zot\.|@[A-Z0-9_]+@')

    def test_the_new_files_name_no_rig_address_registry_or_digest(self):
        for path in (HERE / 'mlp_gateup_card_m.py', HERE / 'readprobe.py', HERE / 'readprobe_reader.cpp', HERE / 'run_card_m.sh', CI / 'tp4_mlp_gateup.py', CI / 'tp4_mlp_fused.py',
                     CI / 'tp4_mlp_fused_input.cpp', CI / 'tp4_mlp_fused_weights.cpp', CI / 'tp4_mlp_bank.py', CI / 'tp4_mlp_bank_weights.cpp'):
            text = path.read_text(encoding='utf-8')
            if path.suffix == '.sh':          # the canonical qual_card.sh block is shared and names the cards by board id; the rest is this harness's
                text = re.sub(r'# >>> qual_card\.sh.*?# <<< qual_card\.sh', '', text, flags=re.S)
            self.assertIsNone(self.BANNED.search(text), '%s: %s' % (path.name, self.BANNED.search(text)))


if __name__ == '__main__':
    unittest.main()
