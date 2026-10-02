"""matmul_tp4_sweep (lever V7): the TP4 per-chip MLP gate/up 64-row matmul sweep. CPU only.

Held here: the shapes are the TP4 widths from tp_shapes (K = 5,120, N = 4,352 = 136 tiles, bfloat4_b for gate and up); the builder
arithmetic equals the graft's transcription and lands on the profile's core counts (the 88-core-requested gate on 68 active cores, the
44-requested up on 34); the candidate generator only emits configs the program accepts (a grid inside the worker grid that holds the
active cores, per_core_N x cores covering every output tile with no empty core, in0_block_w dividing K, a subblock dividing per_core_N
within the register cap, a circular-buffer estimate inside the L1 budget), no duplicates, includes the model's own partitions, and stays
small enough to run in the harness's timeout; the ranking never puts an error or an inexact config first for the answer; the report and
verdict lines read from a fake device run of the whole loop. What only hardware shows is the timing and the byte compare itself.
"""

import json
import math
import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matmul_tp4_sweep as sweep  # noqa: E402
import test_verify_trace_t1_graft as graft  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
BY_NAME = {shape['name']: shape for shape in sweep.shapes()}
GATE, UP, DOWN = BY_NAME['mlp_w1'], BY_NAME['mlp_w3'], BY_NAME['mlp_w2']


def tiles(shape):
    return sweep.base.tiles(shape['K']), sweep.base.tiles(shape['N'])


class ShapeTests(unittest.TestCase):
    def test_the_tp4_per_chip_shapes(self):
        self.assertEqual((GATE['K'], GATE['N'], GATE['dtype'], GATE['silu']), (5120, 4352, 'bfp4', True))
        self.assertEqual((UP['K'], UP['N'], UP['dtype'], UP['silu']), (5120, 4352, 'bfp4', False))
        self.assertEqual((DOWN['K'], DOWN['N'], DOWN['dtype']), (4352, 5120, 'bfp8'))
        self.assertEqual(tiles(GATE), (160, 136))
        self.assertEqual(sweep.TP, 4)

    def test_they_are_the_graft_dims_divided_by_four(self):
        self.assertEqual(GATE['N'], 17408 // 4)
        self.assertEqual(sweep.shapes(2)[0]['N'], 17408 // 2, 'the same table at the pair, for contrast')

    def test_the_dram_floor_matches_the_profiles_bytes(self):
        # v170: gate 802 MB over 64 calls = 12.5 MB per call; 31 us at 400 GB/s; measured 52.9 us = 58% of 405 GB/s
        floor = sweep.dram_floor_us(GATE, 405.0)
        self.assertAlmostEqual(floor, 30.9, delta=0.5)
        self.assertAlmostEqual(floor / 52.9, 0.58, delta=0.02)


class BuilderTests(unittest.TestCase):
    def test_it_equals_the_graft_transcription_for_every_model_arm(self):
        for name, shape in BY_NAME.items():
            for cores in (88, 44, 33):
                with self.subTest(shape=name, num_cores=cores):
                    mine = sweep.builder_config(64, shape['K'], shape['N'], cores, 11)
                    theirs = graft.create_matmul_1d_decode_progcfg(64, shape['K'], shape['N'], cores, grid_w=11)
                    self.assertEqual(mine['grid'], tuple(theirs.compute_with_storage_grid_size))
                    for key in ('in0_block_w', 'per_core_M', 'per_core_N', 'out_subblock_h', 'out_subblock_w'):
                        self.assertEqual(mine[key], getattr(theirs, key), key)

    def test_the_models_gate_and_up_land_on_the_profiles_core_counts(self):
        gate = sweep.builder_config(64, GATE['K'], GATE['N'], sweep.MODEL_CORES['mlp_w1'], 11)
        up = sweep.builder_config(64, UP['K'], UP['N'], sweep.MODEL_CORES['mlp_w3'], 11)
        self.assertEqual(sweep.active_cores(gate, 136), 68, 'the 88-core request is 68 cores of per_core_N 2')
        self.assertEqual(sweep.active_cores(up, 136), 34, 'the 44-core request is 34 cores of per_core_N 4')
        self.assertEqual((gate['per_core_N'], up['per_core_N']), (2, 4))
        self.assertEqual((gate['in0_block_w'], up['in0_block_w']), (8, 8))

    def test_the_models_core_requests_are_the_graft_s(self):
        with open(os.path.join(ROOT, 'docker', 'qwen-c2-graft', 'graft', 'model_config.py'), encoding='utf-8') as handle:
            text = handle.read()
        self.assertIn('num_cores=44,\n            fused_activation=ttnn.UnaryOpType.SILU', text)     # the gate at T1 off
        self.assertIn('num_cores=88,', text)                                                          # the gate at T1 on
        self.assertIn('M64, self.hidden_dim // tp, self.dim, num_cores=33', text)
        self.assertEqual(sweep.MODEL_CORES, {'mlp_w1': 88, 'mlp_w3': 44, 'mlp_w2': 33})


class GeneratorTests(unittest.TestCase):
    def check_legal(self, shape, config):
        k_tiles, n_tiles = tiles(shape)
        cols, rows = config['grid']
        cores = sweep.active_cores(config, n_tiles)
        self.assertLessEqual(cols, sweep.WORKER_GRID[0])
        self.assertLessEqual(rows, sweep.WORKER_GRID[1])
        self.assertLessEqual(cores, cols * rows, 'the grid holds every active core')
        self.assertGreater(cores * config['per_core_N'], n_tiles - 1, 'every output tile column is covered')
        self.assertLess((cores - 1) * config['per_core_N'], n_tiles, 'no core holds nothing')
        self.assertEqual(k_tiles % config['in0_block_w'], 0)
        self.assertEqual(config['per_core_M'], 2)
        self.assertEqual(config['per_core_N'] % config['out_subblock_w'], 0)
        self.assertEqual(config['per_core_M'] % config['out_subblock_h'], 0)
        self.assertLessEqual(config['out_subblock_h'] * config['out_subblock_w'], sweep.SUBBLOCK_CAP)
        self.assertLessEqual(sweep.l1_bytes(config, shape['dtype']), sweep.L1_BUDGET)

    def test_stage_one_is_one_legal_config_per_per_core_n_on_the_models_grid_width_and_block(self):
        for shape in (GATE, UP, DOWN):
            configs = sweep.stage1(shape)
            with self.subTest(shape=shape['name']):
                self.assertTrue(configs)
                self.assertEqual(len({config['per_core_N'] for config in configs}), len(configs))
                for config in configs:
                    self.check_legal(shape, config)
                    self.assertEqual(config['in0_block_w'], sweep.largest_divisor(tiles(shape)[0]), 'the model\'s block')
                    self.assertEqual(config['grid'][0], min(11, sweep.active_cores(config, tiles(shape)[1])), 'the models 11-wide grid')

    def test_stage_one_contains_the_models_partitions_for_gate_and_up(self):
        pcn = {config['per_core_N'] for config in sweep.stage1(GATE)}
        self.assertIn(2, pcn, 'the gate\'s 68 cores')
        self.assertIn(4, pcn, 'the up\'s 34 cores: the gate on the up\'s partition is the first thing to measure')

    def test_per_core_n_values_stay_inside_the_worker_grid(self):
        values = sweep.per_core_n_values(136)
        self.assertEqual(values[0], 2, '136 / 1 = 136 cores does not fit 110')
        for pcn in values:
            cores = math.ceil(136 / pcn)
            self.assertTrue(8 <= cores <= 110)

    def test_stage_two_is_legal_unique_and_varies_width_block_and_subblock(self):
        for shape in (GATE, UP):
            configs = sweep.stage2(shape, [2, 3, 4, 5])
            with self.subTest(shape=shape['name']):
                self.assertEqual(len({sweep.config_key(config) for config in configs}), len(configs))
                for config in configs:
                    self.check_legal(shape, config)
                self.assertEqual({config['grid'][0] for config in configs} & {8, 10, 11}, {8, 10, 11})
                self.assertGreater(len({config['in0_block_w'] for config in configs}), 3)
                self.assertEqual({config['out_subblock_h'] for config in configs}, {1, 2})

    def test_the_candidate_count_fits_the_harness_timeout(self):
        # each config compiles once (about two seconds); the harness's container timeout is 40 minutes for gate and up together
        for shape in (GATE, UP):
            total = len(sweep.stage1(shape)) + len(sweep.stage2(shape, range(2, 2 + sweep.STAGE2_PER_CORE_N)))
            self.assertLess(total, 300)
            self.assertGreater(total, 100)

    def test_the_models_own_in0_block_is_among_the_stage_two_blocks(self):
        configs = sweep.stage2(GATE, [2])
        self.assertIn(8, {config['in0_block_w'] for config in configs})
        self.assertTrue({1, 2, 4, 5, 8, 10, 16, 20} & {config['in0_block_w'] for config in configs})

    def test_l1_budget_prunes_the_largest_blocks_at_wide_per_core_n(self):
        small = sweep.block_choices(160, 2, 'bfp4')
        big = sweep.block_choices(160, 19, 'bfp4')
        self.assertGreaterEqual(max(small), max(big))
        for block in big:
            self.assertLessEqual(sweep.l1_bytes(dict(per_core_M=2, per_core_N=19, in0_block_w=block), 'bfp4'), sweep.L1_BUDGET)

    def test_subblocks_are_distinct_in_height_and_largest_first(self):
        found = sweep.subblocks(2, 4)
        self.assertEqual(found, [(1, 4), (2, 2)])
        self.assertEqual(sweep.subblocks(2, 2, keep=1), [(2, 2)])

    def test_grid_for_refuses_what_does_not_fit_the_worker_grid(self):
        self.assertEqual(sweep.grid_for(68, 11), (11, 7))
        self.assertEqual(sweep.grid_for(68, 8), (8, 9))
        self.assertIsNone(sweep.grid_for(111, 11))
        self.assertEqual(sweep.grid_for(5, 11), (5, 1))

    def test_labels_are_unique_per_config_key(self):
        configs = sweep.stage2(GATE, [2, 3, 4, 5])
        self.assertEqual(len({sweep.label(config) for config in configs}), len(configs))


class BuilderArgumentsTests(unittest.TestCase):
    def test_a_config_the_builder_makes_is_reproduced(self):
        model = sweep.builder_config(64, GATE['K'], GATE['N'], 44, 11)
        found = sweep.builder_arguments(model, 160, 136)
        self.assertEqual((found['num_cores'], found['grid_w'], found['reproduces']), (44, 11, True))

    def test_a_config_it_cannot_make_says_so(self):
        config = dict(grid=(11, 7), in0_block_w=4, per_core_M=2, per_core_N=2, out_subblock_h=2, out_subblock_w=2)
        self.assertFalse(sweep.builder_arguments(config, 160, 136)['reproduces'], 'the builder always takes block 8')


class RankingTests(unittest.TestCase):
    ROWS = [dict(arm='a', us=40.0, exact=True), dict(arm='b', us=30.0, exact=False), dict(arm='c', error='TT_FATAL'),
            dict(arm='d', us=35.0, exact=True), dict(arm='e', us=31.0)]

    def test_errors_last_and_ties_stable(self):
        ranked = sweep.rank(self.ROWS)
        self.assertEqual([row['arm'] for row in ranked], ['b', 'e', 'd', 'a', 'c'])

    def test_the_answer_is_the_fastest_exact_row_not_the_fastest(self):
        exact, anyone = sweep.best(self.ROWS)
        self.assertEqual((exact['arm'], anyone['arm']), ('d', 'b'))

    def test_a_missing_exact_flag_is_not_exact(self):
        exact, _ = sweep.best([dict(arm='e', us=31.0)])
        self.assertIsNone(exact)

    def test_nothing_measured_is_none_and_no_speedup(self):
        self.assertEqual(sweep.best([dict(arm='c', error='x')]), (None, None))
        self.assertIsNone(sweep.speedup(40.0, None))
        self.assertAlmostEqual(sweep.speedup(40.0, dict(us=32.0)), 1.25)


class FakeTensor:
    def __init__(self, tag):
        self.tag = tag

    def clone(self):
        return self

    def __eq__(self, other):
        return isinstance(other, FakeTensor) and self.tag == other.tag

    __hash__ = None


class FakeDevice:
    def compute_with_storage_grid_size(self):
        return types.SimpleNamespace(x=11, y=10)


class FakeTtnn:
    L1_MEMORY_CONFIG = 'l1'
    DRAM_MEMORY_CONFIG = 'dram'
    UnaryOpType = types.SimpleNamespace(SILU='silu')

    def __init__(self):
        self.freed = 0

    def MatmulMultiCoreReuseMultiCast1DProgramConfig(self, **options):
        return options

    def linear(self, x, weight, program_config, **options):
        # the in0_block_w changes the bytes (as it can on hardware); a partition-only change does not
        return FakeTensor(('out', program_config['in0_block_w']))

    def to_torch(self, tensor):
        return tensor

    def deallocate(self, tensor):
        self.freed += 1

    def synchronize_device(self, device):
        pass


class RunLoopTests(unittest.TestCase):
    def run_loop(self, shape):
        ttnn = FakeTtnn()
        torch = types.SimpleNamespace(equal=lambda a, b: a == b)

        def fake_time(_ttnn, _device, once, calls, rounds):
            out = once()
            # a deterministic 'time': fewer cores, wider block is faster; per_core_N 3 errors out
            return dict(us=100.0 - out.tag[1], min_us=99.0)

        records = []

        def fake_program(_ttnn, config, silu):
            if config['per_core_N'] == 3:
                raise RuntimeError('TT_FATAL: bad config')
            records.append(config)
            return dict(config)

        rows = []
        with mock.patch.object(sweep, 'make_weight', lambda *a: 'weight'), \
                mock.patch.object(sweep.base, 'make_activation', lambda *a: 'x'), \
                mock.patch.object(sweep.base, 'compute_kernel_config', lambda _t: 'ckc'), \
                mock.patch.object(sweep, 'time_batch', fake_time), mock.patch.object(sweep, 'program', fake_program):
            sweep.run_shape(ttnn, torch, FakeDevice(), shape, 2, 2, rows)
        return rows, ttnn

    def test_the_loop_measures_current_first_then_both_stages_and_keeps_errors_as_rows(self):
        rows, ttnn = self.run_loop(GATE)
        self.assertEqual(rows[0]['arm'], 'model_current')
        self.assertTrue(rows[0]['is_model_current'])
        self.assertEqual(rows[0]['active_cores'], 68)
        self.assertEqual(rows[0]['exact'], True, 'the current config against itself')
        errors = [row for row in rows if 'error' in row]
        self.assertTrue(errors)
        self.assertTrue(all(row['config']['per_core_N'] == 3 for row in errors))
        keys = [sweep.config_key(row['config']) for row in rows]
        self.assertEqual(len(keys), len(set(keys)), 'no config is timed twice')
        self.assertTrue(any(row['config']['in0_block_w'] != 8 for row in rows), 'stage two varied the block')
        self.assertEqual(sum(1 for row in rows if 'us' in row and row['exact'] is False) > 0, True,
                         'a block change that moves the bytes is reported inexact')

    def test_the_summary_names_the_best_exact_config_and_the_builder_arguments(self):
        rows, _ = self.run_loop(GATE)
        result = sweep.summary(rows, [GATE])['mlp_w1']
        self.assertTrue(result['best_exact']['exact'])
        self.assertEqual(result['best_exact']['config']['in0_block_w'], 8, 'only block 8 is byte-equal in this fake')
        self.assertLessEqual(result['best_any']['us'], result['best_exact']['us'])
        self.assertIsNotNone(result['speedup_exact'])
        self.assertEqual(len(result['top']), 8)
        json.dumps(result)                                   # the report is JSON
        lines = sweep.verdict_lines({'mlp_w1': result})
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith('TP4_SWEEP mlp_w1 current '))
        self.assertIn('best_exact', lines[0])
        self.assertIn('builder {', lines[0])

    def test_a_shape_with_no_result_says_so(self):
        lines = sweep.verdict_lines({'mlp_w3': dict(current=None, best_exact=None)})
        self.assertIn('no result', lines[0])


class HarnessTests(unittest.TestCase):
    def test_the_harness_mounts_the_three_scripts_and_reads_the_image_by_tag(self):
        path = os.path.join(ROOT, 'optimisation', 'ttnn-op', 'matmul_tp4_sweep', 'run_card_m.sh')
        with open(path, encoding='utf-8') as handle:
            text = handle.read()
        self.assertNotIn(chr(13), text)
        for name in ('matmul_tp4_sweep.py', 'matmul64_sweep.py', 'tp_shapes.py'):
            self.assertIn(name, text)
        self.assertIn('qwen38-c2-', text)
        self.assertIn('--network none', text)
        self.assertNotIn('zot', text)

    def test_the_cardm_job_line_parses_with_the_harness_and_its_tag(self):
        import c2_serving_job as job

        with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as handle:
            names = sorted(json.load(handle)['profiles'])
        text = ('C2_CARDS=pair\nC2_ACTIONS=cardm\nC2_IMAGE_TAG=tp4-next-3\n'
                'C2_CARDM_HARNESS=optimisation/ttnn-op/matmul_tp4_sweep/run_card_m.sh\nC2_CARDM_ARGS=--shapes mlp_w1,mlp_w3\n'
                'C2_CARDM_ENV=IMAGE_TAG=tp4-next-3\n')
        found = job.read_job(job.parse_env(text), names)
        self.assertEqual(found['actions'], 'cardm')


if __name__ == '__main__':
    unittest.main()
