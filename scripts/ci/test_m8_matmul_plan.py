"""tp4/m8-phase1: the 128-row matmul plan, the card-M byte-compare harness B1 (m8_matmul_bytes.py + optimisation/ttnn-op/m8_matmul/run_card_m.sh),
and the job pack references/tp4-m8-jobs. CPU only.

Held here: the seven projections' shapes are the TP4 widths; their builder arguments are the graft's own (read from model_config.py, and verify-trace
T1 #11's attn_qkv and gate); the configs at M = 128 are the SAME builder's (per_core_M 4) and keep each projection's in0_block_w, fuse_batch and mcast_in0 (the
K order), moving only the partition and, for the per_core_N = 5 projections, the output subblock; the orchestration (which rows of the 128-row output are
compared with which half and which tile, the per-tile mismatch counts, the verdicts) runs against a fake backend, including a fake that breaks one tile;
the harness embeds the canonical qual_card block, launches with QWEN_C2_SERVING=0 and `env -u TT_MESH_GRAPH_DESC_PATH`, and mounts exactly its five scripts; the
job pack parses with the job parser, runs one card-M job then one quad job that rescans before it resets, and touches no node agent. What only the card shows
is the bytes."""

import json
import os
import re
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402
import m8_matmul_bytes as bytes_run  # noqa: E402
import m8_matmul_plan as plan  # noqa: E402
import test_verify_trace_t1_graft as graft  # noqa: E402
import tp_shapes  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
HARNESS = os.path.join(ROOT, 'optimisation', 'ttnn-op', 'm8_matmul', 'run_card_m.sh')
FOLDER = os.path.join(HERE, 'references', 'tp4-m8-jobs')
PROFILES_PATH = os.path.join(HERE, 'qwen_c2_profiles.json')
IMAGE = 'tp4-262k8-best-1'
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/dev/tenstorrent/by-id|home/|zot\.')
HF_SNAPSHOT = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
GRAFT_MODEL_CONFIG = os.path.join(ROOT, 'docker', 'qwen-c2-graft', 'graft', 'model_config.py')


def read(path):
    with open(path, encoding='utf-8') as handle:
        return handle.read().replace(chr(13) + chr(10), chr(10))


class PlanShapeTests(unittest.TestCase):
    def test_eight_projections_in_model_order_at_the_tp4_widths(self):
        found = tp_shapes.geometry(4)
        table = {entry['name']: entry for entry in plan.projections()}
        self.assertEqual([entry['name'] for entry in plan.projections()],
                         ['attn_qkv', 'attn_wo', 'gdn_qkvzab', 'gdn_out', 'mlp_w1', 'mlp_w3', 'mlp_w2', 'lm_head'])
        self.assertEqual(len(plan.seven()), 7)
        self.assertEqual((table['mlp_w1']['k'], table['mlp_w1']['n'], table['mlp_w1']['dtype'], table['mlp_w1']['silu']), (5120, 4352, 'bfp4', True))
        self.assertEqual((table['mlp_w3']['n'], table['mlp_w3']['dtype'], table['mlp_w3']['silu']), (4352, 'bfp4', False))
        self.assertEqual((table['mlp_w2']['k'], table['mlp_w2']['n'], table['mlp_w2']['dtype']), (4352, 5120, 'bfp8'))
        self.assertEqual((table['attn_qkv']['n'], table['attn_wo']['k']), (3584, 1536))
        self.assertEqual((table['gdn_qkvzab']['n'], table['gdn_out']['k']), (found.gdn_qkvzab_padded, found.gdn_value))
        self.assertEqual((table['gdn_qkvzab']['n'], table['gdn_out']['k']), (4128, 1536))
        self.assertEqual((table['lm_head']['k'], table['lm_head']['n'], table['lm_head']['variants']), (5120, 62080, {}))
        self.assertEqual({entry['compute'] for entry in plan.seven() if entry['name'].startswith('mlp')}, {'lofi'})
        self.assertEqual({entry['compute'] for entry in plan.seven() if not entry['name'].startswith('mlp')}, {'hifi2'})

    def test_the_builder_arguments_are_the_grafts(self):
        text = read(GRAFT_MODEL_CONFIG)
        names = {'attn_qkv': 'attn_qkv_decode_1d_progcfg_64', 'attn_wo': 'attn_wo_decode_1d_progcfg_64', 'gdn_qkvzab': 'gdn_qkvz_decode_1d_progcfg_64',
                 'gdn_out': 'gdn_out_decode_1d_progcfg_64', 'mlp_w1': 'mlp_w1_decode_1d_progcfg_64', 'mlp_w3': 'mlp_w3_decode_1d_progcfg_64',
                 'mlp_w2': 'mlp_w2_decode_1d_progcfg_64'}
        table = {entry['name']: entry for entry in plan.projections()}
        for name, attribute in names.items():
            with self.subTest(name=name):
                start = text.index('self.%s = tpc.create_matmul_1d_decode_progcfg(' % attribute)
                statement = text[start:text.index(chr(10) + '        )', start)]
                cores = int(re.search(r'num_cores=(\d+)', statement).group(1))
                grid = 11 if 'grid_w=self.decode_grid_w' in statement else 8     # the builder's own default grid_w is 8
                self.assertEqual(table[name]['variants']['base'], dict(num_cores=cores, grid_w=grid))
        # verify-trace T1 #11 widens the attn_qkv and the gate; nothing else moves
        t1 = text[text.index('def _qwen_verify_t1_configs'):text.index('class Qwen36ModelArgs')]
        self.assertIn('num_cores=44, grid_w=args.decode_grid_w', t1)
        self.assertIn('num_cores=88', t1)
        self.assertEqual(table['attn_qkv']['variants']['served'], dict(num_cores=44, grid_w=11))
        self.assertEqual(table['mlp_w1']['variants']['served'], dict(num_cores=88, grid_w=11))
        for name in ('attn_wo', 'gdn_qkvzab', 'gdn_out', 'mlp_w3', 'mlp_w2'):
            self.assertEqual(table[name]['variants']['served'], table[name]['variants']['base'], name)

    def test_the_mirror_builder_equals_the_graft_transcription_at_every_row_count(self):
        for entry in plan.seven():
            for variant, args in entry['variants'].items():
                for m in plan.TIMED_ROWS:
                    with self.subTest(name=entry['name'], variant=variant, m=m):
                        mine = plan.config_at(entry, variant, m)
                        theirs = graft.create_matmul_1d_decode_progcfg(m, entry['k'], entry['n'], args['num_cores'], grid_w=args['grid_w'])
                        self.assertEqual((mine['in0_block_w'], mine['per_core_M'], mine['per_core_N'], mine['out_subblock_h'], mine['out_subblock_w']),
                                         (theirs.in0_block_w, theirs.per_core_M, theirs.per_core_N, theirs.out_subblock_h, theirs.out_subblock_w))
                        self.assertEqual(tuple(mine['grid']), tuple(theirs.compute_with_storage_grid_size))


class KOrderTests(unittest.TestCase):
    def test_every_128_row_config_keeps_the_k_order_of_the_64_and_32_row_ones(self):
        for entry in plan.seven():
            for variant in entry['variants']:
                with self.subTest(name=entry['name'], variant=variant):
                    found = plan.plan_for(entry, variant)
                    self.assertIsNone(found['k_order_problem'])
                    self.assertIsNone(found['k_order_problem_tile'])
                    self.assertEqual(found['configs'][128]['per_core_M'], 4)
                    self.assertEqual(found['configs'][64]['per_core_M'], 2)
                    self.assertEqual(found['configs'][32]['per_core_M'], 1)

    def test_only_the_per_core_n_five_projections_change_their_output_subblock(self):
        """Review section 3: down, gdn_out and attn_out go from 2 x 1 to 4 x 1; up stays 1 x 4, the gate 2 x 2, gdn_in 1 x 3."""
        expected = {'mlp_w2': ((2, 1), (4, 1)), 'gdn_out': ((2, 1), (4, 1)), 'attn_wo': ((2, 1), (4, 1)), 'mlp_w3': ((1, 4), (1, 4)),
                    'mlp_w1': ((2, 2), (2, 2)), 'gdn_qkvzab': ((1, 3), (1, 3))}
        table = {entry['name']: entry for entry in plan.seven()}
        for name, (half, whole) in expected.items():
            with self.subTest(name=name):
                moved = plan.plan_for(table[name], 'served')['subblocks']
                self.assertEqual((moved['first'], moved['second']), (half, whole))

    def test_a_changed_in0_block_w_or_fuse_flag_is_refused(self):
        good = plan.config_at(plan.named('mlp_w3'), 'served', 128)
        other = dict(good, in0_block_w=good['in0_block_w'] * 2)
        self.assertIn('in0_block_w', plan.k_order_problem(good, other))
        self.assertIn('fuse_batch', plan.k_order_problem(good, dict(good, fuse_batch=False)))
        self.assertIn('mcast_in0', plan.k_order_problem(good, dict(good, mcast_in0=False)))
        self.assertIsNone(plan.k_order_problem(good, dict(good, per_core_N=good['per_core_N'] + 1, out_subblock_h=1)))

    def test_the_128_row_configs_fit_l1_and_use_the_cores_the_design_expects(self):
        table = {entry['name']: entry for entry in plan.seven()}
        for name, entry in table.items():
            found = plan.plan_for(entry, 'served')
            self.assertLessEqual(found['l1_bytes_128'], 1200000, name)     # the sweep's own L1 budget
        self.assertEqual(plan.plan_for(table['mlp_w3'], 'served')['active_cores_128'], 34)
        self.assertEqual(plan.plan_for(table['mlp_w1'], 'served')['active_cores_128'], 68)

    def test_regrid_candidates_keep_the_k_order_and_the_model_partition_is_among_them(self):
        for name in ('mlp_w1', 'mlp_w3', 'mlp_w2'):
            entry = plan.named(name)
            candidates = plan.regrid_candidates(entry)
            self.assertGreater(len(candidates), 4, name)
            served = plan.config_at(entry, 'served', 128)
            for config in candidates:
                self.assertEqual(config['in0_block_w'], served['in0_block_w'])
                self.assertEqual(config['per_core_M'], 4)
                self.assertIsNone(plan.k_order_problem(served, config))
            self.assertIn(served['per_core_N'], {config['per_core_N'] for config in candidates}, name)
            arguments = plan.builder_arguments(candidates[0], entry)
            self.assertTrue(arguments['reproduces'] in (True, False))

    def test_the_timing_rules_are_the_reviews(self):
        # t32 25.1, t64 28.0, t128 extrapolates to 28.0 + 1.5 * 2.9 = 32.35: the review's rule passes where the design's 1.1 x t64 = 30.8 would not
        verdict = plan.timing_verdict(25.09, 28.02, 32.0)
        self.assertTrue(verdict['ok'])
        self.assertAlmostEqual(verdict['extrapolated'], 28.02 + 1.5 * 2.93, places=2)
        self.assertFalse(plan.timing_verdict(25.0, 28.0, 40.0)['ok'])
        self.assertTrue(plan.timing_verdict(40.0, 40.0, 44.0)['ok'])     # 1.1 x t64
        self.assertIsNone(plan.timing_verdict(None, 1.0, 2.0)['ok'])
        self.assertTrue(plan.total_timing_verdict({'a': (1.0, 10.0, 11.0), 'b': (1.0, 10.0, 11.0)}))
        self.assertFalse(plan.total_timing_verdict({'a': (1.0, 10.0, 12.0)}))
        self.assertIsNone(plan.total_timing_verdict({'a': (None, 1.0, 1.0)}))


class FakeBackend(object):
    """A backend whose 'matmul' is a pure function of the input row (so every split is exact), unless `break_tile` is set: then the 128-row call
    perturbs that 32-row tile of its output. Outputs are lists of per-row tuples."""

    def __init__(self, break_tile=None, break_config=None, auto_breaks=False):
        self.break_tile, self.break_config, self.auto_breaks = break_tile, break_config, auto_breaks
        self.freed, self.linear_calls, self.timed = [], [], []

    def weight(self, entry):
        return ('weight', entry['name'])

    def activation(self, entry, first, last):
        return ('x', entry['name'], first, last)

    def linear(self, entry, x, weight, config):
        rows = x[3] - x[2]
        self.linear_calls.append((entry['name'], rows, None if config is None else config['per_core_M'], None if config is None else config['per_core_N']))
        out = []
        for index in range(rows):
            value = (entry['name'], x[2] + index)
            if rows == plan.BLOCK and (x[2] + index) // 32 == self.break_tile:
                value += ('broken',)
            if config is None and self.auto_breaks and rows == plan.BLOCK and (x[2] + index) // 32 == 3:
                value += ('auto',)
            if config is not None and self.break_config is not None and config['per_core_N'] == self.break_config and rows == plan.BLOCK:
                value += ('grid',)
            out.append(value)
        return out

    def read(self, out):
        return out

    def mismatches(self, whole, first, last, other, other_first, label):
        return sum(1 for a, b in zip(whole[first:last], other[other_first:other_first + (last - first)]) if a != b)

    def time(self, entry, x, weight, config, calls, rounds):
        rows = x[3] - x[2]
        value = {32: 25.0, 64: 28.0, 128: 30.0}[rows]
        self.timed.append((entry['name'], rows))
        return value

    def free(self, handle):
        self.freed.append(handle)


class OrchestrationTests(unittest.TestCase):
    def test_an_exact_backend_is_exact_against_both_oracles_tile_by_tile(self):
        backend = FakeBackend()
        result = bytes_run.run_projection(backend, plan.named('mlp_w2'), 'served', 3, 2)
        self.assertEqual(result['tiles_vs_halves'], [0, 0, 0, 0])
        self.assertEqual(result['tiles_vs_tiles'], [0, 0, 0, 0])
        self.assertTrue(result['exact_halves'] and result['exact_tiles'])
        self.assertEqual((result['t32_us'], result['t64_us'], result['t128_us']), (25.0, 28.0, 30.0))
        self.assertTrue(result['timing']['ok'])
        # one 128-row call, two 64-row calls with the _64 config, four 32-row calls with the M = 32 config, in that order
        calls = [call for call in backend.linear_calls]
        self.assertEqual([(rows, m) for _name, rows, m, _n in calls], [(128, 4)] + [(64, 2)] * 2 + [(32, 1)] * 4)
        self.assertEqual({call[0] for call in calls}, {'mlp_w2'})
        self.assertEqual(result['k_order_problem'], None)

    def test_a_broken_tile_is_named_against_the_half_and_the_tile_that_cover_it(self):
        backend = FakeBackend(break_tile=2)
        result = bytes_run.run_projection(backend, plan.named('mlp_w1'), 'base', 3, 2, timing=False)
        self.assertEqual(result['tiles_vs_halves'], [0, 0, 32, 0])
        self.assertEqual(result['tiles_vs_tiles'], [0, 0, 32, 0])
        self.assertFalse(result['exact_halves'] or result['exact_tiles'])
        self.assertNotIn('t128_us', result)
        self.assertEqual(backend.timed, [])

    def test_the_lm_head_runs_with_no_program_config_at_every_row_count(self):
        backend = FakeBackend(auto_breaks=True)
        result = bytes_run.run_projection(backend, plan.named('lm_head'), None, 3, 2)
        self.assertEqual(result['variant'], 'auto')
        self.assertEqual([call[2:] for call in backend.linear_calls], [(None, None)] * 7)
        self.assertEqual(result['tiles_vs_halves'], [0, 0, 0, 32])
        self.assertFalse(result['exact_halves'])

    def test_every_device_handle_it_made_is_freed_even_when_a_call_raises(self):
        class Raising(FakeBackend):
            def linear(self, entry, x, weight, config):
                if x[3] - x[2] == 64:
                    raise RuntimeError('program refused')
                return FakeBackend.linear(self, entry, x, weight, config)

        backend = Raising()
        with self.assertRaisesRegex(RuntimeError, 'program refused'):
            bytes_run.run_projection(backend, plan.named('mlp_w3'), 'served', 3, 2)
        self.assertIn(('weight', 'mlp_w3'), backend.freed)
        self.assertIn(('x', 'mlp_w3', 0, 128), backend.freed)

    def test_regrid_names_the_best_exact_candidate_and_marks_a_candidate_that_differs(self):
        entry = plan.named('mlp_w3')
        candidates = plan.regrid_candidates(entry)
        slowest_pcn = candidates[0]['per_core_N']

        class Timed(FakeBackend):
            def time(self, entry_, x, weight, config, calls, rounds):
                return 10.0 + config['per_core_N']

        backend = Timed(break_config=min(config['per_core_N'] for config in candidates))
        rows, best = bytes_run.run_regrid(backend, entry, 3, 2, 99)
        self.assertEqual(len(rows), len(candidates))
        broken = [row for row in rows if row['exact'] is False]
        self.assertTrue(broken and all(row['config']['per_core_N'] == backend.break_config for row in broken))
        self.assertTrue(best['exact'])
        self.assertNotEqual(best['config']['per_core_N'], backend.break_config)
        self.assertEqual(best['us'], min(row['us'] for row in rows if row['exact']))
        self.assertIn('num_cores', best['arguments'])
        self.assertIsNotNone(slowest_pcn)

    def test_a_refused_regrid_config_is_a_row_not_the_end_of_the_sweep(self):
        class Refusing(FakeBackend):
            def linear(self, entry, x, weight, config):
                if config is not None and x[3] - x[2] == plan.BLOCK and config['per_core_N'] == 2:
                    raise RuntimeError('TT_FATAL: config refused')
                return FakeBackend.linear(self, entry, x, weight, config)

        rows, best = bytes_run.run_regrid(Refusing(), plan.named('mlp_w1'), 3, 2, 99)
        errors = [row for row in rows if 'error' in row]
        self.assertTrue(errors and all('TT_FATAL' in row['error'] for row in errors))
        self.assertTrue(len(errors) < len(rows) and best is not None)

    def test_the_report_passes_only_when_every_result_is_exact_against_both_oracles(self):
        results = [bytes_run.run_projection(FakeBackend(), plan.named(name), 'served', 3, 2) for name in ('mlp_w1', 'mlp_w2')]
        report = dict(complete=True, errors={}, results=results, regrid={})
        bytes_run.summarize(report)
        self.assertTrue(report['passed'])
        self.assertTrue(report['timing_total_ok'])
        self.assertEqual((report['sum_t64_us'], report['sum_t128_us']), (56.0, 60.0))
        lines = bytes_run.verdict_lines(report)
        self.assertEqual(lines[-1], 'M8_MATMUL verdict=PASS')
        self.assertTrue(lines[0].startswith('M8_MATMUL mlp_w1/served exact_halves=True exact_tiles=True'))
        broken = bytes_run.run_projection(FakeBackend(break_tile=1), plan.named('mlp_w3'), 'served', 3, 2)
        failing = dict(report, results=results + [broken])
        bytes_run.summarize(failing)
        self.assertFalse(failing['passed'])
        self.assertEqual(bytes_run.verdict_lines(failing)[-1], 'M8_MATMUL verdict=FAIL')
        self.assertIn('mismatches_vs_halves=[0, 32, 0, 0]', '\n'.join(bytes_run.verdict_lines(failing)))
        for key, value in (('complete', False), ('errors', {'x': 'boom'}), ('fatal_error', 'device')):
            self.assertFalse(bytes_run.passed(dict(report, **{key: value})), key)
        self.assertFalse(bytes_run.passed(dict(complete=True, errors={}, results=[])))

    def test_the_regrid_lines_name_the_winner_with_its_builder_arguments(self):
        entry = plan.named('mlp_w3')
        rows, best = bytes_run.run_regrid(FakeBackend(), entry, 3, 2, 99)
        report = dict(complete=True, errors={}, results=[], regrid={'mlp_w3': dict(rows=rows, best_exact=best)})
        text = '\n'.join(bytes_run.verdict_lines(report))
        self.assertIn('M8_REGRID mlp_w3 candidates=%d exact=%d best_exact=grid_' % (len(rows), len(rows)), text)
        self.assertIn('"num_cores"', text)

    def test_the_arguments(self):
        args = bytes_run.parse_arguments(['--out', 'x.json'])
        self.assertEqual(args.projection_list, [entry['name'] for entry in plan.projections()])
        self.assertEqual((args.variant_list, args.regrid_list), (['served', 'base'], ['mlp_w1', 'mlp_w3', 'mlp_w2']))
        args = bytes_run.parse_arguments(['--out', 'x.json', '--projections', 'lm_head', '--regrid', '', '--no-timing'])
        self.assertEqual((args.projection_list, args.regrid_list, args.no_timing), (['lm_head'], [], True))
        for bad in (['--projections', 'nope'], ['--variants', 'wide'], ['--regrid', 'attn_nope']):
            with self.assertRaises(SystemExit):
                bytes_run.parse_arguments(['--out', 'x.json'] + bad)

    def test_the_device_layer_makes_the_models_calls(self):
        """The one place the backend is read, not run: the LM head has no program or compute config, the 1D configs carry the model's, the oracle is an int16 view."""
        text = read(os.path.join(HERE, 'm8_matmul_bytes.py'))
        self.assertIn('ttnn.linear(x, weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)', text)
        self.assertIn('compute_kernel_config=self.compute_config(entry[\'compute\'])', text)
        self.assertIn('MatmulMultiCoreReuseMultiCast1DProgramConfig', text)
        self.assertIn('fuse_batch=True', text)
        self.assertIn('mcast_in0=True', text)
        self.assertIn('view(torch.int16)', text)
        self.assertIn('math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True', text)
        self.assertNotIn(chr(13), text)


class HarnessTests(unittest.TestCase):
    def test_it_embeds_the_canonical_qual_card_block_byte_for_byte(self):
        canonical = read(os.path.join(HERE, 'qual_card.sh'))
        harness = read(HARNESS)
        start, end = harness.index('# >>> qual_card.sh'), harness.index('# <<< qual_card.sh')
        self.assertIn(canonical.strip(chr(10)), harness[start:end + len('# <<< qual_card.sh')])
        self.assertIn('qual_card_select', harness)
        self.assertIn('qual_refuse_holders', harness)

    def test_the_launch_turns_the_serving_hook_off_and_unsets_the_four_card_descriptor(self):
        harness = read(HARNESS)
        self.assertIn('-e QWEN_C2_SERVING=0 --entrypoint env "$IMAGE" -u TT_MESH_GRAPH_DESC_PATH python3 -B /bench/m8_matmul_bytes.py', harness)
        self.assertIn('--network none', harness)
        self.assertIn('--device "$node"', harness)
        self.assertIn('--cap-drop ALL', harness)
        self.assertNotIn('--privileged', own_code(harness))

    def test_it_mounts_exactly_its_five_scripts_from_this_checkout_read_only(self):
        harness = read(HARNESS)
        loop = re.search(r'for file in (.*?); do', harness).group(1).split()
        self.assertEqual(loop, ['m8_matmul_bytes.py', 'm8_matmul_plan.py', 'matmul_tp4_sweep.py', 'matmul64_sweep.py', 'tp_shapes.py'])
        self.assertIn('dst=/bench/$file,readonly', harness)
        for name in loop:
            self.assertTrue(os.path.isfile(os.path.join(HERE, name)), name)
        # every module the harness's script imports at load is among them or the standard library
        for module in ('m8_matmul_plan', 'matmul_tp4_sweep', 'matmul64_sweep', 'tp_shapes'):
            self.assertIn(module + '.py', loop)

    def test_the_model_mount_is_optional_and_the_run_is_bounded(self):
        harness = read(HARNESS)
        self.assertIn('MODEL_ARGS=()', harness)
        self.assertIn('WARN: $MODEL_DIR is not a directory', harness)
        self.assertIn('${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}', harness)
        self.assertNotIn('is not a directory" >&2; exit 1', harness)
        self.assertIn('timeout -k 30 "$timeout_s" docker run --rm', harness)
        self.assertIn('qual_reset_hint >&2', harness)
        self.assertIn('m8matmul-$stamp.json', harness)

    def test_it_never_resets_stops_or_serves_anything_itself(self):
        harness = read(HARNESS)
        own = harness[harness.index('# <<< qual_card.sh'):]      # the canonical block only PRINTS a reset hint; what follows is this harness's own code
        executable = chr(10).join(line for line in own.split(chr(10)) if not line.lstrip().startswith('#'))
        for word in ('tt-smi', 'agentstop', 'agentstart', 'systemctl', 'docker stop', 'kill '):
            self.assertNotIn(word, executable)

    def test_it_is_valid_shell_lf_and_executable_in_git(self):
        self.assertEqual(subprocess.run(['bash', '-n', HARNESS], capture_output=True).returncode, 0)
        with open(HARNESS, 'rb') as handle:
            self.assertNotIn(b'\r', handle.read())
        listing = subprocess.run(['git', 'ls-files', '-s', '--', 'optimisation/ttnn-op/m8_matmul/run_card_m.sh'], cwd=ROOT, capture_output=True)
        if listing.stdout:
            self.assertTrue(listing.stdout.startswith(b'100755'), listing.stdout)

    def test_the_job_parser_accepts_it_as_a_card_m_harness(self):
        parsed = job.read_job(job.parse_env(read(os.path.join(FOLDER, 'M1-card-m-matmul-m128.env'))), profile_names(), root=ROOT)
        self.assertEqual(parsed['cardm_harness'], 'optimisation/ttnn-op/m8_matmul/run_card_m.sh')


def own_code(harness):
    """The harness's own text: everything but the canonical qual_card block (which mentions --privileged only to say what it refuses)."""
    return harness[:harness.index('# >>> qual_card.sh')] + harness[harness.index('# <<< qual_card.sh'):]


def profile_names():
    with open(PROFILES_PATH, encoding='utf-8') as handle:
        return sorted(json.load(handle)['profiles'])


def order_lines():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        return [line.split() for line in handle.read().splitlines() if line.strip() and not line.startswith('#')]


def parsed(name):
    return job.read_job(job.parse_env(read(os.path.join(FOLDER, name + '.env'))), profile_names(), root=ROOT)


class JobPackTests(unittest.TestCase):
    def test_order_is_one_card_m_job_then_the_quad_reset(self):
        lines = order_lines()
        self.assertEqual([line[0] for line in lines], ['M1-card-m-matmul-m128', 'Z-status-rescan-reset'])
        self.assertEqual([(line[1], line[2]) for line in lines], [('stop', IMAGE), ('soft', IMAGE)])
        self.assertTrue(all(line[3].isdigit() and int(line[3]) > 0 for line in lines))
        files = sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env'))
        self.assertEqual(files, sorted(line[0] for line in lines))

    def test_the_card_m_job_runs_this_harness_on_the_one_image_with_no_arguments_of_its_own(self):
        result = parsed('M1-card-m-matmul-m128')
        self.assertEqual((result['actions'], result['cards'], result['tag']), ('cardm', 'pair', IMAGE))
        self.assertEqual(result['cardm_harness'], 'optimisation/ttnn-op/m8_matmul/run_card_m.sh')
        self.assertEqual(list(result['cardm_args']), [])
        self.assertEqual(str(result['cardm_env']).split(), ['IMAGE_TAG=' + IMAGE])

    def test_the_first_and_only_quad_job_rescans_before_it_resets(self):
        result = parsed('Z-status-rescan-reset')
        actions = result['actions'].split()
        self.assertEqual((actions, result['cards'], result['tag']), (['status', 'rescan', 'reset'], 'quad', IMAGE))
        self.assertLess(actions.index('rescan'), actions.index('reset'))

    def test_no_job_touches_the_node_agent_or_production(self):
        for name in ('M1-card-m-matmul-m128', 'Z-status-rescan-reset'):
            result = parsed(name)
            self.assertFalse({'agentstop', 'agentstart', 'unserve', 'platform', 'replay', 'priority', 'build', 'smoke', 'gate'} & set(result['actions'].split()), name)
            self.assertEqual(result['bake_default_profile'] or '', '')
        self.assertNotIn(IMAGE, job.PROTECTED)
        self.assertFalse(IMAGE.startswith(job.PROTECTED_PREFIXES))
        with open(PROFILES_PATH, encoding='utf-8') as handle:
            self.assertEqual(json.load(handle)['default'], 'c2-packed-tp4')

    def test_the_templates_are_public_safe_and_lf(self):
        for name in os.listdir(FOLDER):
            path = os.path.join(FOLDER, name)
            with open(path, 'rb') as handle:
                data = handle.read()
            self.assertNotIn(b'\r', data, name)
            self.assertIsNone(BANNED.search(data.decode('utf-8')), name)
        harness = read(HARNESS)
        # the canonical qual_card block names the pair's boards (test_qual_card makes every harness embed it); this harness's own text must not
        own = harness[:harness.index('# >>> qual_card.sh')] + harness[harness.index('# <<< qual_card.sh'):]
        own = own.replace(HF_SNAPSHOT, '<hf-snapshot>')      # the public Hugging Face revision of the model, as every harness names it
        self.assertIsNone(BANNED.search(own), 'the harness names a rig value outside the canonical block')
        for path in (os.path.join(HERE, 'm8_matmul_plan.py'), os.path.join(HERE, 'm8_matmul_bytes.py'), os.path.join(HERE, 'tp4_m8.py'),
                     os.path.join(HERE, 'm8_limits.py')):
            self.assertIsNone(BANNED.search(read(path)), path)


if __name__ == '__main__':
    unittest.main()
