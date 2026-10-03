"""optimisation/ttnn-op/sdpa_tp4_long on the CPU: argument parsing, the pure helpers held to hand-computed values, the verdict logic,
the run_card_m.sh harness text, and the whole flow on a fake device whose attention honours cur_pos, the mask and the page table per
row (so a mis-folded layout, a dropped mask or a call that ignored cur_pos is caught). What only the card can show (that each
configuration is exact and how fast it is) is what the sweep is for.

Run at py 3.11: `py -3.11 -B -m unittest test_sdpa_tp4_long` from scripts/ci.
"""

import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OPS = ROOT / 'optimisation' / 'ttnn-op'
for _path in (OPS / 'sdpa_tp4_long', OPS / 'k64j', OPS / 'sdpa_decode_qwen', HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import sdpa_long_tp  # noqa: E402
import sdpa_tp4_long as sweep  # noqa: E402


def args_for(*words):
    return sweep.parse_args(['--out', 'report.json'] + list(words))


class ParseTests(unittest.TestCase):
    def test_defaults(self):
        args = args_for()
        self.assertEqual(args.extents, [33024, 65792, 131328, 262400])
        self.assertEqual(args.users, 4)
        self.assertEqual(args.starts, [240])
        self.assertEqual(args.arms, list(sweep.ARMS))
        self.assertEqual((args.timing, args.rounds, args.iterations, args.calls), ('trace', 5, 10, 8))
        self.assertEqual(args.watchdog, 300.0)

    def test_the_served_arm_is_always_present_and_first(self):
        self.assertEqual(sweep.parse_arms('multi'), ['served', 'multi'])
        self.assertEqual(sweep.parse_arms('grid8x4,served'), ['grid8x4', 'served'])
        self.assertEqual(sweep.parse_arms('all'), list(sweep.ARMS))

    def test_bad_values_are_refused(self):
        for words in (['--extents', '1000'], ['--extents', '0'], ['--starts', '241'], ['--starts', '-1'], ['--arms', 'nope'],
                      ['--users', '0'], ['--users', '7'], ['--rounds', '0'], ['--calls', '0'], ['--iterations', '0'],
                      ['--timing', 'sometimes']):
            with self.subTest(words=words), self.assertRaises(SystemExit if words[0] in ('--users', '--rounds', '--calls', '--iterations', '--timing') else ValueError):
                args_for(*words)

    def test_the_page_table_width_is_the_served_pools_in_every_case(self):
        self.assertEqual(args_for().page_width, (262400 + 256) // 64, 'the longest family and its poisoned chunk: 4104 pages, above the 4100 floor')
        self.assertEqual(args_for('--extents', '256', '--no-mixed').page_width, 4100)
        self.assertEqual(args_for('--extents', '256').page_width, 4104, 'the mixed case holds a 262,400 family')
        self.assertEqual(sweep.page_width(0, 262400), 4104)
        self.assertEqual(sweep.page_width(0, 131328), 4100, 'the pool width is the floor')
        self.assertEqual(sweep.page_width(0, 300000), (300000 + 256) // 64, 'a case longer than the pool widens it')
        self.assertEqual(args_for('--page-width', '2052', '--extents', '33024', '--no-mixed').page_width, 2052)
        with self.assertRaises(ValueError):
            args_for('--page-width', '100', '--extents', '33024')

    def test_the_grid_must_be_the_serving_meshs(self):
        self.assertIsNone(sweep.grid_problem((11, 10)))
        self.assertIsNone(sweep.grid_problem([11, 10]))
        self.assertIn('[8, 10]', sweep.grid_problem((8, 10)))

    def test_phases_put_the_risky_arms_after_every_safe_case(self):
        self.assertEqual(sweep.phases(sweep.parse_arms('all')),
                         [('safe', ['served', 'grid8x4', 'grid8x10', 'grid11x4', 'grid4x8', 'multi']), ('risky', ['served', 'rowsplit', 'ra'])])
        self.assertEqual(sweep.phases(['served', 'multi']), [('safe', ['served', 'multi'])])
        self.assertEqual(sweep.phases(['served', 'ra']), [('safe', ['served']), ('risky', ['served', 'ra'])])

    def test_the_overrides_parse(self):
        args = args_for('--extents', '512,1024', '--arms', 'multi,rowsplit', '--users', '2', '--timing', 'eager', '--no-mixed',
                        '--seeds', '0,3', '--deadline-s', '5', '--expect-binary-sha256', 'ab')
        self.assertEqual((args.extents, args.arms, args.users, args.timing, args.no_mixed, args.seeds),
                         ([512, 1024], ['served', 'multi', 'rowsplit'], 2, 'eager', True, [0, 3]))
        self.assertEqual((args.deadline_s, args.expect_binary_sha256), (5.0, 'ab'))


class TableTests(unittest.TestCase):
    def test_the_arms_are_the_flags_configurations(self):
        self.assertEqual(sorted(sweep.ARMS), sorted(sdpa_long_tp.names()))

    def test_the_served_arm_is_g8b2_0x23_and_every_other_flag_set_is_named(self):
        self.assertEqual(sweep.ARMS['served'], dict(rows=8, entries=2, flags=0x23, one_launch=False, risky=False))
        self.assertEqual(sweep.ARMS['multi']['flags'], 0x21)
        self.assertEqual(sweep.ARMS['ra']['flags'], 0x2B)
        for name in ('grid8x4', 'grid8x10', 'grid11x4', 'grid4x8'):
            self.assertEqual({k: v for k, v in sweep.ARMS[name].items()}, sweep.ARMS['served'])

    def test_expected_programs(self):
        st = sweep.SERVED_POOL_PAGES * 64 // 32
        self.assertEqual(st, 8200)
        self.assertEqual(sweep.expected_programs('served', 4), {(0x23, 2, 2, st)})
        self.assertEqual(sweep.expected_programs('multi', 4), {(0x21, 4, 3, st)})
        self.assertEqual(sweep.expected_programs('rowsplit', 4), {(0x23, 4, 1, st)})
        self.assertEqual(sweep.expected_programs('ra', 1), {(0x2B, 2, 2, st)})
        self.assertEqual(sweep.expected_programs('served', 1, pages=2052), {(0x23, 2, 2, 4104)})
        for arm in ('grid8x4', 'grid8x10', 'grid11x4', 'grid4x8'):
            self.assertEqual(sweep.expected_programs(arm, 4), sweep.expected_programs('served', 4), 'one key, so the line COUNT is the check')
        self.assertEqual(sweep.launches_per_step('served', 4), 4)
        self.assertEqual(sweep.launches_per_step('multi', 4), 1)


class HelperTests(unittest.TestCase):
    def test_busiest_chunks_match_the_design_table(self):
        self.assertEqual([sweep.busiest_chunks(extent) for extent in (32768, 65536, 131328, 262400)], [8, 16, 33, 65])

    def test_bytes_and_bandwidth(self):
        self.assertAlmostEqual(sweep.kv_bytes(262400) / 1e6, 142.7, places=0)
        self.assertAlmostEqual(sweep.gbps([262400], 1e-3), 142.7, delta=0.2)
        self.assertEqual(sweep.gbps([256], 0.0), 0.0)
        self.assertAlmostEqual(sweep.gbps([262400] * 4, 4e-3), sweep.gbps([262400], 1e-3))

    def test_case_list(self):
        names = [case['name'] for case in sweep.case_list([512, 1024], 4, True)]
        self.assertEqual(names, ['u1@512', 'u1@1024', 'u4@512', 'u4@1024', 'mixed', 'skewed'])
        self.assertEqual([case['name'] for case in sweep.case_list([512], 1, True)], ['u1@512'])
        self.assertEqual([case['name'] for case in sweep.case_list([512], 2, True)], ['u1@512', 'u2@512'])
        mixed = sweep.case_list([512], 4, True)[-2]
        self.assertEqual(mixed['extents'], [262400, 131328, 65792, 33024])
        self.assertEqual(sweep.case_list([512], 4, True)[-1]['extents'], [262400, 4352, 4352, 4352])

    def test_positions_and_words(self):
        self.assertEqual(sweep.extent_positions(2304, 240, 2, 8), [2048 + 240, 2048 + 248][:1] + [2048 + 248])
        self.assertEqual(sweep.extent_positions(2304, 240, 1, 16), [2288])
        self.assertEqual(sweep.words_for('served', [2288, 2296], [2304], 1), [2303, 2303])
        self.assertEqual(sweep.words_for('multi', None, [2304, 512], 2), [2303, 511])
        self.assertEqual(sweep.words_for('rowsplit', [2288] * 4, [2304], 1), [2303] * 4)

    def test_every_start_up_to_240_keeps_a_block_inside_its_family(self):
        for start in (0, 128, 240):
            for arm, spec in sweep.ARMS.items():
                last = sweep.extent_positions(512, start, spec['entries'], spec['rows'])[-1] + spec['rows'] - 1
                self.assertLess(last, 512, arm)

    def test_fit_line(self):
        a, b = sweep.fit_line([(2, 79.2), (3, 92.5), (5, 121.7), (7, 151.4)])
        self.assertAlmostEqual(b, 14.5, delta=0.3)
        self.assertAlmostEqual(a, 49.6, delta=1.5)
        self.assertIsNone(sweep.fit_line([(2, 1.0), (2, 2.0)]))
        self.assertIsNone(sweep.fit_line([]))

    def test_paired_ratios_and_the_timing_summary(self):
        rounds = [dict(served=2.0, multi=1.0), dict(served=2.0, multi=1.5), dict(served=4.0)]
        self.assertEqual(sweep.paired_ratios(rounds, 'multi'), [0.5, 0.75])
        summary = sweep.summarize_timing(rounds, 'multi', [262400])
        self.assertEqual((summary['rounds'], summary['ratio_min'], summary['ratio_max'], summary['ratio_median']), (2, 0.5, 0.75, 0.625))
        self.assertAlmostEqual(summary['median_us'], 1.25e6)
        self.assertIsNone(sweep.summarize_timing(rounds, 'rowsplit', [256]))

    def test_digest_is_stable(self):
        self.assertEqual(sweep.digest(b'abc'), sweep.digest(b'abc'))
        self.assertNotEqual(sweep.digest(b'abc'), sweep.digest(b'abd'))
        self.assertEqual(len(sweep.digest(b'')), 16)


def timing(ratio, gbps=100.0, ratio_max=None):
    return dict(ratio_median=ratio, ratio_min=ratio, ratio_max=ratio if ratio_max is None else ratio_max, gbps_median=gbps, median_us=1.0)


def arm_state(differing=0, ratio=None, status='ok', ratio_max=None, **extra):
    state = dict(status=status, differing_rows=differing, finite=True, **extra)
    if ratio is not None:
        state['timing'] = timing(ratio, ratio_max=ratio_max)
    return state


def case(name, kind, arms, moved=True, extents=(1024,)):
    return dict(name=name, kind=kind, extents=list(extents), arms=arms, liveness=dict(moved=moved))


class DecisionTests(unittest.TestCase):
    def report(self, *cases, **extra):
        return dict(cases=list(cases), failures=[], timing_mode='trace', extents=[1024], **extra)

    def test_a_clean_run_passes_and_the_winner_is_the_fastest_exact_arm(self):
        report = self.report(
            case('u4@1024', 'u4', dict(served=arm_state(ratio=1.0), multi=arm_state(ratio=0.4), grid8x4=arm_state(ratio=0.97))),
            case('u4@2048', 'u4', dict(served=arm_state(ratio=1.0), multi=arm_state(ratio=0.5), grid8x4=arm_state(ratio=0.99))))
        report['decision'] = sweep.decide(report)
        self.assertEqual(report['decision']['verdict'], 'PASS')
        best = sweep.winners(report)
        self.assertEqual(best['u4']['arm'], 'multi')
        self.assertAlmostEqual(best['u4']['ratio_mean'], 0.45)
        line = sweep.verdict_line(report)
        self.assertTrue(line.startswith('SDPA_TP4_LONG verdict=PASS cases=2 arms_ran=grid8x4,multi,served arms_exact=grid8x4,multi,served'), line)
        self.assertIn('winner_u4=multi ratio=0.450', line)

    def test_a_not_exact_arm_is_never_a_winner_and_does_not_fail_the_run(self):
        report = self.report(case('u4@1024', 'u4', dict(served=arm_state(ratio=1.0), multi=arm_state(differing=3, ratio=0.3),
                                                        grid8x4=arm_state(ratio=0.97))))
        report['decision'] = sweep.decide(report)
        self.assertEqual(report['decision']['verdict'], 'PASS')
        self.assertEqual(sweep.winners(report)['u4']['arm'], 'grid8x4')
        self.assertNotIn('multi', sweep.verdict_line(report).split('arms_exact=')[1].split(' ')[0])

    def test_an_arm_that_failed_in_one_case_of_a_kind_is_not_a_winner_of_that_kind(self):
        report = self.report(
            case('a', 'u4', dict(served=arm_state(ratio=1.0), multi=arm_state(ratio=0.4))),
            case('b', 'u4', dict(served=arm_state(ratio=1.0), multi=dict(status='error', error='boom'))))
        self.assertNotIn('u4', sweep.winners(report))

    def test_a_win_inside_the_noise_is_not_a_winner(self):
        # the grid11x4 control sits at about 1.0: 0.995 on the median is noise, and so is a mean win that loses one paired round
        report = self.report(case('u4@1024', 'u4', dict(
            served=arm_state(ratio=1.0), grid11x4=arm_state(ratio=0.995), grid8x4=arm_state(ratio=0.9, ratio_max=1.01),
            grid8x10=arm_state(ratio=0.981), grid4x8=arm_state(ratio=0.98, ratio_max=0.99))))
        best = sweep.winners(report)
        self.assertEqual(best['u4']['arm'], 'grid4x8', best)
        self.assertEqual(sweep.winners(self.report(case('u4@1024', 'u4', dict(
            served=arm_state(ratio=1.0), grid11x4=arm_state(ratio=0.995), grid8x4=arm_state(ratio=0.9, ratio_max=1.0))))), {})
        missing = arm_state(ratio=0.9)
        del missing['timing']['ratio_max']
        self.assertEqual(sweep.winners(self.report(case('u4@1024', 'u4', dict(served=arm_state(ratio=1.0), grid8x4=missing)))), {})

    def test_a_liveness_control_that_did_not_move_is_a_fail(self):
        report = self.report(case('u1@1024', 'u1', dict(served=arm_state(ratio=1.0)), moved=False))
        self.assertEqual(sweep.decide(report)['verdict'], 'FAIL')

    def test_a_non_finite_output_is_a_fail(self):
        bad = arm_state(ratio=1.0)
        bad['finite'] = False
        self.assertEqual(sweep.decide(self.report(case('u1@1024', 'u1', dict(served=bad))))['verdict'], 'FAIL')

    def test_no_served_arm_nothing_timed_a_failure_line_and_no_cases_are_no_decision(self):
        self.assertEqual(sweep.decide(self.report())['verdict'], 'NO-DECISION')
        self.assertEqual(sweep.decide(self.report(case('u1@1024', 'u1', dict(served=dict(status='error', error='x')))))['verdict'],
                         'NO-DECISION')
        self.assertEqual(sweep.decide(self.report(case('u1@1024', 'u1', dict(served=arm_state()))))['verdict'], 'NO-DECISION')
        report = self.report(case('u1@1024', 'u1', dict(served=arm_state(ratio=1.0))))
        report['failures'].append('no [QWEN-SDPA] factory line for flags=0x21 B=4 PNHt=3')
        self.assertEqual(sweep.decide(report)['verdict'], 'NO-DECISION')

    def test_timing_none_needs_no_timing(self):
        report = self.report(case('u1@1024', 'u1', dict(served=arm_state())))
        report['timing_mode'] = 'none'
        self.assertEqual(sweep.decide(report)['verdict'], 'PASS')

    def test_the_per_chunk_fit_is_taken_over_the_one_user_cases(self):
        report = self.report(*[
            case('u1@%d' % extent, 'u1', dict(served=dict(status='ok', timing=dict(median_us=50 + 14.5 * sweep.busiest_chunks(extent)))),
                 extents=(extent,)) for extent in (32768, 65536, 131328, 262400)])
        fitted = sweep.fits(report)['served']
        self.assertAlmostEqual(fitted['per_chunk_us'], 14.5, places=3)
        self.assertAlmostEqual(fitted['fixed_us'], 50.0, places=3)


class HarnessScriptTests(unittest.TestCase):
    path = OPS / 'sdpa_tp4_long' / 'run_card_m.sh'

    def text(self):
        return self.path.read_bytes().decode('utf-8')

    def test_it_mounts_the_five_scripts_reads_the_image_by_tag_and_mounts_no_graft_and_no_model(self):
        text = self.text()
        self.assertNotIn(chr(13), text)
        for name in ('sdpa_tp4_long.py', 'k64j_nkv1_spike.py', 'test_sdpa_decode_qwen_card_m.py', 'sdpa_long_tp.py', 'tp_shapes.py'):
            self.assertIn(name, text)
            self.assertTrue(any((root / name).is_file() for root in (OPS / 'sdpa_tp4_long', OPS / 'k64j', OPS / 'sdpa_decode_qwen', HERE)), name)
        self.assertIn('qwen38-c2-', text)
        self.assertIn('--network none', text)
        self.assertIn('QWEN_SDPA_TREE_SCRATCH_ROUNDS=1', text)
        for banned in ('zot', 'opgraft', 'KOPGRAFT', '_ttnncpp.so,dst', 'models--Qwen', 'HF_MODEL'):
            self.assertNotIn(banned, text)

    def test_it_pins_the_graft_from_the_build_script_not_from_its_own_text(self):
        text = self.text()
        self.assertIn('graft_sha=', text)
        self.assertIsNone(re.search(r'[0-9a-f]{64}', text))
        build = (HERE / 'build-c2-serving-image.sh').read_text(encoding='utf-8')
        self.assertRegex(build, r'(?m)^graft_sha=[0-9a-f]{64}$')

    def test_it_has_one_launch_one_recheck_and_the_hang_hint(self):
        text = self.text()
        self.assertEqual(len(re.findall(r'(?m)^timeout -k 30 "\$timeout_s" docker run', text)), 1)
        code = text[text.index('# <<< qual_card.sh'):]
        self.assertEqual(len(re.findall(r'(?m)^qual_card_recheck\b', code)), 1)
        self.assertIn('qual_reset_hint', text)
        self.assertIn('--deadline-s "$((timeout_s - 600))"', text)

    def test_it_has_a_card_free_image_check_and_reads_the_faulthandler_dump_as_a_hang(self):
        text = self.text()
        self.assertIn('IMAGE_CHECK_ONLY', text)
        resolve = text.index(chr(10) + 'qual_card_resolve' + chr(10))
        self.assertLess(text.index('IMAGE_CHECK_ONLY'), resolve)
        self.assertNotIn('--device', text[text.index('IMAGE_CHECK_ONLY'):resolve])
        self.assertIn("grep -q '^Timeout ('", text)

    def test_the_sweep_reads_what_the_harness_passes(self):
        text = self.text()
        parsed = sweep.parse_args(['--out', '/results/x.json', '--device-id', '0', '--deadline-s', '2400', '--expect-binary-sha256', 'ab'])
        self.assertEqual((parsed.deadline_s, parsed.device_id), (2400.0, 0))
        for flag in ('--out', '--device-id', '--deadline-s', '--expect-binary-sha256'):
            self.assertIn(flag, text)


# ---------------------------------------------------------------------------------------------
# The whole flow on a fake device.

class FakeDevice:
    grid = (11, 10)

    def compute_with_storage_grid_size(self):
        return SimpleNamespace(x=self.grid[0], y=self.grid[1])

    def enable_program_cache(self):
        pass


class FakeTensor:
    def __init__(self, data, dtype):
        self.data, self.dtype = data, dtype

    @property
    def shape(self):
        return self.data.shape


class FakeTTNN:
    bfloat8_b, bfloat16, int32 = 'bf8', 'bf16', 'int32'
    ROW_MAJOR_LAYOUT, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'row', 'tile', 'dram'

    def __init__(self, ignore_cur_pos=False, corrupt_flags=None, fail_flags=None, no_trace=False, grid=(11, 10)):
        self.ignore_cur_pos, self.corrupt_flags, self.fail_flags, self.no_trace = ignore_cur_pos, corrupt_flags, fail_flags, no_trace
        self.grid = grid
        self.launches, self.traces_run, self.opened = [], 0, []
        self.transformer = SimpleNamespace(paged_scaled_dot_product_attention_decode=self.attention)

    def open_device(self, **options):
        self.opened.append(options)
        device = FakeDevice()
        device.grid = self.grid
        return device

    def close_device(self, device):
        pass

    def from_torch(self, host, device=None, dtype=None, layout=None, memory_config=None):
        return FakeTensor(host.clone(), dtype)

    def to_torch(self, tensor):
        return tensor.data

    def deallocate(self, tensor):
        pass

    def synchronize_device(self, device):
        pass

    @staticmethod
    def SDPAProgramConfig(**options):
        return SimpleNamespace(**options)

    def begin_trace_capture(self, device, cq_id):
        if self.no_trace:
            raise RuntimeError('no trace region')
        return 1

    def end_trace_capture(self, device, trace, cq_id):
        pass

    def execute_trace(self, device, trace, cq_id, blocking):
        self.traces_run += 1

    def release_trace(self, device, trace):
        pass

    def attention(self, query, keys, values, *, page_table_tensor, is_causal, attn_mask, scale, program_config, memory_config,
                  cur_pos_tensor=None):
        flags = program_config.q_chunk_size & 0xFF
        self.launches.append(dict(flags=flags, batch=query.shape[1], rows=query.shape[2], grid=program_config.compute_with_storage_grid_size,
                                  pages=page_table_tensor.shape[1]))
        if self.fail_flags is not None and flags == self.fail_flags:
            raise RuntimeError('the factory refuses flags 0x%x' % flags)
        pages, mask = page_table_tensor.data, attn_mask.data
        batch, heads = query.shape[1], query.shape[2]
        output = torch.zeros(1, batch, heads, 256)
        for entry in range(batch):
            width = pages.shape[1] * 64
            if not self.ignore_cur_pos:
                width = int(cur_pos_tensor.data[entry]) + 1
            key = keys.data[pages[entry, :width // 64].long(), 0].reshape(width, 256).float()
            value = values.data[pages[entry, :width // 64].long(), 0].reshape(width, 256).float()
            scores = query.data[0, entry].float() @ key.T * scale
            columns = mask.shape[-1]
            scores[:, width - columns:width] += mask[entry, 0].float()
            output[0, entry] = torch.softmax(scores, dim=-1) @ value
        if self.corrupt_flags is not None and flags == self.corrupt_flags:
            output = output + 1.0
        return FakeTensor(output.to(torch.bfloat16), 'bf16')


class Quiet:
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def text(self):
        return ''


def fabricate(arms=None, users=(1, 2, 4), skip=()):
    """One factory line per (arm, program shape): what a native log has when every arm built its own program."""
    lines = []
    for arm in (sweep.ARMS if arms is None else arms):
        if arm in skip:
            continue
        for count in users:
            for flags, batch, pnht, st in sorted(sweep.expected_programs(arm, count, sweep.SERVED_POOL_PAGES)):
                lines.append(dict(flags='0x%x' % flags, B=batch, PNHt=pnht, St=st, mask_width_t=0, kv_share='x', scratch_slots=4,
                                  cb_bytes=1))
    return lines


def run_flow(device, *extra):
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / 'sweep.json'
        argv = ['--out', str(out), '--extents', '256', '--users', '4', '--no-mixed', '--rounds', '2', '--iterations', '1',
                '--calls', '2', '--watchdog', '0', '--binary', str(Path(directory) / 'none.so')] + list(extra)
        with patch.dict(sys.modules, {'ttnn': device}), patch.dict('os.environ', {'QWEN_SDPA_TREE_SCRATCH_ROUNDS': '1'}):
            import test_sdpa_decode_qwen_card_m as card

            fabricated = fabricate()
            with patch.object(card, 'NativeLog', Quiet), patch.object(card, 'factory_lines', lambda text: fabricated):
                status = sweep.main(argv)
        return status, json.loads(out.read_text())


class FlowTests(unittest.TestCase):
    def test_a_faithful_device_passes_every_arm_is_exact_and_everything_is_timed(self):
        device = FakeTTNN()
        status, report = run_flow(device)
        self.assertEqual(status, 0, report['decision'])
        self.assertEqual(report['decision']['verdict'], 'PASS')
        names = [case['name'] for case in report['cases']]
        self.assertEqual(names, ['u1@256', 'u4@256', 'u1@256', 'u4@256'], 'the safe pass over every case, then the risky pass')
        self.assertEqual([entry['phase'] for entry in report['cases']], ['safe', 'safe', 'risky', 'risky'])
        for entry in report['cases']:
            expected = ['served', 'rowsplit', 'ra'] if entry['phase'] == 'risky' else [arm for arm in sweep.ARMS if arm not in ('rowsplit', 'ra')]
            self.assertEqual(sorted(entry['arms']), sorted(expected), entry['name'])
        for entry in report['cases']:
            for arm, state in entry['arms'].items():
                with self.subTest(case=entry['name'], arm=arm):
                    if arm == 'multi' and len(entry['extents']) < 2:
                        self.assertEqual(state['status'], 'skipped')
                        continue
                    self.assertEqual(state['status'], 'ok', state.get('error'))
                    self.assertEqual(state['differing_rows'], 0)
                    self.assertTrue(state['finite'])
                    self.assertEqual(state['timing']['mode'], 'trace')
                    self.assertEqual(len(state['timing']['rounds'] if isinstance(state['timing']['rounds'], list) else [0] * state['timing']['rounds']), 2)
                    self.assertIn('ratio_median', state['timing'])
            self.assertTrue(entry['liveness']['moved'])
            self.assertEqual(len(entry['arms']['served']['user_hashes']), len(entry['extents']))
            self.assertTrue(all(state.get('trace_equal') for state in entry['arms'].values() if state['status'] == 'ok'))
        self.assertGreater(device.traces_run, 0)
        self.assertIn('trace_region_size', device.opened[0])
        self.assertIn('winners', report)
        self.assertIn('fits', report)
        self.assertTrue(report['verdict_line'].startswith('SDPA_TP4_LONG verdict=PASS'), report['verdict_line'])

    def test_every_launch_of_every_case_uses_the_one_served_page_table_width(self):
        device = FakeTTNN()
        run_flow(device, '--arms', 'served,grid8x4,multi,rowsplit', '--timing', 'none', '--extents', '256,512')
        self.assertEqual({launch['pages'] for launch in device.launches}, {sweep.SERVED_POOL_PAGES})

    def test_a_device_grid_that_is_not_the_serving_meshs_is_no_decision(self):
        status, report = run_flow(FakeTTNN(grid=(8, 10)), '--arms', 'served', '--timing', 'none')
        self.assertEqual(status, 1)
        self.assertEqual(report['decision']['verdict'], 'NO-DECISION')
        self.assertIn('[8, 10]', ' '.join(report['failures']))
        self.assertEqual(report['cases'], [], 'nothing ran on the wrong grid')

    def test_a_grid_arm_without_a_program_of_its_own_is_a_missing_factory_line(self):
        # one line for the shared (flags, B, PNHt, St) satisfies the served arm alone, not the served arm and four grid arms
        every = ['served', 'grid8x4', 'grid8x10', 'grid11x4', 'grid4x8']
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'sweep.json'
            argv = ['--out', str(out), '--extents', '256', '--users', '1', '--arms', ','.join(every),
                    '--timing', 'none', '--watchdog', '0', '--binary', str(Path(directory) / 'none.so')]
            with patch.dict(sys.modules, {'ttnn': FakeTTNN()}), patch.dict('os.environ', {'QWEN_SDPA_TREE_SCRATCH_ROUNDS': '1'}):
                import test_sdpa_decode_qwen_card_m as card

                for lines, verdict in ((fabricate(['served'], users=(1,)), 'NO-DECISION'),
                                       (fabricate(['served', 'grid8x4'], users=(1,)), 'NO-DECISION'),
                                       (fabricate(every, users=(1,)), 'PASS')):
                    with patch.object(card, 'NativeLog', Quiet), patch.object(card, 'factory_lines', lambda text, lines=lines: lines):
                        sweep.main(argv)
                    report = json.loads(out.read_text())
                    self.assertEqual(report['decision']['verdict'], verdict, report['decision'])
                    if verdict != 'PASS':
                        self.assertIn('5 arm(s)', ' '.join(report['decision']['problems']))
                        self.assertIn('grid8x4', ' '.join(report['decision']['problems']))

    def test_the_binary_hashed_is_the_one_mapped(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second, maps = Path(directory) / 'a' / '_ttnncpp.so', Path(directory) / 'b' / '_ttnncpp.so', Path(directory) / 'maps'
            for path, data in ((first, b'graft one'), (second, b'graft two')):
                path.parent.mkdir()
                path.write_bytes(data)
            args = SimpleNamespace(binary=str(Path(directory) / 'fallback.so'), expect_binary_sha256=sweep.sha256_of(first))
            line = '7f00-7f01 r--p 0 00:00 0 %s'
            maps.write_text(chr(10).join([line % first, line % first]) + chr(10))
            report = dict(failures=[])
            sweep.check_binary(args, report, maps=str(maps))
            self.assertEqual((report['failures'], report['binary']['source'], report['binary']['path']), ([], 'mapped', str(first)))
            maps.write_text(chr(10).join([line % first, line % second]) + chr(10))
            report = dict(failures=[])
            sweep.check_binary(args, report, maps=str(maps))
            self.assertIn('more than one _ttnncpp.so is mapped', ' '.join(report['failures']))
            maps.write_text(line % second + chr(10))
            report = dict(failures=[])
            sweep.check_binary(args, report, maps=str(maps))
            self.assertIn('not the expected graft', ' '.join(report['failures']), 'the mapped binary is held to the pin, not the fallback path')

    def test_every_device_step_arms_a_faulthandler_backstop_that_works_without_the_gil(self):
        calls = []
        with patch.object(sweep.faulthandler, 'dump_traceback_later', lambda seconds, **options: calls.append(('arm', seconds, options.get('exit')))), \
                patch.object(sweep.faulthandler, 'cancel_dump_traceback_later', lambda: calls.append(('cancel',))):
            watchdog = sweep.Watchdog(300.0, lambda label: None)
            with watchdog.op('u1@256/served run'):
                self.assertEqual(calls, [('arm', 300.0 + sweep.WATCHDOG_BACKSTOP_S, True)])
            self.assertEqual(calls[-1], ('cancel',))
            calls.clear()
            off = sweep.Watchdog(0, lambda label: None)
            with off.op('x'):
                pass
            self.assertEqual(calls, [], 'watchdog 0 arms nothing')

    def test_the_liveness_control_frees_both_outputs(self):
        freed = []

        class Counting(FakeTTNN):
            def deallocate(self, tensor):
                freed.append(tensor)

        device = Counting()
        run_flow(device, '--arms', 'served', '--timing', 'none', '--extents', '256', '--users', '1')
        outputs = [tensor for tensor in freed if getattr(tensor, 'dtype', None) == 'bf16' and tuple(tensor.shape) == (1, 2, 48, 256)]
        # read() frees one output per user, the liveness control frees two more
        self.assertGreaterEqual(len(outputs), 3)

    def test_the_launch_shapes_are_the_served_ones_per_arm(self):
        device = FakeTTNN()
        run_flow(device, '--arms', 'served,multi,rowsplit,ra,grid4x8', '--timing', 'none', '--extents', '512')
        shapes = {(launch['flags'], launch['batch'], launch['rows']) for launch in device.launches}
        self.assertIn((0x23, 2, 48), shapes)       # served, G8B2
        self.assertIn((0x21, 4, 96), shapes)       # multi: one G16 entry per user
        self.assertIn((0x23, 4, 24), shapes)       # rowsplit, G4B4
        self.assertIn((0x2B, 2, 48), shapes)       # ra
        self.assertIn((4, 8), {launch['grid'] for launch in device.launches}, 'a grid arm launches on its grid')
        self.assertIn((11, 10), {launch['grid'] for launch in device.launches}, 'the served arm launches on the mesh grid')

    def test_the_eager_fallback_and_timing_none(self):
        status, report = run_flow(FakeTTNN(no_trace=True), '--arms', 'served,multi')
        self.assertEqual(status, 0)
        for entry in report['cases']:
            for arm, state in entry['arms'].items():
                if state['status'] == 'ok':
                    self.assertEqual(state['timing']['mode'], 'eager')
                    self.assertIn('trace_error', state)
        status, report = run_flow(FakeTTNN(), '--timing', 'none', '--arms', 'served,multi')
        self.assertEqual(status, 0)
        self.assertEqual(report['timing_mode'], 'none')
        self.assertTrue(all('timing' not in state for entry in report['cases'] for state in entry['arms'].values()))

    def test_a_layout_that_computes_something_else_is_not_exact_and_is_not_a_winner(self):
        status, report = run_flow(FakeTTNN(corrupt_flags=0x21), '--arms', 'served,multi,grid8x4')
        self.assertEqual(status, 0, report['decision'])
        multi = [entry['arms']['multi'] for entry in report['cases'] if entry['name'].startswith('u4')]
        self.assertTrue(all(state['differing_rows'] > 0 for state in multi), multi)
        self.assertNotIn('multi', [winner['arm'] for winner in report['winners'].values()])

    def test_an_arm_the_factory_refuses_is_data_and_the_sweep_goes_on(self):
        status, report = run_flow(FakeTTNN(fail_flags=0x2B), '--arms', 'served,ra,grid8x4')
        self.assertEqual(status, 0, report['decision'])
        for entry in report['cases']:
            if entry['phase'] == 'risky':
                self.assertEqual(entry['arms']['ra']['status'], 'error')
                self.assertIn('0x2b', entry['arms']['ra']['error'])
            else:
                self.assertEqual(entry['arms']['grid8x4']['status'], 'ok')

    def test_a_call_that_ignores_cur_pos_fails_the_liveness_control(self):
        status, report = run_flow(FakeTTNN(ignore_cur_pos=True), '--arms', 'served', '--timing', 'none')
        self.assertEqual(status, 1)
        self.assertEqual(report['decision']['verdict'], 'FAIL')
        self.assertIn('cur_pos', ' '.join(report['decision']['problems']))

    def test_a_missing_factory_line_is_no_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'sweep.json'
            argv = ['--out', str(out), '--extents', '512', '--users', '2', '--no-mixed', '--arms', 'served,multi', '--timing', 'none',
                    '--watchdog', '0', '--binary', str(Path(directory) / 'none.so')]
            with patch.dict(sys.modules, {'ttnn': FakeTTNN()}), patch.dict('os.environ', {'QWEN_SDPA_TREE_SCRATCH_ROUNDS': '1'}):
                import test_sdpa_decode_qwen_card_m as card

                served_only = [dict(flags='0x23', B=2, PNHt=2, St=0, mask_width_t=0, kv_share='true', scratch_slots=4, cb_bytes=1)]
                with patch.object(card, 'NativeLog', Quiet), patch.object(card, 'factory_lines', lambda text: served_only):
                    status = sweep.main(argv)
            report = json.loads(out.read_text())
        self.assertEqual(status, 1)
        self.assertEqual(report['decision']['verdict'], 'NO-DECISION')
        self.assertIn('flags=0x21 B=2 PNHt=3', ' '.join(report['decision']['problems']))

    def test_the_scratch_switch_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'sweep.json'
            with patch.dict(sys.modules, {'ttnn': FakeTTNN()}), patch.dict('os.environ', {'QWEN_SDPA_TREE_SCRATCH_ROUNDS': '0'}):
                status = sweep.main(['--out', str(out), '--extents', '512', '--users', '1', '--watchdog', '0'])
            report = json.loads(out.read_text())
        self.assertEqual(status, 1)
        self.assertIn('QWEN_SDPA_TREE_SCRATCH_ROUNDS=1 is required', ' '.join(report['failures']))

    def test_a_wrong_binary_is_a_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / 'lib.so'
            binary.write_bytes(b'not the graft')
            status, report = run_flow(FakeTTNN(), '--timing', 'none', '--arms', 'served', '--binary', str(binary),
                                      '--expect-binary-sha256', '0' * 64)
        self.assertEqual(status, 1)
        self.assertEqual(report['decision']['verdict'], 'NO-DECISION')
        self.assertIn('not the expected graft', ' '.join(report['failures']))
        self.assertEqual(len(report['binary']['sha256']), 64)

    def test_the_deadline_stops_cleanly_between_cases(self):
        with patch.object(sweep.Deadline, 'reached', lambda self: True):
            status, report = run_flow(FakeTTNN())
        self.assertEqual(report['cases'], [])
        self.assertIn('deadline', report)
        self.assertEqual(report['decision']['verdict'], 'NO-DECISION')


if __name__ == '__main__':
    unittest.main()
