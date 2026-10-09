"""CPU checks for k64j_card_b.py at the OCTO-T8 geometry (--kv-heads 1 --octo: ONE eight-row group per bundle, G8B1, flags 0x21); no device, no ttnn.

K64j is not changed for octo-T8: 0x21 is an existing flag set (K1/K3 qualified it at G8B2 and G4B3, one KV head). What no card has run is a bundle of ONE eight-row group, which is
the qualification job Q1 (scripts/ci/references/tp4-octo-jobs). This file holds the harness that runs it:

  - the geometry: G8B1 is one entry of eight rows, valid at 0x21 and one KV head only (no share to take, no slice at one KV head), refused without --octo; the served ticket
    (K2, X7, Z) is eight rows in one bundle, and set_served_geometry rebinds only the four names and puts them back;
  - the arguments: --octo selects the G8B1 combos and trace combos, needs --kv-heads 1, refuses sections M and K, and the report and the verdict line name the geometry;
  - the masks and the layout: the narrow mask of an eight-row one-entry ticket is the four-card pinned kernel's, bit for bit (extent_attention_replay_tp.narrow_mask_host);
  - the flow on the fake ttnn (test_k64j_card_b.FakeExtentTtnn): N, X, L and T end to end at G8B1, and CB2a's K2 (the native one-row decode against the one-entry extent call, row
    by row), X7 and Z, with the broken variants each section must catch.

    py -3.11 -B -m unittest test_k64j_octo      (from this directory)
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import test_k64j_card_b as base  # noqa: E402 - sys.path (k64j, k64j_probe, sdpa_decode_qwen, scripts/ci), the fake
import test_k64j_cb2a as cb2a  # noqa: E402

card_b, probe, model, card = base.card_b, base.probe, base.model, base.card
FakeExtentTtnn = base.FakeExtentTtnn
ROOT = base.ROOT
CI = ROOT / 'scripts' / 'ci'
if str(CI) not in sys.path:
    sys.path.insert(0, str(CI))


class served_geometry:
    """card_b.set_served_geometry(name) in force and put back; card.set_kv_heads(1) with it when `one_head`."""

    def __init__(self, name='G8B1', one_head=True):
        self.name, self.one_head = name, one_head

    def __enter__(self):
        self.previous_geometry = card_b.set_served_geometry(self.name)
        self.previous_heads = card.set_kv_heads(1) if self.one_head else None
        return self

    def __exit__(self, *exc):
        card_b.set_served_geometry(self.previous_geometry)
        if self.one_head:
            card.set_kv_heads(self.previous_heads)
        return False


class GeometryTests(unittest.TestCase):
    def test_g8b1_is_one_entry_of_eight_rows_valid_at_0x21_and_one_kv_head_only(self):
        self.assertEqual(card_b.SHAPES['G8B1'], (8, 1))
        self.assertEqual((card_b.OCTO_SHAPE, card_b.OCTO_FLAGS), ('G8B1', 0x21))
        self.assertTrue(card_b.valid_combo('G8B1', 0x21, card_b.ONE_KV_HEAD))
        for flags in (0x23, 0x27, 0x2f, 0x25, 0x20, 0x01):
            self.assertFalse(card_b.valid_combo('G8B1', flags, card_b.ONE_KV_HEAD), hex(flags))
        self.assertFalse(card_b.valid_combo('G8B1', 0x21, card_b.PAIR_KV_HEADS), 'a pair chip has two KV heads: not the octo geometry')
        self.assertEqual(card_b.default_combos(card_b.ONE_KV_HEAD), [(shape, flags) for shape in ('G4B3', 'G8B2') for flags in (0x21, 0x23)],
                         'the CB1 defaults do not move')
        self.assertEqual(card_b.default_trace_combos(card_b.ONE_KV_HEAD), (('G4B3', 0x21), ('G8B2', 0x23)))

    def test_the_layout_of_the_octo_ticket_is_one_bundle_of_one_eight_row_group(self):
        import extent_attention_replay as pinned

        self.assertEqual([[(group['offset'], group['rows']) for group in bundle] for bundle in pinned.LAYOUT(8, 8)], [[(0, 8)]])
        self.assertEqual(card_b.SERVED_GEOMETRIES['G8B1'], dict(ticket_rows=8, rows=8, batch=1, offsets=(0,)))
        self.assertEqual(card_b.SHAPES['G8B1'], (card_b.SERVED_GEOMETRIES['G8B1']['rows'], card_b.SERVED_GEOMETRIES['G8B1']['batch']))

    def test_the_flags_of_the_served_ticket(self):
        self.assertEqual(card_b.served_flags_for(card_b.ONE_KV_HEAD, 'G8B1'), (0x21, 0x1))
        self.assertEqual(card_b.served_flags_for(card_b.ONE_KV_HEAD), (0x23, 0x3), 'the M3 ticket, unchanged')
        self.assertEqual(card_b.served_flags_for(card_b.PAIR_KV_HEADS), (0x27, 0x7))
        with self.assertRaisesRegex(ValueError, 'one KV head per chip'):
            card_b.served_flags_for(card_b.PAIR_KV_HEADS, 'G8B1')

    def test_the_switch_rebinds_four_names_and_puts_them_back(self):
        before = (card_b.TICKET_ROWS, card_b.SERVED_ROWS, card_b.SERVED_BATCH, card_b.SERVED_OFFSETS)
        self.assertEqual(before, (16, 8, 2, (0, 8)))
        with served_geometry():
            self.assertEqual((card_b.TICKET_ROWS, card_b.SERVED_ROWS, card_b.SERVED_BATCH, card_b.SERVED_OFFSETS), (8, 8, 1, (0,)))
            self.assertEqual(card_b.ticket_positions(300), list(range(300, 308)))
            self.assertEqual(card_b.accept_limit(300), 8)
            self.assertEqual(card_b.accept_limit(509), 3, 'rows 3..7 lie at or past E = 512: the boundary cap, now at eight rows')
            self.assertEqual(card_b.valid_positions(509), [509, 510, 511])
        self.assertEqual((card_b.TICKET_ROWS, card_b.SERVED_ROWS, card_b.SERVED_BATCH, card_b.SERVED_OFFSETS), before)
        self.assertEqual(card_b.accept_limit(509), 3)
        self.assertEqual(card_b.ticket_positions(300), list(range(300, 316)))
        with self.assertRaisesRegex(ValueError, 'Unknown served geometry'):
            card_b.set_served_geometry('G16B1')

    def test_the_narrow_mask_of_an_eight_row_ticket_is_the_four_card_kernels_bit_for_bit(self):
        import torch

        import extent_attention_replay_tp as quad

        with served_geometry(), mock.patch.dict(os.environ, {'QWEN_FAST_TP': '4'}):
            for start in (128, 135, 200, 255, 300, 511):
                mine = card_b.narrow_mask(torch, start)
                theirs = quad.narrow_mask_host(start & 255, 8, 1, 0)
                self.assertEqual(tuple(mine.shape), (1, 1, 48, 256))
                self.assertTrue(torch.equal(mine.view(torch.int16), theirs.view(torch.int16)), start)
            wide = card_b.wide_mask(torch, 300, 512)
            self.assertEqual(tuple(wide.shape), (1, 1, 48, 512))
            self.assertTrue(torch.equal(wide[..., 256:], card_b.narrow_mask(torch, 300)), 'the narrow mask is the wide mask\'s last chunk')

    def test_the_fold_puts_each_token_on_its_mask_row(self):
        import torch

        with served_geometry():
            tokens = {position: torch.full((6, 256), float(position), dtype=torch.bfloat16) for position in range(300, 308)}
            query = card_b.ticket_query(torch, tokens.__getitem__, 300)
            self.assertEqual(tuple(query.shape), (1, 1, 48, 256))
            for token in range(8):
                self.assertTrue(bool((query[0, 0, token * 6:(token + 1) * 6] == float(300 + token)).all()), token)


class ArgumentTests(unittest.TestCase):
    def parse(self, *argv):
        return card_b.parse_args(['--out', 'x.json'] + list(argv))

    def tearDown(self):
        card_b.set_served_geometry('G8B2')           # parse_args sets the geometry; a parse that never ran main must not leak it

    def test_octo_selects_the_g8b1_combos_and_the_octo_served_ticket(self):
        args = self.parse('--kv-heads', '1', '--octo', '--sections', 'N,X,L,T,K2,X7,Z')
        self.assertEqual(args.combos, [('G8B1', 0x21)])
        self.assertEqual(args.trace_combos, [('G8B1', 0x21)])
        self.assertEqual((args.geometry, args.served_flags, args.compile_flags), ('G8B1', 0x21, 0x1))
        self.assertEqual((card_b.SERVED_BATCH, card_b.TICKET_ROWS), (1, 8))

    def test_without_octo_nothing_moves(self):
        args = self.parse('--kv-heads', '1')
        self.assertEqual((args.geometry, args.served_flags, args.compile_flags), ('G8B2', 0x23, 0x3))
        self.assertEqual((card_b.SERVED_BATCH, card_b.TICKET_ROWS), (2, 16))
        args = self.parse()
        self.assertEqual((args.served_flags, args.compile_flags), (0x27, 0x7))

    def test_octo_is_refused_where_it_is_not_the_geometry(self):
        for argv in (['--octo'],                                                    # a pair chip
                     ['--kv-heads', '1', '--octo', '--sections', 'N,X,M'],          # M and K are the G4B3 families
                     ['--kv-heads', '1', '--octo', '--sections', 'K'],
                     ['--kv-heads', '1', '--combos', 'G8B1:0x21'],                  # a G8B1 combo without --octo
                     ['--kv-heads', '1', '--trace-combos', 'G8B1:0x21'],
                     ['--kv-heads', '1', '--octo', '--combos', 'G8B1:0x23'],
                     ['--kv-heads', '2', '--combos', 'G8B1:0x21']):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                self.parse(*argv)

    def test_the_verdict_line_names_the_octo_geometry_only_for_an_octo_run(self):
        report = dict(comparisons=[], liveness=[], decision=dict(verdict='PASS', first_differing=[], reasons=[], k2='not_run'), geometry='G8B1')
        self.assertIn(' geometry=G8B1 k2_verdict=not_run ', card_b.verdict_line(report))
        report.pop('geometry')
        self.assertNotIn('geometry=', card_b.verdict_line(report))


class OctoFlow(unittest.TestCase):
    """CB1's N, X, L and T at G8B1 on the fake (test_k64j_one_head.OneHeadFlow's setup)."""

    BASE = ['--capacity', '4352', '--extents', '512,2304,4352', '--starts', '7,255', '--seeds', '0',
            '--trace-families', '4', '--trace-references', '2', '--no-timing', '--kv-heads', '1', '--octo', '--sections', 'N,X,L,T']

    setUp = base.DryRunTests.setUp
    tearDown = base.DryRunTests.tearDown
    run_card = base.DryRunTests.run_card
    kinds = base.DryRunTests.kinds

    def test_the_octo_geometry_passes_end_to_end(self):
        fake = FakeExtentTtnn(self.torch, record_calls=True)
        status, report = self.run_card(fake)
        self.assertEqual((report.get('error'), report['failures'], report['warnings']), (None, [], []))
        self.assertEqual((status, report['passed'], report['decision']['verdict']), (0, True, 'PASS'))
        self.assertEqual((report['kv_heads'], report['local_heads'], report['geometry']), (1, 6, 'G8B1'))
        self.assertEqual(report['combos'], ['G8B1:0x21'])
        self.assertEqual(report['trace_combos'], ['G8B1:0x21'])
        self.assertEqual(report['sections_done'], ['N/seed0', 'X/seed0', 'L/seed0', 'T/seed0'])
        kinds = self.kinds(report)
        self.assertEqual(kinds['extent_vs_reference'], (6, 6), '3 extents x 2 starts, one entry each')
        self.assertEqual((kinds['trace_vs_eager'], kinds['trace_vs_reference']), ((4, 4), (2, 2)))
        self.assertEqual(kinds['trace_fence_vs_clean'][0], kinds['trace_fence_vs_clean'][1])
        self.assertEqual(report['fence_extents'], {'G8B1:0x21': [512]})
        self.assertTrue(report['liveness'] and all(entry['live'] for entry in report['liveness']))
        # 6 folded rows per token: an eight-token group is 48 rows, ONE entry; no program is shared and none carries the slice
        # (section N's refusals also issue 0x21 calls at the other shapes; they are refused, so no extent line follows them)
        extent_calls = {(call['rows'], call['batches']) for call in fake.recorded if call['program_config']['q_chunk_size'] in (card.MAGIC | 0x21, card.MAGIC | 0x1)}
        self.assertIn((48, 1), extent_calls)
        self.assertEqual({int(entry['entries']) for entry in report['extent_lines']}, {1}, 'every program that BUILT is the one-entry bundle')
        flags = {program[0] for program in report['requested_programs'] if program[0] in (0x1, 0x21)}
        self.assertEqual(flags, {0x1, 0x21})
        self.assertEqual({line['q_slice'] for line in report['extent_lines']}, {'false'})
        self.assertEqual({line['kv_share'] for line in report['extent_lines']}, {'false'}, 'one entry: nothing to share')
        self.assertIn(' geometry=G8B1 k2_verdict=not_run ', report['verdict_line'])
        self.assertEqual((card_b.SERVED_BATCH, card.KV_HEADS), (2, 2), 'main restored the pair\'s geometry and the served ticket')

    def test_a_program_that_ignores_the_word_fails_at_g8b1(self):
        status, report = self.run_card(FakeExtentTtnn(self.torch, broken={'ignore_word'}), ['--sections', 'X', '--starts', '7'])
        self.assertEqual((status, report['failures'], report['decision']['verdict']), (1, [], 'FAIL'))
        self.assertEqual(self.kinds(report)['extent_vs_reference'], (1, 3), 'only E = C is right')

    def test_a_binary_that_never_logs_the_one_entry_program_was_not_executed(self):
        status, report = self.run_card(FakeExtentTtnn(self.torch, broken={'no_f22'}), ['--sections', 'X', '--extents', '2304', '--starts', '7'])
        self.assertEqual((status, report['decision']['verdict']), (1, 'NO-DECISION'))
        self.assertTrue(report['failures'][0].startswith('extent log: flags=0x21 B=1'), report['failures'])


class OctoCB2aFlow(unittest.TestCase):
    """K2, X7 and Z at G8B1: the native B = 1 one-row decode against the one-entry extent call."""

    BASE = ['--capacity', '4352', '--extents', '2304', '--seeds', '0', '--no-timing', '--sections', 'K2,X7,Z',
            '--cb2-extents', '2304,4352', '--kv-heads', '1', '--octo']
    SMALL = cb2a.SMALL

    setUp = cb2a.DryRunTests.setUp
    tearDown = cb2a.DryRunTests.tearDown
    run_card = cb2a.DryRunTests.run_card
    kinds = cb2a.DryRunTests.kinds

    def test_cb2a_passes_end_to_end_at_g8b1(self):
        fake = FakeExtentTtnn(self.torch, record_calls=True)
        status, report = self.run_card(fake, ['--variants', 'peaky', '--z-starts', '0,32,240'])
        self.assertEqual((report.get('error'), report['failures'], report['warnings']), (None, [], []))
        self.assertEqual((status, report['passed'], report['decision']['verdict'], report['decision']['k2']),
                         (0, True, 'PASS', 'REDUCED-PASS'))
        self.assertEqual((report['kv_heads'], report['local_heads'], report['geometry']), (1, 6, 'G8B1'))
        kinds = self.kinds(report)
        self.assertEqual(kinds['k2_native_vs_extent'], (183, 183))                  # the same tickets: 173 sweep + 10 family
        self.assertEqual(kinds['x7_narrow_vs_wide'][0], kinds['x7_narrow_vs_wide'][1])
        self.assertEqual(kinds['x7_extent_vs_wide'][0], kinds['x7_extent_vs_wide'][1])
        self.assertEqual((kinds['z_trace_vs_eager'], kinds['z_trace_vs_reference']), ((45, 45), (45, 45)))
        self.assertTrue(all(entry['live'] for entry in report['liveness']))
        self.assertEqual({entry['rows'] for entry in report['liveness'] if entry['section'] == 'K2'}, {6})
        self.assertEqual(report['cb2a']['served'], dict(flags='0x21', compile_flags='0x1', rows=8, batch=1, offsets=[0], k_chunk_size=256))
        # K2's rows: each ticket is EIGHT rows now (the M3 ticket's sixteen are two of them): the sweep's tickets at the cap lose their tail rows
        rows = report['k2_rows']
        self.assertEqual(rows['equal'], rows['compared'])
        self.assertGreater(rows['compared'], 0)
        with served_geometry():
            tickets = [ticket for ticket in card_b.k2_tickets(card_b.K2_SWEEP, None, [2304, 4352], card_b.CB2_STARTS) if ticket['kind'] != 'floor']
            self.assertEqual(rows['compared'], sum(len(card_b.valid_positions(ticket['start'])) for ticket in tickets))
            self.assertEqual(rows['capped'], sum(8 - len(card_b.valid_positions(ticket['start'])) for ticket in tickets))
        self.assertIn(' kv_heads=1 geometry=G8B1 k2_verdict=REDUCED-PASS ', report['verdict_line'])
        # The calls: the native one as the model issues it (one row of 6 heads), the subject and its compile-time twin on the 48 folded
        # rows of ONE entry, at 0x21 and 0x1 (never 0x23 / 0x3: no share).
        native_calls = [call for call in fake.recorded if call['program_config']['q_chunk_size'] == card.LEGACY]
        subject = [call for call in fake.recorded if call['program_config']['q_chunk_size'] == card.MAGIC | 0x21]
        compile_time = [call for call in fake.recorded if call['program_config']['q_chunk_size'] == card.MAGIC | 0x1]
        self.assertEqual(len(fake.recorded), len(native_calls) + len(subject) + len(compile_time))
        self.assertTrue(native_calls and subject and compile_time)
        for call in native_calls:
            self.assertEqual((call['options'], call['is_causal'], call['memory_config'], call['rows'], call['batches']),
                             (sorted(card_b.NATIVE_CALL_KWARGS), None, 'l1', 6, 1))
        for group, expected in ((subject, cb2a.SERVED_KWARGS), (compile_time, cb2a.SERVED_KWARGS - {'cur_pos_tensor'})):
            for call in group:
                self.assertEqual((call['options'], call['is_causal'], call['memory_config'], call['rows'], call['batches']),
                                 (sorted(expected), False, 'l1', 48, 1))
        self.assertEqual((card.KV_HEADS, card_b.SERVED_BATCH), (2, 2))

    def test_a_k2_reference_at_the_wrong_cur_pos_fails_at_g8b1(self):
        for name, wrong in (('p+1', lambda position: position + 1), ('p-1', lambda position: position - 1)):
            with self.subTest(wrong=name), mock.patch.object(card_b, 'native_cur_pos', wrong):
                status, report = self.run_card(FakeExtentTtnn(self.torch), ['--sections', 'K2'] + self.SMALL, name='wrong-' + name)
                self.assertEqual(report['decision']['k2'], 'FAIL')

    def test_a_mask_read_at_the_capacity_fails_x7_at_g8b1(self):
        status, report = self.run_card(FakeExtentTtnn(self.torch, broken={'tail_at_capacity'}), ['--sections', 'X7'] + self.SMALL)
        self.assertEqual(report['decision']['verdict'], 'FAIL')

    def test_a_program_that_ignores_the_word_fails_z_at_g8b1(self):
        status, report = self.run_card(FakeExtentTtnn(self.torch, broken={'ignore_word'}),
                                       ['--sections', 'Z'] + self.SMALL + cb2a.Z_PAIR)
        self.assertEqual(report['decision']['verdict'], 'FAIL')


class RunnerAndJobTests(unittest.TestCase):
    """The committed Q jobs (scripts/ci/references/tp4-octo-jobs) through the runner's dry run (K64J_CARD_DRY_RUN=1: the argv it would launch) into the harnesses' own parsers."""

    JOBS = ROOT / 'scripts' / 'ci' / 'references' / 'tp4-octo-jobs'

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.graft = base.make_graft(self.dir)

    def tearDown(self):
        self.tmp.cleanup()
        card_b.set_served_geometry('G8B2')

    def launch(self, **env):
        import subprocess

        environ = {key: value for key, value in os.environ.items() if key not in base.SCRUB}
        environ['QUAL_CARD'] = base.CARD_X
        environ.update(HOME=self.dir.as_posix(), RESULTS=(self.dir / 'results').as_posix(), K64J_CARD_DRY_RUN='1',
                       KOPGRAFT64=self.graft.as_posix(), EXPECT_TTNNCPP_SHA256=base.sha(base.BINARY))
        environ.update(env)
        result = subprocess.run([base.BASH, base.RUNNER.as_posix()], env=environ, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        import shlex

        line = [line for line in result.stdout.splitlines() if line.startswith('### argv: ')]
        return shlex.split(line[0][len('### argv: '):])

    def job(self, name):
        values = {}
        for line in (self.JOBS / name).read_text(encoding='utf-8').splitlines():
            if line and not line.startswith('#'):
                key, _, value = line.partition('=')
                values[key] = value
        words = dict(word.split('=', 1) for word in values['C2_CARDM_ENV'].split())
        placeholders = {'@K64J_GRAFT_DIR@': self.graft.as_posix(), '@K64J_TTNNCPP_SHA256@': base.sha(base.BINARY)}
        env = {key: placeholders.get(value, value) for key, value in words.items() if key in ('WATCHER', 'KOPGRAFT64', 'EXPECT_TTNNCPP_SHA256', 'K64J_HARNESS', 'TP4_WIDTH')}
        return values, words, env

    def test_the_watcher_pass_runs_the_octo_combo_in_place_of_the_m3_ones(self):
        values, words, env = self.job('Q1w-k64j-g8b1-watcher.env')
        self.assertEqual((values['C2_CARDM_HARNESS'], words['K64J_HARNESS'], words['WATCHER']), ('optimisation/ttnn-op/k64j/run_card_b.sh', 'card', '1'))
        argv = self.launch(CARD_B_ARGS=values['C2_CARDM_ARGS'], **env)
        args = card_b.parse_args(argv[argv.index('card') + 1:])
        self.assertEqual((args.kv_heads, args.octo, args.geometry, args.served_flags, args.compile_flags), (1, True, 'G8B1', 0x21, 0x1))
        self.assertEqual((args.combos, args.trace_combos), ([('G8B1', 0x21)], [('G8B1', 0x21)]), 'the runner\'s watcher combos are the octo ones, the job\'s arguments last')
        self.assertEqual((args.sections, args.seeds, args.extents, args.variants, args.no_timing), (['N', 'X', 'L', 'T', 'K2', 'X7', 'Z'], [0], [2304, 131328], ['normal', 'peaky'], True))
        self.assertEqual(args.watchdog, 120.0)
        self.assertEqual((card_b.SERVED_BATCH, card_b.TICKET_ROWS), (1, 8))

    def test_without_octo_the_watcher_pass_is_what_it_was(self):
        argv = self.launch(WATCHER='1', CARD_B_ARGS='--kv-heads 1 --sections N,X --seeds 0')
        args = card_b.parse_args(argv[argv.index('card') + 1:])
        self.assertEqual((args.combos, args.trace_combos, args.octo), ([('G4B3', 0x21), ('G4B3', 0x23), ('G8B2', 0x23)], [('G4B3', 0x21), ('G8B2', 0x23)], False))

    def test_the_evidence_jobs_call_the_harness_with_the_flags_implemented(self):
        for name, sections, seeds in (('Q1a-k64j-g8b1-cb1.env', ['N', 'X', 'L', 'T'], [0, 1, 2, 3, 4]), ('Q1b-k64j-g8b1-cb2a.env', ['K2', 'X7', 'Z'], [0, 1, 2, 3, 4])):
            with self.subTest(job=name):
                values, words, env = self.job(name)
                self.assertNotIn('WATCHER', words)
                argv = self.launch(CARD_B_ARGS=values['C2_CARDM_ARGS'], **env)
                args = card_b.parse_args(argv[argv.index('card') + 1:])
                self.assertEqual((args.kv_heads, args.octo, args.sections, args.seeds, args.no_timing), (1, True, sections, seeds, True))
                self.assertEqual((args.combos, args.trace_combos), ([('G8B1', 0x21)], [('G8B1', 0x21)]))
                if name.startswith('Q1b'):
                    self.assertEqual((args.variants, args.k2_sweep, args.cb2_extents, args.z_families),
                                     (['normal', 'peaky'], card_b.K2_SWEEP, list(card_b.CB2_EXTENTS), list(card_b.Z_FAMILIES)), 'the design\'s whole set: PASS, not REDUCED-PASS')
                else:
                    self.assertEqual((args.extents, args.starts), (list(probe.EXTENTS), list(probe.STARTS)), 'K1\'s six families and starts')

    def test_the_gdn_jobs_call_their_harness_with_the_octo_geometry(self):
        import gdn_tp4_card_test as gdn

        for name, watcher in (('Q2w-gdn-eight-users-watcher.env', True), ('Q2-gdn-eight-users.env', False)):
            with self.subTest(job=name):
                values, words, env = self.job(name)
                self.assertEqual((words['K64J_HARNESS'], words['TP4_WIDTH'], words.get('WATCHER') == '1'), ('gdn_tp4', '4', watcher))
                argv = self.launch(CARD_B_ARGS=values['C2_CARDM_ARGS'], **env)
                self.assertIn('gdn_tp4_card_test.py', ' '.join(argv), 'the container runs the GDN card test')
                arguments = gdn.parse(argv[argv.index('card') + 1:])
                self.assertEqual((arguments.sections, arguments.users, arguments.rows), (['UB', 'NT'], 8, 8))
                if not watcher:
                    self.assertEqual((arguments.seeds, arguments.iterations), ([17, 18, 19], 20))
                else:
                    self.assertEqual(arguments.iterations, 0, 'the watcher pass times nothing')

    def test_every_q_job_pins_the_served_graft_and_names_no_new_one(self):
        for name in sorted(path.name for path in self.JOBS.glob('Q*.env')):
            text = (self.JOBS / name).read_text(encoding='utf-8')
            with self.subTest(job=name):
                self.assertIn('EXPECT_TTNNCPP_SHA256=@K64J_TTNNCPP_SHA256@', text)
                self.assertIn('KOPGRAFT64=@K64J_GRAFT_DIR@', text)
                self.assertIn('NO NEW GRAFT', text)


if __name__ == '__main__':
    unittest.main()
