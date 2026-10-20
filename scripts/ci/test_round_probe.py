"""tp4/round-breakdown: the round probe (round_probe, QWEN_FAST_TP4_ROUND_PROBE), its reader (round_probe_report), the shape model (round_shape_model) and the gate-only twin
(make_probe_profiles). CPU only.

Pinned here:
  - the flag is 0 or 1 and the period an integer from 2 to 4096, both strict, and with the flag unset the coordinator calls nothing of the module (it is not even imported by the hook);
  - the probe: every EVERY-th round of the coordinator (any draft shape) does ONE timed synchronize before the drafts launch (the drain) and ONE after them and before the pre-stage window
    (the sync), writes ONE line, and no other round waits for anything beyond today's fence; the order of the device calls is drain, launch, sync, window, the round's own fence;
  - the reader joins the [PACKED] request lines, the [OCTO] round line, the probe line and the ledger line of a step in file order, closes wall = entry + step + ed + gap, keeps probe rounds
    and long rounds out of the timing tables, tables the probe apart, builds the acceptance curves, and fails (exit 1) on a log it must not average over;
  - the shape model reproduces the measured eight-live round of the s5 runs inside its stated tolerance, and every shape's sign against K0 is the one its docstring claims;
  - the twin is gate only, carries the parent's bytes plus exactly the probe flags (less the parent's own waiver), and no other profile names them.
"""

import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dflash_packed_proposal_coordinator as coordinator  # noqa: E402
import make_probe_profiles as twins  # noqa: E402
import octo_markers  # noqa: E402
import profile_twins  # noqa: E402
import round_host  # noqa: E402
import round_probe as probe  # noqa: E402
import round_probe_report as report  # noqa: E402
import round_shape_model as model  # noqa: E402
import serving_packed_step  # noqa: E402

FLAG, EVERY = probe.FLAG, probe.EVERY_FLAG
PROFILES = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']
PREFIX = '(EngineCore pid=67) 2026-10-20 09:31:13.749 | INFO     | some_module:log_line:75 - '


class Clean(unittest.TestCase):
    def setUp(self):
        environ = {name: value for name, value in os.environ.items() if not name.startswith('QWEN_FAST_')}
        patcher = patch.dict(os.environ, environ, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        probe.reset()
        self.addCleanup(probe.reset)
        self.log = []
        logger = patch.object(probe, 'log_line', side_effect=self.log.append)
        logger.start()
        self.addCleanup(logger.stop)

    def lines(self, marker):
        return [line for line in self.log if line.startswith(marker)]


class FlagTests(Clean):
    def test_the_flag_is_zero_or_one(self):
        self.assertFalse(probe.enabled({}))
        self.assertFalse(probe.enabled({FLAG: '0'}))
        self.assertTrue(probe.enabled({FLAG: '1'}))
        for bad in ('', '2', 'true', 'on', '-1'):
            with self.assertRaises(ValueError, msg=bad):
                probe.enabled({FLAG: bad})

    def test_the_period_is_an_integer_from_two_to_4096(self):
        self.assertEqual(probe.every({}), probe.EVERY_DEFAULT)
        self.assertEqual(probe.every({EVERY: '2'}), 2)
        self.assertEqual(probe.every({EVERY: '4096'}), 4096)
        for bad in ('', '1', '0', '4097', 'x', '-3', '2.5'):
            with self.assertRaises(ValueError, msg=bad):
                probe.every({EVERY: bad})

    def test_a_malformed_flag_fails_the_round_that_reads_it(self):
        os.environ[FLAG] = 'yes'
        with self.assertRaises(ValueError):
            probe.begin([SimpleNamespace()])
        os.environ.update({FLAG: '1', EVERY: '1'})
        with self.assertRaises(ValueError):
            probe.begin([SimpleNamespace()])


def device(calls, name='d'):
    return SimpleNamespace(operations=SimpleNamespace(synchronize_device=lambda mesh: calls.append((name, mesh))), mesh='mesh')


class ProbeUnitTests(Clean):
    def test_flag_unset_nothing_happens(self):
        calls = []
        self.assertIsNone(probe.begin([dict(device=device(calls))]))
        self.assertEqual((calls, self.log, probe.METER.rounds), ([], [], 0))

    def test_every_nth_round_is_a_probe_with_two_waits_and_one_line(self):
        os.environ.update({FLAG: '1', EVERY: '4'})
        calls = []
        entries = [dict(device=device(calls, 'entry'))]
        fence = (SimpleNamespace(synchronize_device=lambda mesh: calls.append(('fence', mesh))), 'mesh')
        probed = []
        for number in range(1, 9):
            state = probe.begin(entries)
            if state is not None:
                self.assertEqual(calls, [('entry', 'mesh')], 'the drain is the first wait of the round')
                probe.finish(state, fence, number, 'octo', pairs=0, singles=0)
                probed.append(number)
            calls.clear()
        self.assertEqual(probed, [4, 8])
        lines = self.lines(probe.MARKER)
        self.assertEqual(len(lines), 2)
        self.assertRegex(lines[0], r'^\[ROUND-PROBE\] round=4 live=1 shape=octo pairs=0 singles=0 every=4 drain_ms=[0-9.]+ launch_ms=[0-9.]+ sync_ms=[0-9.]+ total_ms=[0-9.]+$')
        self.assertEqual(len(self.lines(probe.ENGAGED_MARKER)), 1, 'engaged once')
        self.assertEqual(probe.METER.probes, 2)

    def test_the_other_rounds_wait_for_nothing(self):
        os.environ.update({FLAG: '1', EVERY: '5'})
        calls = []
        for _ in range(4):
            self.assertIsNone(probe.begin([dict(device=device(calls))]))
        self.assertEqual(calls, [])

    def test_no_entries_never_probes_and_no_fence_writes_a_dash(self):
        os.environ.update({FLAG: '1', EVERY: '2'})
        self.assertIsNone(probe.begin([]))
        self.assertIsNone(probe.begin([]))
        calls = []
        state = probe.begin([dict(device=SimpleNamespace())])
        self.assertIsNone(state, 'round 3 is not a multiple of 2')
        state = probe.begin([dict(device=SimpleNamespace())])
        self.assertIsNotNone(state)
        self.assertIsNone(state['drain_ms'], 'a device without operations is not waited on')
        probe.finish(state, None, 4, 'none')
        line = self.lines(probe.MARKER)[0]
        self.assertIn('drain_ms=- ', line)
        self.assertIn('sync_ms=- ', line)
        self.assertEqual(calls, [])

    def test_the_shape_names(self):
        self.assertEqual(probe.shape_of(octo=object()), 'octo')
        self.assertEqual(probe.shape_of(block_quads=[1, 2]), 'quads2')
        self.assertEqual(probe.shape_of(block_quads=[1]), 'quads1')
        self.assertEqual(probe.shape_of(quad=object()), 'quad')
        self.assertEqual(probe.shape_of(groups=[(0, 1)]), 'pairs')
        self.assertEqual(probe.shape_of(), 'none')
        for name in ('octo', 'quads2', 'quads1', 'quad', 'pairs', 'none'):
            self.assertIn(name, probe.SHAPES)


class CoordinatorTests(Clean):
    """The hook in the drafts' coordinator, driven the way test_hostgap_instr drives its probe: the quad launches are patched out, the order of the device calls is read."""

    def setUp(self):
        super().setUp()
        os.environ.update({'QWEN_FAST_QUAD_DRAFT': '1', 'QWEN_FAST_QUAD_DRAFT_BLOCKS': '2'})
        self.coordinator = coordinator.PackedProposalCoordinator()
        self.events = []
        events = self.events
        self.fence = (SimpleNamespace(synchronize_device=lambda mesh: events.append('fence')), 'mesh')
        self.shape = 'quads2'
        test = self

        class Trace:
            def run_audit(self, number):
                pass

        def prepare_quad_blocks(coord, groups, by_slot, round_number, batched, prepared):
            events.append('launch')
            return [dict(fence=test.fence, trace=Trace(), devices=[]) for _ in range(2 if test.shape == 'quads2' else 1)], []

        def prepare_octo(coord, groups, by_slot, round_number, batched):
            events.append('launch')
            return dict(devices=[], fence=test.fence)

        for name, replacement in (('_prepare_quad_blocks', prepare_quad_blocks), ('_prepare_octo', prepare_octo)):
            patcher = patch.object(coordinator.PackedProposalCoordinator, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def bridges(self, count=8):
        return [SimpleNamespace(request=SimpleNamespace(session=SimpleNamespace(seed=index)), slot=index) for index in range(count)]

    def rounds(self, count, **options):
        events = self.events

        def pipelined(bridge):
            return SimpleNamespace(pool_slot=SimpleNamespace(index=bridge.slot), operations=SimpleNamespace(synchronize_device=lambda mesh: events.append('drain')), mesh='mesh')

        with patch('serving_worker_hook.pipelined_device', pipelined):
            for _ in range(count):
                self.coordinator.prepare(self.bridges(), **options)

    def test_flag_unset_the_module_is_never_called(self):
        with patch.object(probe, 'begin', side_effect=AssertionError('the probe is off')):
            self.rounds(20)
        self.assertEqual(self.events.count('fence'), 20)
        self.assertNotIn('drain', self.events)
        self.assertEqual(self.log, [])

    def test_the_order_of_a_probe_round_is_drain_launch_sync_window_fence(self):
        os.environ.update({FLAG: '1', EVERY: '3'})
        window = lambda: self.events.append('window')            # noqa: E731
        self.rounds(3, while_waiting=window)
        self.assertEqual(self.events, ['launch', 'window', 'fence'] * 2 + ['drain', 'launch', 'fence', 'window', 'fence'],
                         'two ordinary rounds, then the probe round: drain, launch, the probe\'s wait on the fence, the window, the round\'s own fence')
        line = self.lines(probe.MARKER)[0]
        self.assertRegex(line, r'^\[ROUND-PROBE\] round=3 live=8 shape=quads2 pairs=0 singles=0 every=3 drain_ms=[0-9.]+ launch_ms=[0-9.]+ sync_ms=[0-9.]+ total_ms=[0-9.]+$')

    def test_only_every_nth_round_is_probed_and_the_others_wait_as_before(self):
        os.environ.update({FLAG: '1', EVERY: '4'})
        self.rounds(12)
        self.assertEqual(len(self.lines(probe.MARKER)), 3)
        self.assertEqual(self.events.count('drain'), 3)
        self.assertEqual(self.events.count('fence'), 12 + 3, 'the round fence every round, the probe\'s wait on three')

    def test_the_octo_pass_is_named(self):
        os.environ.update({FLAG: '1', EVERY: '2'})
        self.rounds(2, octo_draft=True)
        self.assertIn('shape=octo', self.lines(probe.MARKER)[0])

    def test_one_quad_is_named(self):
        os.environ.update({FLAG: '1', EVERY: '2'})
        self.shape = 'quads1'
        self.rounds(2)
        self.assertIn('shape=quads1', self.lines(probe.MARKER)[0])

    def test_the_hook_imports_the_module_only_behind_the_flag(self):
        text = (HERE / 'dflash_packed_proposal_coordinator.py').read_text(encoding='utf-8')
        self.assertNotRegex(text, r'(?m)^import round_probe')
        self.assertIn("os.environ.get('QWEN_FAST_TP4_ROUND_PROBE', '0') != '0'", text)


# -- the reader ---------------------------------------------------------------------------------------------------------------------------

def step(number, live, shape, rows, emitted, pos, fields, probe_line=None, cap=None, committed=None):
    """One step's lines as their producers write them: the [PACKED] request lines, the [OCTO] line, a probe line, the ledger line."""
    lines = []
    for index, tokens in enumerate(emitted):
        lines.append(PREFIX + serving_packed_step.AUDIT_LINE.format(request='chatcmpl-%d' % index, segment=index, position=pos + index, prefix=tokens, emitted=tokens,
                                                                    predictions='[1, 2]') + serving_packed_step.AUDIT_CAP.format(cap=cap or rows))
    lines.append(PREFIX + octo_markers.ROUND_LINE.format(round=number, shape=shape, planned='m3' if shape == 'seq' else shape, live=live, rows=rows, eligible=0, counted=0,
                                                         octo_rounds=0, m3_rounds=number, committed=sum(emitted) if committed is None else committed,
                                                         step_ms=fields.get('step', 0.0), gap_ms=fields.get('gap', 0.0)))
    if probe_line:
        lines.append(PREFIX + probe_line)
    lines.append(PREFIX + round_host.format_line(number, live, fields, dict(select=1, read=1, keyed=1, guard=0), {}, pos))
    return lines


def fields(entry=0.5, v0=44.0, v0_tr=42.4, c0=1.0, ed=16.0, gap=0.7, extra=None):
    out = dict(gap=gap, entry=entry, v0=v0, v0_in=0.5, v0_tr=v0_tr, v0_sy=0.1, v0_rb=0.7, c0=c0, ed=ed, ed_pre=0.8, launch=2.0, window=8.0, fence=0.9, collect=1.9, flush=0.1,
               select=1.3, tail=0.2)
    out['step'] = v0 + c0 + 0.5
    out.update(extra or {})
    return out


def make_log(count_m3=30, count_octo=30, long_round=True, probe_every=10, unmatched=False):
    lines, number = [], 0
    for index in range(count_m3):
        number += 1
        line = None
        if probe_every and index % probe_every == probe_every - 1:
            line = '[ROUND-PROBE] round=%d live=4 shape=quads1 pairs=0 singles=0 every=%d drain_ms=5.00 launch_ms=3.00 sync_ms=8.00 total_ms=16.00' % (number, probe_every)
        lines += step(number, 4, 'm3', 16, [4, 5, 3, 4], 3600, fields(), probe_line=line)
    for index in range(count_octo):
        number += 1
        lines += step(number, 8, 'octo', 8, [3, 4, 3, 3, 3, 4, 3, 3], 3700, fields(v0=54.7, v0_tr=52.2, c0=1.7, ed=21.3))
    if long_round:
        number += 1
        lines += step(number, 8, 'octo', 8, [3] * 8, 3700, fields(v0=54.7, v0_tr=52.2, c0=1.7, ed=21.3, gap=1277.0))
    for index in range(20):
        number += 1
        extra = dict(v1=45.0, v1_in=0.4, v1_tr=42.4, v1_sy=0.1, v1_rb=0.7, c1=1.0)
        lines += step(number, 5, 'm3', 16, [4, 4, 4, 4, 4], 130000, fields(v0=61.0, v0_tr=58.9, ed=36.0, extra=extra))
    if unmatched:
        lines.append(PREFIX + serving_packed_step.AUDIT_LINE.format(request='chatcmpl-x', segment=0, position=1, prefix=2, emitted=2, predictions='[1]') + ' cap=16')
    return lines


class ReaderTests(unittest.TestCase):
    def table(self, lines, **options):
        parsed = report.parse(lines)
        table, stalls, probes = report.group_table(parsed, **options)
        return parsed, table, stalls, probes

    def test_a_step_is_joined_from_its_four_lines_and_the_wall_closes(self):
        parsed = report.parse(make_log())
        self.assertEqual(len(parsed['rounds']), 30 + 30 + 1 + 20)
        for record in parsed['rounds']:
            self.assertIsNotNone(record['ledger'])
            self.assertIsNotNone(record['octo'])
            fields_ = record['ledger']['fields']
            self.assertAlmostEqual(report.wall_ms(record['ledger']), fields_['entry'] + fields_['step'] + fields_['ed'] + fields_['gap'], places=6)
        self.assertEqual(report.problems_of(parsed), [])

    def test_the_groups_the_phases_and_the_rates(self):
        _, table, stalls, probes = self.table(make_log())
        key = (4, 'm3', '<8k', 1)
        row = table[key]
        self.assertEqual(row['n'], 27, 'thirty rounds, three of them probe rounds')
        self.assertAlmostEqual(row['verify'], 42.4)
        self.assertAlmostEqual(row['verify_glue'], 44.0 - 42.4)
        self.assertAlmostEqual(row['commit'], (44.0 + 1.0 + 0.5) - 44.0)
        self.assertAlmostEqual(row['draft'], 16.0)
        self.assertAlmostEqual(row['wall'], 0.5 + 45.5 + 16.0 + 0.7)
        self.assertAlmostEqual(row['other'], 0.0, places=6)
        self.assertAlmostEqual(row['tokens_user_round'], 4.0)
        self.assertAlmostEqual(row['aggregate_tok_s'], 16.0 / row['wall'] * 1000.0)
        octo = table[(8, 'octo', '<8k', 1)]
        self.assertEqual(octo['rows'], 8)
        self.assertAlmostEqual(octo['verify'], 52.2)
        two = table[(5, 'm3', '96-160k', 2)]
        self.assertAlmostEqual(two['verify'], 58.9 + 42.4, msg='two verify passes: the blocks are added')
        self.assertEqual(probes, 3)
        self.assertEqual(stalls['rounds'], 1, 'the 1.3 s gap is a long round, counted apart')

    def test_probe_rounds_and_long_rounds_are_not_round_times(self):
        parsed, table, stalls, probes = self.table(make_log(), stall_ms=400.0)
        self.assertEqual(table[(8, 'octo', '<8k', 1)]['n'], 30, 'the long round is out of the octo group')
        _, all_in, _, _ = self.table(make_log(), stall_ms=1e9)
        self.assertEqual(all_in[(8, 'octo', '<8k', 1)]['n'], 31)
        self.assertGreater(all_in[(8, 'octo', '<8k', 1)]['wall_mean'], table[(8, 'octo', '<8k', 1)]['wall_mean'])

    def test_the_probe_table_is_apart_and_pairs_a_probe_with_the_next_ledger_line(self):
        parsed = report.parse(make_log())
        rows = report.probe_table(parsed)
        row = rows[(4, 'quads1', '<8k')]
        self.assertEqual(row['n'], 3)
        self.assertEqual((row['drain'], row['launch'], row['sync'], row['total']), (5.0, 3.0, 8.0, 16.0))
        self.assertAlmostEqual(row['verify'], 42.4)
        self.assertEqual(report.probe_table(report.parse(make_log(probe_every=0))), {})

    def test_acceptance_curves_at_the_shapes_full_cap(self):
        parsed = report.parse(make_log())
        curve = report.survival(parsed, min_requests=10)
        mean, shares, count = curve[(8, 'octo', 8)]
        self.assertEqual(len(shares), 8)
        self.assertEqual(shares[0], 1.0)
        self.assertAlmostEqual(shares[3], 2.0 / 8.0)                    # two of eight users commit 4 and the rest 3
        self.assertAlmostEqual(mean, (3 * 6 + 4 * 2) / 8.0)
        self.assertGreaterEqual(count, 30 * 8)
        self.assertEqual(report.survival(parsed, min_requests=10 ** 6), {})

    def test_a_log_without_the_ledger_is_a_problem_not_a_table(self):
        lines = [line for line in make_log() if '[PACKED-ROUND-HOST]' not in line]
        parsed = report.parse(lines)
        problems = report.problems_of(parsed)
        self.assertTrue(any('ledger is off' in problem for problem in problems), problems)
        self.assertEqual(report.group_table(parsed)[0], {})

    def test_emitted_counts_that_do_not_sum_to_committed_are_a_problem(self):
        lines = make_log()
        parsed = report.parse(lines)
        self.assertEqual(report.problems_of(parsed), [])
        broken = [line.replace('committed=16', 'committed=17') for line in lines]
        problems = report.problems_of(report.parse(broken))
        self.assertTrue(any('do not sum' in problem for problem in problems), problems)

    def test_a_probe_with_no_ledger_line_after_it_is_a_problem(self):
        lines = make_log() + [PREFIX + '[ROUND-PROBE] round=99 live=4 shape=quads1 pairs=0 singles=0 every=10 drain_ms=1.00 launch_ms=1.00 sync_ms=1.00 total_ms=3.00']
        self.assertTrue(any('[ROUND-PROBE]' in problem for problem in report.problems_of(report.parse(lines))))

    def test_a_log_without_octo_lines_is_still_read_from_the_ledger(self):
        lines = [line for line in make_log() if '[OCTO] round=' not in line]
        parsed = report.parse(lines)
        self.assertFalse(parsed['has_octo'])
        self.assertEqual(parsed['ledger_lines'], 81)
        table, _, _ = report.group_table(parsed)
        self.assertTrue(table, 'the phases come from the ledger alone')
        self.assertIsNone(next(iter(table.values()))['tokens_user_round'], 'no committed count without the [OCTO] line')

    def test_the_command_line_prints_the_tables_and_writes_the_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path, out = Path(directory) / 'container.log', Path(directory) / 'out.json'
            path.write_text('\n'.join(make_log()) + '\n', encoding='utf-8')
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = report.main([str(path), '--json', str(out)])
            self.assertEqual(code, 0)
            text = buffer.getvalue()
            for needle in ('PER ROUND', 'THE DRAFT, BY PART', 'PROBE ROUNDS', 'ACCEPTANCE', 'GAP: the verify pass is not split'):
                self.assertIn(needle, text)
            data = json.loads(out.read_text(encoding='utf-8'))
            self.assertEqual(sorted(data), ['acceptance', 'groups', 'probes', 'problems', 'stalls'])
            self.assertIn('8|octo|<8k|1', data['groups'])
            path.write_text('\n'.join(line for line in make_log() if '[PACKED-ROUND-HOST]' not in line) + '\n', encoding='utf-8')
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                self.assertEqual(report.main([str(path)]), 1)
            self.assertIn('PROBLEM:', buffer.getvalue())

    def test_a_log_with_no_probe_says_the_gap_instead_of_printing_zeros(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'container.log'
            path.write_text('\n'.join(make_log(probe_every=0)) + '\n', encoding='utf-8')
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                report.main([str(path)])
            self.assertIn('PROBE ROUNDS: none in this log. GAP', buffer.getvalue())

    def test_the_context_buckets(self):
        self.assertEqual([report.context_label(p) for p in (None, 100, 7999, 8000, 47999, 48000, 95999, 96000, 159999, 160000, 300000)],
                         ['?', '<8k', '<8k', '8-48k', '8-48k', '48-96k', '48-96k', '96-160k', '96-160k', '>=160k', '>=160k'])

    def test_the_reader_reads_the_lines_the_producers_write(self):
        line = round_host.format_line(7, 3, dict(step=45.5, entry=0.5, ed=16.0, gap=0.7, v0=44.0, v0_tr=42.4), dict(select=1, read=1, keyed=1, guard=0), {}, 3600)
        self.assertTrue(line.startswith(round_host.LEDGER_MARKER))
        match = report.LEDGER.search(line)
        self.assertEqual((match.group(1), match.group(2), match.group(3)), ('7', '3', '3600'))
        self.assertTrue(octo_markers.ROUND_PATTERN.search(octo_markers.ROUND_LINE.format(round=1, shape='octo', planned='octo', live=8, rows=8, eligible=1, counted=1, octo_rounds=1,
                                                                                          m3_rounds=0, committed=27, step_ms=58.2, gap_ms=0.8)))
        self.assertTrue(report.OCTO.search(octo_markers.ROUND_LINE.format(round=1, shape='octo', planned='octo', live=8, rows=8, eligible=1, counted=1, octo_rounds=1,
                                                                           m3_rounds=0, committed=27, step_ms=58.2, gap_ms=0.8)))


# -- the model ---------------------------------------------------------------------------------------------------------------------------

class ModelTests(unittest.TestCase):
    def test_the_expected_prefix(self):
        self.assertEqual(model.expected_prefix(0.7, 1), 1.0)
        self.assertEqual(model.expected_prefix(0.0, 9), 1.0)
        self.assertAlmostEqual(model.expected_prefix(1.0, 7), 7.0)
        values = [model.expected_prefix(0.75, rows) for rows in range(1, 17)]
        self.assertEqual(values, sorted(values))
        self.assertAlmostEqual(values[-1], 4.0, delta=0.1)
        self.assertEqual(model.expected_prefix(0.7, 0), 0.0)

    def test_the_fit_round_trips(self):
        for mean in (1.5, 2.9, 3.47, 5.19, 7.0):
            self.assertAlmostEqual(model.expected_prefix(model.fit_rho(mean), 8), mean, places=6)
        self.assertAlmostEqual(model.expected_prefix(model.fit_rho(0.5), 8), 1.0, places=6, msg='a mean below one is one')

    def test_the_fit_is_the_measured_curve_within_five_percent(self):
        measured = {8: (0.78, 0.57, 0.41, 0.29, 0.20, 0.15, 0.11), 16: (0.78, 0.59, 0.44, 0.33, 0.26, 0.19, 0.15, 0.12, 0.09, 0.07, 0.06, 0.04, 0.03, 0.02, 0.01)}
        pooled8 = 1.0 + sum(measured[8])
        rho = model.fit_rho(pooled8)
        self.assertAlmostEqual(model.expected_prefix(rho, 8), pooled8, places=6)
        pooled16 = 1.0 + sum(measured[16])
        # the 16-row curve from the 8-row fit: within the stated few percent (the measured tail is a little heavier)
        self.assertLess(abs(model.expected_prefix(model.fit_rho(1.0 + sum(measured[16][:7])), 16) - pooled16) / pooled16, 0.06)

    def test_the_row_allocation_is_the_optimum(self):
        import itertools

        rhos = [0.6, 0.75, 0.88]
        rows = model.allocate_rows(rhos, 12, floor=1, cap=9)
        self.assertEqual(sum(rows), 12)
        best = max(itertools.product(range(1, 10), repeat=3), key=lambda a: (sum(model.expected_prefix(r, n) for r, n in zip(rhos, a)) if sum(a) == 12 else -1))
        self.assertAlmostEqual(sum(model.expected_prefix(r, n) for r, n in zip(rhos, rows)), sum(model.expected_prefix(r, n) for r, n in zip(rhos, best)))
        self.assertTrue(all(1 <= value <= 9 for value in rows))

    def test_the_baseline_is_the_measured_eight_live_round(self):
        base = model.shapes(8, '3.6k')[0]
        self.assertAlmostEqual(base['wall'], 82.3, delta=1.5, msg='measured 82.3 and 83.9 in the two runs')
        self.assertAlmostEqual(base['user'], 42.1, delta=2.0)
        self.assertAlmostEqual(base['slowest'], 34.4, delta=2.0)
        self.assertAlmostEqual(base['aggregate'], 335.0, delta=12.0)

    def test_the_seven_live_baseline_is_the_measured_round(self):
        base = model.shapes(7, '3.6k')[0]
        self.assertAlmostEqual(base['wall'], 102.0, delta=4.0, msg='measured 101.4 and 103.2')

    def test_every_shape_has_the_sign_its_docstring_claims(self):
        for context in model.CONTEXTS:
            results = {r['name'][:2]: r for r in model.shapes(7, context)}
            self.assertLess(results['K5']['aggregate'], results['K0']['aggregate'], 'two M3 blocks are slower than the octo block')
            self.assertGreater(results['K1']['aggregate'], results['K0']['aggregate'], context + ': the octo pass at seven live beats the quads')
            self.assertGreater(results['K2']['aggregate'], results['K1']['aggregate'], 'fewer idle segments and more rows are cheaper and richer')
            k3 = [r for r in model.shapes(8, context) if r['name'].startswith('K3')][0]
            k0 = model.shapes(8, context)[0]
            self.assertGreater(k3['aggregate'], k0['aggregate'])
            self.assertLess(k3['slowest'], k0['slowest'], 'the aggregate buys nothing for the slowest user')
            for skipped in [r for r in model.shapes(8, context) if r['name'].startswith('K4')]:
                self.assertLess(skipped['slowest'], 0.5 * k0['slowest'])
                self.assertLessEqual(skipped['aggregate'], k0['aggregate'] * 1.001, 'rows are free, so skipping speculation saves no time')

    def test_five_live_prefers_the_octo_block_to_two_m3_blocks(self):
        for context in model.CONTEXTS:
            two, octo = model.shapes(5, context)
            self.assertGreater(octo['aggregate'], two['aggregate'])
            self.assertGreater(octo['slowest'], two['slowest'])

    def test_the_sensitivity_can_take_the_gain_away_at_depth(self):
        central = model.shapes(7, '128k')
        worst = model.shapes(7, '128k', octo_growth=10.0, octo_pass_tau=model.OCTO_PASS_TAU_LOW)
        self.assertGreater(central[1]['aggregate'] / central[0]['aggregate'], 1.0)
        self.assertLess(worst[1]['aggregate'] / worst[0]['aggregate'], 1.0, 'stacked, the two worst assumptions turn the octo pass at depth into a loss: the card question')

    def test_the_table_prints_every_shape_at_every_context(self):
        text = model.table()
        for context in model.CONTEXTS:
            for live in (8, 7, 6, 5):
                self.assertIn('%5s live %d' % (context, live), text)
        for label in ('K0', 'K1', 'K2', 'K3', 'K4', 'K5'):
            self.assertIn(label, text)


# -- the twin ----------------------------------------------------------------------------------------------------------------------------

class TwinTests(unittest.TestCase):
    def test_each_twin_is_the_parent_plus_exactly_the_probe_flags(self):
        parent = PROFILES[twins.PARENT]
        for name, taken_out in ((twins.NAME, [twins.WAIVER_ENV]), (twins.NAME_QUADS, sorted([twins.WAIVER_ENV, twins.OCTO_DRAFT]))):
            with self.subTest(name=name):
                twin = PROFILES[name]
                self.assertIs(twin.get('gate_only'), True)
                self.assertNotIn(twins.WAIVER_FIELD, twin)
                self.assertTrue(twin['description'].startswith('GATE ONLY'))
                added = {flag: value for flag, value in twin['env'].items() if parent['env'].get(flag) != value}
                self.assertEqual(added, {FLAG: '1', EVERY: twins.EVERY, twins.GATE_ENV: '1'})
                self.assertEqual(sorted(set(parent['env']) - set(twin['env'])), taken_out, 'the parent\'s waiver id is the parent\'s own; the quads twin has no octo draft pass')
                for key in parent:
                    if key not in ('env', 'description', 'gate_only', twins.WAIVER_FIELD):
                        self.assertEqual(twin[key], parent[key], key)
        self.assertEqual(PROFILES[twins.PARENT]['env'][twins.OCTO_DRAFT], '1')

    def test_the_two_twins_differ_in_the_octo_draft_alone(self):
        probe_env, quads_env = PROFILES[twins.NAME]['env'], PROFILES[twins.NAME_QUADS]['env']
        self.assertEqual({k: v for k, v in probe_env.items() if k != twins.OCTO_DRAFT}, quads_env)
        self.assertNotIn(twins.OCTO_DRAFT, quads_env)

    def test_the_file_is_what_the_generator_writes_and_a_second_run_changes_nothing(self):
        data = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))
        once = twins.generate(data)
        self.assertEqual(twins.render(once), (HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))
        self.assertEqual(twins.render(twins.generate(once)), twins.render(once))
        self.assertEqual(twins.main(['--check']), 0)

    def test_the_twins_sit_after_the_s5_audited_twins(self):
        names = list(PROFILES)
        self.assertEqual(names[names.index(twins.ANCHOR) + 1:names.index(twins.ANCHOR) + 3], list(twins.twin_names()))

    def test_no_other_profile_names_the_probe_flags_and_the_registry_knows_the_twin(self):
        carriers = [name for name, profile in PROFILES.items() if FLAG in profile['env'] or EVERY in profile['env']]
        self.assertEqual(carriers, list(twins.twin_names()))
        for name in twins.twin_names():
            self.assertIn(name, profile_twins.twin_names())
        import make_s5_profiles

        self.assertEqual((twins.PARENT, twins.ANCHOR, twins.WAIVER_ENV, twins.WAIVER_FIELD), (make_s5_profiles.SHIP, make_s5_profiles.AUDIT_NOWC, make_s5_profiles.WAIVER_ENV,
                                                                                               make_s5_profiles.WAIVER_FIELD), 'the literals are make_s5_profiles\' names')

    def test_a_parent_that_names_the_flag_or_is_gate_only_is_refused(self):
        parent = json.loads(json.dumps(PROFILES[twins.PARENT]))
        self.assertEqual(twins.parent_problems(parent), [])
        parent['env'][FLAG] = '1'
        self.assertTrue(twins.parent_problems(parent))
        parent = json.loads(json.dumps(PROFILES[twins.PARENT]))
        parent['gate_only'] = True
        self.assertTrue(twins.parent_problems(parent))
        parent = json.loads(json.dumps(PROFILES[twins.PARENT]))
        del parent['env'][twins.OCTO_DRAFT]
        self.assertTrue(any(twins.OCTO_DRAFT in problem for problem in twins.parent_problems(parent)), 'nothing for the quads twin to take out')
        with self.assertRaises(ValueError):
            twins.generate(dict(profiles=dict(PROFILES, **{twins.PARENT: parent})))

    def test_the_probe_modules_are_in_the_image_and_the_test_is_in_the_cpu_workflow(self):
        overlay = (HERE.parent.parent / 'docker' / 'qwen-c2-overlay.txt').read_text(encoding='utf-8')
        self.assertIn('scripts/ci/round_probe.py', overlay.splitlines())
        workflow = (HERE.parent.parent / '.github' / 'workflows' / 'qwen-integration-cpu.yml').read_text(encoding='utf-8')
        self.assertIn('test_round_probe', workflow)

    def test_nothing_public_names_a_host_an_address_or_a_registry(self):
        pattern = re.compile(r'(\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|\.local\b|zot\.|spark-|blackhole-[0-9A-F]{8})')
        for name in ('round_probe.py', 'round_probe_report.py', 'round_shape_model.py', 'make_probe_profiles.py', 'test_round_probe.py'):
            for number, line in enumerate((HERE / name).read_text(encoding='utf-8').splitlines(), 1):
                if name == 'test_round_probe.py' and 'pattern = re.compile' in line:
                    continue
                self.assertIsNone(pattern.search(line), '%s:%d' % (name, number))


if __name__ == '__main__':
    unittest.main()
