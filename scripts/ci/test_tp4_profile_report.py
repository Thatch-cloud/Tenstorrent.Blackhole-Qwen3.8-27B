"""tp4_profile_report on synthetic CSVs (their layout is tt-metal's cpp_device_perf_report.csv), and its positive control:
a slim fixture of TP2's v138 trace, whose table the research file recorded (references/tp4-profile/README.txt)."""
import csv
import gzip
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

import tp4_profile_report as report

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, 'references', 'tp4-profile', 'v138-trace0-slim.csv.gz')
COLUMNS = ['METAL TRACE ID', 'METAL TRACE REPLAY SESSION ID', 'DEVICE ID', 'OP NAME', 'CORE COUNT',
           'DEVICE KERNEL DURATION [ns]', 'DEVICE KERNEL FIRST TO LAST START [ns]', 'DEVICE FW START CYCLE',
           'DEVICE KERNEL START CYCLE', 'DEVICE KERNEL END CYCLE']
GAP_NS = 1000.0


def layer_ops(index, users, sdpa_ns, chip_skew=0.0):
    """One decoder layer's ops as (name, cores, duration ns); attention every fourth layer, as the model is."""
    attn = index % 4 == 3
    ops = [('LayerNormDeviceOperation', 8, 8000), ('AllGatherAsync', 20, 18000), ('MatmulDeviceOperation', 56, 120000)]
    if attn:
        ops.append(('AttnPrepDeviceOperation', 8, 125000))
        ops += [('SdpaDecodeDeviceOperation', 32, sdpa_ns(user)) for user in range(users)]
        ops.append(('ConcatHeadsDeviceOperation', 8, 10000))
    else:
        ops += [('GdnConvGatesDeviceOperation', 4, 35000) for _ in range(users)]
        ops += [('GenericOpDeviceOperation', 4, 18000), ('GenericOpDeviceOperation', 4, 18000),
                ('GenericOpDeviceOperation', 96, 500000)]
    ops += [('MatmulDeviceOperation', 32, 48000), ('ReduceScatterMinimalAsync', 20, 17000 + chip_skew),
            ('BinaryNgDeviceOperation', 32, 4000)]
    ops += [('LayerNormDeviceOperation', 8, 8000), ('AllGatherAsync', 20, 18000)]
    ops += [('MatmulDeviceOperation', 39, 108000), ('MatmulDeviceOperation', 39, 90000),
            ('BinaryNgDeviceOperation', 32, 5000), ('MatmulDeviceOperation', 32, 128000),
            ('ReduceScatterMinimalAsync', 20, 17000 + chip_skew), ('BinaryNgDeviceOperation', 32, 4000)]
    return ops


def replay_ops(users=4, layers=64, sdpa_ns=lambda user: 276000, skew_last_chip=0.0, device=0):
    ops = [('EmbeddingsDeviceOperation', 8, 5000)]
    for index in range(layers):
        ops += layer_ops(index, users, sdpa_ns, skew_last_chip if device else 0.0)
    ops += [('LayerNormDeviceOperation', 8, 8000), ('MatmulDeviceOperation', 108, 1870000),
            ('GenericOpDeviceOperation', 8, 1300000)]
    return ops


class Csv(object):
    """A synthetic cpp_device_perf_report.csv."""

    def __init__(self, chips=4):
        self.chips, self.rows, self.clock = chips, [], collections_clock()

    def add(self, trace, session, ops_by_device, keep=None):
        for device, ops in sorted(ops_by_device.items()):
            cycle = self.clock[device]
            for position, (name, cores, ns) in enumerate(ops):
                if keep is not None and position >= keep:
                    break
                start = cycle + GAP_NS * 1.35
                end = start + ns * 1.35
                self.rows.append({'METAL TRACE ID': trace, 'METAL TRACE REPLAY SESSION ID': session,
                                  'DEVICE ID': str(device), 'OP NAME': name, 'CORE COUNT': cores,
                                  'DEVICE KERNEL DURATION [ns]': ns, 'DEVICE KERNEL FIRST TO LAST START [ns]': 100,
                                  'DEVICE FW START CYCLE': start - 5, 'DEVICE KERNEL START CYCLE': start,
                                  'DEVICE KERNEL END CYCLE': end})
                cycle = end
            self.clock[device] = cycle + 50e6      # the rest of the round: host fences, drafts, commits

    def text(self):
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(self.rows)
        return out.getvalue()

    def write(self, directory, name='cpp_device_perf_report.csv.gz'):
        path = os.path.join(directory, name)
        opener = gzip.open if name.endswith('.gz') else open
        with opener(path, 'wt', newline='', encoding='utf-8') as handle:
            handle.write(self.text())
        return path


def collections_clock():
    import collections
    return collections.defaultdict(float)


def build(chips=4, sessions=('1', '2', '3', '4', '5', '6', '7', '8', '9'), **kwargs):
    data = Csv(chips)
    for session in sessions:
        data.add('0', session, dict((d, replay_ops(device=d, **kwargs)) for d in range(chips)))
    return data


def analyse(data, chips=4, **kwargs):
    sessions, every, columns = report.load(write_temp(data))
    return report.analyse_sessions(sessions, every, columns, chips=chips, **kwargs)


def write_temp(data):
    directory = tempfile.mkdtemp()
    return data.write(directory, 'cpp.csv.gz')


class ClassificationTests(unittest.TestCase):
    def test_a_64_layer_replay_is_a_packed_verify_with_its_users(self):
        result = analyse(build(sessions=('1', '2')))
        listing = result['traces'][0]
        self.assertEqual((listing['kind'], listing['detail']['users'], listing['detail']['layers']),
                         ('verify-packed', 4, 64))

    def test_one_sdpa_launch_per_attention_layer_is_a_lone_user_step(self):
        result = analyse(build(users=1, sessions=('1', '2')))
        self.assertEqual(result['traces'][0]['kind'], 'verify-single')
        self.assertIsNone(result['verify_packed'])
        self.assertEqual(result['verify_lone']['attn_layers'], 16)

    def test_layers_are_typed_by_the_conv_gates_not_by_core_counts(self):
        packed = analyse(build(sessions=('1', '2')))['verify_packed']
        self.assertEqual((packed['gdn_layers'], packed['attn_layers']), (48, 16))
        cats = packed['categories']
        self.assertAlmostEqual(cats['gdn.recurrence']['ms'][0], 48 * 0.5, places=3)
        self.assertAlmostEqual(cats['attn.sdpa']['ms'][0], 16 * 4 * 0.276, places=3)
        self.assertAlmostEqual(cats['gdn.conv_gates']['ms'][0], 48 * 4 * 0.035, places=3)
        self.assertEqual(packed['conv_gates_per_gdn_layer'], 4)

    def test_the_recurrence_is_the_longest_generic_after_the_conv_gates_whatever_its_cores(self):
        # A TP4-style recurrence on 48 cores, and a longer generic BEFORE the conv gates that is not the recurrence.
        ops = [(n, 48 if c == 96 else c, ns) for n, c, ns in layer_ops(0, 4, lambda user: 1)]
        ops.insert(3, ('GenericOpDeviceOperation', 4, 900000))
        replay = [('EmbeddingsDeviceOperation', 8, 5000)] + ops * 3 + [('LayerNormDeviceOperation', 8, 8000)]
        roles, layers, types = report.classify([report.Op(n.replace('DeviceOperation', ''), c, ns, i, i, i, 0)
                                                for i, (n, c, ns) in enumerate(replay)])
        self.assertEqual((layers, types), (3, ['gdn'] * 3))
        recurrences = [i for i, role in enumerate(roles) if role[2] == 'gdn.recurrence']
        self.assertEqual(len(recurrences), 3)
        self.assertTrue(all(replay[i][2] == 500000 and replay[i][1] == 48 for i in recurrences))

    def test_the_norm_free_traces_are_not_verifies(self):
        data = Csv(4)
        for session in ('1', '2'):
            data.add('7', session, dict((d, [('TopKDeviceOperation', 8, 1000), ('SdpaDecodeDeviceOperation', 8, 900)]
                                          + [('MatmulDeviceOperation', 8, 500)] * 30) for d in range(4)))
        result = analyse(data)
        self.assertEqual(result['traces'][0]['kind'], 'drafter')


class CompletenessTests(unittest.TestCase):
    def test_a_truncated_session_and_a_session_missing_a_chip_are_dropped(self):
        data = build(sessions=('1', '2', '3'))
        data.add('0', '4', dict((d, replay_ops(device=d)) for d in range(4)), keep=800)         # truncated on every chip
        data.add('0', '5', dict((d, replay_ops(device=d)) for d in (0, 1, 2)))                   # chip 3 missing
        result = analyse(data)
        packed = result['verify_packed']
        self.assertEqual(packed['complete_sessions'], 3)
        self.assertEqual(packed['sessions_total'], 5)
        self.assertEqual(sorted(packed['per_session']), ['1', '2', '3'])

    def test_fewer_chips_than_asked_is_a_problem(self):
        result = analyse(build(chips=2, sessions=('1', '2')), chips=4)
        self.assertFalse(result['validity']['ok'])
        self.assertIn('2 chips', result['validity']['problems'][0])

    def test_fewer_than_eight_sessions_is_a_note_not_a_failure(self):
        result = analyse(build(sessions=('1', '2', '3')))
        self.assertTrue(result['validity']['ok'])
        self.assertTrue(any('fewer than the 8' in note for note in result['validity']['notes']))

    def test_eight_sessions_need_no_note_about_sessions(self):
        result = analyse(build(sessions=tuple(str(i) for i in range(1, 10))), log_text=None)
        self.assertFalse(any('verify-64 sessions' in note for note in result['validity']['notes']))


class MeasurementTests(unittest.TestCase):
    def test_kernel_sum_span_gaps_and_the_critical_path(self):
        packed = analyse(build(sessions=('1', '2', '3')))['verify_packed']
        kernel = packed['kernel_sum_ms'][0]
        self.assertAlmostEqual(packed['span_ms'][0] - kernel, packed['gap_ms'][0], places=2)
        self.assertGreater(packed['gap_ms'][0], 0.5)
        self.assertAlmostEqual(packed['critical_path_ms'], kernel, places=2)     # identical chips: no skew
        self.assertEqual(packed['collective_skew_ms'], 0.0)

    def test_a_slow_chip_shows_as_collective_skew_and_lengthens_the_critical_path(self):
        packed = analyse(build(sessions=('1', '2', '3'), skew_last_chip=5000.0))['verify_packed']
        self.assertAlmostEqual(packed['collective_skew_ms'], 128 * 0.005, places=3)      # 128 reduce-scatters, 5 us each
        self.assertGreater(packed['critical_path_ms'], packed['kernel_sum_ms'][0] + 0.3)
        self.assertGreater(packed['collectives']['ReduceScatterMinimalAsync']['skew_us'], 4.0)

    def test_collectives_per_call(self):
        packed = analyse(build(sessions=('1', '2')))['verify_packed']
        gather = packed['collectives']['AllGatherAsync']
        self.assertEqual(gather['calls_per_replay'], 128)
        self.assertAlmostEqual(gather['us_min_over_chips'], 18.0, places=1)

    def test_weights_rate_and_ns_per_tile(self):
        packed = analyse(build(sessions=('1', '2')))['verify_packed']
        gate = packed['weights']['mm.mlp.gate']
        self.assertEqual((gate['dtype'], gate['cores']), ('bf4', 39))
        self.assertAlmostEqual(gate['us'], 108.0, places=1)
        self.assertAlmostEqual(gate['gbps'], gate['mbytes'] * 1e6 / 108000.0, places=0)
        self.assertEqual(set(packed['weights']), {'mm.mlp.gate', 'mm.mlp.up', 'mm.mlp.down', 'mm.gdn_in', 'mm.gdn_out',
                                                  'mm.attn_in', 'mm.attn_out', 'mm.lm_head'})

    def test_groups_add_to_the_span(self):
        packed = analyse(build(sessions=('1', '2')))['verify_packed']
        self.assertAlmostEqual(sum(packed['groups'].values()), packed['span_ms'][0], places=2)


def log_of(rounds):
    """A server log: a [PACKED-PHASE] line and its [PACKED] lines per round; rounds = [(round, live, idle, trace_ms,
    {segment: position})]."""
    lines = []
    for number, live, idle, trace_ms, positions in rounds:
        lines.append('[PACKED-PHASE] round=%d users=4 bind_ms=0.10 input_ms=2.0 trace_ms=%.2f sync_ms=0.8 readback_ms=1.0 '
                     'live=%d idle=%s' % (number, trace_ms, live, ','.join(str(i) for i in idle) or '-'))
        for segment, position in sorted(positions.items()):
            lines.append('[PACKED] request=r%d segment=%d position=%d prefix=3 emitted=4' % (segment, segment, position))
    return '\n'.join(lines) + '\n'


class RoundTests(unittest.TestCase):
    CONTEXTS = {0: 4096, 1: 8192, 2: 16384, 3: 24576}

    def data_and_log(self):
        slope = 3.0                                    # us per 1k tokens

        def sdpa(user, contexts=self.CONTEXTS):
            return (60 + slope * contexts[user] / 1024.0) * 1000

        data, rounds = Csv(4), []
        for number in range(1, 9):
            live = 4 if number <= 5 else (3 if number == 6 else 2)
            idle = [] if live == 4 else ([3] if live == 3 else [2, 3])
            positions = dict((u, p + number * 4) for u, p in self.CONTEXTS.items() if u not in idle)
            data.add('0', str(number), dict((d, replay_ops(sdpa_ns=lambda user: sdpa(user))) for d in range(4)))
            # a shorter padded round: an idle segment's SDPA is cheap
            rounds.append((number, live, idle, 62.0, positions))
        return data, rounds

    def test_the_sdpa_fit_recovers_the_fixed_cost_and_the_slope_per_1k(self):
        data, rounds = self.data_and_log()
        fit = analyse(data, log_text=log_of(rounds))['verify_packed']['sdpa_fit']
        self.assertAlmostEqual(fit['intercept_us'], 60.0, delta=1.0)
        self.assertAlmostEqual(fit['slope_us_per_1k'], 3.0, delta=0.05)
        self.assertEqual(fit['contexts_k'], [4.0, 8.0, 16.0, 24.0])

    def test_sessions_map_to_rounds_by_id_and_group_by_live_count(self):
        data = Csv(4)
        for number, users in ((1, 4), (2, 4), (3, 4), (4, 4)):
            data.add('0', str(number), dict((d, replay_ops(device=d)) for d in range(4)))
        rounds = [(1, 4, [], 134.0, {}), (2, 4, [], 134.0, {}), (3, 3, [3], 120.0, {}), (4, 2, [2, 3], 100.0, {})]
        live = analyse(data, log_text=log_of(rounds))['verify_packed']['by_live']
        self.assertEqual(live['table']['4']['sessions'], 2)
        self.assertEqual(sorted(live['table']), ['2', '3', '4'])

    def test_the_round_map_check_flags_an_offset_mapping(self):
        data = build(sessions=('1', '2', '3', '4'))
        span = analyse(data)['verify_packed']['span_ms'][0]
        good = log_of([(n, 4, [], span, {}) for n in range(1, 5)])
        bad = log_of([(n, 4, [], 30.0, {}) for n in range(1, 5)])
        self.assertTrue(analyse(data, log_text=good)['verify_packed']['round_map']['ok'])
        result = analyse(data, log_text=bad)
        self.assertFalse(result['verify_packed']['round_map']['ok'])
        self.assertTrue(any('session id = round' in note for note in result['validity']['notes']))

    def test_the_timeline_splits_the_time_between_replays(self):
        result = analyse(build(sessions=('1', '2', '3', '4')))['verify_packed']['round_timeline']
        self.assertEqual(result['pairs'], 3)
        self.assertAlmostEqual(result['interval_ms'], result['verify_ms'] + result['idle_ms'] + result['other_busy_ms'], delta=0.5)
        self.assertAlmostEqual(result['idle_ms'], 50e6 / 1.35e6, delta=1.0)      # the synthetic round's host time


class ValidityTests(unittest.TestCase):
    def gate(self, **config):
        base = {'QWEN_FAST_TP': '4', 'QWEN_FAST_VERIFY_T1_AUDIT': '0', 'QWEN_FAST_VERIFY_T2_AUDIT': '0'}
        base.update(config)
        return {'qwen_configuration': base, 'streams': [{'text': 'a'}, {'text': 'b'}]}

    def test_an_audit_line_in_the_log_is_a_problem(self):
        result = analyse(build(sessions=('1', '2')), log_text='x\n[PINDIAG] verify t2 audit ok round 3\n')
        self.assertFalse(result['validity']['ok'])
        self.assertTrue(any('audit' in problem for problem in result['validity']['problems']))

    def test_a_launched_configuration_with_an_audit_on_is_a_problem(self):
        result = analyse(build(sessions=('1', '2')), gate_json=self.gate(QWEN_FAST_VERIFY_T2_AUDIT='1'))
        self.assertFalse(result['validity']['ok'])

    def test_the_readback_lines_are_counted_and_too_few_is_a_note(self):
        text = '[PINDIAG] device profiler read back after replay 2 (QWEN_FAST_PROFILE_DUMP_EVERY=2)\n' * 21
        ok = analyse(build(sessions=('1', '2')), log_text=text)
        self.assertEqual(ok['validity']['readbacks'], 21)
        self.assertFalse(any('read-back lines' in note for note in ok['validity']['notes']))
        few = analyse(build(sessions=('1', '2')), log_text='x\n')
        self.assertTrue(any('read-back lines' in note for note in few['validity']['notes']))

    def test_differing_texts_are_a_problem_identical_ones_are_not(self):
        same = analyse(build(sessions=('1', '2')), gate_json=self.gate(), twin_json=self.gate())
        self.assertTrue(same['texts_identical'])
        self.assertTrue(same['validity']['ok'])
        other = self.gate()
        other['streams'][1]['text'] = 'c'
        differ = analyse(build(sessions=('1', '2')), gate_json=self.gate(), twin_json=other)
        self.assertFalse(differ['validity']['ok'])
        self.assertIn('arithmetic', differ['validity']['problems'][0])

    def test_perturbation_is_flagged_above_three_percent(self):
        profiled = log_of([(n, 4, [], 104.0, {}) for n in range(1, 4)])
        twin = log_of([(n, 4, [], 100.0, {}) for n in range(1, 4)])
        result = analyse(build(sessions=('1', '2', '3')), log_text=profiled, twin_log=twin)
        self.assertTrue(result['perturbation']['flagged'])
        twin_close = log_of([(n, 4, [], 103.0, {}) for n in range(1, 4)])
        self.assertFalse(analyse(build(sessions=('1', '2', '3')), log_text=profiled,
                                 twin_log=twin_close)['perturbation']['flagged'])

    def test_dropped_marker_lines_are_counted(self):
        text = 'Profiler: 12 markers dropped on core (1,1)\nfine\n'
        result = analyse(build(sessions=('1', '2')), log_text=text)
        self.assertEqual(result['validity']['drops'], 1)


class LaneProjectionTests(unittest.TestCase):
    def test_the_16_row_lane_takes_per_user_categories_from_the_lone_step_and_interpolates_the_rest(self):
        data = build(sessions=('1', '2'))
        # the lone 4-row step: one user, every row-scaled term at a fifth of the block's
        for session in ('1', '2'):
            data.add('9', session, dict((d, replay_ops(users=1, device=d)) for d in range(4)))
        result = analyse(data)
        self.assertEqual(result['verify_lone']['trace'], '9')
        lane = result['lane_16_row']
        packed, lone = result['verify_packed']['categories'], result['verify_lone']['categories']
        self.assertAlmostEqual(lane['categories']['attn.sdpa'], lone['attn.sdpa']['ms'][0], places=3)
        expect = lone['mm.mlp']['ms'][0] + (packed['mm.mlp']['ms'][0] - lone['mm.mlp']['ms'][0]) * 12.0 / 60.0
        self.assertAlmostEqual(lane['categories']['mm.mlp'], expect, places=3)
        gap = result['verify_lone']['gap_ms'][0] + (result['verify_packed']['gap_ms'][0]
                                                    - result['verify_lone']['gap_ms'][0]) * 12.0 / 60.0
        self.assertAlmostEqual(lane['total_ms'], sum(lane['categories'].values()) + gap, places=1)

    def test_without_a_lone_step_there_is_no_projection_and_a_note_says_so(self):
        result = analyse(build(sessions=('1', '2')))
        self.assertIsNone(result['lane_16_row'])
        self.assertTrue(any('no complete lone-user' in note for note in result['validity']['notes']))


class OutputTests(unittest.TestCase):
    def test_the_markdown_carries_the_tables(self):
        text = report.render_markdown(analyse(build(sessions=('1', '2', '3'))))
        for needle in ('## The packed 64-row verify', '| gdn.recurrence |', 'Weight matmuls', 'Collectives',
                       'Validity: OK', 'weight matmuls'):
            self.assertIn(needle, text)

    def test_the_command_line_reads_the_artifact_layout(self):
        with tempfile.TemporaryDirectory() as results:
            os.makedirs(os.path.join(results, 'ops'))
            os.makedirs(os.path.join(results, 'ops-trace'))
            os.makedirs(os.path.join(results, 'ops-twin'))
            build(sessions=('1', '2', '3')).write(os.path.join(results, 'ops'))
            log = log_of([(n, 4, [], 134.0, {0: 4096, 1: 8192, 2: 16384, 3: 24576}) for n in range(1, 4)])
            for arm in ('ops-trace', 'ops-twin'):
                with open(os.path.join(results, arm, 'server.log'), 'w') as handle:
                    handle.write(log)
                with open(os.path.join(results, arm, 'm3native-gate.json'), 'w') as handle:
                    json.dump({'qwen_configuration': {'QWEN_FAST_TP': '4', 'QWEN_FAST_VERIFY_T1_AUDIT': '0',
                                                      'QWEN_FAST_VERIFY_T2_AUDIT': '0'}, 'streams': [{'text': 't'}]},
                              handle)
            done = subprocess.run([sys.executable, '-B', os.path.join(HERE, 'tp4_profile_report.py'),
                                   '--results', results], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
            self.assertEqual(done.returncode, 0, done.stderr.decode())
            self.assertIn('Validity: OK', done.stdout.decode())
            with open(os.path.join(results, 'ops', 'tp4-profile-report.json')) as handle:
                data = json.load(handle)
            self.assertTrue(data['validity']['ok'])
            self.assertTrue(os.path.isfile(os.path.join(results, 'ops', 'tp4-profile-report.md')))

    def test_a_file_that_is_not_the_cpp_report_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'x.csv')
            with open(path, 'w') as handle:
                handle.write('a,b\n1,2\n')
            with self.assertRaises(report.ReportError):
                report.load(path)


class V138PositiveControl(unittest.TestCase):
    """TP2's v138 replay (4 users x 32k, 64 rows, before the later levers): the research file's table, from the fixture."""

    @classmethod
    def setUpClass(cls):
        sessions, every, columns = report.load(FIXTURE)
        cls.result = report.analyse_sessions(sessions, every, columns, chips=2, table=report.weight_table(2))
        cls.packed = cls.result['verify_packed']

    def test_the_trace_is_a_64_layer_packed_verify_and_the_truncated_session_is_dropped(self):
        self.assertEqual((self.packed['layers'], self.packed['gdn_layers'], self.packed['attn_layers']), (64, 48, 16))
        self.assertEqual(self.packed['ops'], 3605)
        self.assertEqual((self.packed['complete_sessions'], self.packed['sessions_total']), (2, 3))

    def test_kernel_sum_span_and_gaps_are_the_recorded_ones(self):
        self.assertAlmostEqual(self.packed['kernel_sum_ms'][0], 131.42, delta=0.05)
        self.assertAlmostEqual(self.packed['span_ms'][0], 134.24, delta=0.05)
        self.assertAlmostEqual(self.packed['gap_ms'][0], 2.82, delta=0.05)
        self.assertAlmostEqual(self.packed['critical_path_ms'], 132.07, delta=0.05)
        self.assertAlmostEqual(self.packed['collective_skew_ms'], 0.55, delta=0.05)

    def test_the_categories_are_the_recorded_ones(self):
        want = {'gdn.recurrence': 25.24, 'gdn.glue': 24.29, 'mm.mlp': 21.08, 'attn.sdpa': 17.66, 'attn.glue': 12.92,
                'gdn.conv_gates': 6.78, 'mm.gdn_in': 6.00, 'collective': 4.69}
        for name, ms in want.items():
            self.assertAlmostEqual(self.packed['categories'][name]['ms'][0], ms, delta=0.05, msg=name)
        self.assertAlmostEqual(sum(self.packed['groups'].values()), self.packed['span_ms'][0], delta=0.05)

    def test_the_weight_streaming_rates_are_the_recorded_ones(self):
        gate = self.packed['weights']['mm.mlp.gate']
        self.assertAlmostEqual(gate['gbps'], 231, delta=3)
        self.assertAlmostEqual(gate['ns_per_tile_per_core'], 97, delta=2)
        self.assertAlmostEqual(self.packed['weights']['mm.mlp.down']['gbps'], 368, delta=4)
        self.assertEqual(gate['cores'], 39)

    def test_per_layer_totals(self):
        self.assertAlmostEqual(self.packed['by_layer_type']['gdn']['total_us'], 1781, delta=3)
        self.assertAlmostEqual(self.packed['by_layer_type']['attn']['total_us'], 2521, delta=3)
        self.assertEqual(self.packed['sdpa_us_by_user'], dict((str(u), self.packed['sdpa_us_by_user'][str(u)])
                                                              for u in range(4)))
        self.assertAlmostEqual(self.packed['sdpa_us_by_user']['0'], 276, delta=2)


if __name__ == '__main__':
    unittest.main()
