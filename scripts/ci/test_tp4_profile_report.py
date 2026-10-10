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


class StructureTests(unittest.TestCase):
    def test_a_trace_that_is_not_64_layers_is_no_packed_verify(self):
        result = analyse(build(layers=60, sessions=('1', '2')))
        self.assertFalse(result['validity']['ok'])
        self.assertTrue(any('verify-64' in p for p in result['validity']['problems']))

    def test_a_three_user_block_is_a_problem(self):
        result = analyse(build(users=3, sessions=('1', '2')))
        self.assertTrue(any('SDPA launches per attention layer' in p for p in result['validity']['problems']))

    def test_the_lone_trace_is_the_widest_with_a_full_sample(self):
        data = build(sessions=('1', '2'))
        for session in map(str, range(1, 10)):
            data.add('7', session, dict((d, replay_ops(users=1, layers=64, device=d)) for d in range(4)))
        data.add('8', '1', dict((d, replay_ops(users=1, device=d, sdpa_ns=lambda u: 900000)) for d in range(4)))
        result = analyse(data)
        self.assertEqual(result['verify_lone']['trace'], '7')


class LaneProjectionTests(unittest.TestCase):
    def test_the_16_row_lane_takes_chain_and_per_user_terms_from_the_block_and_interpolates_the_rest(self):
        data = build(sessions=('1', '2'))
        # the lone 4-row step: one user, every row-scaled term at a fifth of the block's
        for session in ('1', '2'):
            data.add('9', session, dict((d, replay_ops(users=1, device=d)) for d in range(4)))
        result = analyse(data)
        self.assertEqual(result['verify_lone']['trace'], '9')
        lane = result['lane_16_row']
        packed, lone = result['verify_packed']['categories'], result['verify_lone']['categories']
        self.assertAlmostEqual(lane['categories']['gdn.recurrence'], packed['gdn.recurrence']['ms'][0], places=3)
        for name in ('attn.sdpa', 'gdn.conv_gates'):
            self.assertAlmostEqual(lane['categories'][name], packed[name]['ms'][0] / 4.0, places=3)
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


def analyse_files_of(data, **kwargs):
    """The whole path from a file: load (with its eager rows), analyse."""
    directory = tempfile.mkdtemp()
    path = data.write(directory, 'cpp.csv.gz')
    paths = {}
    for key, text in (('server_log', kwargs.pop('log_text', None)), ('twin_log', kwargs.pop('twin_text', None))):
        if text is not None:
            paths[key] = os.path.join(directory, key + '.log')
            with open(paths[key], 'w', encoding='utf-8') as handle:
                handle.write(text)
    return report.analyse_files(path, **dict(paths, **kwargs))


class PackedPickTests(unittest.TestCase):
    """The packed block is told by what it holds and what the host timed, never by its session count (v170's analyser
    picked the lone user's 4-row step, which had the most sessions)."""

    def test_a_lone_step_with_more_sessions_is_not_the_packed_block(self):
        data = build(sessions=('1', '2', '3'))
        for session in range(1, 10):
            data.add('71', str(session), dict((d, replay_ops(users=1, device=d)) for d in range(4)))
        result = analyse(data)
        self.assertEqual(result['verify_packed']['trace'], '0')
        self.assertEqual(result['verify_lone']['trace'], '71')
        self.assertEqual(result['verify_packed']['pick']['candidates'][0]['structural'], 2)

    def test_a_packed_trace_of_another_user_count_loses_to_the_four_user_block(self):
        data = build(sessions=('1', '2', '3'))
        for session in range(1, 10):
            data.add('5', str(session), dict((d, replay_ops(users=2, device=d)) for d in range(4)))
        result = analyse(data)
        self.assertEqual(result['verify_packed']['trace'], '0')
        table = dict((row['trace'], row) for row in result['verify_packed']['pick']['candidates'])
        self.assertEqual((table['0']['structural'], table['5']['structural']), (2, 0))
        self.assertEqual((table['5']['sdpa_users'], table['5']['conv_users']), (2, 2))

    def test_between_two_four_user_traces_the_one_the_host_timed_wins(self):
        data = build(sessions=('1', '2', '3'))
        for session in range(1, 10):          # more sessions, a slower trace: not what [PACKED-PHASE] timed
            data.add('9', str(session), dict((d, replay_ops(sdpa_ns=lambda user: 900000, device=d)) for d in range(4)))
        fast = analyse(build(sessions=('1', '2', '3')))['verify_packed']['span_ms'][0]
        log = log_of([(n, 4, [], fast, {}) for n in range(1, 4)])
        picked = analyse(data, log_text=log)['verify_packed']
        self.assertEqual(picked['trace'], '0')
        self.assertIn('within', picked['pick']['reason'])
        self.assertLess(picked['pick']['candidates'][0]['span_error'], 0.01)
        self.assertGreater(picked['pick']['candidates'][1]['span_error'], 0.1)

    def test_with_nothing_else_to_go_on_the_session_count_decides_and_the_reason_says_so(self):
        data = build(sessions=('1', '2', '3', '4'))
        for session in ('1', '2'):
            data.add('3', session, dict((d, replay_ops(device=d)) for d in range(4)))
        pick = analyse(data)['verify_packed']['pick']
        self.assertEqual(pick['candidates'][0]['trace'], '0')
        self.assertEqual(pick['candidates'][0]['structural'], 2)

    def test_the_conv_gates_per_gdn_layer_are_recorded_beside_the_sdpa_launches(self):
        listing = analyse(build(sessions=('1', '2')))['traces'][0]
        self.assertEqual((listing['detail']['users'], listing['detail']['conv_users']), (4, 4))


def anatomy_data(rounds=5, with_other=False):
    """Packed verify sessions with the round around them: the publication's eager burst, two drafter traces with eager glue
    between them, four one-op commit traces."""
    data = Csv(4)
    publication = [('CopyDeviceOperation', 110, 22000)] * 40 + [('AllGatherAsync', 12, 280000)] * 4
    glue = [('CopyDeviceOperation', 110, 7000)] * 20
    drafter = [('TopKDeviceOperation', 65, 171000)] * 2 + [('SdpaDecodeDeviceOperation', 64, 173000)] * 5 \
        + [('MatmulDeviceOperation', 108, 64000)] * 47
    for number in range(1, rounds + 1):
        data.add('0', str(number), dict((d, replay_ops(device=d)) for d in range(4)))
        data.add('', '', dict((d, publication) for d in range(4)))
        data.add('89', str(number), dict((d, drafter) for d in range(4)))
        data.add('', '', dict((d, glue) for d in range(4)))
        data.add('114', str(number), dict((d, drafter) for d in range(4)))
        for commit in range(4):
            data.add('c%d' % commit, str(number), dict((d, [('GenericOpDeviceOperation', 96, 650000)]) for d in range(4)))
        if with_other:
            data.add('77', str(number), dict((d, replay_ops(users=1, device=d)) for d in range(4)))
    return data


DRAFTER_MS = 2 * 0.171 + 5 * 0.173 + 47 * 0.064


class RoundAnatomyTests(unittest.TestCase):
    def anatomy(self, **kwargs):
        log = log_of([(n, 4, [], 134.0, {}) for n in range(1, 6)])
        return analyse_files_of(anatomy_data(**kwargs), log_text=log)['verify_packed']['round_anatomy']

    def test_publication_drafters_glue_and_commits_are_attributed_separately(self):
        anatomy = self.anatomy()
        self.assertEqual(anatomy['pairs'], 4)
        self.assertEqual(anatomy['live'], 4)
        kinds = anatomy['kinds']
        self.assertAlmostEqual(kinds['publication (eager)'], 40 * 0.022 + 4 * 0.28, places=3)
        self.assertAlmostEqual(kinds['draft glue (eager)'], 20 * 0.007, places=3)
        self.assertAlmostEqual(kinds['drafters'], 2 * DRAFTER_MS, places=3)
        self.assertAlmostEqual(kinds['commits'], 4 * 0.65, places=3)
        self.assertEqual(kinds['other traces'], 0.0)
        self.assertEqual((anatomy['commit_launches'], anatomy['publication_ops']), (4, 44))

    def test_the_drafter_traces_are_listed_with_their_ops_and_time_and_the_publication_ops_are_named(self):
        anatomy = self.anatomy()
        self.assertEqual([(d['trace'], d['ops']) for d in anatomy['drafter_traces']], [('114', 54), ('89', 54)])
        self.assertTrue(all(abs(d['kernel_ms'] - DRAFTER_MS) < 1e-3 for d in anatomy['drafter_traces']))
        top = anatomy['publication_top']
        self.assertEqual((top[0]['op'], top[0]['cores'], top[0]['per_round']), ('AllGatherAsync', 12, 4.0))
        self.assertAlmostEqual(top[0]['ms_per_round'], 1.12, places=3)
        self.assertEqual(top[1]['op'], 'Copy')

    def test_the_round_is_the_verify_plus_the_interval_and_the_idle_is_what_the_kernels_leave(self):
        anatomy = self.anatomy()
        self.assertAlmostEqual(anatomy['round_ms'], anatomy['verify_ms'] + anatomy['interval_ms'], places=2)
        self.assertAlmostEqual(anatomy['idle_ms'], anatomy['interval_ms'] - anatomy['busy_ms'], places=2)
        self.assertGreater(anatomy['idle_ms'], 100.0)         # the synthetic host gaps between the replays

    def test_another_replayed_trace_in_the_round_is_named_as_such_not_as_a_drafter(self):
        anatomy = self.anatomy(with_other=True)
        self.assertGreater(anatomy['kinds']['other traces'], 50.0)       # a lone 64-layer step
        self.assertAlmostEqual(anatomy['kinds']['drafters'], 2 * DRAFTER_MS, places=3)

    def test_only_rounds_of_the_asked_live_count_are_taken(self):
        log = log_of([(1, 4, [], 134.0, {}), (2, 3, [3], 134.0, {}), (3, 4, [], 134.0, {}), (4, 4, [], 134.0, {}),
                      (5, 4, [], 134.0, {})])
        anatomy = analyse_files_of(anatomy_data(), log_text=log)['verify_packed']['round_anatomy']
        self.assertEqual(anatomy['pairs'], 3)         # the pairs that start at rounds 1, 3 and 4

    def test_without_a_log_every_consecutive_pair_counts(self):
        anatomy = analyse_files_of(anatomy_data())['verify_packed']['round_anatomy']
        self.assertEqual((anatomy['pairs'], anatomy['live']), (4, None))

    def test_without_a_replay_pair_the_anatomy_is_empty(self):
        anatomy = analyse_files_of(anatomy_data(rounds=1))['verify_packed']['round_anatomy']
        self.assertEqual(anatomy, {'pairs': 0})

    def test_the_markdown_carries_the_pick_the_anatomy_and_the_v170_comparison(self):
        log = log_of([(n, 4, [], 134.0, {}) for n in range(1, 6)])
        text = report.render_markdown(analyse_files_of(anatomy_data(), log_text=log))
        for word in ('Picked as the packed block by', 'The round around the verify', 'publication (eager)', 'AllGatherAsync',
                     'Against the v170 profile', 'verify: weight matmuls', 'round_device: drafters'):
            self.assertIn(word, text)


def stamp(total_ms):
    return '2026-10-02 12:%02d:%06.3f' % (int(total_ms // 60000), (total_ms % 60000) / 1000.0)


def phase_log(rounds):
    """[PHASE] lines: rounds = [(live, sequential steps, verify, commit, draft, period)] in ms."""
    lines, clock = [], 0.0
    for number, (live, steps, verify, commit, draft, period) in enumerate(rounds, 1):
        lines.append('%s [PHASE] packed_verify r%d begin' % (stamp(clock), number))
        lines.append('%s [PACKED-PHASE] round=%d users=4 bind_ms=0.1 input_ms=2 trace_ms=60 sync_ms=1 readback_ms=1 '
                     'live=%d idle=-' % (stamp(clock + 1), number, live))
        lines.append('%s [PHASE] packed_verify r%d end %.1f ms' % (stamp(clock + verify), number, verify))
        lines.append('%s [PHASE] packed_commit r%d end %.1f ms' % (stamp(clock + verify + commit), number, commit))
        lines.append('%s [PHASE] early_draft r%d end %.1f ms' % (stamp(clock + verify + commit + draft), number, draft))
        for _ in range(steps):
            lines.append('%s [PHASE] step r%d end 5.0 ms' % (stamp(clock + period - 1), number))
        clock += period
    lines.append('%s [PHASE] packed_verify r%d begin' % (stamp(clock), len(rounds) + 1))
    return '\n'.join(lines) + '\n'


class HostBudgetTests(unittest.TestCase):
    def test_the_phases_are_medians_over_four_live_rounds_without_sequential_steps(self):
        text = phase_log([(4, 0, 64, 24, 32, 125), (4, 0, 66, 25, 33, 127), (4, 0, 62, 23, 31, 123),
                          (3, 0, 50, 19, 20, 95), (4, 1, 70, 30, 40, 150)])
        budget = report.host_budget(text)
        self.assertEqual((budget['rounds'], budget['sampled'], budget['live']), (5, 3, 4))
        self.assertEqual((budget['packed_verify_ms'], budget['packed_commit_ms'], budget['early_draft_ms']),
                         (64.0, 24.0, 32.0))
        self.assertAlmostEqual(budget['period_ms'], 125.0, delta=1.0)
        self.assertAlmostEqual(budget['unphased_ms'], 125.0 - 64 - 24 - 32, delta=1.0)

    def test_a_log_without_phase_lines_has_no_budget(self):
        self.assertIsNone(report.host_budget('nothing\n'))
        self.assertIsNone(report.host_budget(''))

    def test_the_budget_reaches_the_report_beside_the_anatomy(self):
        log = phase_log([(4, 0, 64, 24, 32, 125)] * 5)
        result = analyse_files_of(anatomy_data(), log_text=log)['verify_packed']
        self.assertEqual(result['host_budget']['packed_commit_ms'], 24.0)
        self.assertEqual(result['vs_v170']['round_host']['packed_commit_ms']['delta'], -0.5)


def eight_seat_log(steps):
    """Two 64-row blocks per engine step, one '[PHASE] execute' line at the end of each step: steps = [(live0, live1, period)]."""
    lines, clock, number = [], 0.0, 0
    for live0, live1, period in steps:
        for offset, live in ((0.0, live0), (period / 2.0, live1)):
            number += 1
            lines.append('%s [PHASE] packed_verify r%d begin' % (stamp(clock + offset), number))
            lines.append('%s [PACKED-PHASE] round=%d users=4 trace_ms=60 live=%d idle=-' % (stamp(clock + offset + 1), number, live))
            lines.append('%s [PHASE] packed_verify r%d end 60.0 ms' % (stamp(clock + offset + 60), number))
            lines.append('%s [PHASE] packed_commit r%d end 5.0 ms' % (stamp(clock + offset + 65), number))
            lines.append('%s [PHASE] early_draft r%d end 20.0 ms' % (stamp(clock + offset + 85), number))
        lines.append('%s [PHASE] execute total=8 new=0 cached=8 spec=64' % stamp(clock + period - 1))
        clock += period
    lines.append('%s [PHASE] packed_verify r%d begin' % (stamp(clock), number + 1))
    lines.append('%s [PHASE] execute total=8 new=0 cached=8 spec=64' % stamp(clock + 1))
    return '\n'.join(lines) + '\n'


class EightSeatHostBudgetTests(unittest.TestCase):
    def test_both_blocks_of_a_step_are_one_round_of_eight_live_users(self):
        budget = report.host_budget(eight_seat_log([(4, 4, 300), (4, 4, 296), (4, 4, 304), (4, 3, 250)]))
        self.assertEqual((budget['live'], budget['blocks_per_round']), (8, 2))
        self.assertEqual(budget['sampled'], 3)
        self.assertAlmostEqual(budget['period_ms'], 300.0, delta=1.0)
        self.assertEqual((budget['packed_verify_ms'], budget['packed_commit_ms'], budget['early_draft_ms']), (120.0, 10.0, 40.0))

    def test_four_seat_logs_with_execute_lines_are_not_grouped(self):
        text = phase_log([(4, 0, 64, 24, 32, 125)] * 4)
        text = ''.join(line + '\n' + ('%s [PHASE] execute total=4 new=0 cached=4 spec=64\n' % stamp(0) if 'early_draft' in line else '')
                       for line in text.splitlines())
        budget = report.host_budget(text)
        self.assertEqual(budget['live'], 4)
        self.assertNotIn('blocks_per_round', budget)


class V170ComparisonTests(unittest.TestCase):
    def test_the_comparison_is_per_category_and_carries_deltas(self):
        packed = analyse(build(sessions=('1', '2')))['verify_packed']
        comparison = report.vs_v170(packed['groups'], {'kinds': {'publication (eager)': 10.0, 'drafters': 25.0,
                                                                  'commits': 2.6}}, {'packed_commit_ms': 20.0})
        self.assertEqual(list(comparison['verify']), list(report.V170_VERIFY))
        self.assertAlmostEqual(comparison['round_device']['publication (eager)']['delta'], -2.0)
        self.assertAlmostEqual(comparison['round_host']['packed_commit_ms']['delta'], -4.5)
        self.assertIsNone(comparison['round_host']['early_draft_ms']['now'])
        self.assertIsNone(comparison['round_host']['early_draft_ms']['delta'])
        self.assertEqual(packed['vs_v170']['verify']['weight matmuls']['v170'], 17.0)


MULTI_FIXTURE = os.path.join(HERE, 'references', 'tp4-profile', 'v676-multi-sdpa-slim.csv.gz')


def multi_fixture():
    return report.load(MULTI_FIXTURE)


class MultiSdpaBlockTests(unittest.TestCase):
    """v676 (the shipped multi-SDPA stack): its 64-row blocks hold ONE SDPA launch per attention layer and no named
    conv-gates launch (F1 is a generic op), so the per-user count read them as lone steps; the 1,599-launch 4-row step,
    which holds four SDPA launches per attention layer (one per row) and one conv-gates launch, was picked as the
    "packed verify". The fixture is trace 0 (two replays) and the 4-row step (one replay) on all four chips."""

    @classmethod
    def setUpClass(cls):
        sessions, every, columns = multi_fixture()
        cls.sessions, cls.every, cls.columns = sessions, every, columns
        cls.result = report.analyse_sessions(sessions, every, columns, chips=4)
        cls.block = cls.result['verify_packed']
        cls.lone = cls.result['verify_lone']

    def ops_of(self, trace):
        return next(ops for (t, _, _), ops in sorted(self.sessions.items()) if t == trace)

    def test_the_block_is_the_packed_verify_and_the_four_row_step_is_not(self):
        listing = dict((row['trace'], row) for row in self.result['traces'])
        self.assertEqual((listing['0']['kind'], listing['0']['ops']), ('verify-packed', 1681))
        self.assertEqual((listing['344']['kind'], listing['344']['ops']), ('verify-single', 1599))
        self.assertEqual(listing['0']['detail']['layout'], 'multi')
        self.assertEqual((listing['344']['detail']['layout'], listing['344']['detail']['sdpa_launches'],
                          listing['344']['detail']['conv_users']), ('lone', 4, 1))
        self.assertEqual(report.trace_signature(self.ops_of('0'))[0], 'verify-packed')
        self.assertEqual(report.trace_signature(self.ops_of('344'))[0], 'verify-single')

    def test_the_packed_verify_is_the_1681_launch_block_at_about_45_6_ms_per_chip(self):
        self.assertEqual((self.block['trace'], self.block['ops']), ('0', 1681))
        self.assertEqual(len(self.block['kernel_sum_ms']), 4)
        for ms in self.block['kernel_sum_ms']:
            self.assertAlmostEqual(ms, 45.55, delta=0.1)
        self.assertAlmostEqual(self.block['span_ms'][0], 46.7, delta=0.1)
        self.assertAlmostEqual(self.block['gap_ms'][0], 1.15, delta=0.03)
        self.assertEqual(self.block['complete_sessions'], 2)

    def test_the_lone_step_is_analysed_separately(self):
        self.assertEqual((self.lone['trace'], self.lone['ops'], self.lone['layout']), ('344', 1599, 'lone'))
        self.assertAlmostEqual(self.lone['kernel_sum_ms'][0], 34.78, delta=0.05)
        self.assertEqual(len(self.lone['sdpa_us_by_user']), 4)
        self.assertEqual(self.lone['conv_gates_per_gdn_layer'], 1)
        self.assertEqual((self.lone['gdn_layers'], self.lone['attn_layers']), (48, 16))

    def test_the_block_layers_are_48_gdn_and_16_attention_and_each_attention_layer_has_one_sdpa(self):
        self.assertEqual((self.block['layers'], self.block['gdn_layers'], self.block['attn_layers']), (64, 48, 16))
        self.assertEqual(self.block['sdpa_us_by_user'].keys(), {'0'})
        self.assertAlmostEqual(self.block['categories']['attn.sdpa']['ms'][0], 16 * 0.1015, delta=0.05)
        self.assertAlmostEqual(self.block['categories']['gdn.recurrence']['ms'][0], 48 * 0.1935, delta=0.05)
        self.assertIsNone(self.block['conv_gates_per_gdn_layer'])

    def test_the_categories_are_the_measured_census(self):
        cats = self.block['categories']
        # weight matmuls 17.03, GDN recurrence 9.29, collectives 4.71 + sampler's 0.68, sampler argmax and glue 1.03 (M676)
        weights = sum(cats[k]['ms'][0] for k in cats if k.startswith('mm.'))
        self.assertAlmostEqual(weights, 17.03, delta=0.1)
        self.assertAlmostEqual(cats['collective']['ms'][0], 4.71, delta=0.05)
        self.assertAlmostEqual(cats['sampler.argmax/glue']['ms'][0], 1.03, delta=0.05)

    def test_the_validity_has_no_structure_problem(self):
        self.assertEqual(self.result['validity']['problems'], [])

    def test_a_block_without_a_log_has_four_assumed_users_and_with_a_log_the_logged_ones(self):
        self.assertEqual((self.block['users'], self.block['users_from']), (4, 'assumed (no host log)'))
        text = log_of([(1, 4, [], self.block['span_ms'][0], {}), (2, 4, [], self.block['span_ms'][0], {})])
        with_log = report.analyse_sessions(self.sessions, self.every, self.columns, chips=4, log_text=text)['verify_packed']
        self.assertEqual((with_log['users'], with_log['users_from']), (4, 'host log users='))
        self.assertIn('within', with_log['pick']['reason'])
        self.assertLess(with_log['pick']['candidates'][0]['span_error'], 0.01)

    def test_there_is_no_per_user_sdpa_fit_for_a_multi_launch(self):
        rounds = [(1, 4, [], 46.7, {0: 4100, 1: 8200, 2: 12300, 3: 16400}),
                  (2, 4, [], 46.7, {0: 4110, 1: 8210, 2: 12310, 3: 16410})]
        got = report.analyse_sessions(self.sessions, self.every, self.columns, chips=4,
                                      log_text=log_of(rounds))['verify_packed']
        self.assertIsNone(got['sdpa_fit'])

    def test_the_16_row_projection_divides_the_per_user_terms_by_the_block_users_not_by_the_sdpa_launches(self):
        lane = self.result['lane_16_row']
        self.assertAlmostEqual(lane['categories']['attn.sdpa'], self.block['categories']['attn.sdpa']['ms'][0] / 4.0,
                               places=3)

    def test_the_markdown_names_the_layout_and_the_multi_launch(self):
        text = report.render_markdown(self.result)
        self.assertIn('multi layout', text)
        self.assertIn('one multi launch per attention layer for all 4 users', text)
        self.assertNotIn('SDPA us by user', text)
        self.assertIn('| 0 | verify-packed | 1681 |', text)
        self.assertIn('| 344 | verify-single | 1599 |', text)

    def test_the_file_path_reads_the_same_fixture(self):
        got = report.analyse_files(MULTI_FIXTURE)
        self.assertEqual((got['verify_packed']['trace'], got['verify_lone']['trace']), ('0', '344'))
        self.assertTrue(got['validity']['ok'])

    def copy_lone(self, sessions, trace, count, scale=1.0):
        """`count` complete replays of the 4-row step as `trace`, durations scaled by `scale`."""
        for (t, sid, device), ops in list(self.sessions.items()):
            if t != '344':
                continue
            for n in range(1, count + 1):
                sessions[(trace, str(n), device)] = [op._replace(k=op.k * scale) for op in ops]

    def lone_variants(self, spec):
        """The block plus lone-step traces: spec = [(trace, replays, scale)]."""
        sessions = dict((key, ops) for key, ops in self.sessions.items() if key[0] == '0')
        for trace, count, scale in spec:
            self.copy_lone(sessions, trace, count, scale)
        return report.analyse_sessions(sessions, self.every, self.columns, chips=4)

    def test_three_lone_traces_of_one_kernel_sum_are_not_three_widths(self):
        got = self.lone_variants([('320', 3, 1.0), ('272', 9, 1.0), ('344', 5, 1.0)])
        self.assertTrue(all(row['kind'] == 'verify-single' for row in got['traces'] if row['trace'] != '0'))
        self.assertFalse(any('label' in row for row in got['traces']))
        self.assertEqual(got['single_user_labels'], {})
        self.assertTrue(any('widths cannot be told apart' in note for note in got['validity']['notes']))
        # the lone step is the trace with the most complete sessions among equals (and a full sample)
        self.assertEqual(got['verify_lone']['trace'], '272')
        self.assertEqual(got['verify_packed']['trace'], '0')

    def test_three_lone_traces_of_three_kernel_sums_are_the_1_2_and_4_row_steps(self):
        got = self.lone_variants([('1', 9, 0.9), ('2', 9, 1.0), ('4', 3, 1.1)])
        labels = dict((row['trace'], row.get('label')) for row in got['traces'])
        self.assertEqual((labels['1'], labels['2'], labels['4']), ('verify-w1', 'verify-w2', 'verify-w4'))
        self.assertEqual(got['verify_lone']['trace'], '2')    # the widest with a full sample (the w4 has only 3)

    def test_the_structure_checks_know_the_three_layouts(self):
        analysis = dict(layers=64, gdn_layers=48, attn_layers=16, conv_gates_per_gdn_layer=None,
                        sdpa_us_by_user={'0': 1.0})
        self.assertEqual(report.structure_problems('b', analysis, 4, layout='multi'), [])
        two = dict(analysis, sdpa_us_by_user={'0': 1.0, '1': 1.0})
        self.assertEqual(len(report.structure_problems('b', two, 4, layout='multi')), 1)
        lone = dict(analysis, conv_gates_per_gdn_layer=1, sdpa_us_by_user=dict((str(i), 1.0) for i in range(4)))
        self.assertEqual(report.structure_problems('l', lone, 1, layout='lone'), [])
        three = dict(lone, sdpa_us_by_user=dict((str(i), 1.0) for i in range(3)))
        self.assertEqual(len(report.structure_problems('l', three, 1, layout='lone')), 1)
        per_user = dict(analysis, conv_gates_per_gdn_layer=4, sdpa_us_by_user=dict((str(i), 1.0) for i in range(4)))
        self.assertEqual(report.structure_problems('p', per_user, 4), [])
        self.assertEqual(len(report.structure_problems('p', dict(per_user, conv_gates_per_gdn_layer=1), 4)), 1)


class BlockLayoutSignatureTests(unittest.TestCase):
    """trace_signature on synthetic replays: what makes a 64-layer replay the block, whatever its SDPA count."""

    def ops_of(self, rows):
        return [report.Op(n.replace('DeviceOperation', ''), c, ns, i, i, i, 0) for i, (n, c, ns) in enumerate(rows)]

    def per_user_rows(self, sdpa, conv):
        rows = [('EmbeddingsDeviceOperation', 8, 5000)]
        for index in range(64):
            layer = layer_ops(index, max(sdpa, conv), lambda user: 100000)
            if index % 4 == 3:
                keep = [r for r in layer if not r[0].startswith('SdpaDecode')]
                at = next(i for i, r in enumerate(keep) if r[0].startswith('AttnPrep')) + 1
                layer = keep[:at] + [('SdpaDecodeDeviceOperation', 32, 100000)] * sdpa + keep[at:]
            else:
                keep = [r for r in layer if not r[0].startswith('GdnConvGates')]
                at = next(i for i, r in enumerate(keep) if r[0] == 'MatmulDeviceOperation') + 1
                layer = keep[:at] + [('GdnConvGatesDeviceOperation', 4, 35000)] * conv + keep[at:]
            rows += layer
        return rows + [('LayerNormDeviceOperation', 8, 8000), ('MatmulDeviceOperation', 108, 1870000),
                       ('GenericOpDeviceOperation', 8, 1300000)]

    def test_four_sdpa_launches_with_one_conv_gates_launch_is_a_four_row_step_not_a_block(self):
        kind, detail = report.trace_signature(self.ops_of(self.per_user_rows(sdpa=4, conv=1)))
        self.assertEqual((kind, detail['layout']), ('verify-single', 'lone'))

    def test_one_launch_of_each_is_a_one_row_step(self):
        self.assertEqual(report.trace_signature(self.ops_of(self.per_user_rows(sdpa=1, conv=1)))[0], 'verify-single')

    def test_four_launches_of_each_is_the_per_user_block(self):
        kind, detail = report.trace_signature(self.ops_of(self.per_user_rows(sdpa=4, conv=4)))
        self.assertEqual((kind, detail['layout'], detail['users']), ('verify-packed', 'per-user', 4))

    def test_the_f1_conv_gates_as_a_generic_op_with_per_user_sdpa_is_still_the_block(self):
        rows = [(n.replace('GdnConvGates', 'GenericOp'), c, ns) for n, c, ns in self.per_user_rows(sdpa=4, conv=4)]
        kind, detail = report.trace_signature(self.ops_of(rows))
        self.assertEqual((kind, detail['layout'], detail['users']), ('verify-packed', 'per-user', 4))

    def test_one_sdpa_launch_without_a_fold_and_without_conv_gates_is_not_called_a_block(self):
        rows = [(n.replace('GdnConvGates', 'GenericOp'), c, ns) for n, c, ns in self.per_user_rows(sdpa=1, conv=1)]
        self.assertEqual(report.trace_signature(self.ops_of(rows))[0], 'verify-single')

    def test_one_folded_sdpa_launch_without_conv_gates_is_the_multi_block(self):
        rows = []
        for name, cores, ns in self.per_user_rows(sdpa=1, conv=1):
            if name.startswith('GdnConvGates'):
                continue
            if name.startswith('SdpaDecode'):
                rows += [('GenericOpDeviceOperation', 96, 15000), (name, cores, ns), ('GenericOpDeviceOperation', 110, 17000)]
            else:
                rows.append((name, cores, ns))
        kind, detail = report.trace_signature(self.ops_of(rows))
        self.assertEqual((kind, detail['layout'], detail['users']), ('verify-packed', 'multi', None))
        roles, layers, types = report.classify(self.ops_of(rows))
        self.assertEqual((layers, types.count('gdn'), types.count('attn')), (64, 48, 16))

    def test_a_mixer_of_generic_ops_alone_is_a_gdn_layer_not_an_attention_layer(self):
        self.assertEqual(report.layer_type(['Matmul', 'GenericOp', 'GenericOp', 'GenericOp', 'Matmul']), 'gdn')
        self.assertEqual(report.layer_type(['Matmul', 'AttnPrep', 'GenericOp', 'Matmul']), 'attn')
        self.assertEqual(report.layer_type(['Matmul', 'GenericOp', 'SdpaDecode', 'GenericOp']), 'attn')
        self.assertEqual(report.layer_type(['Matmul', 'GdnConvGates', 'GenericOp']), 'gdn')
        self.assertEqual(report.layer_type(['Matmul', 'Matmul']), 'attn')


if __name__ == '__main__':
    unittest.main()
