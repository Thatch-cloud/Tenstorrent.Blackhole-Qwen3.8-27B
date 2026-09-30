"""llk_profile_report: the device-log parser and the LLK-vs-dataflow report.

references/llk/profile_log_device.v077-format.csv is written in tt-metal v0.77.0's exact device-log format
(impl/profiler/profiler.cpp writeCSVHeader and dumpDeviceResultsToCSV: the preamble, the column line with its
spaces, one row per marker, enchantum names for the RISC and marker type, and the counter meta data as
nlohmann's sorted-key JSON with ',' written as ';'), with rows the report must ignore (firmware and kernel
markers, a TS_DATA row, a row whose cycle happens to read 9090). No hardware log with QWEN_LLK zones exists
yet: the first comes from the combined window's llk-* arms. Its numbers are chosen so every ratio below can be
checked by hand. Synthetic rows (kernel_rows, counter_lines) cover the classification branches and the refusals.
"""
import ast
import gzip
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import llk_kernels as kernels  # noqa: E402
import llk_profile_report as report  # noqa: E402
import llk_zones as zones  # noqa: E402

FIXTURE = os.path.join(HERE, 'references', 'llk', 'profile_log_device.v077-format.csv')
HEADER = ('ARCH: blackhole, CHIP_FREQ[MHz]: 1350, Max Compute Cores: 140\n'
          'PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset], data, run host ID, '
          'trace id, trace id counter, zone name, type, source line, source file, meta data\n')


def read(path):
    with open(os.path.join(ROOT, path), 'rb') as handle:
        return zones.decode(handle.read())


def fixture_records():
    """What the arm's manifest holds for the fixture's kernels: the real transforms' records."""
    compute = read('scripts/ci/gdn_seq_block_compute.cpp').replace('// @@GDN_SEQ_BLOCK_BUILD@@\n', '')
    records = [kernels.instrument_entry(kernels.BY_KEY['K5A'], compute, 'stages')[1],
               kernels.instrument_entry(kernels.BY_KEY['K5A_RD'], read('scripts/ci/gdn_seq_block_reader.cpp'), 'stages')[1],
               kernels.instrument_entry(kernels.BY_KEY['SDPA_DEC'], read(kernels.BY_KEY['SDPA_DEC']['repo']), 'stages')[1]]
    return records


def line(chip, x, y, risc, timer, cycle, data, host, zone, phase, meta='', trace=3, replay=7):
    return '%d,%d,%d,%s,%d,%d,%d,%d,%s,%s,%s,%s,0,k.cpp,%s\n' % (
        chip, x, y, risc, timer, cycle, data, host, '' if trace is None else trace, '' if replay is None else replay,
        zone, phase, meta)


def kernel_rows(kernel, chips=(0, 1), envelope=1000, waits=None, core=(1, 1), host=10, start=100, stages=(),
                sums=True, trace=3, replay=7, skip_sum=()):
    """A compute kernel's envelope on the three TRISCs of each chip, its sums and stage instances ((stage,
    [(start, end) per thread offset])). Each thread writes the sum rows of the slots it waits in (THREAD_SLOTS), as
    the device does: the value from waits ({risc: (in, out)}), else 1 (the zone's own cycles); sums False writes
    none, and skip_sum ({(chip, risc, slot)}) leaves single rows out."""
    text = []
    for chip in chips:
        for offset, risc in enumerate(report.COMPUTE_THREADS):
            begin = start + offset

            def row(timer, cycle, data, zone, phase):
                return line(chip, core[0], core[1], risc, timer, cycle, data, host, zone, phase, trace=trace,
                            replay=replay)
            text.append(row(1, begin, 0, 'QWEN_LLK_' + kernel, 'ZONE_START'))
            for stage, intervals in stages:
                for first, last in intervals[offset]:
                    text.append(row(2, first, 0, 'QWEN_LLK_%s_%s' % (kernel, stage), 'ZONE_START'))
                    text.append(row(2, last, 0, 'QWEN_LLK_%s_%s' % (kernel, stage), 'ZONE_END'))
            text.append(row(1, begin + envelope, 0, 'QWEN_LLK_' + kernel, 'ZONE_END'))
            if not sums:
                continue
            given = (waits or {}).get(risc, (0, 0))
            for slot, timer, zone, value in (('wait_in', 3, zones.WAIT_IN, given[0]),
                                             ('wait_out', 4, zones.WAIT_OUT, given[1])):
                if (slot in report.THREAD_SLOTS[risc] or value) and (chip, risc, slot) not in skip_sum:
                    text.append(row(timer, begin + envelope, value or 1, zone, 'ZONE_TOTAL'))
    return text


def counter_lines(values, chip=0, core=(1, 1), host=10, ref=1000, trace=3, replay=7):
    return [line(chip, core[0], core[1], 'BRISC', 9090, 5000 + index, value, host, '', 'TS_DATA_16B',
                 '{"counter type":"%s";"ref cnt":%d;"value":%d}' % (kind, ref, value), trace=trace, replay=replay)
            for index, (kind, value) in enumerate(sorted(values.items()))]


def record(kernel, stages=(), sums=True):
    zone_list = [dict(name='QWEN_LLK_' + kernel, kind='envelope', stage=None, multiplicity=1, reconfig_static=3)]
    zone_list += [dict(name='QWEN_LLK_%s_%s' % (kernel, stage), kind='region', stage=stage, multiplicity=4,
                       reconfig_static=static) for stage, static in stages]
    return dict(key=kernel, kernel=kernel, zones=zone_list, sync=dict(enabled=sums))


def analyse_text(body, records, console='', twin=None, profiled=None, window='all'):
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'profile_log_device.csv')
        with open(path, 'w', encoding='utf-8', newline='') as handle:
            handle.write(HEADER + ''.join(body))
        preamble, rows = report.read_rows(path)
    return report.analyse(rows, records, console, twin, profiled, preamble, window=window)


class FixtureTests(unittest.TestCase):
    def setUp(self):
        preamble, rows = report.read_rows(FIXTURE)
        self.preamble, self.rows = preamble, rows
        self.report = report.analyse(rows, fixture_records(), '', None, None, preamble)
        self.kernels = dict((kernel['kernel'], kernel) for kernel in self.report['kernels'])

    def test_preamble_and_filter(self):
        self.assertEqual(self.preamble, dict(arch='blackhole', chip_freq_mhz=1350, max_compute_cores=140))
        # 91 marker rows: the 79 QWEN_LLK rows and the 7 counter rows are read; the two firmware rows, the TRISC
        # and BRISC kernel rows (the latter at cycle 9090) and the TS_DATA row are not.
        self.assertEqual(len(self.rows), 79 + 7)
        self.assertEqual(sorted(set(row['risc'] for row in self.rows)), ['BRISC', 'NCRISC', 'TRISC_0', 'TRISC_1', 'TRISC_2'])

    def test_complete_and_ranked(self):
        self.assertTrue(self.report['complete'])
        self.assertEqual(self.report['drops'], dict(unmatched_end=0, unmatched_start=0, console=False))
        self.assertEqual([rank['kernel'] for rank in self.report['ranking']], ['SDPA_DEC', 'K5A', 'K5A_RD'])
        self.assertEqual([rank['share_of_instrumented'] for rank in self.report['ranking']], [0.5755, 0.2878, 0.1367])
        self.assertEqual(self.report['unknown_zones'], [])
        self.assertEqual(self.report['unattributed'], dict(sums=0, counters=0))
        self.assertEqual(self.report['counters_seen'], 7)

    def test_k5a_is_starved_by_its_reader(self):
        k5a = self.kernels['K5A']
        self.assertEqual(k5a['chips'], [0, 1])
        self.assertEqual(k5a['invocations'], 2)
        unpack = k5a['threads']['TRISC_0']
        self.assertEqual(unpack['envelope']['total'], 4000)
        self.assertEqual(unpack['wait_in'], 2400)
        self.assertEqual(unpack['wait_in_fraction'], 0.6)
        self.assertEqual(unpack['busy'], 1600)
        self.assertEqual(k5a['threads']['TRISC_1']['wait_out_fraction'], 0.05)
        self.assertEqual(k5a['threads']['TRISC_2']['wait_in_fraction'], 0.15)
        self.assertEqual(k5a['bound'], 'dataflow-in')
        self.assertFalse(k5a['llk_bound'])
        stage = k5a['stages'][0]
        self.assertEqual(stage['name'], 'T1')
        self.assertEqual(stage['instances_per_thread'], dict(TRISC_0=6, TRISC_1=6, TRISC_2=6))
        self.assertEqual(stage['threads']['TRISC_1']['mean'], 200.0)
        self.assertEqual(stage['reconfig_total'], 0)
        self.assertEqual(stage['lockstep']['mean_math_end_after_unpack_end'], 20.0)

    def test_sdpa_is_math_bound_by_its_counters(self):
        sdpa = self.kernels['SDPA_DEC']
        self.assertEqual(sdpa['attribution']['pack_waiting_on_math'], 0.5)
        self.assertEqual(sdpa['attribution']['math_waiting_on_unpack'], 0.1)
        self.assertEqual(sdpa['attribution']['math_waiting_on_unpack_source'], 'counters')
        self.assertEqual(sdpa['counters']['derived']['fpu_util'], 0.6)
        self.assertEqual(sdpa['counters']['derived']['sfpu_util'], 0.05)
        self.assertEqual(sdpa['bound'], 'llk-math')
        self.assertTrue(sdpa['llk_bound'])

    def test_reader_is_a_data_movement_kernel(self):
        reader = self.kernels['K5A_RD']
        self.assertEqual(list(reader['threads']), ['NCRISC'])
        self.assertEqual(reader['threads']['NCRISC']['wait_in_fraction'], 0.7895)
        self.assertEqual(reader['bound'], 'dataflow-read')

    def test_coverage(self):
        self.assertEqual(report.coverage(self.report, ('K5A', 'SDPA_DEC')), [])
        problems = report.coverage(self.report, ('K5A', 'MM'))
        self.assertEqual(len(problems), 1)
        self.assertIn('MM: no QWEN_LLK_MM zone', problems[0])

    def test_render_and_json(self):
        text = report.render(self.report)
        self.assertIn('#1 SDPA_DEC 57.6% llk-math', text)
        json.dumps(self.report)


class ClassificationTests(unittest.TestCase):
    def classify(self, waits, counters=None, stages=()):
        body = kernel_rows('X', waits=waits, stages=stages)
        if counters:
            body += counter_lines(counters) + counter_lines(counters, chip=1)
        result = analyse_text(body, [record('X', [(stage, 5) for stage, _ in stages])])
        return result['kernels'][0]

    def test_dataflow_out(self):
        kernel = self.classify({'TRISC_2': (0, 600)})
        self.assertEqual(kernel['bound'], 'dataflow-out')

    def test_pack_bound(self):
        kernel = self.classify({'TRISC_1': (0, 400), 'TRISC_2': (100, 0)})
        self.assertEqual((kernel['bound'], kernel['llk_bound']), ('llk-pack', True))

    def test_unpack_bound_from_counters(self):
        kernel = self.classify({'TRISC_2': (400, 0)}, counters=dict(WAITING_FOR_SRCA_VALID=500, FPU_COUNTER=300))
        self.assertEqual(kernel['bound'], 'llk-unpack')

    def test_sfpu_math(self):
        kernel = self.classify({'TRISC_2': (400, 0)}, counters=dict(WAITING_FOR_SRCB_VALID=10, SFPU_COUNTER=600,
                                                                     FPU_COUNTER=100))
        self.assertEqual(kernel['bound'], 'llk-math-sfpu')

    def test_without_counters_the_unpack_math_split_stays_open(self):
        waves = [(100 + 100 * i, 150 + 100 * i + 7 * i) for i in range(6)]
        stage = ('S', [waves, [(first + 3, last + 4) for first, last in waves], waves])
        kernel = self.classify({'TRISC_2': (400, 0)}, stages=(stage,))
        self.assertEqual(kernel['bound'], 'llk-unpack-or-math')
        self.assertIn('lockstep in S', kernel['reason'])
        lock = kernel['stages'][0]['lockstep']
        self.assertTrue(lock['lockstep'])
        self.assertGreaterEqual(lock['r_unpack_math'], 0.99)
        self.assertTrue(kernel['stages'][0]['reconfig_heavy'])
        self.assertEqual(kernel['reconfig_heavy_stages'], ['S'])
        self.assertEqual(kernel['stages'][0]['reconfig_total'], 5 * 12)

    def test_balanced(self):
        self.assertEqual(self.classify({'TRISC_0': (100, 0)})['bound'], 'llk-balanced')

    def test_tag_level_without_counters_is_undetermined(self):
        # Level tag compiles no sum zone, so the device writes no ZONE_TOTAL row (kernel_rows now writes the rows
        # a stages-level thread emits unless told not to).
        body = kernel_rows('X', sums=False)
        result = analyse_text(body, [record('X', sums=False)])
        self.assertEqual(result['kernels'][0]['bound'], 'undetermined')
        self.assertIsNone(result['kernels'][0]['threads']['TRISC_0']['wait_in'])

    def test_sums_on_with_no_rows_is_undetermined_never_zero(self):
        """Sums asked for and no ZONE_TOTAL row arrived (the define never reached the worker, the zones compiled out,
        a zone-id collision, or dropped rows): zero waits would read LLK-balanced."""
        result = analyse_text(kernel_rows('X', sums=False), [record('X')])
        kernel = result['kernels'][0]
        self.assertIsNone(kernel['threads']['TRISC_0']['wait_in'])
        self.assertEqual(kernel['threads']['TRISC_0']['sums_missing'], dict(wait_in=2))
        self.assertIsNone(kernel['threads']['TRISC_0']['busy'])
        self.assertEqual((kernel['bound'], kernel['llk_bound']), ('undetermined', False))
        self.assertIn('wait sums missing: TRISC_0 wait_in (2 invocations)', kernel['reason'])
        self.assertTrue(kernel['sums_missing'])
        problems = report.coverage(result, ('X',), kind='zones')
        self.assertTrue(any('X: wait sums missing' in problem for problem in problems), problems)
        self.assertFalse(any('wait sums' in problem for problem in report.coverage(result, ('X',), kind='counters')))

    def test_one_missing_row_makes_its_slot_undetermined(self):
        body = kernel_rows('X', waits={'TRISC_2': (400, 0)}, skip_sum={(1, 'TRISC_2', 'wait_out')})
        kernel = analyse_text(body, [record('X')])['kernels'][0]
        self.assertIsNone(kernel['threads']['TRISC_2']['wait_out'])
        self.assertEqual(kernel['threads']['TRISC_2']['wait_in'], 800)
        self.assertEqual(kernel['threads']['TRISC_2']['sums_missing'], dict(wait_out=1))
        self.assertEqual(kernel['bound'], 'undetermined')

    def test_a_slot_the_thread_does_not_wait_in_reads_zero(self):
        kernel = analyse_text(kernel_rows('X'), [record('X')])['kernels'][0]
        self.assertEqual(kernel['threads']['TRISC_0']['wait_out'], 0)
        self.assertEqual(kernel['threads']['TRISC_1']['wait_in'], 0)
        self.assertIsNone(kernel['threads']['TRISC_0']['sums_missing'])
        self.assertEqual(kernel['bound'], 'llk-balanced')

    def test_a_mover_without_its_sums_is_undetermined(self):
        rows = []
        for chip in (0, 1):
            rows.append(line(chip, 2, 2, 'NCRISC', 1, 100, 0, 10, 'QWEN_LLK_R', 'ZONE_START'))
            rows.append(line(chip, 2, 2, 'NCRISC', 1, 900, 0, 10, 'QWEN_LLK_R', 'ZONE_END'))
        mover = dict(record('R'), sync=dict(enabled=True, wait_in=2, wait_out=0))
        kernel = analyse_text(rows, [mover])['kernels'][0]
        self.assertEqual(kernel['bound'], 'undetermined')
        self.assertEqual(kernel['threads']['NCRISC']['sums_missing'], dict(wait_in=2))
        self.assertEqual(kernel['threads']['NCRISC']['wait_out'], 0)


def sessions(kernels_, replays, chips=(0, 1), short=(), dropped=(), counters=False):
    """Rows of trace 3, one session per replay counter: each kernel on its own core; short: (chip, replay, kernel)
    left out; dropped: (chip, replay) whose first kernel on TRISC_0 lost its ZONE_END."""
    body = []
    for replay in replays:
        for chip in chips:
            for index, kernel in enumerate(kernels_):
                if (chip, replay, kernel) in short:
                    continue
                rows = kernel_rows(kernel, chips=(chip,), core=(1 + index, 1), host=10 + index, replay=replay,
                                   start=100 * replay)
                if (chip, replay) in dropped and index == 0:
                    rows = [row for row in rows if not (',TRISC_0,' in row and 'QWEN_LLK_%s,ZONE_END' % kernel in row)]
                body += rows
                if counters:
                    body += counter_lines(dict(FPU_COUNTER=5), chip=chip, core=(1 + index, 1), host=10 + index,
                                          replay=replay)
    return body


class WindowTests(unittest.TestCase):
    def records(self):
        return [record('A'), record('B')]

    def test_the_traced_window_keeps_complete_sessions(self):
        body = sessions(('A', 'B'), (7, 8, 9), short={(1, 9, 'B')})
        body += kernel_rows('A', trace=None, replay=None, core=(5, 5), host=99)   # a prefill: not a replay
        result = analyse_text(body, self.records(), window='traced')
        chips = result['selection']['chips']
        self.assertEqual(chips['0'], dict(sessions=3, dropped=0, short=0, used=3))
        self.assertEqual(chips['1'], dict(sessions=3, dropped=0, short=1, used=2))
        self.assertGreater(result['selection']['untraced_rows'], 0)
        a = dict((kernel['kernel'], kernel) for kernel in result['kernels'])['A']
        self.assertEqual(a['sessions_by_chip'], {'0': 3, '1': 2})
        self.assertEqual(a['invocations'], 5)        # the untraced prefill invocation is not read
        self.assertTrue(result['complete'])
        self.assertEqual(report.coverage(result, ('A', 'B'), min_sessions=2), [])
        self.assertEqual(report.coverage(result, ('A', 'B'), min_sessions=3),
                         ['A: 2 complete traced sessions on chip 1, 3 needed',
                          'B: 2 complete traced sessions on chip 1, 3 needed'])

    def test_a_session_with_a_drop_is_left_out_and_the_rest_stay_complete(self):
        body = sessions(('A', 'B'), (7, 8, 9), dropped={(0, 8)})
        console = 'Metal | Profiler DRAM buffers were full, markers were dropped! device 0'
        result = analyse_text(body, self.records(), console=console, window='traced')
        self.assertEqual(result['selection']['chips']['0'], dict(sessions=3, dropped=1, short=0, used=2))
        self.assertEqual(result['drops'], dict(unmatched_end=0, unmatched_start=0, console=True))
        self.assertTrue(result['complete'])
        # The same rows read whole: the drop and the console line make them incomplete.
        whole = analyse_text(body, self.records(), console=console)
        self.assertFalse(whole['complete'])
        self.assertEqual(whole['drops']['unmatched_start'], 1)

    def test_no_complete_session_is_not_complete(self):
        body = sessions(('A',), (7,), dropped={(0, 7), (1, 7)})
        result = analyse_text(body, [record('A')], window='traced')
        self.assertFalse(result['complete'])
        self.assertEqual(result['kernels'], [])
        self.assertIn('A: no QWEN_LLK_A zone in a complete traced session', report.coverage(result, ('A',))[0])

    def test_the_untraced_window(self):
        body = sessions(('A',), (7,)) + kernel_rows('A', trace=None, replay=None, core=(5, 5), host=99)
        result = analyse_text(body, [record('A')], window='untraced')
        self.assertEqual(result['kernels'][0]['invocations'], 2)
        self.assertEqual(result['selection']['window'], 'untraced')
        with self.assertRaisesRegex(report.ReportError, 'window must be one of'):
            analyse_text(body, [record('A')], window='rounds')

    def test_counter_coverage_needs_every_chip(self):
        body = sessions(('A',), (7, 8), counters=True)
        result = analyse_text(body, [record('A', sums=False)], window='traced')
        self.assertEqual(result['kernels'][0]['counter_chips'], [0, 1])
        self.assertEqual(report.coverage(result, ('A',), kind='counters'), [])
        body = [row for row in body if not (row.startswith('1,') and ',9090,' in row)]
        result = analyse_text(body, [record('A', sums=False)], window='traced')
        self.assertEqual(report.coverage(result, ('A',), kind='counters'), ['A: no counter rows on chip 1'])

    def test_nothing_required_is_a_problem(self):
        result = analyse_text(kernel_rows('A'), [record('A')])
        self.assertEqual(report.coverage(result, ()), ['no required kernel named: an arm with nothing to cover '
                                                       'measures nothing'])


class DropAndRefusalTests(unittest.TestCase):
    def test_drops_are_counted(self):
        body = kernel_rows('X')
        body.append(line(0, 1, 1, 'TRISC_0', 1, 99999, 0, 10, 'QWEN_LLK_X', 'ZONE_END'))
        body.append(line(0, 2, 2, 'TRISC_0', 1, 5, 0, 10, 'QWEN_LLK_X', 'ZONE_START'))
        result = analyse_text(body, [record('X')], console='Profiler DRAM buffers were full, markers were dropped!')
        self.assertEqual(result['drops'], dict(unmatched_end=1, unmatched_start=1, console=True))
        self.assertFalse(result['complete'])

    def test_unknown_zones_are_listed(self):
        body = kernel_rows('X') + kernel_rows('Y', core=(4, 4))
        result = analyse_text(body, [record('X')])
        self.assertEqual(result['unknown_zones'], ['QWEN_LLK_Y'])

    def test_sums_and_counters_without_an_envelope_are_unattributed(self):
        body = [line(0, 9, 9, 'TRISC_0', 3, 10, 50, 10, zones.WAIT_IN, 'ZONE_TOTAL')]
        body += counter_lines(dict(FPU_COUNTER=1), core=(9, 9))
        result = analyse_text(body + kernel_rows('X'), [record('X')])
        self.assertEqual(result['unattributed'], dict(sums=1, counters=1))

    def test_refusals(self):
        cases = (
            ('no preamble\n', 'line 1 must start'),
            ('ARCH: wormhole_b0, CHIP_FREQ[MHz]: 1000, Max Compute Cores: 64\n', 'Blackhole device log'),
            ('ARCH: blackhole, Max Compute Cores: 1\nx\n', 'CHIP_FREQ'),
            ('ARCH: blackhole, CHIP_FREQ[MHz]: 1350, Max Compute Cores: 140\nPCIe slot, core_x\n', 'column names'),
            (HEADER + '0,1,1,TRISC_0,1,5,0,10,3,7,QWEN_LLK_X,ZONE_START\n', 'fields'),
            (HEADER + '0,1,1,TRISC_9,1,5,0,10,3,7,QWEN_LLK_X,ZONE_START,0,k.cpp,\n', 'RISC'),
            (HEADER + '0,1,1,TRISC_0,1,x5,0,10,3,7,QWEN_LLK_X,ZONE_START,0,k.cpp,\n', 'not an integer'),
            (HEADER + line(0, 1, 1, 'TRISC_0', 1, 5, 0, 10, 'QWEN_LLK_X', 'TS_DATA'), 'scopes and sums only'),
            (HEADER + line(0, 1, 1, 'BRISC', 9090, 5, 0, 10, '', 'TS_DATA_16B', '{"counter type":"FPU_COUNTER"}'),
             'needs "counter type"'),
            (HEADER + line(0, 1, 1, 'BRISC', 9090, 5, 0, 10, '', 'TS_DATA_16B', 'not json'), 'not JSON'),
        )
        for text, reason in cases:
            with self.subTest(reason=reason):
                with tempfile.TemporaryDirectory() as directory:
                    path = os.path.join(directory, 'log.csv')
                    with open(path, 'w', encoding='utf-8', newline='') as handle:
                        handle.write(text)
                    with self.assertRaisesRegex(report.ReportError, reason):
                        preamble, rows = report.read_rows(path)
                        report.analyse(rows, [record('X')], '', None, None, preamble)

    def test_two_l1_banks_in_one_pass_are_refused(self):
        body = kernel_rows('X') + counter_lines(dict(L1_0_UNPACKER_0=1, L1_1_EXT_UNPACKER_1=1))
        with self.assertRaisesRegex(report.ReportError, 'one L1 bank per run'):
            analyse_text(body, [record('X')])

    def test_the_size_cap_refuses_before_parsing(self):
        with self.assertRaisesRegex(report.ReportError, 'refused before parsing'):
            report.read_rows(FIXTURE, max_bytes=100)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(report.ReportError, 'refused before parsing'):
                report.export_filtered(FIXTURE, os.path.join(directory, 'x.csv.gz'), max_bytes=100)


class ExportTests(unittest.TestCase):
    def test_a_truncated_last_line_is_left_out(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'profile_log_device.csv')
            with open(FIXTURE, 'rb') as handle:
                data = handle.read()
            with open(source, 'wb') as handle:
                handle.write(data + b'0,1,2,TRISC_0,20001,1000,0,1024,3,7,QWEN_LLK_K5A,ZONE_ST')
            out = os.path.join(directory, 'e.csv.gz')
            result = report.export_filtered(source, out)
            self.assertTrue(result['source']['truncated_tail'])
            self.assertEqual(result['export']['rows'], 87)
            self.assertEqual(report.read_rows(out)[1], report.read_rows(FIXTURE)[1])

    def test_export_keeps_the_header_and_the_wanted_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            out = os.path.join(directory, 'llk-export.csv.gz')
            result = report.export_filtered(FIXTURE, out)
            with open(FIXTURE, 'rb') as handle:
                data = handle.read()
            self.assertEqual(result['source']['sha256'], hashlib.sha256(data).hexdigest())
            self.assertEqual(result['source']['bytes'], len(data))
            # The 86 wanted rows plus the cycle-9090 row the byte filter cannot tell apart (the parser drops it).
            self.assertEqual(result['export']['rows'], 87)
            with gzip.open(out, 'rt', encoding='utf-8') as handle:
                self.assertTrue(handle.readline().startswith('ARCH: blackhole'))
            self.assertEqual(report.read_rows(out)[1], report.read_rows(FIXTURE)[1])


class PerturbationTests(unittest.TestCase):
    def test_ratios(self):
        twin = dict(packed_phase=dict(trace_ms_mean=160.0), streams=[dict(ttft_s=10.0), dict(ttft_s=12.0)])
        profiled = dict(packed_phase=dict(trace_ms_mean=200.0), streams=[dict(ttft_s=15.0)])
        result = report.perturbation(twin, profiled)
        self.assertEqual(result['trace_ratio'], 1.25)
        self.assertEqual(result['ttft_ratio'], 1.25)
        self.assertIsNone(report.perturbation(None, profiled))


class CliTests(unittest.TestCase):
    def test_report_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = os.path.join(directory, 'llk-manifest.json')
            with open(manifest, 'w', encoding='utf-8') as handle:
                json.dump(dict(files=fixture_records(), generated=[]), handle)
            out = os.path.join(directory, 'report.json')
            printed = io.StringIO()
            with redirect_stdout(printed), redirect_stderr(printed):
                self.assertEqual(report.main(['report', '--csv', FIXTURE, '--manifest', manifest, '--out', out]), 0)
                self.assertEqual(report.main(['export', '--csv', FIXTURE, '--out', os.path.join(directory, 'e.csv.gz')]), 0)
                self.assertEqual(report.main(['report', '--csv', FIXTURE, '--manifest', manifest, '--out', out,
                                              '--max-bytes', '10']), 1)
            with open(out, encoding='utf-8') as handle:
                self.assertEqual(json.load(handle)['schema'], report.SCHEMA)
            self.assertIn('#1 SDPA_DEC', printed.getvalue())
            self.assertIn('refused before parsing', printed.getvalue())


class SyntaxTests(unittest.TestCase):
    def test_module_parses_as_python_37(self):
        with open(os.path.join(HERE, 'llk_profile_report.py'), encoding='utf-8') as handle:
            ast.parse(handle.read(), 'llk_profile_report.py', feature_version=(3, 7))


if __name__ == '__main__':
    unittest.main()
