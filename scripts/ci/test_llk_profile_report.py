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


def kernel_rows(kernel, chips=(0, 1), envelope=1000, waits=None, core=(1, 1), host=10, start=100, stages=()):
    """A compute kernel's envelope on the three TRISCs of each chip, its sums (waits: {risc: (in, out)}) and
    stage instances ((stage, [(start, end) per thread offset]))."""
    text = []
    for chip in chips:
        for offset, risc in enumerate(report.COMPUTE_THREADS):
            begin = start + offset
            text.append(line(chip, core[0], core[1], risc, 1, begin, 0, host, 'QWEN_LLK_' + kernel, 'ZONE_START'))
            for stage, intervals in stages:
                for first, last in intervals[offset]:
                    text.append(line(chip, core[0], core[1], risc, 2, first, 0, host, 'QWEN_LLK_%s_%s' % (kernel, stage),
                                     'ZONE_START'))
                    text.append(line(chip, core[0], core[1], risc, 2, last, 0, host, 'QWEN_LLK_%s_%s' % (kernel, stage),
                                     'ZONE_END'))
            text.append(line(chip, core[0], core[1], risc, 1, begin + envelope, 0, host, 'QWEN_LLK_' + kernel, 'ZONE_END'))
            wait_in, wait_out = (waits or {}).get(risc, (0, 0))
            if wait_in:
                text.append(line(chip, core[0], core[1], risc, 3, begin + envelope, wait_in, host, zones.WAIT_IN, 'ZONE_TOTAL'))
            if wait_out:
                text.append(line(chip, core[0], core[1], risc, 4, begin + envelope, wait_out, host, zones.WAIT_OUT,
                                 'ZONE_TOTAL'))
    return text


def counter_lines(values, chip=0, core=(1, 1), host=10, ref=1000):
    return [line(chip, core[0], core[1], 'BRISC', 9090, 5000 + index, value, host, '', 'TS_DATA_16B',
                 '{"counter type":"%s";"ref cnt":%d;"value":%d}' % (kind, ref, value))
            for index, (kind, value) in enumerate(sorted(values.items()))]


def record(kernel, stages=(), sums=True):
    zone_list = [dict(name='QWEN_LLK_' + kernel, kind='envelope', stage=None, multiplicity=1, reconfig_static=3)]
    zone_list += [dict(name='QWEN_LLK_%s_%s' % (kernel, stage), kind='region', stage=stage, multiplicity=4,
                       reconfig_static=static) for stage, static in stages]
    return dict(key=kernel, kernel=kernel, zones=zone_list, sync=dict(enabled=sums))


def analyse_text(body, records, console='', twin=None, profiled=None):
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'profile_log_device.csv')
        with open(path, 'w', encoding='utf-8', newline='') as handle:
            handle.write(HEADER + ''.join(body))
        preamble, rows = report.read_rows(path)
    return report.analyse(rows, records, console, twin, profiled, preamble)


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
        body = kernel_rows('X')
        result = analyse_text(body, [record('X', sums=False)])
        self.assertEqual(result['kernels'][0]['bound'], 'undetermined')
        self.assertIsNone(result['kernels'][0]['threads']['TRISC_0']['wait_in'])

    def test_sums_on_with_no_rows_means_zero_waits(self):
        result = analyse_text(kernel_rows('X'), [record('X')])
        self.assertEqual(result['kernels'][0]['threads']['TRISC_0']['wait_in'], 0)
        self.assertEqual(result['kernels'][0]['bound'], 'llk-balanced')


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
