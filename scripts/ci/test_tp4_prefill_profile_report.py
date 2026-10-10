"""tp4_prefill_profile_report on a SYNTHETIC cpp_device_perf_report.csv: four chips, warm-up prefill groups, a prompt of N chunks of 64 layers (three GDN layers
then one attention layer), a final norm and eager decode glue after the last chunk, decode rows with a trace id mixed in. The real TP4 prefill CSV has never been
seen, so these tests hold the split, the categories, the SDPA fit, the collectives, the matmul efficiency, the validity problems and the graceful degradation on
the shapes the generator can say."""
import csv
import gzip
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tp4_prefill_profile_report as report_mod  # noqa: E402
import tp4_profile_report as base  # noqa: E402

CLK = 1.35               # cycles per ns
SDPA_FIXED_NS = 200000
SDPA_SLOPE_NS_PER_1K = 3000
GATE_NS, UP_NS, DOWN_NS = 400000, 400000, 450000
IN_GDN_NS, OUT_GDN_NS, IN_ATTN_NS, OUT_ATTN_NS = 500000, 300000, 350000, 250000
AG_NS, RS_NS = 90000, 120000
HEADER = ['GLOBAL CALL COUNT', 'DEVICE ID', 'OP NAME', 'DEVICE KERNEL DURATION [ns]', 'METAL TRACE ID', 'CORE COUNT', 'DEVICE KERNEL START CYCLE',
          'DEVICE KERNEL END CYCLE']
SHAPE_HEADER = ['INPUT_0_W_PAD[LOGICAL]', 'INPUT_0_Z_PAD[LOGICAL]', 'INPUT_0_Y_PAD[LOGICAL]', 'INPUT_0_X_PAD[LOGICAL]',
                'INPUT_1_W_PAD[LOGICAL]', 'INPUT_1_Z_PAD[LOGICAL]', 'INPUT_1_Y_PAD[LOGICAL]', 'INPUT_1_X_PAD[LOGICAL]']


def table():
    return base.weight_table(4)


class Synthetic(object):
    """Rows of one chip at a time; build() makes the CSV text."""

    def __init__(self, chips=4, chunks=8, warmup=2, shapes=False, fidelity=None, order=True, final_norm=True, extra_ops=(), cores=64):
        self.chips, self.chunks, self.warmup, self.shapes, self.fidelity = chips, chunks, warmup, shapes, fidelity
        self.order, self.final_norm, self.extra_ops, self.cores = order, final_norm, tuple(extra_ops), cores
        self.drop = set()        # (chip, op name, nth occurrence) to remove
        self.lost_chunk = None   # (chip, chunk index) whose SDPA rows are removed

    def ops_of_layer(self, layer, context_k):
        """[(op name, ns, matmul shape or None)] of one layer, mixer then MLP."""
        gdn = layer % 4 != 3
        wt = table()
        pre = [('LayerNormPreAllGatherDeviceOperation', 20000, None), ('LayerNormPostAllGatherDeviceOperation', 15000, None)]
        if gdn:
            k_in, n_in, _ = wt['mm.gdn_in']
            k_out, n_out, _ = wt['mm.gdn_out']
            mixer = pre + [('MatmulDeviceOperation', IN_GDN_NS, (k_in, n_in)), ('TypecastDeviceOperation', 8000, None),
                           ('GdnPrefillConvExactDeviceOperation', 60000, None), ('GdnPrefillConvExactDeviceOperation', 60000, None),
                           ('GdnChunkScanDeviceOperation', 700000, None), ('UntilizeDeviceOperation', 12000, None),
                           ('MatmulDeviceOperation', OUT_GDN_NS, (k_out, n_out)), ('ReduceScatterMinimalAsyncDeviceOperation', RS_NS, None)]
        else:
            k_in, n_in, _ = wt['mm.attn_in']
            k_out, n_out, _ = wt['mm.attn_out']
            mixer = pre + [('MatmulDeviceOperation', IN_ATTN_NS, (k_in, n_in)), ('TransposeDeviceOperation', 9000, None),
                           ('PermuteDeviceOperation', 7000, None),
                           ('SDPAOperation', SDPA_FIXED_NS + int(SDPA_SLOPE_NS_PER_1K * context_k), None), ('ReshapeViewDeviceOperation', 3000, None),
                           ('MatmulDeviceOperation', OUT_ATTN_NS, (k_out, n_out)), ('ReduceScatterMinimalAsyncDeviceOperation', RS_NS, None)]
        kg, ng, _ = wt['mm.mlp.gate']
        ku, nu, _ = wt['mm.mlp.up']
        kd, nd, _ = wt['mm.mlp.down']
        mlp = pre + [('AllGatherAsyncDeviceOperation', AG_NS, None), ('MatmulDeviceOperation', GATE_NS, (kg, ng)), ('MatmulDeviceOperation', UP_NS, (ku, nu)),
                     ('TernaryDeviceOperation', 40000, None), ('MatmulDeviceOperation', DOWN_NS, (kd, nd)),
                     ('ReduceScatterMinimalAsyncDeviceOperation', RS_NS, None), ('BinaryNgDeviceOperation', 10000, None)]
        return mixer + mlp

    def chip_rows(self, chip):
        out = []
        groups = ([('warm', index) for index in range(self.warmup)] + [('traced-block', 0)] + [('prompt', index) for index in range(self.chunks)])
        for kind, index in groups:
            if kind == 'traced-block':
                out += [('MatmulDeviceOperation', 1000, None, '7') for _ in range(50)]
                continue
            context_k = index * 2.0
            for layer in range(report_mod.LAYERS):
                for name, ns, mm in self.ops_of_layer(layer, context_k):
                    skew = 1.0 + 0.01 * chip if ('Async' in name) else 1.0
                    if name == 'SDPAOperation' and kind == 'prompt' and self.lost_chunk == (chip, index):
                        continue
                    out.append((name, int(ns * skew), mm, ''))
                if kind == 'prompt':
                    out += [(name, 50000, None, '') for name in self.extra_ops]         # unknown ops inside every layer (the end of its MLP half)
            if kind == 'prompt' and index == self.chunks - 1 and self.final_norm:
                out += [('LayerNormPreAllGatherDeviceOperation', 18000, None, ''), ('LayerNormPostAllGatherDeviceOperation', 12000, None, ''),
                        ('MatmulDeviceOperation', 900000, None, ''), ('ArgMaxDeviceOperation', 30000, None, '')]
                out += [('CopyDeviceOperation', 2000, None, '')] * 5
        return out

    def text(self):
        stream = io.StringIO()
        columns = list(HEADER)
        if not self.order:
            columns.remove('GLOBAL CALL COUNT')
        if self.shapes:
            columns += SHAPE_HEADER
        if self.fidelity:
            columns.append('MATH FIDELITY')
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for chip in range(self.chips):
            clock, count = 1000.0, 0
            for name, ns, mm, trace in self.chip_rows(chip):
                count += 1
                row = {'GLOBAL CALL COUNT': count, 'DEVICE ID': str(chip), 'OP NAME': name, 'DEVICE KERNEL DURATION [ns]': ns,
                       'METAL TRACE ID': trace, 'CORE COUNT': self.cores, 'DEVICE KERNEL START CYCLE': int(clock),
                       'DEVICE KERNEL END CYCLE': int(clock + ns * CLK)}
                clock += ns * CLK + 100
                if self.shapes and mm:
                    k, n = mm
                    row.update({'INPUT_0_W_PAD[LOGICAL]': '1[1]', 'INPUT_0_Z_PAD[LOGICAL]': '1[1]', 'INPUT_0_Y_PAD[LOGICAL]': '2048[2048]',
                                'INPUT_0_X_PAD[LOGICAL]': '%d[%d]' % (k, k), 'INPUT_1_W_PAD[LOGICAL]': '1[1]', 'INPUT_1_Z_PAD[LOGICAL]': '1[1]',
                                'INPUT_1_Y_PAD[LOGICAL]': '%d[%d]' % (k, k), 'INPUT_1_X_PAD[LOGICAL]': '%d[%d]' % (n, n)})
                if self.fidelity and name == 'MatmulDeviceOperation':
                    row['MATH FIDELITY'] = self.fidelity
                if not self.order:
                    del row['GLOBAL CALL COUNT']
                writer.writerow(row)
        return stream.getvalue()

    def write(self, directory, name='cpp_device_perf_report.prefill.csv.gz'):
        path = os.path.join(directory, name)
        data = self.text().encode('utf-8')
        if name.endswith('.gz'):
            with gzip.open(path, 'wb') as handle:
                handle.write(data)
        else:
            with open(path, 'wb') as handle:
                handle.write(data)
        return path


def analysed(synthetic, **kwargs):
    with tempfile.TemporaryDirectory() as directory:
        path = synthetic.write(directory)
        kwargs.setdefault('chips', synthetic.chips)
        kwargs.setdefault('prompt_tokens', synthetic.chunks * report_mod.CHUNK)
        return report_mod.analyse_files(path, **kwargs)


class ClassifyTests(unittest.TestCase):
    def test_the_substring_rules(self):
        cases = {'LayerNormPreAllGatherDeviceOperation': 'norm', 'RMSNormDeviceOperation': 'norm', 'MatmulDeviceOperation': 'matmul',
                 'AllGatherMinimalMatmulAsyncDeviceOperation': 'matmul', 'SDPAOperation': 'attn.sdpa', 'ScaledDotProductAttentionDeviceOperation': 'attn.sdpa',
                 'GdnPrefillConvExactDeviceOperation': 'gdn.conv', 'GdnChunkScanDeviceOperation': 'gdn.scan',
                 'AllGatherAsyncDeviceOperation': 'collective', 'ReduceScatterMinimalAsyncDeviceOperation': 'collective',
                 'TilizeWithValPaddingDeviceOperation': 'glue', 'UntilizeDeviceOperation': 'glue', 'TypecastDeviceOperation': 'glue',
                 'ReshapeViewDeviceOperation': 'glue', 'PermuteDeviceOperation': 'glue', 'SliceDeviceOperation': 'glue',
                 'ConcatDeviceOperation': 'glue', 'CopyDeviceOperation': 'glue', 'TernaryDeviceOperation': 'eltwise',
                 'BinaryNgDeviceOperation': 'eltwise', 'EmbeddingsDeviceOperation': 'embedding', 'MysteryDeviceOperation': 'other',
                 'SdpaDecodeDeviceOperation': 'other'}       # a decode op carries a trace id in a real CSV and is dropped before it matters
        for name, category in cases.items():
            with self.subTest(op=name):
                self.assertEqual(report_mod.classify_op(name), category)

    def test_a_dimension_is_the_logical_size(self):
        self.assertEqual(report_mod._dim('2048[2000]'), 2000)
        self.assertEqual(report_mod._dim('5120'), 5120)
        self.assertIsNone(report_mod._dim(''))
        self.assertIsNone(report_mod._dim('x'))

    def test_the_fidelity_factor(self):
        self.assertEqual([report_mod.fidelity_divisor(name) for name in ('LoFi', 'MathFidelity.HiFi2', 'HiFi4', 'weird', '')], [1, 2, 4, 2, 2])

    def test_a_line_fit(self):
        a, b, r2 = report_mod.fit_line([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0])
        self.assertAlmostEqual((a, b, r2), (1.0, 2.0, 1.0), places=9)
        self.assertIsNone(report_mod.fit_line([1.0], [1.0]))
        self.assertIsNone(report_mod.fit_line([2.0, 2.0], [1.0, 3.0]))


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = analysed(Synthetic(chunks=8, warmup=2))

    def test_the_prompt_is_the_last_chunks_and_the_warmup_is_counted_apart(self):
        report = self.report
        self.assertEqual(report['chunks_found_per_chip'], {'0': 8, '1': 8, '2': 8, '3': 8})
        self.assertEqual(set(report['warmup_groups_per_chip'].values()), {2})
        self.assertEqual(report['split_method'], ['sdpa'])
        self.assertEqual(report['rows_traced_dropped'], 4 * 50)
        self.assertTrue(report['validity']['ok'], report['validity'])

    def test_the_chunk_is_the_64_layers_and_not_the_final_norm_or_the_eager_tail(self):
        rows = self.report['by_chunk']
        self.assertEqual(len(rows), 8)
        per_chunk_ops = sum(1 for layer in range(64) for _ in Synthetic().ops_of_layer(layer, 0))
        for row in rows:
            self.assertEqual(row['ops'], per_chunk_ops, row)

    def test_the_layer_pattern_is_checked(self):
        ops = self.report['per_op']['0']
        by_name = dict((item['op'], item) for item in ops)
        self.assertEqual(by_name['SDPAOperation']['calls'], 16)
        self.assertEqual(by_name['GdnPrefillConvExactDeviceOperation']['calls'], 96)
        self.assertEqual(by_name['MatmulDeviceOperation']['calls'], 48 * 2 + 16 * 2 + 64 * 3)

    def test_the_chunk_ms_are_the_sum_of_its_categories(self):
        for row in self.report['by_chunk']:
            self.assertAlmostEqual(sum(row[c] for c in report_mod.CATEGORIES), row['ms'], places=6)
        first = self.report['by_chunk'][0]
        self.assertAlmostEqual(first['gdn.conv'], 96 * 60000 / 1e6, places=6)
        self.assertEqual(first['gdn.conv.calls'], 96)
        self.assertAlmostEqual(first['other'], 0.0)

    def test_the_span_comes_from_the_kernel_cycles(self):
        first = self.report['by_chunk'][0]
        self.assertGreater(first['span_ms'], first['ms'])               # the 100-cycle gaps sit between the kernels
        self.assertLess(first['span_ms'], first['ms'] * 1.05)

    def test_the_sampled_chunks(self):
        self.assertEqual(self.report['sampled_chunks'], [0, 1, 2, 4, 6, 7])
        self.assertEqual(report_mod.sample_chunks(64), [0, 1, 16, 32, 48, 63])
        self.assertEqual(report_mod.sample_chunks(1), [0])
        self.assertEqual(report_mod.sample_chunks(0), [])

    def test_without_the_call_count_column_the_rows_are_ordered_by_the_kernel_start(self):
        report = analysed(Synthetic(chunks=4, warmup=1, order=False))
        self.assertTrue(report['validity']['ok'], report['validity'])
        self.assertEqual(report['order_by'], 'DEVICE KERNEL START CYCLE')
        self.assertTrue(any('no GLOBAL CALL COUNT' in note for note in report['validity']['notes']))

    def test_a_prompt_without_an_eager_tail_or_a_final_norm(self):
        report = analysed(Synthetic(chunks=4, warmup=0, final_norm=False))
        self.assertTrue(report['validity']['ok'], report['validity'])
        self.assertEqual(len(report['by_chunk']), 4)


class SectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = analysed(Synthetic(chunks=8, warmup=1))

    def test_the_sdpa_fit_recovers_the_fixed_cost_and_the_slope(self):
        fit = self.report['sdpa_fit']
        self.assertAlmostEqual(fit['fixed_ms_per_chunk'], 16 * SDPA_FIXED_NS / 1e6, places=4)
        # the context of chunk c is c x 2,048 tokens; the generator's slope is per 2 x c (thousand-token units): 2.048 units of context per generator unit
        self.assertAlmostEqual(fit['slope_ms_per_1k_per_chunk'], 16 * SDPA_SLOPE_NS_PER_1K / 1e6 * 2.0 / 2.048, places=5)
        self.assertAlmostEqual(fit['r2'], 1.0, places=6)
        self.assertAlmostEqual(fit['fixed_ms_per_attention_layer'], SDPA_FIXED_NS / 1e6, places=5)
        self.assertEqual(fit['points'], 8)

    def test_the_gdn_section_counts_conv_calls_per_chunk_and_layer(self):
        gdn = self.report['gdn']
        self.assertEqual(gdn['conv_calls_per_chunk'], 96)
        self.assertEqual(gdn['conv_calls_per_gdn_layer'], 2.0)
        self.assertTrue(gdn['conv_calls_constant'])
        self.assertEqual(gdn['scan_ops_per_chunk'], 48)
        self.assertGreater(gdn['gdn_layers_ms'], gdn['attention_layers_ms'] * 0.5)

    def test_collectives_have_a_call_count_a_minimum_over_the_chips_and_a_skew(self):
        collectives = self.report['collectives']
        self.assertTrue(collectives['aligned'])
        kinds = dict((item['op'], item) for item in collectives['kinds'])
        rs = kinds['ReduceScatterMinimalAsyncDeviceOperation']
        self.assertEqual(rs['calls_per_chunk'], 64 * 2)
        self.assertAlmostEqual(rs['min_ms_per_call'], RS_NS / 1e6, places=5)                # chip 0 has no skew
        self.assertAlmostEqual(rs['skew_ms_per_call'], RS_NS * 0.03 / 1e6, places=5)        # chip 3 runs 3% longer
        ag = kinds['AllGatherAsyncDeviceOperation']
        self.assertEqual(ag['calls_per_chunk'], 64)

    def test_matmul_efficiency_from_the_weight_table(self):
        section = self.report['matmul_efficiency']['4']
        weights = dict((item['weight'], item) for item in section['weights'])
        self.assertEqual(set(weights), {'gdn_in', 'gdn_out', 'attn_in', 'attn_out', 'mlp_gate', 'mlp_up', 'mlp_down'})
        k, n, _ = table()['mm.mlp.gate']
        gate = weights['mlp_gate']
        self.assertEqual(gate['calls_per_chunk'], 64)
        self.assertAlmostEqual(gate['flops_per_call'], 2.0 * 2048 * k * n)
        self.assertAlmostEqual(gate['achieved_tflops'], 2.0 * 2048 * k * n / GATE_NS / 1e3, places=6)
        self.assertEqual(gate['flops_source'], 'weight table (assumed)')
        self.assertEqual(gate['peak_tflops'], report_mod.LOFI_PEAK_TFLOPS / 2)          # HiFi2 assumed
        self.assertAlmostEqual(gate['pct_of_peak'], 100.0 * gate['achieved_tflops'] / gate['peak_tflops'], places=6)
        self.assertEqual(gate['fidelity'], ['HiFi2 (assumed)'])
        self.assertIn('assumption', section['shapes'])
        self.assertGreater(section['overall']['achieved_tflops'], 0)

    def test_matmul_efficiency_from_the_csv_shapes_and_the_fidelity_column(self):
        report = analysed(Synthetic(chunks=4, warmup=0, shapes=True, fidelity='HiFi4'))
        section = report['matmul_efficiency']['2']
        down = dict((item['weight'], item) for item in section['weights'])['mlp_down']
        k, n, _ = table()['mm.mlp.down']
        self.assertEqual(down['flops_source'], 'csv shapes')
        self.assertAlmostEqual(down['flops_per_call'], 2.0 * 2048 * k * n)
        self.assertEqual(down['peak_tflops'], report_mod.LOFI_PEAK_TFLOPS / 4)
        self.assertEqual(down['fidelity'], ['HiFi4'])
        self.assertEqual(down['cores'], 64)
        self.assertIn('input-shape', section['shapes'])
        self.assertFalse(any('no INPUT_0/1 shape columns' in note for note in report['validity']['notes']))

    def test_the_markdown_has_every_section(self):
        text = report_mod.render_markdown(self.report)
        for heading in ('# TP4 prefill op profile', '## Validity', '## Kernel ms per chunk by category', '## Attention prefill against context',
                        '## GDN chunked prefill', '## Collectives', '## Weight matmul efficiency', '## Per-op ms, chunk 0', '## Per-op ms, chunk 7'):
            self.assertIn(heading, text)
        self.assertIn('ESTIMATE', text)
        self.assertIn('Verdict: OK', text)
        json.dumps(self.report)           # serialisable


class ValidityTests(unittest.TestCase):
    def test_a_prompt_longer_than_the_capture_is_a_problem_that_says_how_many_chunks(self):
        report = analysed(Synthetic(chunks=4, warmup=0), prompt_tokens=8 * 2048)
        self.assertFalse(report['validity']['ok'])
        self.assertTrue(any('4 prefill chunk(s) found' in text and '8 expected' in text for text in report['validity']['problems']), report['validity'])

    def test_a_missing_chip_is_a_problem(self):
        report = analysed(Synthetic(chips=3, chunks=4, warmup=0), chips=4)
        self.assertTrue(any('3 chip(s)' in text for text in report['validity']['problems']))

    def test_a_chip_that_lost_a_chunks_attention_rows_is_found(self):
        synthetic = Synthetic(chunks=6, warmup=0)
        synthetic.lost_chunk = (2, 3)
        report = analysed(synthetic)
        self.assertFalse(report['validity']['ok'])
        text = ' '.join(report['validity']['problems'])
        self.assertIn('5 prefill chunk(s) found', text)
        self.assertIn('different chunk counts', text)

    def test_dropped_markers_in_the_server_log_are_a_problem_and_the_flush_markers_are_counted(self):
        log = 'x\n' + (report_mod.FLUSH_MARKER + ': prompt 1 chunk 0 begin\n') * 7 + 'profiler dropped 12 markers\n'
        with tempfile.TemporaryDirectory() as directory:
            synthetic = Synthetic(chunks=4, warmup=0)
            path = synthetic.write(directory)
            log_path = os.path.join(directory, 'server.log')
            with open(log_path, 'w') as handle:
                handle.write(log)
            report = report_mod.analyse_files(path, server_log=log_path, prompt_tokens=4 * 2048)
        self.assertEqual(report['server_log']['flush_marker_count'], 7)
        self.assertTrue(any('dropped or lost profiler markers' in text for text in report['validity']['problems']))
        self.assertFalse(any('flush marker' in text for text in report['validity']['notes']))

    def test_no_flush_marker_is_a_note_not_a_problem(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Synthetic(chunks=4, warmup=0).write(directory)
            log_path = os.path.join(directory, 'server.log')
            with open(log_path, 'w') as handle:
                handle.write('nothing here\n')
            report = report_mod.analyse_files(path, server_log=log_path, prompt_tokens=4 * 2048)
        self.assertTrue(report['validity']['ok'])
        self.assertTrue(any('0 flush marker line(s)' in text for text in report['validity']['notes']))

    def test_unknown_ops_are_reported_as_other_and_a_large_share_is_a_note(self):
        report = analysed(Synthetic(chunks=4, warmup=0, extra_ops=('MysteryAlphaDeviceOperation',) * 16))
        self.assertTrue(report['unclassified_share'] > 0.0)
        names = set(item['op'] for item in report['per_op']['0'])
        self.assertIn('MysteryAlphaDeviceOperation', names)
        self.assertGreater(report['category_totals_ms']['other'], 0.0)
        self.assertGreater(report['unclassified_share'], report_mod.UNCLASSIFIED_NOTE_SHARE)
        self.assertTrue(any('"other" category' in note and 'MysteryAlphaDeviceOperation' in note for note in report['validity']['notes']), report['validity'])
        self.assertTrue(report['validity']['ok'], 'unclassified ops are a note, never a problem')

    def test_a_csv_without_the_required_columns_is_refused_by_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'x.csv')
            with open(path, 'w') as handle:
                handle.write('FOO,BAR\n1,2\n')
            with self.assertRaisesRegex(report_mod.ReportError, 'OP NAME'):
                report_mod.analyse_files(path)

    def test_a_csv_of_decode_rows_only_says_so_and_does_not_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'x.csv')
            with open(path, 'w') as handle:
                handle.write('DEVICE ID,OP NAME,DEVICE KERNEL DURATION [ns],METAL TRACE ID\n0,MatmulDeviceOperation,100,5\n')
            report = report_mod.analyse_files(path)
        self.assertFalse(report['validity']['ok'])
        self.assertTrue(any('no untraced row' in text for text in report['validity']['problems']))
        self.assertIn('# TP4 prefill op profile', report_mod.render_markdown(report))

    def test_no_norm_op_names_what_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'x.csv')
            with open(path, 'w') as handle:
                handle.write('DEVICE ID,OP NAME,DEVICE KERNEL DURATION [ns]\n' + '0,MatmulDeviceOperation,100\n' * 10)
            report = report_mod.analyse_files(path, chips=1)
        self.assertFalse(report['validity']['ok'])
        self.assertTrue(any('LayerNormPreAllGather' in text for text in report['validity']['notes'] + report['validity']['problems']))
        report_mod.render_markdown(report)

    def test_a_trace_without_the_sdpa_op_falls_back_to_the_norms(self):
        synthetic = Synthetic(chunks=4, warmup=1, final_norm=False)
        original = synthetic.ops_of_layer

        def renamed(layer, context_k):
            return [(name.replace('SDPAOperation', 'FlashMysteryOperation'), ns, mm) for name, ns, mm in original(layer, context_k)]

        synthetic.ops_of_layer = renamed
        report = analysed(synthetic)
        self.assertEqual(report['split_method'], ['norm'])
        self.assertEqual(report['chunks_found_per_chip']['0'], 4)


class CommandLineTests(unittest.TestCase):
    def run_main(self, argv):
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = report_mod.main(argv)
        return code, stream.getvalue()

    def test_a_csv_in_writes_the_json_and_the_markdown_beside_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Synthetic(chunks=4, warmup=0).write(directory)
            code, out = self.run_main([path, '--prompt-tokens', str(4 * 2048)])
            self.assertEqual(code, 0, out)
            self.assertIn('# TP4 prefill op profile', out)
            self.assertEqual(sorted(os.listdir(directory)), ['cpp_device_perf_report.prefill.csv.gz', 'prefill-profile-report.json',
                                                             'prefill-profile-report.md'])
            with open(os.path.join(directory, 'prefill-profile-report.json')) as handle:
                self.assertTrue(json.load(handle)['validity']['ok'])

    def test_the_exit_status_is_one_on_a_validity_problem(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Synthetic(chunks=4, warmup=0).write(directory)
            code, _ = self.run_main([path, '--prompt-tokens', str(8 * 2048), '--out', os.path.join(directory, 'out')])
            self.assertEqual(code, 1)
            self.assertTrue(os.path.isfile(os.path.join(directory, 'out', 'prefill-profile-report.md')))

    def test_results_reads_the_artifact_layout_and_the_gate_json_sets_the_prompt(self):
        with tempfile.TemporaryDirectory() as results:
            os.makedirs(os.path.join(results, 'ops'))
            os.makedirs(os.path.join(results, 'ops-prefill-trace'))
            Synthetic(chunks=4, warmup=0).write(os.path.join(results, 'ops'))
            with open(os.path.join(results, 'ops-prefill-trace', 'm3native-gate.json'), 'w') as handle:
                json.dump({'streams': [{'prompt_tokens': 4 * 2048, 'ttft': 1.5}]}, handle)
            with open(os.path.join(results, 'ops-prefill-trace', 'server.log'), 'w') as handle:
                handle.write((report_mod.FLUSH_MARKER + ' x\n') * 8)
            code, out = self.run_main(['--results', results])
            self.assertEqual(code, 0, out)
            self.assertIn('1.5', out)
            self.assertTrue(os.path.isfile(os.path.join(results, 'ops', 'prefill-profile-report.json')))

    def test_a_missing_csv_is_status_two(self):
        from contextlib import redirect_stderr
        with redirect_stderr(io.StringIO()):
            code, _ = self.run_main(['/nonexistent/x.csv'])
        self.assertEqual(code, 2)

    def test_the_arguments_are_one_of_csv_or_results(self):
        from contextlib import redirect_stderr
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            report_mod.main([])


class SourceTests(unittest.TestCase):
    def test_it_is_stdlib_python_37_syntax_and_public_safe(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tp4_prefill_profile_report.py')
        with open(path, encoding='utf-8') as handle:
            text = handle.read()
        import ast
        tree = ast.parse(text, feature_version=(3, 7))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add((node.module or '').split('.')[0])
        self.assertLessEqual(imports, {'argparse', 'bisect', 'collections', 'csv', 'json', 'os', 're', 'statistics', 'sys', 'tp4_profile_report'})
        for banned in ('/home/', 'thatch.local', 'zot.'):
            self.assertNotIn(banned, text)


if __name__ == '__main__':
    unittest.main()
