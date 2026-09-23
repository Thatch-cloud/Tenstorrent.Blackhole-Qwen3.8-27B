"""lever_n_prefill_profile_report: per-op, coverage, signposted-chunk and norm-count grouping of a
prefill device profile, from synthetic CSVs using the pinned profiler's column names (the header
of a real cpp_device_perf_report.csv) and the ';' / '`' tracy_ops_data.csv format."""

import csv
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lever_n_prefill_profile_report as report  # noqa: E402

COLUMNS = ('GLOBAL CALL COUNT', 'METAL TRACE ID', 'METAL TRACE REPLAY SESSION ID', 'DEVICE ID', 'OP NAME',
           'CORE COUNT', 'DEVICE FW DURATION [ns]', 'DEVICE FW START CYCLE', 'DEVICE FW END CYCLE',
           'DEVICE KERNEL DURATION [ns]')

LAYERS, GDN = 4, 3          # a small model: 4 layers, 3 of them GDN


WARMUP_SDPA_NS = 7777


def synthetic(chunks=3, chips=('0', '1'), sdpa_ns=(1000, 2000, 3000), warmup=0, final_norm=False):
    """Rows for `warmup` warm-up forwards then `chunks` prompt chunks: per layer 2 norms; per GDN
    layer 3 ternaries; the full-attention layer (the 4th) runs one SDPA whose cost grows with the
    prompt chunk index (a warm-up forward's SDPA costs WARMUP_SDPA_NS). With final_norm, every
    forward ends with one more LayerNormPreAllGather and an lm_head matmul (the drift the norm
    split cannot see). Each op is 1000 cycles at 1 cycle/ns, back to back."""
    rows, call, cycle = [], 1000, 0
    for forward in range(warmup + chunks):
        chunk = forward - warmup
        forward_ops = []
        for layer in range(LAYERS):
            forward_ops += ['LayerNormPreAllGatherDeviceOperation']
            forward_ops += ['TernaryDeviceOperation'] * 3 if layer < GDN else ['SDPAOperation']
            forward_ops += ['LayerNormPreAllGatherDeviceOperation', 'MatmulDeviceOperation']
        if final_norm:
            forward_ops += ['LayerNormPreAllGatherDeviceOperation', 'LmHeadMatmulDeviceOperation']
        for op in forward_ops:
            call += 1
            duration = (sdpa_ns[chunk] if chunk >= 0 else WARMUP_SDPA_NS) if op == 'SDPAOperation' else 1000
            for chip in chips:
                rows.append({'GLOBAL CALL COUNT': str(call), 'METAL TRACE ID': '', 'METAL TRACE REPLAY SESSION ID': '',
                             'DEVICE ID': chip, 'OP NAME': op, 'CORE COUNT': '110',
                             'DEVICE FW DURATION [ns]': str(duration), 'DEVICE FW START CYCLE': str(cycle),
                             'DEVICE FW END CYCLE': str(cycle + duration),
                             'DEVICE KERNEL DURATION [ns]': str(duration)})
            cycle += duration
    # One decode row under trace: must be excluded from everything.
    rows.append({'GLOBAL CALL COUNT': '99999', 'METAL TRACE ID': '7', 'METAL TRACE REPLAY SESSION ID': '1',
                 'DEVICE ID': '0', 'OP NAME': 'SDPAOperation', 'CORE COUNT': '110',
                 'DEVICE FW DURATION [ns]': '9', 'DEVICE FW START CYCLE': '0', 'DEVICE FW END CYCLE': '9',
                 'DEVICE KERNEL DURATION [ns]': '999999999'})
    return rows


def tracy_text(rows, signposted=(0, 1), ends=True, cached=True, prompt=1):
    """tracy_ops_data.csv: each chunk's ops, bracketed by signposts for the signposted chunks."""
    out = io.StringIO()
    out.write('MessageName;total_ns\n')
    calls = sorted({int(r['GLOBAL CALL COUNT']) for r in rows if not r['METAL TRACE ID']})
    per_chunk = GDN * 6 + (LAYERS - GDN) * 4
    time = 0
    for index, call in enumerate(calls):
        chunk, offset = divmod(index, per_chunk)
        if offset == 0 and chunk in signposted:
            time += 1
            out.write('`TT_SIGNPOST: qwen_prefill_p%d_chunk_%d_begin`;%d\n' % (prompt, chunk, time))
        time += 1
        if cached:
            out.write('`TT_DNN_DEVICE_OP: MatmulDeviceOperation, 1234, 0, %d, `;%d\n' % (call, time))
        else:
            out.write('`TT_DNN_DEVICE_OP: "Matmul" ->\n%s`;%d\n' % (json.dumps({'global_call_count': call}), time))
        if offset == per_chunk - 1 and chunk in signposted and ends:
            time += 1
            out.write('`TT_SIGNPOST: qwen_prefill_p%d_chunk_%d_end`;%d\n' % (prompt, chunk, time))
    return out.getvalue()


def write(directory, rows, tracy=None, log=None):
    logs = Path(directory) / '.logs'
    logs.mkdir()
    with open(logs / 'cpp_device_perf_report.csv', 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    if tracy is not None:
        (logs / 'tracy_ops_data.csv').write_text(tracy, encoding='utf-8')
    if log is not None:
        (Path(directory) / 'stdout.log').write_text(log, encoding='utf-8')
    return Path(directory) / 'stdout.log'


class PrefillProfileReportTests(unittest.TestCase):
    def build(self, rows, tracy=None, log=None, chunks=3):
        with tempfile.TemporaryDirectory() as directory:
            log_path = write(directory, rows, tracy, log)
            return report.build_report(directory, log_path, chunks=chunks, layers=LAYERS, gdn_layers=GDN,
                                       wanted=(0, 1, 2))

    def test_per_op_excludes_traced_decode_rows_and_sums_per_chip(self):
        result = self.build(synthetic())
        self.assertEqual(result['rows_prefill'], result['rows_total'] - 1)
        ops = {entry['op']: entry for entry in result['per_op']}
        self.assertEqual(ops['SDPAOperation']['calls'], 6)
        self.assertAlmostEqual(ops['SDPAOperation']['total_ms'], 2 * (1000 + 2000 + 3000) / 1e6)
        self.assertEqual(ops['SDPAOperation']['per_chip']['0']['calls'], 3)
        self.assertEqual(result['per_op'][0]['op'], 'TernaryDeviceOperation')

    def test_coverage_is_one_when_every_row_is_present_and_drops_when_rows_are_lost(self):
        rows = synthetic()
        result = self.build(rows)
        self.assertEqual(result['coverage']['TernaryDeviceOperation']['expected_per_chip'], 3 * GDN * 3)
        self.assertEqual(result['coverage']['TernaryDeviceOperation']['ratio_per_chip'], {'0': 1.0, '1': 1.0})
        self.assertEqual(result['coverage']['LayerNormPreAllGather']['ratio_per_chip'], {'0': 1.0, '1': 1.0})
        lost = [r for r in rows if not (r['DEVICE ID'] == '1' and r['OP NAME'] == 'TernaryDeviceOperation'
                                        and int(r['GLOBAL CALL COUNT']) > 1040)]
        ratio = self.build(lost)['coverage']['TernaryDeviceOperation']['ratio_per_chip']
        self.assertEqual(ratio['0'], 1.0)
        self.assertLess(ratio['1'], 1.0)

    def test_signposted_chunks_are_joined_by_global_call_count(self):
        rows = synthetic()
        for cached in (True, False):
            with self.subTest(cached=cached):
                result = self.build(rows, tracy=tracy_text(rows, cached=cached))
                self.assertEqual(result['signposts_found'], ['qwen_prefill_p1_chunk_0_begin', 'qwen_prefill_p1_chunk_0_end',
                                                             'qwen_prefill_p1_chunk_1_begin', 'qwen_prefill_p1_chunk_1_end'])
                groups = result['signposted']
                self.assertEqual(sorted(groups), ['0', '1'])
                self.assertEqual(groups['1']['sdpa']['0'], dict(calls=1, mean_ms=0.002, median_ms=0.002))
                self.assertEqual(groups['0']['rows'], 2 * (GDN * 6 + 4))

    def test_a_missing_end_signpost_runs_to_the_next_begin(self):
        rows = synthetic()
        groups = self.build(rows, tracy=tracy_text(rows, signposted=(0, 1), ends=False))['signposted']
        self.assertEqual(groups['0']['rows'], 2 * (GDN * 6 + 4))
        # Chunk 1 has no end and no next begin, so it runs to the end (chunk 2's rows too).
        self.assertEqual(groups['1']['rows'], 2 * 2 * (GDN * 6 + 4))

    def test_busy_union_and_span_come_from_the_device_cycles(self):
        result = self.build(synthetic())
        chip = result['busy']['0']
        self.assertAlmostEqual(chip['cycles_per_ns'], 1.0)
        self.assertAlmostEqual(chip['busy_union_ms'], chip['span_ms'])
        self.assertAlmostEqual(chip['span_ms'], sum(int(r['DEVICE KERNEL DURATION [ns]']) for r in synthetic()
                                                    if r['DEVICE ID'] == '0' and not r['METAL TRACE ID']) / 1e6)
        overlapping = synthetic()
        for row in overlapping:
            if row['GLOBAL CALL COUNT'] == '1002':
                row['DEVICE FW START CYCLE'] = str(int(row['DEVICE FW START CYCLE']) - 500)
                row['DEVICE FW DURATION [ns]'] = '1500'
        chip = self.build(overlapping)['busy']['0']
        self.assertAlmostEqual(chip['busy_union_ms'], chip['span_ms'])  # still one merged interval
        gapped = synthetic()
        for row in gapped:
            if int(row['GLOBAL CALL COUNT']) > 1010 and not row['METAL TRACE ID']:
                for name in ('DEVICE FW START CYCLE', 'DEVICE FW END CYCLE'):
                    row[name] = str(int(row[name]) + 7000)
        chip = self.build(gapped)['busy']['0']
        self.assertAlmostEqual(chip['span_ms'] - chip['busy_union_ms'], 7000 / 1e6)

    def test_norm_count_split_needs_no_host_log(self):
        result = self.build(synthetic())
        by_norm = result['by_norm']
        self.assertEqual(by_norm['norms_per_chunk'], 2 * LAYERS)
        self.assertEqual(by_norm['consistency']['0'], dict(norm_calls=3 * 2 * LAYERS, chunks_inferred=3,
                                                           warmup_groups=0, divides_evenly=True))
        self.assertEqual(by_norm['chunks']['2']['sdpa']['1']['mean_ms'], 0.003)
        self.assertIsNone(result['signposted'])
        self.assertTrue(any('tracy_ops_data.csv' in caveat for caveat in result['caveats']))
        self.assertTrue(all(cell['agree'] for table in by_norm['agreement'].values() for cell in table.values()))

    def test_the_sdpa_split_is_the_per_chunk_split_without_any_tracy_file(self):
        """The arm exports no tracy_ops_data.csv, so the device-only SDPA split must carry the
        per-chunk attribution: whole chunks, layer 0's attention norm first, one SDPA each."""
        result = self.build(synthetic())
        by_sdpa = result['by_sdpa']
        self.assertEqual(sorted(by_sdpa['chunks'], key=int), ['0', '1', '2'])
        per_chip = GDN * 6 + 4
        for chunk, sdpa_ms in (('0', 0.001), ('1', 0.002), ('2', 0.003)):
            self.assertEqual(by_sdpa['chunks'][chunk]['rows'], 2 * per_chip)
            self.assertEqual(by_sdpa['chunks'][chunk]['sdpa']['0'], dict(calls=1, mean_ms=sdpa_ms, median_ms=sdpa_ms))
        self.assertEqual(by_sdpa['consistency']['0'], dict(sdpa_calls=3, sdpa_per_chunk=1, groups=3, remainder=0,
                                                           warmup_groups=0, complete=True, gap_rows=0))
        self.assertEqual(result['coverage_scope'], 'prompt span (by_sdpa)')

    def test_warm_up_prefills_and_a_per_forward_final_norm_do_not_shift_the_sdpa_split(self):
        """Two warm-up forwards before the prompt and a final norm + lm_head after every forward:
        the SDPA split still finds the prompt's chunks 0-2 (from the end), coverage over the
        prompt span stays exactly 1 while the all-rows count is inflated, and the norm
        cross-check's drift is reported, not silently used."""
        rows = synthetic(warmup=2, final_norm=True)
        result = self.build(rows)
        by_sdpa = result['by_sdpa']
        self.assertEqual(by_sdpa['consistency']['1']['warmup_groups'], 2)
        self.assertEqual(by_sdpa['consistency']['1']['gap_rows'], 2 * 2)  # final norm + lm_head, twice
        for chunk, sdpa_ms in (('0', 0.001), ('1', 0.002), ('2', 0.003)):
            self.assertEqual(by_sdpa['chunks'][chunk]['sdpa']['1']['mean_ms'], sdpa_ms)
            self.assertEqual(by_sdpa['chunks'][chunk]['rows'], 2 * (GDN * 6 + 4))
        # Chunk 2's span stops before the final norm (the (2 x layers + 1)-th norm).
        self.assertNotIn('LmHeadMatmulDeviceOperation', {e['op'] for e in by_sdpa['chunks']['2']['ops']})
        coverage = result['coverage']
        self.assertEqual(coverage['TernaryDeviceOperation']['ratio_per_chip'], {'0': 1.0, '1': 1.0})
        self.assertEqual(coverage['LayerNormPreAllGather']['ratio_per_chip'], {'0': 1.0, '1': 1.0})
        self.assertEqual(coverage['TernaryDeviceOperation']['observed_all_untraced_per_chip']['0'], 3 * GDN * 5)
        agreement = result['by_norm']['agreement']['0']
        self.assertTrue(agreement['0']['agree'] is False or agreement['1']['agree'] is False)
        self.assertTrue(any('cross-check' in caveat for caveat in result['caveats']))
        # Without the extra norm the two splits agree, warm-up or not.
        clean = self.build(synthetic(warmup=2))
        self.assertTrue(all(cell['agree'] for table in clean['by_norm']['agreement'].values()
                            for cell in table.values()))
        self.assertFalse(any('cross-check' in caveat for caveat in clean['caveats']))

    def test_lost_sdpa_rows_make_the_split_untrustworthy_and_say_so(self):
        rows = [r for r in synthetic() if not (r['DEVICE ID'] == '0' and r['OP NAME'] == 'SDPAOperation'
                                               and r['DEVICE KERNEL DURATION [ns]'] == '1000')]
        result = self.build(rows)
        self.assertFalse(result['by_sdpa']['consistency']['0']['complete'])
        self.assertTrue(result['by_sdpa']['consistency']['1']['complete'])
        self.assertTrue(any(caveat.startswith('chip 0:') for caveat in result['caveats']))

    def test_only_the_last_prompts_signposts_are_joined(self):
        """A warm-up prompt's signposts (p1) must not claim the real prompt's (p2) chunk numbers."""
        rows = synthetic()
        events = report.parse_tracy_ops(tracy_text(rows, signposted=(0,), prompt=1))
        later = report.parse_tracy_ops(tracy_text(rows, signposted=(1,), prompt=2))
        shift = max(when for when, _, _ in events) + 1
        mapping = report.chunk_of_calls(events + [(when + shift, kind, value) for when, kind, value in later])
        self.assertEqual(set(mapping.values()), {1})

    def test_the_log_is_scanned_for_the_flush_hook_and_dropped_markers(self):
        log = ('x [PINDIAG] prefill profile flush: first flush at chunk 0 layer 15 (every 16 layers)\n'
               'Warning: Profiler DRAM buffers were full, markers were dropped\n')
        result = self.build(synthetic(), log=log)
        self.assertEqual(result['log']['flush_marker_count'], 1)
        self.assertEqual(len(result['log']['dropped_or_full']), 1)
        self.assertTrue(any('incomplete' in caveat for caveat in result['caveats']))
        result = self.build(synthetic(), log='nothing here\n')
        self.assertTrue(any('did not run' in caveat for caveat in result['caveats']))

    def test_a_missing_device_csv_is_an_error_not_a_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            result = report.build_report(directory)
        self.assertIn('not exported', result['error'])

    def test_main_writes_the_json_and_prints_one_summary_line(self):
        rows = synthetic()
        with tempfile.TemporaryDirectory() as directory:
            write(directory, rows, tracy_text(rows))
            out = Path(directory) / 'prefill.json'
            from contextlib import redirect_stdout
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                status = report.main(['--profile-dir', directory, '--out', str(out), '--chunks', '3',
                                      '--layers', str(LAYERS), '--gdn-layers', str(GDN)])
            self.assertEqual(status, 0)
            self.assertIn('"0"', out.read_text(encoding='utf-8'))
        line = buffer.getvalue().strip()
        self.assertTrue(line.startswith('PREFILL_PROFILE '))
        self.assertEqual(json.loads(line.split(' ', 1)[1])['signposts'], 4)


if __name__ == '__main__':
    unittest.main()
