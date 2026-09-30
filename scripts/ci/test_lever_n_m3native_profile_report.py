"""lever_n_m3native_profile_report: op/bucket attribution, per-chip and total
sums, the sanity line against the log's own [PACKED-PHASE] trace_ms, and the
missing-column fallbacks - built from a synthetic fixture using the real column
names request_verifier_profile_report.py reads (OP NAME, CORE COUNT, DEVICE ID,
DEVICE KERNEL DURATION [ns], METAL TRACE ID, METAL TRACE REPLAY SESSION ID)."""

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lever_n_m3native_profile_report import (  # noqa: E402
    attribute_rows, bucket_attribution, build_report, classify_op, find_csv,
    parse_packed_rounds, select_round_rowsets,
)

COLUMNS = ('OP NAME', 'CORE COUNT', 'DEVICE ID', 'DEVICE KERNEL DURATION [ns]',
           'METAL TRACE ID', 'METAL TRACE REPLAY SESSION ID')

LOG_TWO_ROUNDS = (
    '[PACKED-PHASE] round=1 users=4 bind_ms=1.00 input_ms=2.00 trace_ms=3.30 '
    'sync_ms=0.10 readback_ms=0.20\n'
    '[PACKED-PHASE] round=2 users=4 bind_ms=1.00 input_ms=2.00 trace_ms=3.40 '
    'sync_ms=0.10 readback_ms=0.20\n'
    '[PHASE] packed_verify [0, 1, 2, 3] begin\n'
    '[PHASE] packed_verify [0, 1, 2, 3] end 3.5 ms\n'
)


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def two_round_rows():
    """Two packed rounds (METAL TRACE REPLAY SESSION ID 1 and 2) of the same
    METAL TRACE ID, on two chips, plus one large untracked prefill row (no trace
    id) that a correct attribution must exclude."""
    rows = []
    # Round 1 (replay session 1).
    rows.append(dict(zip(COLUMNS, ('sdpa', '16', '0', '1000000', '42', '1'))))
    rows.append(dict(zip(COLUMNS, ('decode_gated_delta_rule', '96', '0', '2000000', '42', '1'))))
    rows.append(dict(zip(COLUMNS, ('sdpa', '16', '1', '1100000', '42', '1'))))
    rows.append(dict(zip(COLUMNS, ('decode_gated_delta_rule', '96', '1', '2100000', '42', '1'))))
    # Round 2 (replay session 2).
    rows.append(dict(zip(COLUMNS, ('sdpa', '16', '0', '1050000', '42', '2'))))
    rows.append(dict(zip(COLUMNS, ('decode_gated_delta_rule', '96', '0', '2050000', '42', '2'))))
    rows.append(dict(zip(COLUMNS, ('sdpa', '16', '1', '1150000', '42', '2'))))
    rows.append(dict(zip(COLUMNS, ('decode_gated_delta_rule', '96', '1', '2150000', '42', '2'))))
    # Untracked prefill: no METAL TRACE ID / REPLAY SESSION ID at all, and huge -
    # a bug that folded this in would blow the sums up by two orders of magnitude.
    rows.append(dict(zip(COLUMNS, ('Matmul', '64', '0', '500000000', '', ''))))
    return rows


class ClassifyOpTests(unittest.TestCase):
    def test_full_attention_keywords(self):
        self.assertEqual(classify_op('sdpa_decode'), 'full_attention')
        self.assertEqual(classify_op('nlp_concat_heads_decode'), 'full_attention')
        self.assertEqual(classify_op('attn_decode_prep'), 'full_attention')

    def test_gdn_keywords(self):
        self.assertEqual(classify_op('decode_gated_delta_rule_packed'), 'gdn')
        self.assertEqual(classify_op('gdn_norm_gate'), 'gdn')
        self.assertEqual(classify_op('project_qkvzab_raw'), 'gdn')

    def test_mlp_keywords(self):
        self.assertEqual(classify_op('fused_mlp_down'), 'mlp')
        self.assertEqual(classify_op('lazy_gate_up_loader'), 'mlp')

    def test_norm_keywords_and_case_insensitivity(self):
        self.assertEqual(classify_op('RmsNorm'), 'norm')
        self.assertEqual(classify_op('LayerNorm'), 'norm')

    def test_collective_keywords_including_camel_case(self):
        self.assertEqual(classify_op('AllGather'), 'collective')
        self.assertEqual(classify_op('ReduceScatter'), 'collective')
        self.assertEqual(classify_op('all_gather_minimal_matmul_async'), 'collective')

    def test_sampling_keywords(self):
        self.assertEqual(classify_op('sharded_lm_head'), 'sampling_lm_head')
        self.assertEqual(classify_op('force_argmax_sampling'), 'sampling_lm_head')

    def test_unmatched_name_is_other_not_a_guess(self):
        self.assertEqual(classify_op('Matmul'), 'other')
        self.assertEqual(classify_op(''), 'other')

    def test_gdn_norm_gate_is_gdn_not_norm(self):
        # gdn is checked before the generic norm bucket, so its own norm-gate op
        # must not be reclassified into the generic norm bucket.
        self.assertEqual(classify_op('gdn_norm_gate'), 'gdn')


class AttributeRowsTests(unittest.TestCase):
    def test_per_chip_and_total_sums(self):
        rows = [row for row in two_round_rows() if row['OP NAME'] != 'Matmul']
        result = attribute_rows(rows, has_core_count=True)
        sdpa = next(entry for entry in result['ops'] if entry['op'] == 'sdpa')
        gdn = next(entry for entry in result['ops'] if entry['op'] == 'decode_gated_delta_rule')
        self.assertAlmostEqual(sdpa['per_chip_ms']['0'], 2.05)
        self.assertAlmostEqual(sdpa['per_chip_ms']['1'], 2.25)
        self.assertAlmostEqual(sdpa['total_ms'], 4.30)
        self.assertEqual(sdpa['calls'], 4)
        self.assertAlmostEqual(gdn['total_ms'], 8.30)
        self.assertAlmostEqual(result['grand_total_ms'], 12.60)
        self.assertAlmostEqual(result['per_chip_total_ms']['0'], 6.10)
        self.assertAlmostEqual(result['per_chip_total_ms']['1'], 6.50)
        self.assertEqual(result['rows_skipped_invalid_duration'], 0)

    def test_invalid_duration_is_skipped_not_zeroed(self):
        rows = [dict(zip(COLUMNS, ('sdpa', '16', '0', '', '42', '1'))),
                dict(zip(COLUMNS, ('sdpa', '16', '0', '1000000', '42', '2')))]
        result = attribute_rows(rows, has_core_count=True)
        self.assertEqual(result['rows_skipped_invalid_duration'], 1)
        self.assertEqual(result['rows_used'], 1)
        self.assertAlmostEqual(result['grand_total_ms'], 1.0)

    def test_bucket_attribution_shares_sum_to_one(self):
        rows = [row for row in two_round_rows() if row['OP NAME'] != 'Matmul']
        attribution = attribute_rows(rows, has_core_count=True)
        buckets = bucket_attribution(attribution)
        shares = sum(entry['share_of_attributed_total'] for entry in buckets)
        self.assertAlmostEqual(shares, 1.0)
        by_name = {entry['bucket']: entry for entry in buckets}
        self.assertIn('full_attention', by_name)
        self.assertIn('gdn', by_name)
        self.assertAlmostEqual(by_name['full_attention']['total_ms'], 4.30)
        self.assertAlmostEqual(by_name['gdn']['total_ms'], 8.30)


class ParsePackedRoundsTests(unittest.TestCase):
    def test_extracts_round_and_trace_ms_in_order(self):
        rounds = parse_packed_rounds(LOG_TWO_ROUNDS)
        self.assertEqual([entry['round'] for entry in rounds], [1, 2])
        self.assertEqual([entry['trace_ms'] for entry in rounds], [3.30, 3.40])
        self.assertEqual(rounds[0]['users'], 4)


class FindCsvTests(unittest.TestCase):
    def test_checks_direct_metadata_then_dot_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'metadata').mkdir()
            (root / 'metadata' / 'cpp_device_perf_report.csv').write_text('x', encoding='utf-8')
            found = find_csv(root, 'cpp_device_perf_report.csv')
            self.assertEqual(found, root / 'metadata' / 'cpp_device_perf_report.csv')

    def test_missing_everywhere_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(find_csv(Path(tmp), 'cpp_device_perf_report.csv'))


class BuildReportPerRoundTests(unittest.TestCase):
    """The full pipeline, with a CSV that carries METAL TRACE ID / REPLAY SESSION
    ID data consistent with two clean packed rounds - the best-case
    per_round_trace_replay path, including the sanity line."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.csv_path = root / 'cpp_device_perf_report.csv'
        self.log_path = root / 'm3native-gate-stdout.log'
        write_csv(self.csv_path, two_round_rows())
        self.log_path.write_text(LOG_TWO_ROUNDS, encoding='utf-8')
        self.report = build_report(root, self.log_path)

    def test_method_is_per_round_trace_replay(self):
        self.assertEqual(self.report['attribution_method'], 'per_round_trace_replay')
        self.assertIsNone(self.report.get('error'))

    def test_prefill_row_is_excluded_from_totals(self):
        # 12.60 ms, not 512.60 ms - the untracked Matmul prefill row must not leak in.
        self.assertAlmostEqual(self.report['grand_total_attributed_ms'], 12.60)

    def test_per_op_top_list_and_per_chip_breakdown(self):
        ops = {entry['op']: entry for entry in self.report['per_op']}
        self.assertAlmostEqual(ops['sdpa']['total_ms'], 4.30)
        self.assertAlmostEqual(ops['sdpa']['per_chip_ms']['0'], 2.05)
        self.assertAlmostEqual(ops['decode_gated_delta_rule']['total_ms'], 8.30)

    def test_per_bucket_grouping(self):
        buckets = {entry['bucket']: entry for entry in self.report['per_bucket']}
        self.assertAlmostEqual(buckets['full_attention']['total_ms'], 4.30)
        self.assertAlmostEqual(buckets['gdn']['total_ms'], 8.30)

    def test_sanity_line_per_round_and_mean(self):
        sanity = self.report['sanity']
        self.assertEqual(len(sanity['per_round']), 2)
        first = sanity['per_round'][0]
        self.assertEqual(first['round'], 1)
        self.assertAlmostEqual(first['reported_trace_ms'], 3.30)
        self.assertAlmostEqual(first['attributed_device_ms'], 6.2)
        self.assertAlmostEqual(first['delta_ms'], 6.2 - 3.30)
        second = sanity['per_round'][1]
        self.assertAlmostEqual(second['attributed_device_ms'], 6.4)
        self.assertAlmostEqual(sanity['mean_reported_trace_ms'], 3.35)
        self.assertAlmostEqual(sanity['mean_attributed_device_ms_per_round'], 6.30)
        self.assertAlmostEqual(self.report['mean_per_round']['grand_total_ms'], 6.30)


class BuildReportMissingColumnTests(unittest.TestCase):
    def test_missing_trace_columns_falls_back_to_averaged_over_all_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / 'cpp_device_perf_report.csv'
            log_path = root / 'log.txt'
            columns = ('OP NAME', 'CORE COUNT', 'DEVICE ID', 'DEVICE KERNEL DURATION [ns]')
            with open(csv_path, 'w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerow(dict(zip(columns, ('sdpa', '16', '0', '1000000'))))
                writer.writerow(dict(zip(columns, ('sdpa', '16', '1', '1100000'))))
            log_path.write_text(LOG_TWO_ROUNDS, encoding='utf-8')

            report = build_report(root, log_path)
            self.assertEqual(report['attribution_method'], 'averaged_over_all_rows')
            self.assertIsNone(report.get('error'))
            joined_caveats = ' '.join(report['caveats'])
            self.assertIn('METAL TRACE ID', joined_caveats)
            self.assertIn('METAL TRACE REPLAY SESSION ID', joined_caveats)
            self.assertIn('not exported by the pinned profiler', joined_caveats)
            # 2.1 ms total across both chips, divided evenly over the 2 observed rounds.
            self.assertAlmostEqual(report['grand_total_attributed_ms'], 2.1)
            self.assertAlmostEqual(report['mean_per_round']['grand_total_ms'], 1.05)
            self.assertIn('note', report['sanity'])

    def test_missing_required_duration_column_is_a_hard_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / 'cpp_device_perf_report.csv'
            log_path = root / 'log.txt'
            columns = ('OP NAME', 'DEVICE ID')
            with open(csv_path, 'w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerow(dict(zip(columns, ('sdpa', '0'))))
            log_path.write_text(LOG_TWO_ROUNDS, encoding='utf-8')

            report = build_report(root, log_path)
            self.assertIsNone(report['attribution_method'])
            self.assertEqual(report['per_op'], [])
            self.assertEqual(report['per_bucket'], [])
            self.assertIn('not exported by the pinned profiler', report['error'])
            self.assertIn('DEVICE KERNEL DURATION [ns]', report['error'])

    def test_missing_csv_entirely_is_reported_not_invented(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_path = root / 'log.txt'
            log_path.write_text(LOG_TWO_ROUNDS, encoding='utf-8')
            report = build_report(root, log_path)
            self.assertIsNone(report['cpp_device_perf_report_csv'])
            self.assertIn('not exported by the pinned profiler', report['error'])


if __name__ == '__main__':
    unittest.main()
