import unittest

from fused_t16_admission import REPORT_SHA256 as FUSION_SHA256
from shared_qk_norm_comparison_scope import SCATTER_SHA256, PREFETCH_SHA256
from shared_qk_norm_report import validate, validate_route, validate_audit, summarize, summarize_requests
from shared_qk_norm_report import TAIL_SHA256
from test_mlp_read_order_report import fixture as read_fixture


def fixture():
    report = read_fixture()
    report.pop('read_order_comparison')
    report['norm_reader_sources'] = report.pop('read_order_sources')
    report['norm_reader_sources_after'] = report.pop('read_order_sources_after')
    for request in report['request_checks']:
        enabled = request.pop('weight_read_order')['staggered']
        request.pop('gdn_norm_prefetch')
        expected = SCATTER_SHA256 if enabled else PREFETCH_SHA256
        request['norm_reader'] = dict(scatter=enabled, simulator_report_sha256=expected,
            builds=96, restored=True)
        request['fused_t16_mlp']['passed_simulator'] = FUSION_SHA256
        request['gdn_shared_qk'] = dict(restored=True, released=True,
            admission=dict(report_sha256=expected),
            loads=[dict(rows=16, programs=3, retained_preparation_buffers=2)] * 96)
        request['draft_tail'] = dict(enabled=True, calls=20, report_sha256=TAIL_SHA256)
        request['incremental_history']['records'] = [dict(initial_position=4096, capacity=4352,
            failed=False, restored=True, prepared=16, committed=12, discarded=4, max_touched_rows=32)]
    report['norm_reader_comparison'] = summarize(report['request_checks'], summarize_requests, validate_audit, validate_route)
    return report


class NormReportTests(unittest.TestCase):
    def test_complete_report(self):
        result = validate(fixture())
        self.assertEqual(result['arms']['prefetched']['committed_tg'], 100)
        self.assertEqual(result['arms']['scatter']['committed_tg'], 120)

    def test_reject_missing_winning_components_or_evidence(self):
        for mutate in (
                lambda report: report.update(closed_cleanly=False),
                lambda report: report['request_checks'][0]['gdn_shared_qk']['loads'].pop(),
                lambda report: report['request_checks'][1]['norm_reader'].update(simulator_report_sha256=PREFETCH_SHA256),
                lambda report: report['request_checks'][1]['dspark']['feature_checks'].pop(),
                lambda report: report['request_checks'][2]['draft_tail'].update(enabled=False),
                lambda report: report['request_checks'][3]['incremental_history']['records'][0].update(discarded=0),
                lambda report: report['request_checks'][4]['fused_t16_mlp']['weight_audit']['checks'].pop(),
                lambda report: report['norm_reader_comparison']['arms']['scatter'].update(committed_tg=200)):
            report = fixture()
            mutate(report)
            with self.assertRaises(ValueError):
                validate(report)
