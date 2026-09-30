import unittest

from gdn_qk_double_buffer_comparison import summarize
from gdn_qk_double_buffer_gate import REPORT_SHA256
from gdn_qk_double_buffer_report import validate, validate_audit, validate_route, summarize_requests
from mlp_weight_pipeline_report import BASELINE_SHA256
from test_mlp_weight_pipeline_report import fixture as baseline_fixture


def fixture():
    report = baseline_fixture()
    for request in report['request_checks']:
        enabled = request.pop('weight_weight_pipeline')['pipelined']
        request['fused_t16_mlp']['passed_simulator'] = BASELINE_SHA256
        request['gdn_qk_double_buffer'] = dict(double_buffered_qk=enabled, builds=96 if enabled else 0,
            report_sha256=REPORT_SHA256 if enabled else None, restored=True)
    report['qk_double_buffer_sources'] = report.pop('weight_pipeline_sources')
    report['qk_double_buffer_sources_after'] = report.pop('weight_pipeline_sources_after')
    report.pop('weight_pipeline_comparison')
    report['qk_double_buffer_comparison'] = summarize(report['request_checks'], summarize_requests, validate_audit, validate_route)
    return report


class DoubleBufferReportTests(unittest.TestCase):
    def test_complete_result_recomputes_committed_tg(self):
        result = validate(fixture())
        self.assertEqual(result['arms']['double_buffered_qk']['committed_tg'], 120)
        self.assertFalse(result['performance_promoted'])

    def test_missing_identity_incomplete_layers_or_forged_tg_rejected(self):
        for mutate in (lambda report: report.update(closed_cleanly=False),
                lambda report: report['request_checks'][1]['gdn_qk_double_buffer'].update(builds=48),
                lambda report: report['request_checks'][1]['gdn_qk_double_buffer'].update(restored=False),
                lambda report: report['request_checks'][1]['gdn_qk_double_buffer'].update(report_sha256='bad'),
                lambda report: report['request_checks'][1]['fused_t16_mlp']['weight_audit']['checks'].pop(),
                lambda report: report['request_checks'][1]['dspark']['feature_checks'].pop(),
                lambda report: report['qk_double_buffer_comparison']['arms']['double_buffered_qk'].update(committed_tg=200)):
            report = fixture()
            mutate(report)
            with self.assertRaises(ValueError):
                validate(report)
