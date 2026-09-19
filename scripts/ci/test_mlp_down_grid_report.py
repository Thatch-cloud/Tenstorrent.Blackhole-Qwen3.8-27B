import unittest

from mlp_down_grid_comparison import summarize
from mlp_down_grid_gate import REPORT_SHA256
from mlp_down_grid_report import validate, validate_audit, validate_route, summarize_requests
from mlp_weight_pipeline_report import BASELINE_SHA256
from test_mlp_weight_pipeline_report import fixture as baseline_fixture


def fixture():
    report = baseline_fixture()
    for request in report['request_checks']:
        enabled = request.pop('weight_weight_pipeline')['pipelined']
        request['fused_t16_mlp']['passed_simulator'] = BASELINE_SHA256
        request['mlp_down_grid'] = dict(wider_down=enabled,
            hits=list(request['fused_t16_mlp']['hits']) if enabled else [0] * 64,
            report_sha256=REPORT_SHA256 if enabled else None, restored=True)
    report['mlp_down_grid_sources'] = report.pop('weight_pipeline_sources')
    report['mlp_down_grid_sources_after'] = report.pop('weight_pipeline_sources_after')
    report.pop('weight_pipeline_comparison')
    report['mlp_down_grid_comparison'] = summarize(report['request_checks'], summarize_requests, validate_audit, validate_route)
    return report


class DownGridReportTests(unittest.TestCase):
    def test_complete_result_recomputes_committed_tg(self):
        result = validate(fixture())
        self.assertEqual(result['arms']['wider_down']['committed_tg'], 120)
        self.assertFalse(result['performance_promoted'])

    def test_missing_identity_incomplete_layers_or_forged_tg_rejected(self):
        for mutate in (lambda report: report.update(closed_cleanly=False),
                lambda report: report['request_checks'][1]['mlp_down_grid'].update(hits=[0] * 64),
                lambda report: report['request_checks'][1]['mlp_down_grid'].update(restored=False),
                lambda report: report['request_checks'][1]['mlp_down_grid'].update(report_sha256='bad'),
                lambda report: report['request_checks'][1]['fused_t16_mlp']['weight_audit']['checks'].pop(),
                lambda report: report['request_checks'][1]['dspark']['feature_checks'].pop(),
                lambda report: report['mlp_down_grid_comparison']['arms']['wider_down'].update(committed_tg=200)):
            report = fixture()
            mutate(report)
            with self.assertRaises(ValueError):
                validate(report)
