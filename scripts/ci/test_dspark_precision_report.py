import copy
import json
from pathlib import Path
import unittest

from dspark_precision_comparison import summarize
from dspark_precision_report import validate, validate_audit, validate_route, summarize_requests
from mlp_weight_pipeline_report import BASELINE_SHA256, TAPS
from test_dspark_precision_comparison import fixture as request_fixture
from test_mlp_weight_pipeline_report import fixture as baseline_fixture


def fixture():
    report = baseline_fixture()
    baseline = report['request_checks'][0]
    requests = request_fixture()
    reviewed = json.loads(Path(__file__).with_name('dspark-precision-reviewed.json').read_bytes())
    for request in requests:
        for name in ('fused_t16_mlp', 'gdn_norm_prefetch', 'incremental_history'):
            request[name] = copy.deepcopy(baseline[name])
        request['fused_t16_mlp']['passed_simulator'] = BASELINE_SHA256
        candidate = request['drafter_precision'] == 'hifi2'
        request['drafter_precision_evidence'] = dict(closed=True,
            layer_calls=10 if candidate else 0, report_sha256=reviewed if candidate else None)
        blocks = request['blocks']
        request['dspark'].update(proposal_checks=[dict(position=position, tensors=6, exact=True)
            for position in [4096, *(block['position'] for block in blocks)]],
            feature_checks=[dict(position=block['position'], rows=block['committed'], tap=tap, chip=chip, exact=True)
                for block in blocks for tap in TAPS for chip in (0, 1)])
        request['gdn_verify_checks'] = [dict(position=block['position'], rows=16, unchanged=True) for block in blocks]
    report.pop('weight_pipeline_comparison')
    report['request_checks'] = requests
    report['drafter_precision_sources'] = report.pop('weight_pipeline_sources')
    report['drafter_precision_sources_after'] = report.pop('weight_pipeline_sources_after')
    report['drafter_precision_comparison'] = summarize(requests, summarize_requests, validate_audit, validate_route)
    return report


class PrecisionReportTests(unittest.TestCase):
    def test_different_acceptance_with_exact_target_recomputes(self):
        result = validate(fixture())
        self.assertEqual(result['arms']['hifi2']['committed_tg'], 120)
        self.assertNotEqual(result['arms']['hifi2']['acceptance'], result['arms']['hifi4']['acceptance'])
        self.assertFalse(result['performance_promoted'])

    def test_missing_correctness_source_identity_or_false_rate_rejected(self):
        for mutate in (lambda report: report.update(closed_cleanly=False),
                lambda report: report['request_checks'][1]['dspark']['feature_checks'].pop(),
                lambda report: report['request_checks'][1]['fused_t16_mlp']['weight_audit']['checks'].pop(),
                lambda report: report['request_checks'][1]['drafter_precision_evidence'].update(layer_calls=0),
                lambda report: report['request_checks'][1]['drafter_precision_evidence'].update(report_sha256={}),
                lambda report: report['native_sources_after'].clear(),
                lambda report: report['drafter_precision_comparison']['arms']['hifi2'].update(committed_tg=200)):
            report = fixture()
            mutate(report)
            with self.assertRaises(ValueError):
                validate(report)
