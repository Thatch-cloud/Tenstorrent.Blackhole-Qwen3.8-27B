import unittest

from mlp_read_order_comparison import summarize
from mlp_read_order_report import validate, validate_audit, validate_route, summarize_requests
from mlp_read_order_report import BASELINE_SHA256, REPORT_SHA256, NORM_SHA256, HISTORY_SHA256, TAPS
from test_mlp_read_order_comparison import fixture as request_fixture


def fixture():
    requests = request_fixture()
    for request in requests:
        enabled = request['weight_read_order']['staggered']
        request['weight_read_order']['simulator_report_sha256'] = REPORT_SHA256 if enabled else None
        request['fused_t16_mlp'] = dict(passed_simulator=REPORT_SHA256 if enabled else BASELINE_SHA256,
            rows=16, layers=64, restored=True, native_bindings_unchanged=True, extra_weight_allocations=0,
            hits=[2] * 64, weight_audit=dict(passed=True, checks=[dict(layer=layer, offset=offset, chip=chip,
                exact=True, pages=43520, mismatched_words=0) for layer in range(64) for offset in (0, 1) for chip in (0, 1)]))
        request['gdn_norm_prefetch'] = dict(enabled=True, builds=96, report_sha256=NORM_SHA256)
        request['incremental_history'] = dict(enabled=True, report_sha256=HISTORY_SHA256)
        request['dspark'].update(proposal_checks=[dict(position=4096, tensors=6, exact=True)] * 2,
            feature_checks=[dict(position=4096, rows=12, tap=tap, chip=chip, exact=True) for tap in TAPS for chip in (0, 1)])
        request['gdn_verify_checks'] = [dict(position=4096, rows=16, unchanged=True)]
    comparison = summarize(requests, summarize_requests, validate_audit, validate_route)
    return dict(passed=True, closed_cleanly=True, stage='complete', streams=1, ctx_tokens=4096,
        sampler_links=4, fresh_context_audit=True, request_checks=requests, read_order_comparison=comparison,
        sources={'one': 'a' * 64}, sources_after={'one': 'a' * 64},
        native_sources={'two': 'b' * 64}, native_sources_after={'two': 'b' * 64},
        read_order_sources={'three': 'c' * 64}, read_order_sources_after={'three': 'c' * 64})


class ReadOrderReportTests(unittest.TestCase):
    def test_complete_report_recomputes_tg(self):
        self.assertEqual(validate(fixture())['arms']['staggered']['committed_tg'], 120)

    def test_missing_evidence_or_forged_rate_rejected(self):
        for mutate in (
                lambda report: report.update(closed_cleanly=False),
                lambda report: report.update(sampler_links=1),
                lambda report: report['sources_after'].update(one='changed'),
                lambda report: report['request_checks'][0]['dspark']['feature_checks'].pop(),
                lambda report: report['request_checks'][1]['fused_t16_mlp']['weight_audit']['checks'].pop(),
                lambda report: report['request_checks'][2]['gdn_norm_prefetch'].update(enabled=False),
                lambda report: report['read_order_comparison']['arms']['staggered'].update(committed_tg=200)):
            report = fixture()
            mutate(report)
            with self.assertRaises(ValueError):
                validate(report)
