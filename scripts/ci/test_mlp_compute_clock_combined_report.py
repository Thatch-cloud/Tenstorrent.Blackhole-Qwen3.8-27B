import copy
import unittest

from mlp_compute_clock_combined_report import validate
from test_mlp_compute_clock_report import fixture as isolated_fixture
from dspark_intake import TAPS


def fixture():
    records = []
    for replay, position in enumerate((4096, 4107)):
        layers = []
        for index in range(64):
            capture = copy.deepcopy(isolated_fixture()['compute_clock_samples'][0])
            capture['label'] = f'verify-{replay}-layer-{index}'
            layers.append(dict(layer=index, capture=capture))
        records.append(dict(position=position, rows=16, layers=layers))
    summary = dict(diagnostic_only=True, committed_tg=None, layers=64,
        sampled_verifier_replays=2, records=records, compute_samples_releasable=True)
    request = dict(length=4096, arm='publication', exact=True, state_exact=True, inactive_exact=True,
        instrumented_timing=True, committed_tokens_per_second=None,
        fused_t16_mlp=dict(rows=16, restored=True, native_bindings_unchanged=True, hits=[2] * 64),
        gdn_norm_prefetch=dict(enabled=True), incremental_history=dict(enabled=True),
        blocks=[dict(position=4096, rows=16), dict(position=4107, rows=16)], mlp_compute_clock_combined=summary)
    for block in request['blocks']:
        block['committed'] = 11
    request['dspark'] = dict(
        proposal_checks=[dict(position=position, tensors=6, exact=True) for position in (4096, 4096, 4107)],
        feature_checks=[dict(position=position, rows=11, tap=tap, chip=chip, exact=True)
            for position in (4096, 4107) for tap in TAPS for chip in range(2)])
    request['gdn_verify_checks'] = [dict(position=position, rows=16, unchanged=True) for position in (4096, 4107)]
    return dict(passed=True, closed_cleanly=True, ctx_tokens=4096, streams=1, fresh_context_audit=True,
        pp=None, committed_tg=None, sources={'one': 'a' * 64}, sources_after={'one': 'a' * 64},
        native_sources={'two': 'b' * 64}, native_sources_after={'two': 'b' * 64},
        mlp_compute_clock_sources={'three': 'c' * 64}, mlp_compute_clock_sources_after={'three': 'c' * 64},
        request_checks=[request], mlp_compute_clock_combined=summary)


class CombinedReportTests(unittest.TestCase):
    def test_complete_report_summarizes_samples_without_claiming_tg(self):
        result = validate(fixture())
        self.assertEqual(result['sample_count'], 4608)
        self.assertIsNone(result['committed_tg'])
        self.assertTrue(all(group['median_cycles'] == 20 for group in result['groups']))

    def test_partial_capture_changed_recipe_or_timing_claim_rejected(self):
        mutations = (
            lambda report: report['request_checks'][0]['dspark']['feature_checks'].pop(),
            lambda report: report['request_checks'][0]['dspark']['proposal_checks'].pop(),
            lambda report: report['request_checks'][0]['gdn_verify_checks'].pop(),
            lambda report: report.update(committed_tg=200),
            lambda report: report.update(closed_cleanly=False),
            lambda report: report['sources_after'].update(one='different'),
            lambda report: report['request_checks'][0].update(state_exact=False),
            lambda report: report['request_checks'][0]['gdn_norm_prefetch'].update(enabled=False),
            lambda report: report['mlp_compute_clock_combined']['records'][0]['layers'].pop(),
            lambda report: report['mlp_compute_clock_combined']['records'][0].update(position=0),
            lambda report: report['mlp_compute_clock_combined']['records'][0]['layers'][0]['capture'].update(poisoned_before_execution=False),
            lambda report: report['mlp_compute_clock_combined']['records'][0]['layers'][0]['capture']['samples'][0].update(duration_cycles=30),
        )
        for mutate in mutations:
            report = fixture()
            mutate(report)
            with self.assertRaises(ValueError):
                validate(report)
