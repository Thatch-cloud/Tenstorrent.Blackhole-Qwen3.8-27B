from copy import deepcopy
import unittest

from fused_t16_admission import REPORT_SHA256 as MLP_SHA256
from gdn_recurrence_clock_combined_report import LAYERS, NORM_SHA256, HISTORY_SHA256, validate
from gdn_recurrence_clock_gate import KERNEL, REPORT_SHA256
from test_gdn_recurrence_clock_report import fixture as isolated_fixture
from test_mlp_compute_clock_combined_report import fixture as request_fixture


def fixture():
    report = request_fixture()
    report.update(stage='complete', sampler_links=4,
        gdn_recurrence_clock_sources={'helper': 'a' * 64},
        gdn_recurrence_clock_sources_after={'helper': 'a' * 64})
    records = []
    for replay, position in enumerate((4096, 4107)):
        layers = []
        for layer in LAYERS:
            capture = deepcopy(isolated_fixture()['recurrence_clock_samples'][0])
            capture['label'] = f'verify-{replay}-layer-{layer}'
            layers.append(dict(layer=layer, capture=capture))
        records.append(dict(position=position, rows=16, layers=layers))
    summary = dict(diagnostic_only=True, committed_tg=None, report_sha256=REPORT_SHA256,
        layers=list(LAYERS), sampled_verifier_replays=2, samples_releasable=True, records=records,
        builds=[dict(layer=LAYERS[index % 48], ordinal=index % 48, kernel=dict(KERNEL)) for index in range(96)])
    report['gdn_recurrence_clock_combined'] = summary
    request = report['request_checks'][0]
    request.update(gdn_recurrence_clock_combined=summary,
        gdn_norm_prefetch=dict(enabled=True, builds=96, report_sha256=NORM_SHA256),
        incremental_history=dict(enabled=True, report_sha256=HISTORY_SHA256))
    request['fused_t16_mlp'].update(passed_simulator=MLP_SHA256, layers=64, extra_weight_allocations=0,
        weight_audit=dict(passed=True, checks=[dict(layer=layer, offset=offset, chip=chip,
            exact=True, pages=43520, mismatched_words=0)
            for layer in range(64) for offset in (0, 1) for chip in range(2)]))
    return report


class CombinedRecurrenceReportTests(unittest.TestCase):
    def test_complete_combined_diagnostic_is_not_throughput(self):
        result = validate(fixture())
        self.assertEqual(result['sample_count'], 4032)
        self.assertEqual(result['layers'], 48)
        self.assertEqual(len(result['groups']), 42)
        self.assertTrue(all(group['samples'] == 96 and group['median_cycles'] == 20 for group in result['groups']))
        self.assertIsNone(result['committed_tg'])

    def test_incomplete_samples_recipe_changes_and_numerical_failure_rejected(self):
        mutations = (
            lambda report: report.update(committed_tg=200),
            lambda report: report.update(sampler_links=1),
            lambda report: report.update(closed_cleanly=False),
            lambda report: report['gdn_recurrence_clock_sources_after'].clear(),
            lambda report: report['request_checks'][0].update(state_exact=False),
            lambda report: report['request_checks'][0]['gdn_norm_prefetch'].update(builds=48),
            lambda report: report['request_checks'][0]['incremental_history'].update(enabled=False),
            lambda report: report['request_checks'][0]['dspark']['feature_checks'].pop(),
            lambda report: report['request_checks'][0]['dspark']['proposal_checks'].pop(),
            lambda report: report['request_checks'][0]['fused_t16_mlp']['weight_audit']['checks'].pop(),
            lambda report: report['gdn_recurrence_clock_combined']['builds'][0]['kernel'].update(token=0),
            lambda report: report['gdn_recurrence_clock_combined']['records'][0]['layers'].pop(),
            lambda report: report['gdn_recurrence_clock_combined']['records'][0].update(position=0),
            lambda report: report['gdn_recurrence_clock_combined']['records'][0]['layers'][0]['capture'].update(poisoned_before_execution=False),
            lambda report: report['gdn_recurrence_clock_combined']['records'][0]['layers'][0]['capture']['samples'][0].update(duration_cycles=21),
        )
        for mutate in mutations:
            report = fixture()
            mutate(report)
            with self.assertRaises(ValueError):
                validate(report)


if __name__ == '__main__':
    unittest.main()
