import copy
import unittest

from dspark_fusion_variants import POLICIES, SCHEDULE, summarize_variants
from fused_t16_admission import REPORT_SHA256
import test_dspark_score_layout_variants as fixtures


class FusionVariantTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.ScoreLayoutVariantTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        originals = fixture.requests()
        self.requests = []
        for arm, audit in SCHEDULE:
            value = copy.deepcopy(next(item for item in originals
                if item['arm'] == 'scores' and item['instrumented_timing'] is audit))
            value['arm'] = arm
            if arm == 'fusion':
                value['fused_t16_mlp'] = dict(restored=True, native_bindings_unchanged=True,
                    passed_simulator=REPORT_SHA256, rows=16, layers=64, extra_weight_allocations=0,
                    hits=[2] * 64, weight_audit=dict(passed=True, checks=[
                        dict(layer=layer, offset=offset, chip=chip, exact=True, pages=43520)
                        for layer in range(64) for offset in (0, 1) for chip in (0, 1)]))
            self.requests.append(value)

    def test_matched_full_request_schedule(self):
        self.assertTrue(all(policy['score_layout'] for policy in POLICIES.values()))
        self.assertEqual(set(summarize_variants(self.requests)['arms']), {'control', 'fusion'})

    def test_unqualified_layers_and_weight_coverage_fail(self):
        for change in (dict(restored=False), dict(hits=[0] + [2] * 63),
                dict(native_bindings_unchanged=False), dict(passed_simulator='other'),
                dict(weight_audit=dict(passed=True, checks=[]))):
            values = copy.deepcopy(self.requests)
            values[1]['fused_t16_mlp'].update(change)
            with self.assertRaises(ValueError):
                summarize_variants(values)

    def test_duplicate_weight_check_cannot_hide_missing_chip(self):
        values = copy.deepcopy(self.requests)
        checks = values[1]['fused_t16_mlp']['weight_audit']['checks']
        checks[1] = copy.deepcopy(checks[0])
        with self.assertRaises(ValueError):
            summarize_variants(values)
