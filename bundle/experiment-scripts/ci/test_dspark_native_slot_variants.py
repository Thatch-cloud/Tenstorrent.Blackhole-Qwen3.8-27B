import copy
import unittest
from unittest.mock import patch

from dspark_native_slot_variants import POLICIES, SCHEDULE, summarize_variants
from test_dspark_score_layout_variants import ScoreLayoutVariantTests


class NativeSlotVariantTests(unittest.TestCase):
    def setUp(self):
        fixture = ScoreLayoutVariantTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        originals = fixture.requests()
        self.requests = []
        for arm, audit in SCHEDULE:
            value = copy.deepcopy(next(item for item in originals if item['arm'] == 'scores' and item['instrumented_timing'] is audit))
            value['arm'] = arm
            if arm == 'direct':
                value['native_slot_gdn'] = dict(restored=True, qualification={'synthetic': True}, rows=16,
                    calls_by_layer=[2] * 48, publication_prefixes=list(range(17)))
            self.requests.append(value)
        qualifier = patch('dspark_native_slot_variants.qualify', return_value={'synthetic': True})
        qualifier.start()
        self.addCleanup(qualifier.stop)

    def test_matched_combined_schedule(self):
        self.assertTrue(all(policy['score_layout'] for policy in POLICIES.values()))
        self.assertEqual(set(summarize_variants(self.requests)['arms']), {'control', 'direct'})

    def test_missing_layer_prefix_or_restoration_fails(self):
        for change in (dict(restored=False), dict(calls_by_layer=[2] * 47),
                dict(calls_by_layer=[0] + [2] * 47), dict(publication_prefixes=list(range(1, 17))), dict(qualification={})):
            values = copy.deepcopy(self.requests)
            values[1]['native_slot_gdn'].update(change)
            with self.assertRaises(ValueError):
                summarize_variants(values)
