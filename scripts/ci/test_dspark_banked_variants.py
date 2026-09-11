import copy
import unittest
from unittest.mock import patch

from dspark_banked_variants import POLICIES, SCHEDULE, summarize_variants
from test_dspark_score_layout_variants import ScoreLayoutVariantTests


class BankedVariantTests(unittest.TestCase):
    def setUp(self):
        fixture = ScoreLayoutVariantTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        originals = fixture.requests()
        self.requests = []
        for arm, audit in SCHEDULE:
            value = copy.deepcopy(next(item for item in originals
                if item['arm'] == 'scores' and item['instrumented_timing'] is audit))
            value['arm'] = arm
            if arm == 'banked':
                count = sum(block['rows'] > 1 for block in value['blocks'])
                value['dspark']['banked_proposal'] = dict(evidence={'synthetic': True}, replay_counts=[1, count])
            self.requests.append(value)
        qualifier = patch('dspark_banked_gate.qualify', return_value={'synthetic': True})
        qualifier.start()
        self.addCleanup(qualifier.stop)

    def test_complete_schedule_preserves_fused_scores_in_both_arms(self):
        self.assertTrue(all(policy['score_layout'] for policy in POLICIES.values()))
        result = summarize_variants(self.requests)
        self.assertEqual(set(result['arms']), {'control', 'banked'})
        self.assertEqual(result['committed_tg_change_percent'], 0)

    def test_missing_bank_usage_scope_and_cross_arm_changes_fail(self):
        for mutate in (
                lambda values: values[1]['dspark']['banked_proposal'].update(replay_counts=[0, 3]),
                lambda values: values[1]['dspark']['banked_proposal'].update(replay_counts=[1, 999]),
                lambda values: values[1]['dspark']['banked_proposal'].update(evidence={}),
                lambda values: values[0].pop('score_layout'),
                lambda values: values[0]['dspark'].update(banked_proposal={}),
                lambda values: values[1]['blocks'][0]['input_tokens'].__setitem__(-1, 999)):
            values = copy.deepcopy(self.requests)
            mutate(values)
            with self.assertRaises(ValueError):
                summarize_variants(values)
