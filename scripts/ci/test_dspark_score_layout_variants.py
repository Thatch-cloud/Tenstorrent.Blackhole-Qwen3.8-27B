import copy
import unittest
from unittest.mock import patch

from dspark_score_layout_variants import POLICIES, SCHEDULE, summarize_variants
from test_dspark_request_variants import DSparkVariantTests


class ScoreLayoutVariantTests(unittest.TestCase):
    def setUp(self):
        patcher = patch('dspark_score_layout_hardware_gate.qualify', return_value={'test_gate': True})
        patcher.start()
        self.addCleanup(patcher.stop)
        validator = patch('dspark_score_layout_hardware_gate.validate_hardware', return_value='digest')
        validator.start()
        self.addCleanup(validator.stop)

    def requests(self):
        existing = DSparkVariantTests().requests()
        result = []
        for arm, audit in SCHEDULE:
            value = copy.deepcopy(next(entry for entry in existing
                if entry['arm'] == 'trace_commit' and entry['instrumented_timing'] is audit))
            value['arm'] = arm
            value['dspark']['native_attention'] = True
            for field in ('target_attention_t16', 'attention_replay', 'family_routing'):
                value[field] = True
            if arm == 'scores':
                value['score_layout'] = dict(calls=2, restored=True, qualification={'test_gate': True},
                    hardware_audit={}, hardware_audit_sha256='digest')
            result.append(value)
        return result

    def test_only_score_layout_changes_between_policies(self):
        self.assertIs(POLICIES['control']['score_layout'], False)
        self.assertIs(POLICIES['scores']['score_layout'], True)
        self.assertEqual({key: value for key, value in POLICIES['control'].items() if key != 'score_layout'},
            {key: value for key, value in POLICIES['scores'].items() if key != 'score_layout'})

    def test_complete_matched_schedule(self):
        result = summarize_variants(self.requests())
        self.assertEqual(set(result['arms']), {'control', 'scores'})
        self.assertEqual(result['committed_tg_change_percent'], 0)
        self.assertFalse(result['serving_qualified'])

    def test_missing_failed_or_stale_scope_rejected(self):
        for mutation in (
                lambda records: records[1].pop('score_layout'),
                lambda records: records[1]['score_layout'].update(restored=False),
                lambda records: records[1]['score_layout'].update(calls=0),
                lambda records: records[1]['score_layout'].update(qualification={}),
                lambda records: records[0].update(score_layout=records[1]['score_layout']),
                lambda records: records[1].update(target_attention_t16=False),
                lambda records: records[1].update(state_exact=False),
                lambda records: records.pop()):
            records = self.requests()
            mutation(records)
            with self.assertRaises(ValueError):
                summarize_variants(records)

    def test_cross_arm_proposal_changes_rejected(self):
        records = self.requests()
        for record in records:
            if record['arm'] == 'scores':
                record['blocks'][0]['input_tokens'][-1] = 99
        with self.assertRaises(ValueError):
            summarize_variants(records)


if __name__ == '__main__':
    unittest.main()
