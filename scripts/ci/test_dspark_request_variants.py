import copy
from pathlib import Path
import unittest

import yaml

from dspark_request_variants import POLICIES, SCHEDULE, summarize_variants


class DSparkVariantTests(unittest.TestCase):
    def requests(self):
        records = []
        for ordinal, (arm, audit) in enumerate(SCHEDULE):
            policy = POLICIES[arm]
            checks = [dict(position=4096, tensors=6, exact=True)] * 2 if audit and policy['proposal_trace'] else []
            records.append(dict(arm=arm, instrumented_timing=audit, exact=True, state_exact=True, inactive_exact=True,
                prompt_tokens=[1] * 4096, emitted=[2, 3, 4], length=4096, committed_decode_tokens=2,
                blocks=[dict(position=4096, rows=16, input_tokens=list(range(2, 18)), accepted=1, committed=2)],
                commit_only_gdn=policy['commit_only_gdn'],
                dspark=dict(proposal_trace=policy['proposal_trace'], proposal_checks=checks),
                gdn_verify_checks=[dict(position=4096, rows=16, unchanged=True)] if audit and policy['commit_only_gdn'] else [],
                decode_ms=100 + ordinal, prefill_ms=1000, feature_setup_ms=100, engine_setup_ms=200,
                prefill_setup_decode_ms=1400 + ordinal, proposed=15, accepted=1))
        return records

    def test_exact_three_arm_schedule_preserves_every_slow_sample_and_setup(self):
        report = summarize_variants(self.requests())
        self.assertEqual(report['order'], ['eager', 'trace', 'trace_commit', 'eager', 'trace', 'trace_commit',
            'trace_commit', 'trace', 'eager'])
        self.assertEqual(set(report['arms']), set(POLICIES))
        for arm, policy in POLICIES.items():
            result = report['arms'][arm]
            self.assertEqual(result['committed_tg'], 4000 / 211)
            self.assertEqual(result['mean_setup_inclusive_ms'], 1405.5)
            self.assertEqual(result['proposal_trace'], policy['proposal_trace'])
            self.assertEqual(result['commit_only_gdn'], policy['commit_only_gdn'])
        self.assertFalse(report['serving_qualified'])
        self.assertFalse(report['held_out_coding_quality'])

    def test_missing_misordered_changed_policy_or_native_mismatch_never_qualifies(self):
        for index, field, altered in ((3, 'arm', 'trace'), (4, 'instrumented_timing', True),
                (7, 'commit_only_gdn', True), (5, 'exact', False), (8, 'emitted', [2, 3, 8]),
                (6, 'state_exact', False), (6, 'decode_ms', float('nan'))):
            records = self.requests()
            records[index][field] = altered
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize_variants(records)
        with self.assertRaises(ValueError):
            summarize_variants(self.requests()[:-1])

    def test_missing_replay_or_premature_gdn_write_fails_the_instrumented_arm(self):
        baseline = self.requests()
        for mutation in ('missing_replay', 'wrong_position', 'missing_gdn', 'changed_gdn', 'wrong_rows'):
            records = copy.deepcopy(baseline)
            if mutation == 'missing_replay':
                records[1]['dspark']['proposal_checks'].pop()
            elif mutation == 'wrong_position':
                records[1]['dspark']['proposal_checks'][0]['position'] = 4109
            elif mutation == 'missing_gdn':
                records[2]['gdn_verify_checks'] = []
            elif mutation == 'changed_gdn':
                records[2]['gdn_verify_checks'][0]['unchanged'] = False
            else:
                records[2]['gdn_verify_checks'][0]['rows'] = 8
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                summarize_variants(records)

    def test_commit_only_cannot_claim_a_gain_from_changed_proposals_or_acceptance(self):
        for change in ('repeat', 'state_policy'):
            records = self.requests()
            selected = [records[7]] if change == 'repeat' else [value for value in records if value['arm'] == 'trace_commit']
            for record in selected:
                record['blocks'][0]['input_tokens'][-1] = 18
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'audited proposals'):
                summarize_variants(records)

    def test_ci_route_is_opt_in_and_preserves_original_control_suite(self):
        root = Path(__file__).resolve().parents[2]
        workflow = yaml.safe_load((root / '.github/workflows/qwen-experiments.yml').read_text())
        dispatch = workflow.get('on', workflow.get(True))['workflow_dispatch']['inputs']['suite']
        self.assertIn('dspark-request-variants', dispatch['options'])
        self.assertIn('dspark-request', dispatch['options'])
        self.assertEqual(dispatch['default'], 'inventory')
        suite = (root / 'scripts/ci/dspark-hardware-suite.sh').read_text()
        self.assertIn('request_options+=(--request-variants)', suite)
        self.assertIn('report_name=dspark-request-variants-hardware', suite)


if __name__ == '__main__':
    unittest.main()
