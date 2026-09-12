import unittest
from pathlib import Path

import yaml

from dspark_combined_variants import POLICIES, summarize_variants
from test_dspark_norm_request_variants import NormRequestVariantTests


class CombinedVariantTests(unittest.TestCase):
    def requests(self):
        records = NormRequestVariantTests().requests()
        for value in records:
            value.update(target_attention_t16=True, attention_replay=True, family_routing=True)
        return records

    def test_both_policies_keep_parallel_attention_enabled(self):
        self.assertEqual(set(POLICIES), {'control', 'scatter'})
        for policy in POLICIES.values():
            self.assertIs(policy['target_attention_t16'], True)
            self.assertIs(policy['native_attention'], True)
            self.assertIs(policy['commit_only_gdn'], True)
        result = summarize_variants(self.requests())
        self.assertEqual(result['committed_tg_change_percent'], 0)
        self.assertFalse(result['serving_qualified'])

    def test_missing_parallel_attention_rejected(self):
        for field in ('target_attention_t16', 'attention_replay', 'family_routing'):
            records = self.requests()
            records[0][field] = False
            with self.assertRaises(ValueError):
                summarize_variants(records)

    def test_missing_norm_engagement_rejected(self):
        records = self.requests()
        records[1]['norm_scatter_kernel'] = None
        with self.assertRaises(ValueError):
            summarize_variants(records)

    def test_combined_ci_route_is_explicit_and_keeps_inventory_default(self):
        root = Path(__file__).resolve().parents[2]
        workflow = yaml.safe_load((root / '.github/workflows/qwen-experiments.yml').read_text())
        dispatch = workflow.get('on', workflow.get(True))['workflow_dispatch']
        self.assertEqual(dispatch['inputs']['suite']['default'], 'inventory')
        self.assertIn('dspark-combined-request', dispatch['inputs']['suite']['options'])
        steps = [step for job in workflow['jobs'].values() for step in job.get('steps', [])
            if step.get('run', '').endswith('bash scripts/ci/run-dspark-hardware.sh')]
        self.assertEqual(len(steps), 1)
        self.assertIn("inputs.suite == 'dspark-combined-request'", steps[0]['if'])
        self.assertIn("inputs.suite == 'dspark-combined-request' && 'request-combined'",
            steps[0]['env']['QWEN_DSPARK_MODE'])
        suite = (root / 'scripts/ci/dspark-hardware-suite.sh').read_text()
        self.assertIn('report_name=dspark-combined-request-hardware', suite)
        self.assertIn('request_options+=(--combined-variants)', suite)


if __name__ == '__main__':
    unittest.main()
