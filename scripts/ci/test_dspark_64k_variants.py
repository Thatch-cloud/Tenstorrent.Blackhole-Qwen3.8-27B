import unittest
from unittest.mock import patch

import dspark_64k_variants as candidate


class VariantTests(unittest.TestCase):
    def test_both_arms_keep_capture_without_folded_target(self):
        for policy in candidate.POLICIES.values():
            self.assertTrue(policy['captured_publication'])
            self.assertTrue(policy['native_attention'])
            self.assertTrue(policy['proposal_trace'])
            self.assertTrue(policy['commit_only_gdn'])
            self.assertFalse(policy.get('target_attention_t16', False))

    def test_publication_audits_are_required_before_norm_summary(self):
        record = dict(prompt_tokens=[1] * 65536, length=65536, instrumented_timing=True, blocks=[{}],
            captured_publication=dict(enabled=True, checks=[dict(exact=True, tensors=20)] * 2))
        with patch.object(candidate, 'summarize_norm', return_value={'verified': True}) as summarize:
            self.assertEqual(candidate.summarize_variants([record]), {'verified': True})
            record['captured_publication']['checks'] = []
            with self.assertRaises(ValueError):
                candidate.summarize_variants([record])
            self.assertEqual(summarize.call_count, 1)

    def test_rejects_scalar_count_and_inconsistent_length(self):
        record = dict(prompt_tokens=65536, length=65536, instrumented_timing=False,
            blocks=[], captured_publication=dict(enabled=True, checks=[]))
        with patch.object(candidate, 'summarize_norm') as summarize:
            for tokens, length in ((65536, 65536), ([1] * 65535, 65536),
                    ([1] * 65536, 8192), ([True] * 65536, 65536)):
                record.update(prompt_tokens=tokens, length=length)
                with self.assertRaises(ValueError):
                    candidate.summarize_variants([record])
            summarize.assert_not_called()
