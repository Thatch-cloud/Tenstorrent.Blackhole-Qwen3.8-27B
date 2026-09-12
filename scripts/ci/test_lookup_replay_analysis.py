import unittest

from lookup_replay_analysis import analyze, replay


class LookupReplayTests(unittest.TestCase):
    def fixture(self):
        request = dict(prompt_tokens=[1, 2, 3] * 12, emitted=[1, 2, 3, 1, 2, 3, 1, 2],
            committed_decode_tokens=7, exact=True, state_exact=True, inactive_exact=True, family_routing=False)
        request['blocks'] = replay(request)
        return request

    def test_known_suffix_can_commit_multiple_tokens(self):
        request = self.fixture()
        self.assertEqual(sum(block['committed'] for block in request['blocks']), 7)
        self.assertGreater(sum(block['accepted'] for block in request['blocks']), 0)
        self.assertTrue(analyze(request)['recorded_routing_exact'])

    def test_target_only_has_no_proposals_or_acceptance(self):
        blocks = replay(self.fixture(), max_rows=1)
        self.assertEqual(len(blocks), 7)
        self.assertTrue(all(block['source'] == 'target' and block['accepted'] == 0 for block in blocks))

    def test_mismatched_record_cannot_be_used_for_counterfactual_claim(self):
        request = self.fixture()
        request['blocks'][0]['accepted'] += 1
        with self.assertRaisesRegex(ValueError, 'reproduce every'):
            analyze(request)

    def test_future_oracle_does_not_select_first_proposal(self):
        request = self.fixture()
        original = replay(request)[0]
        request['emitted'][1:] = [9] * 7
        changed = replay(request)[0]
        self.assertEqual(changed['input_tokens'], original['input_tokens'])
        self.assertEqual(changed['accepted'], 0)

    def test_rejects_unqualified_measurements_and_policy(self):
        for key, value in (('family_routing', True), ('exact', False), ('committed_decode_tokens', 99)):
            request = self.fixture()
            request[key] = value
            with self.assertRaises(ValueError):
                replay(request)
        for options in (dict(min_match=True), dict(min_match=0), dict(max_rows=3)):
            with self.assertRaises(ValueError):
                replay(self.fixture(), **options)
