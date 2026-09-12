"""Exact lookup policy and request lifecycle, without accelerator dependencies."""

import unittest
import random

from lookup_draft import LookupDraft


class LookupTests(unittest.TestCase):
    def test_match_telemetry_is_current_and_does_not_change_proposals(self):
        draft = LookupDraft('request', [1, 2, 3, 4, 2, 9, 1, 2])
        self.assertEqual(draft.propose_with_match('request', 3), ([3, 4, 2], 2))
        self.assertEqual(draft.propose('request', 3), [3, 4, 2])
        draft.commit('request', [99])
        self.assertEqual(draft.propose_with_match('request', 3), ([], 0))
        with self.assertRaises(ValueError):
            draft.propose_with_match('other', 3)

    def test_longest_suffix_wins(self):
        draft = LookupDraft("request", [1, 2, 3, 4, 2, 9, 1, 2])
        self.assertEqual(draft.propose("request", 3), [3, 4, 2])

    def test_most_recent_match_breaks_ties(self):
        draft = LookupDraft("request", [1, 2, 3, 1, 2, 4, 1, 2])
        self.assertEqual(draft.propose("request", 1), [4])

    def test_full_suffix_opt_in_uses_only_known_contiguous_history(self):
        history = [0, 1, 2] * 100
        default = LookupDraft('request', history, max_proposals=31)
        candidate = LookupDraft('request', history, max_proposals=31, prefer_full_suffix=True)
        self.assertEqual(default.propose('request', 31), [0, 1, 2])
        self.assertEqual(candidate.propose('request', 31), (history * 2)[:31])
        self.assertEqual(candidate.history, history)
        self.assertIn(candidate.propose('request', 31), [history[start:start + 31] for start in range(len(history))])

    def test_longer_match_still_beats_longer_available_continuation(self):
        history = [0, 1, 2] * 12 + [0]
        candidate = LookupDraft('request', history, max_proposals=31, prefer_full_suffix=True)
        self.assertEqual(candidate.propose('request', 31), [1, 2, 0])
        with self.assertRaises(ValueError):
            LookupDraft('request', history, prefer_full_suffix=1)

    def test_full_suffix_policy_matches_exhaustive_rank(self):
        generator = random.Random(456)
        for _ in range(500):
            history = [generator.randrange(3) for _ in range(generator.randrange(60))]
            candidates = []
            for end in range(max(0, len(history) - 1)):
                for length in range(1, min(8, end + 1, len(history) - 1) + 1):
                    if history[end - length + 1:end + 1] == history[-length:]:
                        candidates.append((length, min(7, len(history) - end - 1), end))
            expected = []
            if candidates:
                end = max(candidates)[2]
                expected = history[end + 1:end + 8]
            candidate = LookupDraft('request', history, match_limit=8, prefer_full_suffix=True)
            self.assertEqual(candidate.propose('request', 7), expected)
            self.assertEqual(candidate.propose_with_match('request', 7)[1], max(candidates)[0] if candidates else 0)

    def test_no_match_falls_back(self):
        self.assertEqual(LookupDraft("request", [1, 2, 3]).propose("request", 7), [])

    def test_only_committed_history_changes_proposals(self):
        draft = LookupDraft("request", [1, 2, 3, 1, 2])
        original = list(draft.history)
        draft.propose("request", 3)
        self.assertEqual(draft.history, original)
        draft.commit("request", [3])
        self.assertEqual(draft.history, original + [3])

    def test_owner_cannot_read_or_write_other_request(self):
        draft = LookupDraft("request", [1, 2, 1])
        for action in (lambda: draft.propose("other", 1), lambda: draft.commit("other", [3]),
                       lambda: draft.close("other")):
            with self.assertRaises(ValueError):
                action()

    def test_cancel_clears_history_and_rejects_reuse(self):
        draft = LookupDraft("request", [1, 2, 1])
        draft.close("request")
        self.assertEqual(draft.history, [])
        with self.assertRaises(ValueError):
            draft.propose("request", 1)

    def test_history_is_bounded(self):
        draft = LookupDraft("request", range(10), history_limit=4)
        self.assertEqual(draft.history, [6, 7, 8, 9])

    def test_invalid_ids_do_not_mutate_history(self):
        draft = LookupDraft("request", [1])
        for token in (-1, True, 1.5, "1"):
            with self.assertRaises(ValueError):
                draft.commit("request", [token])
            self.assertEqual(draft.history, [1])

    def test_no_shared_history(self):
        first = LookupDraft("first", [1, 2, 1])
        second = LookupDraft("second", [5])
        self.assertEqual(second.propose("second", 1), [])
        first.commit("first", [7])
        self.assertEqual(second.history, [5])

    def test_proposals_are_bounded(self):
        draft = LookupDraft("request", [1, 2, 3, 4, 1])
        self.assertEqual(draft.propose("request", 1), [2])
        for count in (0, 16, True):
            with self.assertRaises(ValueError):
                draft.propose("request", count)

    def test_matches_exhaustive_policy(self):
        generator = random.Random(123)
        for _ in range(500):
            history = [generator.randrange(5) for _ in range(generator.randrange(40))]
            expected = []
            for length in range(min(8, len(history) - 1), 0, -1):
                suffix_start = len(history) - length
                for start in range(suffix_start - 1, -1, -1):
                    if history[start:start + length] == history[suffix_start:]:
                        expected = history[start + length:start + length + 7]
                        break
                if expected:
                    break
            self.assertEqual(LookupDraft("request", history, match_limit=8).propose("request", 7), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
