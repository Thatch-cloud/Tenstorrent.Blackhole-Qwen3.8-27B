"""Routing contract tests; these do not certify accelerator verification."""

import unittest
from unittest.mock import Mock

from hybrid_draft import HybridDraft, Proposal


class HybridTests(unittest.TestCase):
    def make(self, prompt=(1, 2, 3), **options):
        return HybridDraft("request", prompt, vocab_size=10, **options)

    def propose(self, draft, **options):
        return draft.propose("request", 3, greedy=True, verifier_ready=True, **options)

    def test_lookup_skips_all_neural_work(self):
        adapter = Mock(side_effect=AssertionError("Must not run"))
        draft = self.make((1, 2, 3, 1, 2), neural={"mtp": adapter})
        self.assertEqual(self.propose(draft, selected="mtp"), Proposal("lookup", (3, 1, 2)))
        adapter.assert_not_called()

    def test_only_selected_adapter_runs(self):
        first, second = Mock(return_value=[4, 5]), Mock()
        draft = self.make(neural={"first": first, "second": second})
        self.assertEqual(self.propose(draft, selected="first"), Proposal("first", (4, 5)))
        first.assert_called_once_with("request", (1, 2, 3), 3)
        second.assert_not_called()
        self.assertEqual(draft.lookup.history, [1, 2, 3])

    def test_default_and_unsupported_modes_do_not_draft(self):
        adapter = Mock()
        draft = self.make((1, 2, 1), neural={"mtp": adapter})
        for options in ({}, {"greedy": True}, {"verifier_ready": True},
                        {"greedy": 1, "verifier_ready": True}):
            self.assertEqual(draft.propose("request", 3, selected="mtp", **options),
                             Proposal("target", ()))
        adapter.assert_not_called()

    def test_unavailable_and_empty_neural_fall_back(self):
        draft = self.make(neural={"empty": Mock(return_value=[])})
        for selected in (None, "missing", "empty"):
            self.assertEqual(self.propose(draft, selected=selected), Proposal("target", ()))

    def test_invalid_neural_output_fails_without_commit(self):
        for output in ([True], [-1], [10], [1.5], [1, 2, 3, 4]):
            draft = self.make(neural={"bad": Mock(return_value=output)})
            with self.assertRaises(ValueError):
                self.propose(draft, selected="bad")
            self.assertEqual(draft.lookup.history, [1, 2, 3])

    def test_adapter_failure_is_not_silently_retried(self):
        adapter = Mock(side_effect=RuntimeError("Device failure"))
        draft = self.make(neural={"bad": adapter})
        with self.assertRaises(RuntimeError):
            self.propose(draft, selected="bad")
        self.assertEqual(draft.lookup.history, [1, 2, 3])

    def test_only_verifier_committed_tokens_enter_history(self):
        draft = self.make(neural={"draft": Mock(return_value=[4, 5, 6])})
        self.propose(draft, selected="draft")
        draft.commit_verified("request", [4, 9])
        self.assertEqual(draft.lookup.history, [1, 2, 3, 4, 9])
        with self.assertRaises(ValueError):
            draft.commit_verified("request", [10])
        self.assertEqual(draft.lookup.history, [1, 2, 3, 4, 9])

    def test_isolation_and_close(self):
        draft = self.make()
        other = HybridDraft("other", [7], vocab_size=10)
        for action in (lambda: draft.propose("other", 3),
                       lambda: draft.commit_verified("other", [4]),
                       lambda: draft.close("other")):
            with self.assertRaises(ValueError):
                action()
        draft.commit_verified("request", [4])
        self.assertEqual(other.lookup.history, [7])
        draft.close("request")
        self.assertEqual(draft.lookup.history, [])
        with self.assertRaises(ValueError):
            self.propose(draft)

    def test_configuration_and_bounds(self):
        for options in ({"vocab_size": 0}, {"vocab_size": True},
                        {"vocab_size": 10, "neural": {"lookup": Mock()}},
                        {"vocab_size": 10, "neural": {"bad": None}}):
            with self.assertRaises(ValueError):
                HybridDraft("request", [1], **options)
        draft = self.make()
        for count in (0, 16, True):
            with self.assertRaises(ValueError):
                draft.propose("request", count)


if __name__ == "__main__":
    unittest.main(verbosity=2)
