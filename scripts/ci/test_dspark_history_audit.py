from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from dspark_history_audit import AuditedHistoryDrafter, HistoryMismatch


class HistoryAuditTests(unittest.TestCase):
    def setUp(self):
        self.operations = SimpleNamespace(get_device_tensors=lambda value: (value, value.clone()),
            to_torch=lambda value: value)
        self.layers = self.values(32)
        self.drafter = SimpleNamespace(position=32, history=SimpleNamespace(layers=self.layers),
            propose=Mock(return_value=(11, 12)), prepare_publication=Mock(side_effect=self.prepare),
            commit_publication=Mock(side_effect=self.commit), discard_publication=Mock(), close=Mock())
        self.records = []
        self.auditor = AuditedHistoryDrafter(self.operations, self.drafter, self.records)

    def values(self, rows):
        return tuple(tuple(torch.full((1, 4, rows, 128), layer * 2 + operand, dtype=torch.bfloat16)
            for operand in range(2)) for layer in range(5))

    def prepare(self, features, prefix, *, position):
        return SimpleNamespace(position=position, prefix=prefix, layers=tuple(tuple(
            torch.cat((value, torch.full((1, 4, prefix, 128), 13, dtype=torch.bfloat16)), dim=2)
            for value in pair) for pair in self.drafter.history.layers))

    def commit(self, publication):
        self.drafter.position += publication.prefix
        self.drafter.history.layers = publication.layers

    def test_complete_history_checked_across_proposal_verify_and_publication(self):
        self.assertEqual(self.auditor.propose(10, 2), (11, 12))
        publication = self.auditor.prepare_publication(None, 3, position=32)
        self.auditor.commit_publication(publication)
        self.auditor.propose(12, 2)
        self.assertEqual(self.auditor.position, 35)
        self.assertEqual(len(self.records), 8)
        self.assertTrue(all(value['tensor_checks'] == 20 and value['exact'] for value in self.records))
        self.assertIn('prepared_history_preserves_committed_prefix', [value['phase'] for value in self.records])
        self.auditor.close()
        self.drafter.close.assert_called_once()

    def test_capture_or_verifier_cannot_overwrite_any_cached_position(self):
        self.layers[4][1][0, 3, 0, 127] = -55
        for call, phase in ((lambda: self.auditor.propose(10, 2), 'before_proposal_after_capture_or_publication'),
                (lambda: self.auditor.prepare_publication(None, 1, position=32), 'after_target_verifier')):
            with self.subTest(phase=phase), self.assertRaises(HistoryMismatch) as caught:
                call()
            self.assertEqual(caught.exception.evidence['phase'], phase)
            self.assertEqual(caught.exception.evidence['layer'], 4)
            self.assertEqual(caught.exception.evidence['operand'], 'value')
        self.drafter.propose.assert_not_called()
        self.drafter.prepare_publication.assert_not_called()

    def test_proposal_cannot_mutate_borrowed_history(self):
        def corrupt(anchor, count):
            self.layers[0][0][..., -1, -1] = 5
            return (11, 12)
        self.drafter.propose.side_effect = corrupt
        with self.assertRaises(HistoryMismatch) as caught:
            self.auditor.propose(10, 2)
        self.assertEqual(caught.exception.evidence['phase'], 'after_proposal')

    def test_prepared_cache_cannot_be_overwritten_by_target_commit_trace(self):
        publication = self.auditor.prepare_publication(None, 3, position=32)
        publication.layers[2][0][..., -1, -1] = 9
        with self.assertRaises(HistoryMismatch) as caught:
            self.auditor.commit_publication(publication)
        self.assertEqual(caught.exception.evidence['phase'], 'after_target_publication')
        self.drafter.commit_publication.assert_not_called()
        self.assertEqual(self.auditor.position, 32)

    def test_corrupt_prepared_prefix_is_discarded_without_commit(self):
        def corrupt(features, prefix, *, position):
            publication = self.prepare(features, prefix, position=position)
            publication.layers[0][0][..., 0, 0] = 17
            return publication
        self.drafter.prepare_publication.side_effect = corrupt
        with self.assertRaises(HistoryMismatch) as caught:
            self.auditor.prepare_publication(None, 3, position=32)
        self.assertEqual(caught.exception.evidence['phase'], 'prepared_history_preserves_committed_prefix')
        self.drafter.discard_publication.assert_called_once()
        self.drafter.commit_publication.assert_not_called()

    def test_nonfinite_history_fails_with_physical_operand_evidence(self):
        self.layers[3][1][..., 0, 0] = float('nan')
        with self.assertRaises(HistoryMismatch) as caught:
            self.auditor.propose(10, 2)
        self.assertFalse(caught.exception.evidence['finite'])
        self.assertEqual(caught.exception.evidence['layer'], 3)

    def test_second_chip_corruption_is_not_hidden_by_first_chip(self):
        def shards(value):
            second = value.clone()
            second[..., 0, 0] = 77
            return value, second
        self.operations.get_device_tensors = shards
        with self.assertRaises(HistoryMismatch) as caught:
            self.auditor.propose(10, 2)
        self.assertEqual(caught.exception.evidence['chip'], 1)

    def test_fixed_capacity_audit_checks_the_complete_valid_prefix_and_rejects_nonfinite_padding(self):
        self.drafter.history.capacity = 64
        self.drafter.history.layers = tuple(tuple(torch.nn.functional.pad(value, (0, 0, 0, 32))
            for value in pair) for pair in self.layers)
        self.auditor.propose(10, 2)
        self.drafter.history.layers[0][0][..., -1, -1] = float('nan')
        with self.assertRaises(HistoryMismatch) as caught:
            self.auditor.propose(10, 2)
        self.assertFalse(caught.exception.evidence['finite'])


if __name__ == '__main__':
    unittest.main()
