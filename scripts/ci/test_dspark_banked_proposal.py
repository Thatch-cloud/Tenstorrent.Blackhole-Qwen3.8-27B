from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import dspark_banked_proposal as banked


class BankedProposalTests(unittest.TestCase):
    def fixture(self):
        banks = tuple(tuple(tuple(object() for operand in range(2)) for layer in range(5)) for bank in range(2))
        device = SimpleNamespace(closed=False, operations=object(), history=SimpleNamespace(
            pending=None, layers=banks[0], spare_layers=banks[1]))
        return device, banks

    def test_bank_selection_swaps_without_copy_or_rebinding(self):
        device, banks = self.fixture()
        traces = [SimpleNamespace(checks=[], propose=Mock(return_value=(11, 12)), close=Mock()) for bank in banks]
        with patch.object(banked, 'addresses', side_effect=lambda operations, value: (id(value), id(value))), \
                patch.object(banked, 'BoundProposal', side_effect=traces):
            candidate = banked.BankedDSparkProposal(device, 10)
            self.assertEqual(candidate.propose(10, 2), (11, 12))
            device.history.layers, device.history.spare_layers = banks[1], banks[0]
            candidate.propose(12, 2)
            candidate.propose(14, 2)
            self.assertEqual(candidate.replay_counts, [1, 2])
            device.history.pending = object()
            with self.assertRaises(ValueError):
                candidate.propose(16, 2)
            device.history.pending = None
            device.history.layers = tuple(tuple(object() for operand in range(2)) for layer in range(5))
            with self.assertRaises(ValueError):
                candidate.propose(16, 2)
            candidate.close()
            candidate.close()
            for trace in traces:
                trace.close.assert_called_once()
            self.assertFalse(device.closed)

    def test_aliases_rejected_and_second_capture_failure_closes_first(self):
        device, banks = self.fixture()
        first = SimpleNamespace(close=Mock())
        with patch.object(banked, 'addresses', side_effect=lambda operations, value: (id(value), id(value))), \
                patch.object(banked, 'BoundProposal', side_effect=[first, RuntimeError('capture')]):
            with self.assertRaisesRegex(RuntimeError, 'capture'):
                banked.BankedDSparkProposal(device, 10)
            first.close.assert_called_once()
            device.history.spare_layers = device.history.layers
            with self.assertRaisesRegex(ValueError, 'independent'):
                banked.BankedDSparkProposal(device, 10)

    def test_bound_proposal_borrows_bank_and_reuses_native_execution(self):
        from test_dspark_prepared_proposal import PreparedProposalTests
        fixture = PreparedProposalTests()
        fixture.setUp()
        try:
            with patch.object(banked, 'addresses', side_effect=lambda operations, value: (id(value), id(value))):
                proposal = banked.BoundProposal(fixture.device, 10, fixture.bank, audit=True)
                self.assertIs(proposal.history, fixture.bank)
                self.assertEqual(proposal.propose(20, 7), tuple(range(21, 28)))
                fixture.operations.copy.assert_not_called()
                self.assertEqual(len(proposal.checks), 1)
                proposal.close()
                self.assertFalse(fixture.device.closed)
        finally:
            fixture.doCleanups()
