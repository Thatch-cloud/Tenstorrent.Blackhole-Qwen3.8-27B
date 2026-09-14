from types import SimpleNamespace
from copy import deepcopy
import unittest
from unittest.mock import patch

from dspark_proposal_phase_profile import (
    profile_proposals, probe_three_replays, stop_after_prepared_probe, ProposalProbeComplete)


class PhaseProfileTests(unittest.TestCase):
    def test_loaded_hook_stops_after_probe_and_restores_prepare(self):
        from dspark_prepared_proposal import TracedDSparkDevice

        device, clock = self.fixture()
        with patch.object(TracedDSparkDevice, 'prepare_trace') as prepare:
            with self.assertRaises(ProposalProbeComplete) as caught:
                with stop_after_prepared_probe(lambda report: None):
                    TracedDSparkDevice.prepare_trace(device, 7, audit=True)
            prepare.assert_called_once_with(device, 7, audit=False)
            self.assertIs(TracedDSparkDevice.prepare_trace, prepare)
        self.assertEqual(caught.exception.evidence['completed_replays'], 3)

    def fixture(self, fail=False):
        elapsed = [0.]

        def advance(seconds):
            elapsed[0] += seconds

        operations = SimpleNamespace(execute_trace=lambda: advance(.020))
        prepared = SimpleNamespace(trace=1, closed=False, audit=False,
            update=lambda anchor: advance(.003), read_tokens=lambda output: advance(.002))

        def propose(anchor, count):
            prepared.update(anchor)
            operations.execute_trace()
            if fail:
                raise RuntimeError('replay failed')
            prepared.read_tokens(None)
            advance(.001)
            return [7] * count

        prepared.propose = propose
        prepared.expected_warmup = (7,) * 15
        return SimpleNamespace(prepared=prepared, operations=operations, position=65536,
            max_drafts=15, history=SimpleNamespace(pending=None)), lambda: elapsed[0]

    def test_bounded_probe_checkpoints_each_replay_without_claiming_tg(self):
        device, clock = self.fixture()
        checkpoints = []
        result = probe_three_replays(device, 7, lambda report: checkpoints.append(deepcopy(report)), clock=clock)
        self.assertTrue(result['passed'])
        self.assertEqual(result['completed_replays'], 3)
        self.assertEqual(len(result['records']), 3)
        self.assertFalse(result['performance_qualified'])
        self.assertIsNone(result['committed_tg'])
        self.assertEqual([item['ordinal'] for item in checkpoints if item['phase'] == 'replay'
            and item['completed_replays'] == item['ordinal']], [0, 1, 2])

    def test_probe_persists_failure_and_stops(self):
        device, clock = self.fixture(fail=True)
        checkpoints = []
        with self.assertRaisesRegex(RuntimeError, 'replay failed'):
            probe_three_replays(device, 7, lambda report: checkpoints.append(deepcopy(report)), clock=clock)
        self.assertEqual(checkpoints[-1]['phase'], 'failed')
        self.assertEqual(checkpoints[-1]['completed_replays'], 0)
        self.assertEqual(len(checkpoints[-1]['records']), 1)

    def test_probe_rejects_divergent_tokens(self):
        device, clock = self.fixture()
        device.prepared.expected_warmup = (8,) * 15
        with self.assertRaisesRegex(AssertionError, 'warmup tokens'):
            probe_three_replays(device, 7, lambda report: None, clock=clock)

    def test_partition_and_restore(self):
        device, clock = self.fixture()
        originals = (device.prepared.propose, device.prepared.update, device.operations.execute_trace)
        records = []
        with profile_proposals(device, records, clock=clock):
            device.operations.execute_trace()
            self.assertEqual(device.prepared.propose(7, 15), [7] * 15)
        self.assertEqual(originals, (device.prepared.propose, device.prepared.update, device.operations.execute_trace))
        self.assertEqual(len(records), 1)
        for name, expected in dict(update_inputs_history_ms=3, trace_replay_ms=20,
                token_readback_ms=2, other_host_ms=1, total_ms=26).items():
            self.assertAlmostEqual(records[0][name], expected)
        self.assertTrue(records[0]['passed'])

    def test_failed_replay_is_not_success_and_bindings_restore(self):
        device, clock = self.fixture(fail=True)
        original = device.prepared.propose
        records = []
        with self.assertRaisesRegex(RuntimeError, 'replay failed'):
            with profile_proposals(device, records, clock=clock):
                device.prepared.propose(7, 15)
        self.assertFalse(records[0]['passed'])
        self.assertIs(device.prepared.propose, original)

    def test_audit_mode_is_rejected(self):
        device, clock = self.fixture()
        device.prepared.audit = True
        with self.assertRaises(ValueError):
            with profile_proposals(device, [], clock=clock):
                self.fail('Audit profiling must not be confused with timed replay')
