import os
from types import SimpleNamespace
import unittest
import unittest.mock
from unittest.mock import Mock


class FakeTrace:
    """Stand-in for dflash_proposal_trace.PreparedPackedDFlashProposal - records calls
    without any device machinery, so PackedProposalCoordinator's own pairing/gating/
    install logic is tested in isolation from the trace class itself (covered by
    test_dflash_packed_proposal_trace.py)."""
    instances = []

    def __init__(self, device_a, device_b):
        self.device_a, self.device_b = device_a, device_b
        self.prepared = []
        self.discard_calls = 0
        self.closed = False
        self.ready = True
        self.fail = None
        FakeTrace.instances.append(self)

    def prepare_device(self, seed_a, seed_b):
        if self.fail is not None:
            raise self.fail
        self.prepared.append((seed_a, seed_b))
        return self.ready

    def has_pending(self, which, seed):
        return bool(self.prepared) and seed == self.prepared[-1][0 if which == 'a' else 1]

    def finish(self, which, count):
        return ('tokens', which, count)

    def discard_pending(self):
        self.discard_calls += 1

    def close(self):
        self.closed = True


def make_device(operations, mesh, *, slot, history_rows=2048, native=True):
    return SimpleNamespace(operations=operations, mesh=mesh, history_rows=history_rows,
        native_proposal_attention=native, closed=False, pending=None, progress=None,
        pool_slot=(SimpleNamespace(index=slot) if slot is not None else None),
        proposal_capture=SimpleNamespace(discard_pending=Mock(), close=Mock()),
        prepare_device=Mock(return_value=True))


def make_bridge(name, device, seed):
    session = SimpleNamespace(request_id=name, seed=seed, pending=None, finished=False)
    request = SimpleNamespace(session=session, runtime=SimpleNamespace(drafter=device),
        closed=False, cancelled=False)
    return SimpleNamespace(request=request, failed=False)


class PackedProposalCoordinatorTests(unittest.TestCase):
    def setUp(self):
        FakeTrace.instances = []
        self.patcher = unittest.mock.patch('dflash_proposal_trace.PreparedPackedDFlashProposal', FakeTrace)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.operations = SimpleNamespace(synchronize_device=Mock())
        self.mesh = object()

    def bridges(self, slots, *, history_rows=2048, native=True):
        """One bridge per (name, slot) pair, slot None for an unpooled device."""
        out = {}
        for index, (name, slot) in enumerate(slots):
            device = make_device(self.operations, self.mesh, slot=slot, history_rows=history_rows, native=native)
            out[name] = make_bridge(name, device, seed=100 + index)
        return out

    def test_four_active_packable_users_pack_into_two_pairs(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator, _PackedCaptureView

        bridges = self.bridges([('a', 0), ('b', 1), ('c', 2), ('d', 3)])
        coordinator = PackedProposalCoordinator()
        coordinator.prepare(list(bridges.values()))
        self.assertEqual(len(FakeTrace.instances), 2, 'one trace per fixed pair')
        for name in 'abcd':
            capture = bridges[name].request.runtime.drafter.proposal_capture
            self.assertIsInstance(capture, _PackedCaptureView)
        self.operations.synchronize_device.assert_called_once_with(self.mesh)

    def test_not_yet_at_steady_state_context_falls_back_to_single_user(self):
        """history_rows below 2048 (still ramping): both members of a full pair still
        run their own device.prepare_device(), no trace is ever built."""
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        bridges = self.bridges([('a', 0), ('b', 1)], history_rows=1024)
        coordinator = PackedProposalCoordinator()
        coordinator.prepare(list(bridges.values()))
        self.assertEqual(FakeTrace.instances, [])
        for name in 'ab':
            bridges[name].request.runtime.drafter.prepare_device.assert_called_once()

    def test_three_active_degrades_the_broken_pair_to_single_user(self):
        """Slot 2 finished (absent from this round): (0, 1) still packs, slot 3 runs
        its own single-user prepare_device() - never paired with 0 or 1."""
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        bridges = self.bridges([('a', 0), ('b', 1), ('d', 3)])
        coordinator = PackedProposalCoordinator()
        coordinator.prepare(list(bridges.values()))
        self.assertEqual(len(FakeTrace.instances), 1)
        self.assertEqual(sorted([id(FakeTrace.instances[0].device_a), id(FakeTrace.instances[0].device_b)]),
            sorted([id(bridges['a'].request.runtime.drafter), id(bridges['b'].request.runtime.drafter)]))
        bridges['d'].request.runtime.drafter.prepare_device.assert_called_once()

    def test_one_active_is_single_user(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        bridges = self.bridges([('a', 0)])
        coordinator = PackedProposalCoordinator()
        coordinator.prepare(list(bridges.values()))
        self.assertEqual(FakeTrace.instances, [])
        bridges['a'].request.runtime.drafter.prepare_device.assert_called_once()

    def test_eager_mode_devices_never_pack(self):
        """proposal_capture is None (QWEN_FAST_EAGER_PROPOSAL): both members of an
        otherwise-full, otherwise-packable pair still fall back to single-user
        prepare_device() - packing them would leave a _PackedCaptureView wrapping
        None as its fallback, which crashes the moment the pair later breaks up and
        either side needs its single-user path back."""
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        bridges = self.bridges([('a', 0), ('b', 1)])
        for name in 'ab':
            bridges[name].request.runtime.drafter.proposal_capture = None
        coordinator = PackedProposalCoordinator()
        coordinator.prepare(list(bridges.values()))
        self.assertEqual(FakeTrace.instances, [])
        for name in 'ab':
            bridges[name].request.runtime.drafter.prepare_device.assert_called_once()

    def test_two_active_across_fixed_pairs_never_packs(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        bridges = self.bridges([('a', 0), ('c', 2)])
        coordinator = PackedProposalCoordinator()
        coordinator.prepare(list(bridges.values()))
        self.assertEqual(FakeTrace.instances, [])
        for name in 'ac':
            bridges[name].request.runtime.drafter.prepare_device.assert_called_once()

    def test_an_unpooled_device_is_never_paired(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        bridges = self.bridges([('a', None), ('b', None)])
        coordinator = PackedProposalCoordinator()
        coordinator.prepare(list(bridges.values()))
        self.assertEqual(FakeTrace.instances, [])
        for name in 'ab':
            bridges[name].request.runtime.drafter.prepare_device.assert_called_once()

    def test_the_same_pair_of_devices_reuses_one_trace_across_rounds(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        bridges = self.bridges([('a', 0), ('b', 1)])
        coordinator = PackedProposalCoordinator()
        coordinator.prepare(list(bridges.values()))
        coordinator.prepare(list(bridges.values()))
        self.assertEqual(len(FakeTrace.instances), 1)
        self.assertEqual(len(FakeTrace.instances[0].prepared), 2, 'both rounds replayed the same trace')

    def test_a_new_request_on_a_slot_retires_the_old_pair_trace(self):
        """The device occupying slot 1 changes identity (its old request finished,
        a new one acquired the slot): the stale trace is closed and a fresh one built
        for the new pairing, never silently reused across two different requests."""
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        bridges = self.bridges([('a', 0), ('b', 1)])
        coordinator = PackedProposalCoordinator()
        coordinator.prepare(list(bridges.values()))
        first = FakeTrace.instances[0]
        bridges['b'] = make_bridge('b2', make_device(self.operations, self.mesh, slot=1), seed=999)
        coordinator.prepare(list(bridges.values()))
        self.assertEqual(len(FakeTrace.instances), 2)
        self.assertTrue(first.closed)
        self.assertFalse(FakeTrace.instances[1].closed)

    def test_prepare_device_declining_falls_back_to_single_user_for_both(self):
        """trace.prepare_device() returning False (e.g. a device closed mid-round)
        must not silently drop either user - both fall back to their own
        prepare_device(), exactly as an unpaired bridge always has."""
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        bridges = self.bridges([('a', 0), ('b', 1)])
        coordinator = PackedProposalCoordinator()

        real_trace_for = coordinator._trace_for
        def declining(*args, **kwargs):
            trace = real_trace_for(*args, **kwargs)
            trace.ready = False
            return trace
        coordinator._trace_for = declining
        coordinator.prepare(list(bridges.values()))
        for name in 'ab':
            bridges[name].request.runtime.drafter.prepare_device.assert_called_once()

    def test_a_failure_preparing_one_pair_fences_and_discards_everything_already_prepared(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        bridges = self.bridges([('a', 0), ('b', 1), ('c', 2), ('d', 3)])
        coordinator = PackedProposalCoordinator()
        failure = RuntimeError('device fault preparing the second pair')

        real_trace_for = coordinator._trace_for
        calls = []
        def sometimes_failing(pair, device_a, device_b):
            trace = real_trace_for(pair, device_a, device_b)
            calls.append(pair)
            if pair == (2, 3):
                trace.fail = failure
            return trace
        coordinator._trace_for = sometimes_failing
        with self.assertRaises(RuntimeError) as caught:
            coordinator.prepare(list(bridges.values()))
        self.assertIs(caught.exception, failure)
        self.assertEqual(FakeTrace.instances[0].discard_calls, 1, 'the already-prepared first pair is discarded')
        self.operations.synchronize_device.assert_called_once_with(self.mesh)

    def test_audit_line_is_silent_by_default(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        bridges = self.bridges([('a', 0), ('b', 1)])
        coordinator = PackedProposalCoordinator()
        with unittest.mock.patch('dflash_packed_proposal_coordinator.audit_log') as audit_log:
            self.assertNotIn('QWEN_FAST_PACKED_AUDIT', os.environ)
            coordinator.prepare(list(bridges.values()))
        audit_log.assert_not_called()

    def test_audit_line_reports_round_pairs_and_propose_ms(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator, AUDIT_LINE

        bridges = self.bridges([('a', 0), ('b', 1), ('c', 2), ('d', 3)])
        coordinator = PackedProposalCoordinator()
        with unittest.mock.patch.dict(os.environ, {'QWEN_FAST_PACKED_AUDIT': '1'}), \
                unittest.mock.patch('dflash_packed_proposal_coordinator.audit_log') as audit_log:
            coordinator.prepare(list(bridges.values()))
        audit_log.assert_called_once()
        args, kwargs = audit_log.call_args
        self.assertEqual(args[0], AUDIT_LINE)
        self.assertEqual(kwargs['round'], 1)
        self.assertEqual(kwargs['pairs'], [[0, 1], [2, 3]])
        self.assertEqual(len(kwargs['propose_ms']), 2)

    def test_audit_round_number_increments_and_only_counts_actually_packed_rounds(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        bridges = self.bridges([('a', 0), ('b', 1)])
        coordinator = PackedProposalCoordinator()
        with unittest.mock.patch.dict(os.environ, {'QWEN_FAST_PACKED_AUDIT': '1'}), \
                unittest.mock.patch('dflash_packed_proposal_coordinator.audit_log') as audit_log:
            coordinator.prepare(list(bridges.values()))
            coordinator.prepare([])
            coordinator.prepare(list(bridges.values()))
        self.assertEqual(audit_log.call_count, 2)
        self.assertEqual(audit_log.call_args_list[0].kwargs['round'], 1)
        self.assertEqual(audit_log.call_args_list[1].kwargs['round'], 3, 'the empty round still counts')

    def test_phase_log_wraps_each_pairs_prepare_device_call(self):
        import serving_worker_hook
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        bridges = self.bridges([('a', 0), ('b', 1)])
        coordinator = PackedProposalCoordinator()
        lines = []
        stub = SimpleNamespace(info=lambda template, *values: lines.append(template.format(*values)))
        with unittest.mock.patch.dict('sys.modules', {'loguru': SimpleNamespace(logger=stub)}), \
                unittest.mock.patch.object(serving_worker_hook, 'PHASE_LOG', True):
            coordinator.prepare(list(bridges.values()))
        self.assertEqual(lines[0], '[PHASE] propose_pair a,b begin')
        self.assertTrue(lines[1].startswith('[PHASE] propose_pair a,b end '))

    def test_close_closes_every_cached_pair_trace(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        bridges = self.bridges([('a', 0), ('b', 1), ('c', 2), ('d', 3)])
        coordinator = PackedProposalCoordinator()
        coordinator.prepare(list(bridges.values()))
        self.assertEqual(len(FakeTrace.instances), 2)
        coordinator.close()
        self.assertTrue(all(trace.closed for trace in FakeTrace.instances))
        self.assertEqual(coordinator.pairs, {})


class PackedCaptureViewTests(unittest.TestCase):
    def test_has_pending_and_finish_delegate_to_the_trace_when_matched(self):
        from dflash_packed_proposal_coordinator import _PackedCaptureView

        trace = FakeTrace(object(), object())
        trace.prepared.append((11, 22))
        original = SimpleNamespace(has_pending=Mock(return_value=False))
        view = _PackedCaptureView(trace, 'a', original)
        self.assertTrue(view.has_pending(11))
        self.assertEqual(view.finish(3), ('tokens', 'a', 3))
        original.has_pending.assert_not_called()

    def test_has_pending_falls_through_to_the_original_when_not_matched(self):
        from dflash_packed_proposal_coordinator import _PackedCaptureView

        trace = FakeTrace(object(), object())
        original = SimpleNamespace(has_pending=Mock(return_value=True), finish=Mock(return_value='original-tokens'))
        view = _PackedCaptureView(trace, 'a', original)
        self.assertTrue(view.has_pending(999))
        self.assertEqual(view.finish(3), 'original-tokens')
        original.finish.assert_called_once_with(3)

    def test_close_only_closes_the_original_never_the_shared_trace(self):
        from dflash_packed_proposal_coordinator import _PackedCaptureView

        trace = FakeTrace(object(), object())
        original = SimpleNamespace(close=Mock())
        view = _PackedCaptureView(trace, 'a', original)
        view.close()
        original.close.assert_called_once()
        self.assertFalse(trace.closed, "closing one device's capture must not close the shared pair trace")

    def test_discard_pending_forwards_to_the_original_only(self):
        from dflash_packed_proposal_coordinator import _PackedCaptureView

        trace = FakeTrace(object(), object())
        original = SimpleNamespace(discard_pending=Mock())
        view = _PackedCaptureView(trace, 'a', original)
        view.discard_pending()
        original.discard_pending.assert_called_once()
        self.assertEqual(trace.discard_calls, 0)

    def test_install_unwraps_a_previous_view_instead_of_nesting(self):
        from dflash_packed_proposal_coordinator import _PackedCaptureView, _install

        trace_one = FakeTrace(object(), object())
        trace_two = FakeTrace(object(), object())
        original = SimpleNamespace(name='original')
        device = SimpleNamespace(proposal_capture=original)
        _install(device, trace_one, 'a')
        _install(device, trace_two, 'b')
        self.assertIsInstance(device.proposal_capture, _PackedCaptureView)
        self.assertIs(device.proposal_capture._original, original, 'never nests over a previous view')
        self.assertIs(device.proposal_capture._trace, trace_two)


if __name__ == '__main__':
    unittest.main()
