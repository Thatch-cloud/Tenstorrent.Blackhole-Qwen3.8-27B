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


class FakeSingleUserCapture:
    """Stand-in for dflash_proposal_trace.PreparedDFlashProposal - records what it
    was built for (device, max_new_tokens) and that it can be close()d, without any
    device machinery."""
    instances = []

    def __init__(self, device, *, max_new_tokens):
        self.device, self.max_new_tokens = device, max_new_tokens
        self.closed = False
        FakeSingleUserCapture.instances.append(self)

    def close(self):
        self.closed = True

    def prepare_device(self, seed):
        return True

    def has_pending(self, seed):
        return False

    def finish(self, count):
        return ()

    def propose(self, seed, count):
        return ()

    def discard_pending(self):
        pass


def make_device(operations, mesh, *, slot, history_rows=2048, native=True, committed_kv=True):
    layers = [object()] * 5
    kv_history = SimpleNamespace(active=(list(layers) if committed_kv else [])) if committed_kv is not None else None
    return SimpleNamespace(operations=operations, mesh=mesh, history_rows=history_rows,
        native_proposal_attention=native, closed=False, pending=None, progress=None,
        layers=layers, kv_history=kv_history,
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

    def bridges(self, slots, *, history_rows=2048, native=True, committed_kv=True):
        """One bridge per (name, slot) pair, slot None for an unpooled device."""
        out = {}
        for index, (name, slot) in enumerate(slots):
            device = make_device(self.operations, self.mesh, slot=slot, history_rows=history_rows,
                native=native, committed_kv=committed_kv)
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

    def test_a_pair_warmup_failure_falls_back_to_single_user_without_propagating(self):
        """Run 35581352016: a pair's first traced execution raised on hardware despite
        packable() passing every host-side check. The four-user gate must never lose
        the engine to a proposal-path refusal - the failing pair's own two devices run
        their own single-user prepare_device() for this round instead, and the round
        completes normally; only ONE pair's own trace.prepare_device() ever runs
        (the failure happens the first time it is called for this pair, so a second
        FakeTrace for that same pair is never even needed)."""
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        bridges = self.bridges([('a', 0), ('b', 1), ('c', 2), ('d', 3)])
        coordinator = PackedProposalCoordinator()
        failure = RuntimeError('execute_proposal refused the packed cached_history')

        real_trace_for = coordinator._trace_for
        def sometimes_failing(pair, device_a, device_b):
            trace = real_trace_for(pair, device_a, device_b)
            if pair == (2, 3):
                trace.fail = failure
            return trace
        coordinator._trace_for = sometimes_failing
        prepared = coordinator.prepare(list(bridges.values()))
        self.assertEqual(len(FakeTrace.instances), 2, 'both pairs still got a trace object')
        self.assertEqual(FakeTrace.instances[1].prepared, [], 'the failing pair never actually prepared')
        self.assertEqual(FakeTrace.instances[1].discard_calls, 1)
        # (0, 1) packed normally; (2, 3) fell back - both its own devices' single-user
        # prepare_device() ran instead.
        self.assertEqual(len(FakeTrace.instances[0].prepared), 1)
        for name in 'cd':
            bridges[name].request.runtime.drafter.prepare_device.assert_called_once()
        self.operations.synchronize_device.assert_called_once_with(self.mesh)

    def test_a_pair_warmup_failure_logs_the_fallback_line_under_the_audit_flag(self):
        import os
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator, PAIR_FALLBACK_LINE

        bridges = self.bridges([('a', 0), ('b', 1)])
        coordinator = PackedProposalCoordinator()
        failure = ValueError('Every prepared learned layer requires a committed K/V cache')
        real_trace_for = coordinator._trace_for
        def failing(pair, device_a, device_b):
            trace = real_trace_for(pair, device_a, device_b)
            trace.fail = failure
            return trace
        coordinator._trace_for = failing
        with unittest.mock.patch.dict(os.environ, {'QWEN_FAST_PACKED_AUDIT': '1'}), \
                unittest.mock.patch('dflash_packed_proposal_coordinator.audit_log') as audit_log:
            coordinator.prepare(list(bridges.values()))
        audit_log.assert_called_once()
        args, kwargs = audit_log.call_args
        self.assertEqual(args[0], PAIR_FALLBACK_LINE)
        self.assertEqual(kwargs['pair'], [0, 1])
        self.assertIn('Every prepared learned layer requires a committed K/V cache', kwargs['fallback'])

    def test_a_pair_warmup_failure_is_silent_without_the_audit_flag(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        bridges = self.bridges([('a', 0), ('b', 1)])
        coordinator = PackedProposalCoordinator()
        real_trace_for = coordinator._trace_for
        def failing(pair, device_a, device_b):
            trace = real_trace_for(pair, device_a, device_b)
            trace.fail = RuntimeError('boom')
            return trace
        coordinator._trace_for = failing
        with unittest.mock.patch('dflash_packed_proposal_coordinator.audit_log') as audit_log:
            self.assertNotIn('QWEN_FAST_PACKED_AUDIT', __import__('os').environ)
            coordinator.prepare(list(bridges.values()))
        audit_log.assert_not_called()

    def test_a_failing_pair_is_re_evaluated_fresh_next_round(self):
        """packable() is re-checked every round, and a failed pair's trace stays
        usable (FakeTrace.fail is reset here to simulate the underlying condition
        clearing) - the SAME trace object is reused, not rebuilt, and this round
        packs normally."""
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        bridges = self.bridges([('a', 0), ('b', 1)])
        coordinator = PackedProposalCoordinator()
        real_trace_for = coordinator._trace_for
        injected = []
        def once_failing(pair, device_a, device_b):
            trace = real_trace_for(pair, device_a, device_b)
            if trace not in injected:
                trace.fail = RuntimeError('first round only')
                injected.append(trace)
            return trace
        coordinator._trace_for = once_failing
        coordinator.prepare(list(bridges.values()))
        self.assertEqual(FakeTrace.instances[0].prepared, [])
        FakeTrace.instances[0].fail = None
        coordinator.prepare(list(bridges.values()))
        self.assertEqual(len(FakeTrace.instances), 1, 'the same trace object was reused, not rebuilt')
        self.assertEqual(len(FakeTrace.instances[0].prepared), 1, 'the second round packed successfully')

    def test_a_failure_preparing_an_unpaired_devices_own_prepare_device_fences_and_discards_everything_already_prepared(self):
        """Unlike a packed pair's own trace.prepare_device() (now non-fatal, above),
        an UNPAIRED device's own single-user prepare_device() raising is still a real
        phase-A failure - prepare_pipelined_drafts' own contract, unchanged."""
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        bridges = self.bridges([('a', 0), ('b', 1), ('c', None)])
        coordinator = PackedProposalCoordinator()
        failure = RuntimeError('device fault preparing the unpaired entry')
        bridges['c'].request.runtime.drafter.prepare_device = Mock(side_effect=failure)
        with self.assertRaises(RuntimeError) as caught:
            coordinator.prepare(list(bridges.values()))
        self.assertIs(caught.exception, failure)
        self.assertEqual(FakeTrace.instances[0].discard_calls, 1, 'the already-prepared pair is discarded')
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


def committed_device(*, committed=True):
    """A bare device with exactly the two attributes _committed_kv_history reads:
    layers (the fixed 5 draft layers) and kv_history.active (populated once at
    DraftKVHistory construction, per-layer - not by any later commit)."""
    layers = [object()] * 5
    kv_history = SimpleNamespace(active=(list(layers) if committed else []))
    return SimpleNamespace(history_rows=2048, native_proposal_attention=True,
        layers=layers, kv_history=kv_history, proposal_capture=object())


class ReleaseAndRecaptureTests(unittest.TestCase):
    """PackedProposalCoordinator._release_single_user / _ensure_single_user - the
    DRAM-neutral half of run 35585107688's fix: a pair's placeholder set duplicates
    the two single-user traced-proposal placeholder sets, so once a pair captures
    successfully the single-user ones must go, and come back exactly once if a
    later round needs the fallback again."""

    def setUp(self):
        self.patcher = unittest.mock.patch('dflash_proposal_trace.PreparedDFlashProposal', FakeSingleUserCapture)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        FakeSingleUserCapture.instances = []
        self.operations = SimpleNamespace(synchronize_device=Mock())
        self.mesh = object()

    def device(self, *, position=32780):
        device = make_device(self.operations, self.mesh, slot=0)
        device.position = position
        return device

    def test_release_direct_closes_and_marks_the_device(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        device = self.device()
        original = device.proposal_capture
        coordinator = PackedProposalCoordinator()
        coordinator._release_single_user(device)
        original.close.assert_called_once()
        self.assertIsNone(device.proposal_capture)
        self.assertTrue(device._packed_capture_released)

    def test_release_through_a_view_closes_original_and_keeps_the_view_installed(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator, _PackedCaptureView

        device = self.device()
        original = device.proposal_capture
        view = _PackedCaptureView(trace=object(), which='a', original=original)
        device.proposal_capture = view
        coordinator = PackedProposalCoordinator()
        coordinator._release_single_user(device)
        original.close.assert_called_once()
        self.assertIs(device.proposal_capture, view, 'the view itself stays installed - only its original is freed')
        self.assertIsNone(view._original)
        self.assertTrue(device._packed_capture_released)

    def test_release_is_idempotent(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        device = self.device()
        original = device.proposal_capture
        coordinator = PackedProposalCoordinator()
        coordinator._release_single_user(device)
        coordinator._release_single_user(device)
        original.close.assert_called_once()

    def test_ensure_is_a_noop_for_a_device_never_released_by_this_coordinator(self):
        """proposal_capture is None for an ordinary reason (QWEN_FAST_EAGER_PROPOSAL,
        or simply a device this coordinator has never touched) must never be treated
        as 'needs a rebuild' - only device._packed_capture_released says that."""
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        device = self.device()
        device.proposal_capture = None
        coordinator = PackedProposalCoordinator()
        self.assertFalse(coordinator._ensure_single_user(device))
        self.assertIsNone(device.proposal_capture)
        self.assertEqual(FakeSingleUserCapture.instances, [])

    def test_ensure_rebuilds_after_a_direct_release(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        device = self.device(position=32780)
        coordinator = PackedProposalCoordinator()
        coordinator._release_single_user(device)
        self.assertTrue(coordinator._ensure_single_user(device))
        self.assertIsInstance(device.proposal_capture, FakeSingleUserCapture)
        self.assertIs(device.proposal_capture.device, device)
        self.assertEqual(device.proposal_capture.max_new_tokens, 1)
        self.assertFalse(device._packed_capture_released)

    def test_ensure_rebuilds_the_views_original_after_a_release_through_it(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator, _PackedCaptureView

        device = self.device(position=32780)
        original = device.proposal_capture
        view = _PackedCaptureView(trace=object(), which='b', original=original)
        device.proposal_capture = view
        coordinator = PackedProposalCoordinator()
        coordinator._release_single_user(device)
        self.assertTrue(coordinator._ensure_single_user(device))
        self.assertIs(device.proposal_capture, view, 'the view stays installed')
        self.assertIsInstance(view._original, FakeSingleUserCapture)
        self.assertFalse(device._packed_capture_released)

    def test_ensure_is_a_noop_once_already_rebuilt(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        device = self.device(position=32780)
        coordinator = PackedProposalCoordinator()
        coordinator._release_single_user(device)
        coordinator._ensure_single_user(device)
        rebuilt = device.proposal_capture
        self.assertFalse(coordinator._ensure_single_user(device))
        self.assertIs(device.proposal_capture, rebuilt)
        self.assertEqual(len(FakeSingleUserCapture.instances), 1)

    def test_recapture_line_under_the_audit_flag_names_the_slot(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator, RECAPTURE_LINE

        device = self.device(position=32780)
        coordinator = PackedProposalCoordinator()
        coordinator._release_single_user(device)
        with unittest.mock.patch.dict(os.environ, {'QWEN_FAST_PACKED_AUDIT': '1'}), \
                unittest.mock.patch('dflash_packed_proposal_coordinator.audit_log') as audit_log:
            coordinator._ensure_single_user(device)
        audit_log.assert_called_once_with(RECAPTURE_LINE, slot=0)

    def test_recapture_is_silent_without_the_audit_flag(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        device = self.device(position=32780)
        coordinator = PackedProposalCoordinator()
        coordinator._release_single_user(device)
        with unittest.mock.patch('dflash_packed_proposal_coordinator.audit_log') as audit_log:
            self.assertNotIn('QWEN_FAST_PACKED_AUDIT', os.environ)
            coordinator._ensure_single_user(device)
        audit_log.assert_not_called()


class ReleaseAndRecaptureIntegrationTests(unittest.TestCase):
    """The same, through coordinator.prepare() end to end: a pair's first success
    releases both single-user captures; a later round where the pair breaks up (or
    its own capture fails) rebuilds whichever side needs its fallback."""

    def setUp(self):
        FakeTrace.instances = []
        FakeSingleUserCapture.instances = []
        self.patcher_pair = unittest.mock.patch('dflash_proposal_trace.PreparedPackedDFlashProposal', FakeTrace)
        self.patcher_single = unittest.mock.patch('dflash_proposal_trace.PreparedDFlashProposal', FakeSingleUserCapture)
        self.patcher_pair.start()
        self.patcher_single.start()
        self.addCleanup(self.patcher_pair.stop)
        self.addCleanup(self.patcher_single.stop)
        self.operations = SimpleNamespace(synchronize_device=Mock())
        self.mesh = object()

    def bridges(self, slots):
        out = {}
        for index, (name, slot) in enumerate(slots):
            device = make_device(self.operations, self.mesh, slot=slot)
            device.position = 32780
            out[name] = make_bridge(name, device, seed=100 + index)
        return out

    def test_a_successful_pair_releases_both_single_user_captures(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        bridges = self.bridges([('a', 0), ('b', 1)])
        originals = [bridges[name].request.runtime.drafter.proposal_capture for name in 'ab']
        coordinator = PackedProposalCoordinator()
        coordinator.prepare(list(bridges.values()))
        for original in originals:
            original.close.assert_called_once()
        for name in 'ab':
            device = bridges[name].request.runtime.drafter
            self.assertTrue(device._packed_capture_released)
            self.assertIsInstance(device.proposal_capture, __import__('dflash_packed_proposal_coordinator')._PackedCaptureView)
            self.assertIsNone(device.proposal_capture._original)

    def test_release_happens_only_once_not_every_successful_round(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        bridges = self.bridges([('a', 0), ('b', 1)])
        originals = [bridges[name].request.runtime.drafter.proposal_capture for name in 'ab']
        coordinator = PackedProposalCoordinator()
        coordinator.prepare(list(bridges.values()))
        coordinator.prepare(list(bridges.values()))
        coordinator.prepare(list(bridges.values()))
        for original in originals:
            original.close.assert_called_once()

    def test_a_pair_that_breaks_up_recaptures_the_survivors_fallback_once(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator, RECAPTURE_LINE

        bridges = self.bridges([('a', 0), ('b', 1)])
        coordinator = PackedProposalCoordinator()
        coordinator.prepare(list(bridges.values()))  # a,b pack and release
        device_a = bridges['a'].request.runtime.drafter
        # b finishes (removed from this round) - a survives alone in its fixed pair.
        with unittest.mock.patch.dict(os.environ, {'QWEN_FAST_PACKED_AUDIT': '1'}), \
                unittest.mock.patch('dflash_packed_proposal_coordinator.audit_log') as audit_log:
            coordinator.prepare([bridges['a']])
        self.assertFalse(device_a._packed_capture_released, 'rebuilt for the fallback')
        self.assertTrue(any(call.args[0] == RECAPTURE_LINE for call in audit_log.call_args_list))
        device_a.prepare_device.assert_called_once_with(bridges['a'].request.session.seed)

    def test_a_later_capture_failure_recaptures_both_sides_fallback(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        bridges = self.bridges([('a', 0), ('b', 1)])
        coordinator = PackedProposalCoordinator()
        coordinator.prepare(list(bridges.values()))  # packs and releases
        (trace,) = FakeTrace.instances
        trace.fail = RuntimeError('later round fails')
        coordinator.prepare(list(bridges.values()))
        for name in 'ab':
            device = bridges[name].request.runtime.drafter
            self.assertFalse(device._packed_capture_released, '%s recaptured its own fallback' % name)
            device.prepare_device.assert_called_once_with(bridges[name].request.session.seed)


class DramReserveTests(unittest.TestCase):
    def setUp(self):
        FakeTrace.instances = []
        self.patcher = unittest.mock.patch('dflash_proposal_trace.PreparedPackedDFlashProposal', FakeTrace)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.operations = SimpleNamespace(synchronize_device=Mock())
        self.mesh = object()

    def bridges(self, slots):
        out = {}
        for index, (name, slot) in enumerate(slots):
            device = make_device(self.operations, self.mesh, slot=slot)
            out[name] = make_bridge(name, device, seed=100 + index)
        return out

    def test_default_reserve_is_256mb(self):
        from dflash_packed_proposal_coordinator import dram_reserve_bytes

        self.assertEqual(dram_reserve_bytes({}), 256 * 1024 * 1024)

    def test_reserve_reads_a_custom_megabyte_value(self):
        from dflash_packed_proposal_coordinator import dram_reserve_bytes

        self.assertEqual(dram_reserve_bytes({'QWEN_FAST_PACKED_PROPOSAL_DRAM_RESERVE_MB': '512'}), 512 * 1024 * 1024)
        self.assertEqual(dram_reserve_bytes({'QWEN_FAST_PACKED_PROPOSAL_DRAM_RESERVE_MB': '0'}), 0)

    def test_reserve_rejects_non_integer_or_negative_values(self):
        from dflash_packed_proposal_coordinator import dram_reserve_bytes

        for bad in ('not-a-number', '1.5', '-1'):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    dram_reserve_bytes({'QWEN_FAST_PACKED_PROPOSAL_DRAM_RESERVE_MB': bad})

    def test_headroom_below_the_estimate_plus_reserve_refuses_the_pair(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator, estimated_pair_capture_bytes

        bridges = self.bridges([('a', 0), ('b', 1)])
        coordinator = PackedProposalCoordinator()
        too_small = estimated_pair_capture_bytes()  # exactly the estimate, no room for ANY reserve
        with unittest.mock.patch('dflash_packed_proposal_coordinator.dram_headroom', return_value=too_small), \
                unittest.mock.patch.dict(os.environ, {'QWEN_FAST_PACKED_AUDIT': '1'}), \
                unittest.mock.patch('dflash_packed_proposal_coordinator.audit_log') as audit_log:
            coordinator.prepare(list(bridges.values()))
        self.assertEqual(FakeTrace.instances, [], 'no capture was even attempted')
        for name in 'ab':
            bridges[name].request.runtime.drafter.prepare_device.assert_called_once()
        args, kwargs = audit_log.call_args
        self.assertEqual(kwargs['pair'], [0, 1])
        self.assertEqual(kwargs['fallback'], 'dram_reserve')

    def test_headroom_above_the_estimate_plus_reserve_allows_the_pair(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator, estimated_pair_capture_bytes, dram_reserve_bytes

        bridges = self.bridges([('a', 0), ('b', 1)])
        coordinator = PackedProposalCoordinator()
        plenty = estimated_pair_capture_bytes() + dram_reserve_bytes() + 1
        with unittest.mock.patch('dflash_packed_proposal_coordinator.dram_headroom', return_value=plenty):
            coordinator.prepare(list(bridges.values()))
        self.assertEqual(len(FakeTrace.instances), 1)
        for name in 'ab':
            bridges[name].request.runtime.drafter.prepare_device.assert_not_called()

    def test_unavailable_statistics_do_not_block_packing(self):
        """serving_buffer_pool.dram_statistics's own contract: a diagnostic, never a
        gate. dram_headroom returns None when it cannot read the allocator (the
        ordinary case in every host test, and this module's own default)."""
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator, dram_headroom

        bridges = self.bridges([('a', 0), ('b', 1)])
        device_a = bridges['a'].request.runtime.drafter
        self.assertIsNone(dram_headroom(device_a), 'no allocator behind a plain host fake')
        coordinator = PackedProposalCoordinator()
        coordinator.prepare(list(bridges.values()))
        self.assertEqual(len(FakeTrace.instances), 1)

    def test_the_reserve_check_only_runs_for_a_fresh_capture_not_a_replay(self):
        """Replaying an already-built trace allocates no new placeholder buffers, so
        the reserve check must not even be consulted on the second round of an
        established pair - confirmed by making the mocked headroom always refuse and
        checking the pair still packs on round two."""
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator, estimated_pair_capture_bytes

        bridges = self.bridges([('a', 0), ('b', 1)])
        coordinator = PackedProposalCoordinator()
        with unittest.mock.patch('dflash_packed_proposal_coordinator.dram_headroom',
                                 return_value=estimated_pair_capture_bytes() + 10**9):
            coordinator.prepare(list(bridges.values()))  # round 1: plenty of room, captures
        self.assertEqual(len(FakeTrace.instances), 1)
        with unittest.mock.patch('dflash_packed_proposal_coordinator.dram_headroom', return_value=0):
            coordinator.prepare(list(bridges.values()))  # round 2: headroom would refuse a FRESH build
        self.assertEqual(len(FakeTrace.instances), 1, 'no second trace was built')
        self.assertEqual(len(FakeTrace.instances[0].prepared), 2, 'round two still replayed the existing trace')


EXTENT = {'QWEN_FAST_EXTENT_REPLAY': '1'}


def extent_environment(on):
    """os.environ with QWEN_FAST_EXTENT_REPLAY set to 1, or absent (the flag-off path)."""
    environment = {name: value for name, value in os.environ.items() if name != 'QWEN_FAST_EXTENT_REPLAY'}
    if on:
        environment.update(EXTENT)
    return unittest.mock.patch.dict(os.environ, environment, clear=True)


class ReleaseClosedTests(unittest.TestCase):
    """S2 W6c: release_closed closes, at detach, the quad if any of its devices has closed and every pair trace
    bound to a closed device; live traces stay, and the line is logged every time."""

    def devices(self, count=4):
        operations, mesh = SimpleNamespace(synchronize_device=Mock()), object()
        return [make_device(operations, mesh, slot=slot) for slot in range(count)]

    def coordinator(self, devices):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        coordinator = PackedProposalCoordinator()
        pairs = {(0, 1): FakeTrace(devices[0], devices[1]), (2, 3): FakeTrace(devices[2], devices[3])}
        for group, trace in pairs.items():
            coordinator.pairs[group] = (devices[group[0]], devices[group[1]], trace, True)
        quad = SimpleNamespace(closed=False)
        quad.close = lambda: setattr(quad, 'closed', True)
        coordinator.quad = (tuple(devices), quad, True)
        return coordinator, pairs, quad

    def test_a_closed_device_releases_the_quad_and_its_own_pair_and_keeps_the_live_pair(self):
        from dflash_packed_proposal_coordinator import RELEASED_LINE

        devices = self.devices()
        coordinator, pairs, quad = self.coordinator(devices)
        devices[3].closed = True
        with unittest.mock.patch('dflash_packed_proposal_coordinator.audit_log') as log:
            self.assertEqual(coordinator.release_closed(), dict(quad=1, pairs=[[2, 3]]))
        self.assertTrue(quad.closed)
        self.assertIsNone(coordinator.quad)
        self.assertTrue(pairs[(2, 3)].closed)
        self.assertFalse(pairs[(0, 1)].closed)
        self.assertEqual(list(coordinator.pairs), [(0, 1)])
        log.assert_called_once_with(RELEASED_LINE, quad=1, pairs=[[2, 3]])

    def test_with_every_device_live_nothing_is_released_and_the_line_says_so(self):
        devices = self.devices()
        coordinator, pairs, quad = self.coordinator(devices)
        with unittest.mock.patch('dflash_packed_proposal_coordinator.audit_log') as log:
            self.assertEqual(coordinator.release_closed(), dict(quad=0, pairs=[]))
        self.assertFalse(quad.closed or any(trace.closed for trace in pairs.values()))
        self.assertEqual(sorted(coordinator.pairs), [(0, 1), (2, 3)])
        log.assert_called_once()
        empty = __import__('dflash_packed_proposal_coordinator').PackedProposalCoordinator()
        with unittest.mock.patch('dflash_packed_proposal_coordinator.audit_log'):
            self.assertEqual(empty.release_closed(), dict(quad=0, pairs=[]))

    def test_two_closed_devices_release_both_pairs(self):
        devices = self.devices()
        coordinator, pairs, quad = self.coordinator(devices)
        devices[0].closed = devices[2].closed = True
        with unittest.mock.patch('dflash_packed_proposal_coordinator.audit_log'):
            self.assertEqual(coordinator.release_closed(), dict(quad=1, pairs=[[0, 1], [2, 3]]))
        self.assertEqual(coordinator.pairs, {})
        self.assertTrue(all(trace.closed for trace in pairs.values()))

    def test_the_survivors_draft_on_after_a_release(self):
        """A released pair's survivor rebuilds its single-user capture once, the next round it drafts alone."""
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        FakeTrace.instances, FakeSingleUserCapture.instances = [], []
        with unittest.mock.patch('dflash_proposal_trace.PreparedPackedDFlashProposal', FakeTrace), \
                unittest.mock.patch('dflash_proposal_trace.PreparedDFlashProposal', FakeSingleUserCapture), \
                unittest.mock.patch('dflash_packed_proposal_coordinator.audit_log'):
            operations, mesh = SimpleNamespace(synchronize_device=Mock()), object()
            bridges = [make_bridge(name, make_device(operations, mesh, slot=slot), seed=100 + slot)
                       for slot, name in enumerate('ab')]
            coordinator = PackedProposalCoordinator()
            coordinator.prepare(bridges)
            self.assertEqual(len(FakeTrace.instances), 1)
            survivor, departed = (bridge.request.runtime.drafter for bridge in bridges)
            departed.closed = True
            self.assertEqual(coordinator.release_closed(), dict(quad=0, pairs=[[0, 1]]))
            self.assertTrue(FakeTrace.instances[0].closed)
            coordinator.prepare(bridges[:1])
            self.assertEqual(len(FakeSingleUserCapture.instances), 1, 'the survivor rebuilt its own capture once')
            self.assertEqual(len(FakeTrace.instances), 1, 'no pair re-formed')


class LedgerPointTests(unittest.TestCase):
    """S2 W6d: a fresh pair or quad capture and a single-user rebuild are ledger before/after points under
    QWEN_FAST_EXTENT_REPLAY=1; a replay is not; with the flag off memory_ledger is never asked."""

    def setUp(self):
        FakeTrace.instances, FakeSingleUserCapture.instances = [], []
        for target, value in (('dflash_proposal_trace.PreparedPackedDFlashProposal', FakeTrace),
                              ('dflash_proposal_trace.PreparedDFlashProposal', FakeSingleUserCapture)):
            patcher = unittest.mock.patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.operations, self.mesh = SimpleNamespace(synchronize_device=Mock()), object()
        self.points = []

    def ledger(self):
        import memory_ledger

        def before(op, **options):
            self.points.append(('before', op, options))
            return dict(op=op)

        return unittest.mock.patch.multiple(memory_ledger, before=Mock(side_effect=before),
                                            after=Mock(side_effect=lambda token: self.points.append(('after', token['op']))))

    def pair_bridges(self):
        return [make_bridge(name, make_device(self.operations, self.mesh, slot=slot), seed=100 + slot)
                for slot, name in enumerate('ab')]

    def test_a_fresh_pair_capture_is_read_either_side_and_its_replay_is_not(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator, estimated_pair_capture_bytes

        bridges, coordinator = self.pair_bridges(), PackedProposalCoordinator()
        with extent_environment(True), self.ledger():
            coordinator.prepare(bridges)
            coordinator.prepare(bridges)
        self.assertEqual(self.points, [('before', 'pair', dict(estimate=estimated_pair_capture_bytes(), point='slots=0,1')),
                                       ('after', 'pair')])

    def test_a_failed_pair_capture_still_closes_its_point(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        bridges, coordinator = self.pair_bridges(), PackedProposalCoordinator()
        original = FakeTrace.__init__

        def failing(trace, device_a, device_b):
            original(trace, device_a, device_b)
            trace.fail = RuntimeError('capture refused')

        with extent_environment(True), self.ledger(), unittest.mock.patch.object(FakeTrace, '__init__', failing):
            coordinator.prepare(bridges)
        self.assertEqual([(kind, op) for kind, op, *_ in self.points], [('before', 'pair'), ('after', 'pair')])

    def test_a_single_user_rebuild_is_read_either_side_even_when_it_fails(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator, estimated_single_capture_bytes

        device = make_device(self.operations, self.mesh, slot=2)
        device._packed_capture_released, device.proposal_capture = True, None
        with extent_environment(True), self.ledger():
            self.assertTrue(PackedProposalCoordinator()._ensure_single_user(device))
            device._packed_capture_released, device.proposal_capture = True, None
            with unittest.mock.patch('dflash_proposal_trace.PreparedDFlashProposal', side_effect=RuntimeError('oom')), \
                    self.assertRaisesRegex(RuntimeError, 'oom'):
                PackedProposalCoordinator()._ensure_single_user(device)
        self.assertEqual(self.points, [('before', 'single', dict(estimate=estimated_single_capture_bytes(), point='slot=2')),
                                       ('after', 'single')] * 2)

    def test_the_quad_capture_is_read_either_side(self):
        import quad_draft
        from test_quad_draft import FLAG, REQUIRED, FakeQuadTrace, clean_environment, pair_trace_class, quad_bridges
        from test_quad_draft import selected_tokens
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        FakeQuadTrace.instances, FakeQuadTrace.failures = [], []
        with clean_environment(**{FLAG: '1'}, **REQUIRED, **EXTENT), self.ledger(), \
                unittest.mock.patch('quad_draft.PreparedQuadDFlashProposal', FakeQuadTrace), \
                unittest.mock.patch('dflash_proposal_trace.PreparedPackedDFlashProposal', pair_trace_class()), \
                unittest.mock.patch('dflash_packed_proposal.select_packed_batched', Mock(side_effect=selected_tokens)), \
                unittest.mock.patch('dflash_packed_proposal_coordinator.audit_log'):
            coordinator, bridges = PackedProposalCoordinator(), quad_bridges(self.operations, self.mesh)
            coordinator.prepare(bridges)
            coordinator.prepare(bridges)
        self.assertEqual(len(FakeQuadTrace.instances), 1)
        self.assertEqual(self.points, [('before', 'quad', dict(estimate=quad_draft.QUAD_CAPTURE_BYTES_EST,
                                                               point='slots=0,1,2,3')), ('after', 'quad')])

    def test_with_the_flag_off_the_ledger_is_never_asked(self):
        import memory_ledger
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        with extent_environment(False), \
                unittest.mock.patch.object(memory_ledger, 'before', side_effect=AssertionError('asked')), \
                unittest.mock.patch.object(memory_ledger, 'after', side_effect=AssertionError('asked')):
            coordinator = PackedProposalCoordinator()
            coordinator.prepare(self.pair_bridges())
            device = make_device(self.operations, self.mesh, slot=2)
            device._packed_capture_released, device.proposal_capture = True, None
            self.assertTrue(coordinator._ensure_single_user(device))
        self.assertEqual(len(FakeTrace.instances), 1)

    def test_the_single_estimate_is_its_placeholders_and_one_publication_transient(self):
        from dflash_packed_proposal_coordinator import (PREPARE_PUBLICATION_TRANSIENT_BYTES,
                                                        SINGLE_CAPTURE_PLACEHOLDER_BYTES, estimated_single_capture_bytes)

        history = 2080 * 5120 * 2
        mask, rope_q, rope_k = 32 * 2080 * 2, 2 * 32 * 128 * 2, 2 * 2080 * 128 * 2
        cache = 5 * 2 * 4 * 2048 * 128 * 2
        self.assertEqual(SINGLE_CAPTURE_PLACEHOLDER_BYTES, 64 + history + mask + rope_q + rope_k + cache)
        self.assertEqual(estimated_single_capture_bytes(), SINGLE_CAPTURE_PLACEHOLDER_BYTES + PREPARE_PUBLICATION_TRANSIENT_BYTES)


class CommittedKvHistoryTests(unittest.TestCase):
    """_committed_kv_history: the exact precondition dflash_device.DFlashDevice.
    execute_proposal itself checks (dflash_device.py:661-662) before it accepts a
    packed cached_history for one device - kv_history exists and holds one active
    bank per prepared draft layer. Mocked exactly at that marker, not at a higher
    level, since draft_kv_history.py has no separate 'committed' flag to check."""

    def test_kv_history_none_is_refused(self):
        from dflash_packed_proposal_coordinator import _committed_kv_history

        device = committed_device()
        device.kv_history = None
        self.assertFalse(_committed_kv_history(device))

    def test_fewer_active_banks_than_layers_is_refused(self):
        """The exact shape run 35581352016 hit: execute_proposal's own guard is
        `len(cached_history) != len(self.layers)` - mirrored here as kv_history.active
        holding fewer entries than device.layers."""
        from dflash_packed_proposal_coordinator import _committed_kv_history

        device = committed_device(committed=False)
        self.assertEqual(len(device.kv_history.active), 0)
        self.assertEqual(len(device.layers), 5)
        self.assertFalse(_committed_kv_history(device))

    def test_matching_active_banks_and_layers_is_accepted(self):
        from dflash_packed_proposal_coordinator import _committed_kv_history

        device = committed_device(committed=True)
        self.assertTrue(_committed_kv_history(device))

    def test_no_kv_history_attribute_at_all_is_refused(self):
        from dflash_packed_proposal_coordinator import _committed_kv_history

        device = SimpleNamespace(layers=[object()] * 5)
        self.assertFalse(_committed_kv_history(device))


class PackableTests(unittest.TestCase):
    def test_both_committed_and_steady_state_is_packable(self):
        from dflash_packed_proposal_coordinator import packable

        self.assertTrue(packable(committed_device(), committed_device()))

    def test_either_side_without_a_committed_kv_history_refuses_the_pair(self):
        from dflash_packed_proposal_coordinator import packable

        self.assertFalse(packable(committed_device(committed=False), committed_device()))
        self.assertFalse(packable(committed_device(), committed_device(committed=False)))
        self.assertFalse(packable(committed_device(committed=False), committed_device(committed=False)))


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
