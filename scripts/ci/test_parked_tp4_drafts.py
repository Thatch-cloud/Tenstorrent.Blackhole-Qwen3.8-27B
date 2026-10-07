"""Engine reuse, 2c: the draft trace book (QWEN_FAST_PARKED_DRAFTS=1; design 4, review fixes 4 and N5).

The pair and block-quad drafter traces are bound to the pool slots for the process. Held here:
  - the book is the SOLE holder: a coordinator built while it is registered uses the book's own dicts, so a trace is listed once; every close of
    a pair or block-quad trace goes through the book's retire, which closes it ONCE, unlists it and unwraps the surviving members' capture views;
  - hook close drops views and the per-hook give-up counters and leaves the traces; release_parked retires nothing; singles are never released;
  - a source test: no coordinator path closes a pair or block-quad trace directly (the allowlist of close sites is pinned, so a new one is a review);
  - end to end on the census world (the real factory, engine, device, proposal captures, pool, hook detach and coordinator; pair traces faked):
    a hundred rebinds per slot with no pair recaptured, every trace closed exactly once across an unpark, the kill switch and the shutdown, the
    singles never released, and a flag-off control that does recapture;
  - the policy and contract refuse the drafts switch without what it needs.
"""

import ast
from collections import Counter
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dflash_packed_proposal_coordinator as coordinator_module  # noqa: E402
import serving_parked_engines as parked  # noqa: E402
from test_dflash_packed_proposal_coordinator import FakeTrace, make_bridge, make_device  # noqa: E402
from test_parked_tp4_census import World  # noqa: E402
from test_parked_tp4_set import make_set  # noqa: E402
from test_parked_tp4_wiring import CensusPair, Serving  # noqa: E402

DRAFTS = {'QWEN_FAST_PARKED_ENGINES': '1', 'QWEN_FAST_PARKED_DRAFTS': '1'}


class CountingPair(CensusPair):
    """CensusPair that counts its closes."""
    instances = []

    def __init__(self, device_a, device_b):
        super().__init__(device_a, device_b)
        self.closes = 0
        CountingPair.instances.append(self)

    def close(self):
        self.closes += 1
        super().close()


class CountingTrace(FakeTrace):
    def __init__(self, device_a, device_b):
        super().__init__(device_a, device_b)
        self.closes = 0

    def close(self):
        self.closes += 1
        super().close()


class FakeQuad:
    def __init__(self):
        self.closes = 0
        self.buckets = {1: 1}

    def close(self):
        self.closes += 1


def book_for_test():
    book = parked.DraftTraceBook(log=lambda *arguments: None)
    unregister = coordinator_module.register_draft_book(book)
    return book, unregister


class RegistryTests(unittest.TestCase):
    def test_one_book_at_a_time_and_none_by_default(self):
        self.assertIsNone(coordinator_module.draft_book())
        book, unregister = book_for_test()
        try:
            self.assertIs(coordinator_module.draft_book(), book)
            with self.assertRaisesRegex(ValueError, 'already registered'):
                coordinator_module.register_draft_book(parked.DraftTraceBook())
        finally:
            unregister()
        self.assertIsNone(coordinator_module.draft_book())
        unregister()

    def test_the_set_registers_its_book_under_the_flag_and_unregisters_it_at_close(self):
        with World(environment=DRAFTS) as world:
            engines = make_set(world, environ={'QWEN_FAST_PARKED_DRAFTS': '1'})
            self.assertIs(coordinator_module.draft_book(), engines.book)
            engines.build()
            engines.close()
            self.assertIsNone(coordinator_module.draft_book())
            self.assertTrue(engines.book.closed)
        with World(environment={'QWEN_FAST_PARKED_ENGINES': '1'}) as world:
            engines = make_set(world)
            self.assertIsNone(engines.book)
            self.assertIsNone(coordinator_module.draft_book())

    def test_a_coordinator_uses_the_books_dicts_and_a_flag_off_one_its_own(self):
        book, unregister = book_for_test()
        try:
            coordinator = coordinator_module.PackedProposalCoordinator()
            self.assertIs(coordinator.pairs, book.pairs)
            self.assertIs(coordinator.quad_blocks, book.quad_blocks)
            self.assertIs(coordinator.book, book)
        finally:
            unregister()
        plain = coordinator_module.PackedProposalCoordinator()
        self.assertIsNone(plain.book)
        self.assertEqual((plain.pairs, plain.quad_blocks), ({}, {}))


class BookTests(unittest.TestCase):
    def setUp(self):
        FakeTrace.instances = []
        patcher = patch('dflash_proposal_trace.PreparedPackedDFlashProposal', CountingTrace)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.operations = SimpleNamespace(synchronize_device=Mock())
        self.mesh = object()
        self.book, unregister = book_for_test()
        self.addCleanup(unregister)

    def bridges(self, slots):
        return [make_bridge('r%d' % slot, make_device(self.operations, self.mesh, slot=slot), seed=100 + slot) for slot in slots]

    def test_hook_close_drops_views_and_per_hook_state_and_leaves_the_traces(self):
        coordinator = coordinator_module.PackedProposalCoordinator()
        bridges = self.bridges((0, 1, 2, 3))
        coordinator.prepare(bridges)
        pair_a, pair_b = FakeTrace.instances
        devices = [bridge.request.runtime.drafter for bridge in bridges]
        self.assertTrue(all(isinstance(device.proposal_capture, coordinator_module._PackedCaptureView) for device in devices))
        coordinator.quad_blocked[(0, 1, 2, 3)] = 'reason'
        coordinator.quad_block_failures[(0, 1, 2, 3)] = 1
        coordinator.quad_failures, coordinator.quad_disabled = 1, True
        coordinator.close()
        self.assertEqual((pair_a.closes, pair_b.closes), (0, 0), 'the traces are the slots\' for the process')
        self.assertEqual(sorted(self.book.pairs), [(0, 1), (2, 3)])
        self.assertFalse(any(isinstance(device.proposal_capture, coordinator_module._PackedCaptureView) for device in devices))
        self.assertEqual((coordinator.quad_blocked, coordinator.quad_block_failures, coordinator.quad_failures,
                          coordinator.quad_disabled), ({}, {}, 0, False), 'a drain gives a blocked quad another chance')
        # the next hook's coordinator finds the same traces and builds none
        again = coordinator_module.PackedProposalCoordinator()
        again.prepare(bridges)
        self.assertEqual(len(FakeTrace.instances), 2)
        self.assertEqual(pair_a.prepared, [(100, 101), (100, 101)])

    def test_retire_closes_a_trace_once_and_unwraps_the_survivors(self):
        coordinator = coordinator_module.PackedProposalCoordinator()
        bridges = self.bridges((0, 1))
        coordinator.prepare(bridges)
        (pair,) = FakeTrace.instances
        a, b = (bridge.request.runtime.drafter for bridge in bridges)
        self.assertIsInstance(b.proposal_capture, coordinator_module._PackedCaptureView)
        self.assertTrue(self.book.retire('pair', (0, 1)))
        self.assertFalse(self.book.retire('pair', (0, 1)), 'nothing left to close')
        self.assertEqual(pair.closes, 1)
        self.assertEqual(self.book.pairs, {})
        self.assertNotIsInstance(a.proposal_capture, coordinator_module._PackedCaptureView)
        self.assertNotIsInstance(b.proposal_capture, coordinator_module._PackedCaptureView)

    def test_retire_member_closes_every_pair_and_quad_the_device_is_in_once(self):
        coordinator = coordinator_module.PackedProposalCoordinator()
        bridges = self.bridges((0, 1, 2, 3))
        coordinator.prepare(bridges)
        pair_a, pair_b = FakeTrace.instances
        devices = [bridge.request.runtime.drafter for bridge in bridges]
        quad = FakeQuad()
        self.book.quad_blocks[(0, 1, 2, 3)] = (tuple(devices), quad, True)
        self.assertEqual(self.book.retire_member(devices[1]), 2, 'the quad and its pair')
        self.assertEqual((quad.closes, pair_a.closes, pair_b.closes), (1, 1, 0))
        self.assertEqual(self.book.retire_member(devices[1]), 0)
        self.assertEqual(sorted(self.book.pairs), [(2, 3)])
        self.assertEqual(self.book.retire_all(), 1)
        self.assertEqual(pair_b.closes, 1)

    def test_release_closed_retires_a_closed_members_traces_through_the_book(self):
        coordinator = coordinator_module.PackedProposalCoordinator()
        bridges = self.bridges((0, 1, 2, 3))
        coordinator.prepare(bridges)
        pair_a, pair_b = FakeTrace.instances
        devices = [bridge.request.runtime.drafter for bridge in bridges]
        quad = FakeQuad()
        self.book.quad_blocks[(0, 1, 2, 3)] = (tuple(devices), quad, True)
        devices[2].closed = True
        self.assertEqual(coordinator.release_closed(), dict(quad=1, pairs=[[2, 3]]))
        self.assertEqual((quad.closes, pair_a.closes, pair_b.closes), (1, 0, 1))
        self.assertEqual(self.book.quad_blocks, {})
        self.assertEqual(sorted(self.book.pairs), [(0, 1)])

    def test_the_rebound_generation_retires_nothing_the_book_holds_and_release_parked_does_nothing(self):
        coordinator = coordinator_module.PackedProposalCoordinator()
        bridges = self.bridges((0, 1))
        coordinator.prepare(bridges)
        (pair,) = FakeTrace.instances
        devices = [bridge.request.runtime.drafter for bridge in bridges]
        devices[0].rebind_generation = 5
        with patch.dict(os.environ, {'QWEN_FAST_PARKED_ENGINES': '1'}):
            coordinator.prepare(bridges)
            self.assertEqual(coordinator.release_closed(), dict(quad=0, pairs=[]))
            self.assertEqual(coordinator.release_parked(devices[0]), dict(quad=0, pairs=[]))
        self.assertEqual((len(FakeTrace.instances), pair.closes, coordinator.generations), (1, 0, {}))

    def test_singles_are_never_released_under_the_book(self):
        coordinator = coordinator_module.PackedProposalCoordinator()
        device = make_device(self.operations, self.mesh, slot=0)
        capture = device.proposal_capture
        coordinator._release_single_user(device)
        self.assertIs(device.proposal_capture, capture)
        self.assertFalse(getattr(device, '_packed_capture_released', False))
        capture.close.assert_not_called()

    def test_a_failed_close_still_unwraps_and_the_book_unlists_it(self):
        coordinator = coordinator_module.PackedProposalCoordinator()
        bridges = self.bridges((0, 1))
        coordinator.prepare(bridges)
        (pair,) = FakeTrace.instances
        pair.close = Mock(side_effect=RuntimeError('close failed'))
        with self.assertRaisesRegex(RuntimeError, 'close failed'):
            self.book.retire('pair', (0, 1))
        self.assertEqual(self.book.pairs, {}, 'never listed once its close has been tried')
        self.assertFalse(isinstance(bridges[1].request.runtime.drafter.proposal_capture, coordinator_module._PackedCaptureView))


class SourceTests(unittest.TestCase):
    # Every call to .close() in the coordinator's methods, by (function, receiver). A new one is a decision for a reviewer: a trace the book holds
    # must be closed through DraftTraceBook.retire (via _close_pair / _close_quad_block), never directly.
    ALLOWED = {
        ('_PackedCaptureView.close', 'self._original'),
        ('PackedProposalCoordinator.close', 'trace'),
        ('PackedProposalCoordinator.close', 'self.quad[1]'),
        ('PackedProposalCoordinator.close', 'state[1]'),
        ('PackedProposalCoordinator._close_pair', "self.pairs.pop(group)[2]"),
        ('PackedProposalCoordinator._close_quad_block', 'self.quad_blocks.pop(slots)[1]'),
        ('PackedProposalCoordinator.release_closed', 'self.quad[1]'),
        ('PackedProposalCoordinator.release_parked', 'self.quad[1]'),
        ('PackedProposalCoordinator._retire_quad', 'trace'),
        ('PackedProposalCoordinator._disable_quad', 'self.quad[1]'),
        ('PackedProposalCoordinator._prepare_quad', 'state[1]'),
        ('PackedProposalCoordinator._release_single_user', 'capture._original'),
        ('PackedProposalCoordinator._release_single_user', 'capture'),
        ('PackedProposalCoordinator.prepare', 'singles_audit'),
        ('_discard_round', None),
    }

    def sites(self):
        tree = ast.parse((HERE / 'dflash_packed_proposal_coordinator.py').read_text(encoding='utf-8'))
        found = set()

        def visit(node, scope):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.ClassDef):
                    visit(child, scope + [child.name])
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    visit(child, scope + [child.name])
                else:
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and child.func.attr == 'close':
                        found.add(('.'.join(scope), ast.unparse(child.func.value)))
                    visit(child, scope)
        visit(tree, [])
        return found

    def test_no_coordinator_path_closes_a_book_owned_trace_directly(self):
        found = {site for site in self.sites() if site[0] != '_discard_round'}
        allowed = {site for site in self.ALLOWED if site[0] != '_discard_round'}
        self.assertEqual(found - allowed, set(), 'a new close site: route it through _close_pair / _close_quad_block (the book)')

    def test_the_block_quad_and_pair_paths_close_only_through_the_helpers(self):
        paths = ('PackedProposalCoordinator._block_quad', 'PackedProposalCoordinator._trace_for',
                 'PackedProposalCoordinator._prepare_quad_blocks')
        direct = {site for site in self.sites() if site[0] in paths}
        self.assertEqual(direct, set())
        source = (HERE / 'dflash_packed_proposal_coordinator.py').read_text(encoding='utf-8')
        for helper in ('_close_pair', '_close_quad_block'):
            self.assertIn('def %s(self' % helper, source)


class EndToEndTests(unittest.TestCase):
    """The real factory, engine, device and hook on the census world with the book registered: a hundred rebinds per slot, no pair recaptured."""

    LENGTHS = (1, 17, 300, 2047, 2048, 2049, 3000)
    BUDGETS = (2, 3, 5, 16, 40)

    def run_churn(self, drafts, minimum, seed):
        CountingPair.instances = []
        environment = DRAFTS if drafts else {'QWEN_FAST_PARKED_ENGINES': '1'}
        rng = random.Random(seed)
        with World(environment=environment) as world, patch('dflash_proposal_trace.PreparedPackedDFlashProposal', CountingPair):
            engines = make_set(world, environ={'QWEN_FAST_PARKED_DRAFTS': '1'} if drafts else {})
            engines.build()
            serving = Serving(world, engines)
            attached = Counter((tensor.label, tuple(tensor.shape)) for tensor in world.ops.live.values())
            events = []
            while min(entry.rebinds for entry in engines.slots) < minimum:
                while len(serving.hook.bridges) < 4:
                    prompt = rng.choice(self.LENGTHS)
                    serving.arrive(prompt, min(rng.choice(self.BUDGETS), 68 * 64 - prompt - 1))
                serving.round()
                if serving.hook.bridges and rng.random() < 0.15:
                    serving.leave(rng.choice(sorted(serving.hook.bridges)))
            for request_id in sorted(serving.hook.bridges):
                serving.leave(request_id)
            serving.drain()
            return world, engines, serving, attached

    def test_a_hundred_rebinds_per_slot_recapture_no_pair_and_close_every_trace_once_at_shutdown(self):
        CountingPair.instances = []
        with World(environment=DRAFTS) as world, patch('dflash_proposal_trace.PreparedPackedDFlashProposal', CountingPair):
            engines = make_set(world, environ={'QWEN_FAST_PARKED_DRAFTS': '1'})
            engines.build()
            serving = Serving(world, engines)
            rng = random.Random(31)
            while min(entry.rebinds for entry in engines.slots) < 100:
                while len(serving.hook.bridges) < 4:
                    prompt = rng.choice(self.LENGTHS)
                    serving.arrive(prompt, min(rng.choice(self.BUDGETS), 68 * 64 - prompt - 1))
                serving.round()
                if serving.hook.bridges and rng.random() < 0.12:
                    serving.leave(rng.choice(sorted(serving.hook.bridges)))
            for request_id in sorted(serving.hook.bridges):
                serving.leave(request_id)
            serving.drain()
            # the pairs formed once per slot pair; no hook, no detach and no rebind retired or recaptured one
            slots = [tuple(sorted((trace.device_a.pool_slot.index, trace.device_b.pool_slot.index))) for trace in CountingPair.instances]
            self.assertLessEqual(len(CountingPair.instances), 6, 'at most one trace per pair of slots: %r' % (slots,))
            self.assertEqual(len(set(slots)), len(slots), 'no pair was captured twice')
            self.assertTrue(all(trace.closes == 0 for trace in CountingPair.instances), 'every trace outlived every request and every hook')
            self.assertEqual(sorted(engines.book.pairs), sorted(set(slots)))
            self.assertFalse(any(engines.single_released(entry) for entry in engines.slots), 'singles are kept')
            self.assertEqual(world.ops.violations, [])
            self.assertEqual(engines.unparks, 0)
            # the traces were served to members rebound since their capture: that is the point of the book (the flag-off control below
            # retires them at every park instead)
            self.assertTrue(any(served != trace.captured for trace in CountingPair.instances for served in trace.served))
            # an unpark retires the slot's pair once, and the slot's next request recaptures it once
            traces = list(CountingPair.instances)
            victim = traces[0]
            index = victim.device_a.pool_slot.index
            key = tuple(sorted((victim.device_a.pool_slot.index, victim.device_b.pool_slot.index)))
            engines.unpark(engines.slots[index], 'test')
            self.assertEqual(victim.closes, 1)
            self.assertNotIn(key, engines.book.pairs)
            engines.repark_idle()
            # the kill switch retires every remaining trace, once; the shutdown closes none twice
            engines.switch_off('test')
            self.assertEqual(engines.book.pairs, {})
            engines.close()
            self.assertTrue(all(trace.closes == 1 for trace in CountingPair.instances),
                            'each trace closed exactly once: %r' % [trace.closes for trace in CountingPair.instances])

    def test_the_flag_off_control_retires_and_recaptures_the_pairs(self):
        world, engines, serving, attached = self.run_churn(False, 12, 29)
        try:
            self.assertGreater(len(CountingPair.instances), 6, 'E1 retires at every park: pairs are recaptured')
            self.assertTrue(any(trace.closes for trace in CountingPair.instances))
        finally:
            world.__exit__(None, None, None)


class PolicyTests(unittest.TestCase):
    def problems(self, **extra):
        import serving_fast_policy as policy

        environ = {'QWEN_FAST_ANY_REQUEST': '1', 'QWEN_FAST_EXTENT_REPLAY': '1', 'QWEN_FAST_PACKED_STEP': '1', 'QWEN_FAST_SHARED_CCL': '1',
                   'QWEN_FAST_TP': '4', 'QWEN_FAST_M3_REQUEST_WARM': '1', 'QWEN_FAST_VERIFY_T1': '1', 'QWEN_FAST_M3_BLOCKS': '2',
                   'QWEN_FAST_PARKED_ENGINES': '1', 'QWEN_FAST_PARKED_DRAFTS': '1', 'QWEN_FAST_QUAD_DRAFT': '1',
                   'QWEN_FAST_QUAD_DRAFT_BLOCKS': '2', 'QWEN_FAST_FUSED_COMMIT_LIVE_BANKS': '1'}
        environ.update(extra)
        return policy.parked_engine_problems({name: value for name, value in environ.items() if value is not None}, 8)

    def test_the_drafts_switch_needs_the_engines_the_quad_over_both_blocks_the_live_banks_and_no_wide_drafter(self):
        self.assertEqual(self.problems(), [])
        for name, value, text in (('QWEN_FAST_PARKED_ENGINES', None, 'needs QWEN_FAST_PARKED_ENGINES=1'),
                                  ('QWEN_FAST_QUAD_DRAFT', None, 'QWEN_FAST_QUAD_DRAFT=1'),
                                  ('QWEN_FAST_QUAD_DRAFT_BLOCKS', '0', 'QWEN_FAST_QUAD_DRAFT_BLOCKS=2'),
                                  ('QWEN_FAST_FUSED_COMMIT_LIVE_BANKS', None, 'LIVE_BANKS=1'),
                                  ('QWEN_FAST_TP4_DRAFT_WIDE', '1', 'DRAFT_WIDE')):
            with self.subTest(name=name):
                problems = self.problems(**{name: value})
                self.assertTrue(any(text in problem for problem in problems), problems)


if __name__ == '__main__':
    unittest.main()
