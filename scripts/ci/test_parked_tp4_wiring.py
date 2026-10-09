"""Engine reuse, E4': the parked engines wired into serving at four chips (design 3.4, 5.9, 5.11, 5.12; the build table's E4').

  - FastRequest(release_engine=): what close() calls in place of engine.close(); None keeps close() the base
    commit's, event for event;
  - serving_request_factory.from_prefill(parked=): every host check in its order (the same refusals flag on and
    off, the parked engine untouched), the backstop's parked terms, then the taken slot rebound (rebind_parked)
    instead of built, or today's build on an unparked slot; with the E4 modules at the base commit the census
    trail of today's churn is unchanged;
  - the park in two halves (park_engine, park_drafter): an engine that cannot park is closed with its device, a
    slot left unfit (a failed page binding) is unparked, an engine with a block in flight raises as close() does;
  - after a park (after_park, the worker hook's release_parked): the device's pair and quad traces retired
    (the coordinator's release_parked) and its released single rebuilt when the split allows, else kept released
    until the next rebind rebuilds it; the lifecycle's idle moment does the same and re-parks unparked slots;
  - the coordinator's generation: a rebound device never replays a pair or quad trace captured for its previous
    request, even when release_parked is skipped (_cached_trace, _retire_quad, release_closed); flag off, identity
    and `closed` decide as before;
  - THE REGRESSION: a device whose single was released at 2048 and that is rebound at P = 300 rebuilds a (2048,)
    bucket and survives its history growing past 512; the coordinator's own rebuild below 2048 is scoped the same
    way under the flag, and unscoped (flag off) it builds (512,) and the round past 512 raises;
  - the cold-build fallback: a rebind refused on the host (before any write and before take()) or failed after its first write
    unparks the slot, and today's build runs on the same slot over the SAME prefill capture, which is closed exactly once, admitted on today's
    DRAM terms; a request short of them is refused (quarantined) and the engine lives;
  - end to end on the census world (the real factory, engine, device, proposal captures, pool, hook detach and
    coordinator; pair traces faked): four users, twenty rebinds per slot and more, pairs formed and retired,
    singles released and rebuilt, the hook closed and the idle moment between waves - no census violation, no
    unpark, nothing unowned, and the attach's allocations exactly what is left;
  - serving_runtime's bridge factory (the set passed on; a rebound request logs no build time and walks no engine
    ledger; a failed binding marks the slot unfit), the DRAM registration's and the lifecycle's keywords, and the
    lifecycle's idle moment.
"""

from collections import Counter
from functools import partial
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import serving_parked_engines as parked  # noqa: E402
from test_dflash_packed_proposal_coordinator import FakeTrace, make_bridge, make_device  # noqa: E402
from test_parked_tp4_census import PARITY_MODULES, CensusCapture, World, base_module  # noqa: E402
from test_parked_tp4_set import make_set  # noqa: E402

PARKED = {'QWEN_FAST_PARKED_ENGINES': '1'}
CPU_WORKFLOW = ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml'


def flag_environment(on):
    """QWEN_FAST_PARKED_ENGINES exactly as given (on: '1'; off: unset), the rest of the environment kept."""
    environment = {name: value for name, value in os.environ.items() if name != parked.FLAG}
    if on:
        environment[parked.FLAG] = '1'
    return patch.dict(os.environ, environment, clear=True)


def admit(world, engines, request_id, prompt, budget, *, first_block=None, sampling=None, blocks=None):
    """serving_runtime's bridge factory as far as the request: from_prefill beside the four-user block, with the
    attach's parked set (`engines`, None for today's call). Each slot's requests take their own page range."""
    from serving_request_factory import from_prefill

    if blocks is None:
        if first_block is None:
            entry = engines.peek() if engines is not None else None
            first_block = 1 + 70 * (entry.index if entry is not None else 0)
        blocks = list(range(first_block, first_block + -(-(prompt + budget) // 64) + 1))
    state = world.state(request_id, prompt, budget, blocks=blocks)
    if sampling is not None:
        state.sampling_params = sampling
    capture = CensusCapture(world.ops, world.mesh, prompt)
    world.last_capture = capture
    return from_prefill(world.ops, world.model, world.sampler, world.pages(state), world.helpers, state=state,
                        capture=capture, fixtures=world.fixtures, eos_ids=(99,), collectives=world.collectives,
                        buffer_pool=world.pool, shared_weights=world.weights, capture_rows=4,
                        **({} if engines is None else dict(parked=engines)))


def run_to_end(request):
    request_id = request.session.request_id
    steps = 0
    while not request.session.finished:
        request.step(request_id, cancelled=lambda: False)
        steps += 1
    return steps


def bare_hook(coordinator):
    """A FastWorkerHook holding bridges and the coordinator, without a worker: detach and close are the real ones."""
    from serving_worker_hook import FastWorkerHook

    hook = FastWorkerHook.__new__(FastWorkerHook)
    hook.bridges, hook.saved, hook.closed = {}, [], False
    if coordinator is not None:
        hook._packed_coordinator = coordinator
    return hook


def bridge_of(request):
    return SimpleNamespace(request=request, failed=False, close=partial(request.close, request.session.request_id))


class CensusPair(FakeTrace):
    """PreparedPackedDFlashProposal as the coordinator drives it, over two real census devices: the draft is zeros (the
    census's own selection), nothing is allocated, and every prepare records its members' rebind_generation beside
    the ones it was captured at - the property this suite holds: never served to a rebound member."""
    instances = []

    def __init__(self, device_a, device_b):
        from dflash_packed_proposal_coordinator import generations

        super().__init__(device_a, device_b)
        self.captured = generations((device_a, device_b))
        self.served, self.pending = [], {}
        CensusPair.instances.append(self)

    def prepare_device(self, seed_a, seed_b):
        from dflash_packed_proposal_coordinator import generations

        if self.closed:
            raise AssertionError('a closed pair trace was prepared')
        self.served.append(generations((self.device_a, self.device_b)))
        self.pending = {'a': seed_a, 'b': seed_b}
        return super().prepare_device(seed_a, seed_b)

    def has_pending(self, which, seed):
        return not self.closed and self.pending.get(which) == seed

    def finish(self, which, count):
        self.pending.pop(which, None)
        return tuple(0 for _ in range(count))

    def discard_pending(self):
        self.pending = {}
        super().discard_pending()


# -----------------------------------------------------------------------------------------------------------------
# FastRequest(release_engine=)
# -----------------------------------------------------------------------------------------------------------------
class FastRequestReleaseTests(unittest.TestCase):
    def parts(self, events, pending=False):
        session = SimpleNamespace(phase='idle', verifier_rows=16, request_id='r', pending='ticket' if pending else None,
                                  fail_verification=Mock(side_effect=lambda *args: events.append('fail')),
                                  close=Mock(side_effect=lambda request_id: events.append('session_close')))
        engine = SimpleNamespace(phase='idle', close=Mock(side_effect=lambda: events.append('engine_close')))
        engine.session = session
        runtime = SimpleNamespace(drafter_name='dflash2', bind=Mock())
        return session, engine, runtime

    def close_events(self, module, pending=False, **options):
        events = []
        session, engine, runtime = self.parts(events, pending)
        request = module.FastRequest(session, engine, runtime, release_drafter=lambda: events.append('drafter'),
                                     **options)
        if pending:
            session.phase = 'pending'
        request.close('r')
        request.close('r')
        return events

    def test_without_it_close_is_the_base_commits_event_for_event(self):
        import serving_fast_request

        base = base_module('serving_fast_request')
        for pending in (False, True):
            with self.subTest(pending=pending):
                self.assertEqual(self.close_events(serving_fast_request, pending),
                                 self.close_events(base, pending))
        self.assertEqual(self.close_events(serving_fast_request), ['engine_close', 'drafter', 'session_close'])

    def test_with_it_the_engine_is_released_in_place_of_its_close(self):
        import serving_fast_request

        released = []
        events = self.close_events(serving_fast_request, release_engine=lambda: released.append(1))
        self.assertEqual(events, ['drafter', 'session_close'])
        self.assertEqual(released, [1], 'once, however often close is called')
        events = []
        session, engine, runtime = self.parts(events)
        request = serving_fast_request.FastRequest(session, engine, runtime, release_drafter=lambda: events.append('d'),
                                                   release_engine=lambda: events.append('e'))
        request.close('r')
        self.assertEqual(events, ['e', 'd', 'session_close'])

    def test_a_release_that_is_not_callable_is_refused(self):
        import serving_fast_request

        session, engine, runtime = self.parts([])
        with self.assertRaisesRegex(ValueError, 'explicit cleanup required'):
            serving_fast_request.FastRequest(session, engine, runtime, release_drafter=lambda: None,
                                             release_engine='engine.close')


# -----------------------------------------------------------------------------------------------------------------
# from_prefill(parked=)
# -----------------------------------------------------------------------------------------------------------------
class FlagOffTrailTests(unittest.TestCase):
    def test_todays_churn_with_the_e4_modules_at_the_base_commit_is_todays_trail(self):
        """The census trail - every allocation, read, write, free, upload by its bytes, capture, replay and fence -
        of today's per-request churn with serving_request_factory and serving_fast_request at the base commit (and
        verifier_engine_tp, serving_buffer_pool and the coordinator, the census's own swap) equals today's."""
        from test_parked_tp4_census import TodayChurnCensusTests

        fast = base_module('serving_fast_request')
        with patch.dict(sys.modules, {'serving_fast_request': fast}):
            factory = base_module('serving_request_factory')
        base = {name: base_module(name) for name in PARITY_MODULES if name != 'serving_request_factory'}
        base.update(serving_fast_request=fast, serving_request_factory=factory)
        def trail(modules):
            with World(modules=modules, tracking=True) as world:
                TodayChurnCensusTests().churn(world)
                if modules:
                    from serving_request_factory import from_prefill

                    self.assertIs(sys.modules['serving_request_factory'], factory)
                    self.assertIs(from_prefill, factory.from_prefill)
                return world.ops.trail

        before = trail(base)
        self.assertGreater(len(before), 10000)
        self.assertEqual(trail({}), before)


class ParkedFactoryTests(unittest.TestCase):
    def test_a_request_on_a_parked_slot_is_rebound_not_built_and_its_close_parks_it(self):
        with World(environment=PARKED) as world:
            engines = make_set(world)
            engines.build()
            entry = engines.slots[0]
            captures, generation = world.ops.captures, entry.device.rebind_generation
            request = admit(world, engines, 'request', 300, 24)
            self.assertEqual(world.ops.captures, captures, 'nothing captured: rebound')
            self.assertIs(request.parked_slot, entry)
            self.assertIs(request.engine, entry.engine)
            self.assertIs(request.runtime.drafter, entry.device)
            self.assertEqual(entry.device.rebind_generation, generation + 1)
            self.assertEqual((entry.state, entry.rebinds), ('serving', 1))
            self.assertEqual(request.release_engine.func, engines.park_engine)
            self.assertEqual(request.release_drafter.func, engines.park_drafter)
            buckets = [line for line in world.lines if line.startswith('[PINDIAG] proposal buckets built request=')]
            self.assertEqual(buckets, ['[PINDIAG] proposal buckets built request=request contexts=(2048,)'])
            self.assertTrue(any(line.startswith(parked.REBIND_MARKER + 'req=request slot=0 ') for line in world.lines))
            self.assertFalse([line for line in world.lines if line.startswith('[PINDIAG] any-request engine for')],
                             'the build-only lines are not logged for a rebind')
            run_to_end(request)
            request.close('request')
            self.assertEqual((entry.state, entry.parks, engines.unparks), ('parked', 1, 0))
            self.assertEqual(entry.engine.phase, 'parked')
            self.assertEqual(world.ops.violations, [])

    def test_a_rebound_session_honours_eos_only_without_ignore_eos(self):
        """D4 (B4 5b) on the parked path: under ignore_eos the session keeps only the budget, else it stops at the EOS."""
        from test_parked_tp4_census import sampling

        for ignore, expected in ((True, ()), (False, (99,))):
            with self.subTest(ignore_eos=ignore), World(environment=PARKED) as world:
                engines = make_set(world)
                engines.build()
                request = admit(world, engines, 'request', 300, 24, sampling=sampling(24, ignore_eos=ignore))
                self.assertEqual(tuple(request.session.eos_ids), expected)
                run_to_end(request)
                request.close('request')

    def test_host_refusals_are_todays_and_leave_the_parked_engine_untouched(self):
        from serving_request_factory import RequestRefused
        from test_parked_tp4_census import sampling

        hot = sampling(16)
        hot.temperature = 0.7
        cases = (dict(sampling=hot), dict(sampling=sampling(0)), dict(blocks=[1, 2]))
        for case in cases:
            outcomes = []
            for on in (False, True):
                with self.subTest(case=sorted(case), parked=on), World(environment=PARKED if on else {}) as world:
                    engines = make_set(world) if on else None
                    if on:
                        engines.build()
                    captures = world.ops.captures
                    with self.assertRaises(RequestRefused) as caught:
                        admit(world, engines, 'refused', 300, 16, **case)
                    outcomes.append(str(caught.exception))
                    self.assertEqual(world.ops.captures, captures)
                    if on:
                        self.assertEqual([entry.state for entry in engines.slots], ['parked'] * 4)
                        self.assertEqual(sum(entry.rebinds for entry in engines.slots), 0)
                        self.assertFalse([line for line in world.lines if line.startswith(parked.REBIND_MARKER)])
            self.assertEqual(outcomes[0], outcomes[1])

    def test_an_unparked_slot_is_built_as_today_on_that_slot(self):
        with World(environment=PARKED) as world:
            engines = make_set(world)
            engines.build()
            engines.unpark(engines.slots[0], 'test')
            captures = world.ops.captures
            request = admit(world, engines, 'fresh', 300, 16)
            self.assertIsNone(getattr(request, 'parked_slot', None))
            self.assertIsNone(request.release_engine)
            self.assertIs(request.runtime.drafter.pool_slot, world.pool.slots[0])
            self.assertGreater(world.ops.captures, captures, 'built')
            run_to_end(request)
            request.close('fresh')
            self.assertFalse(world.pool.slots[0].lent)
            self.assertEqual(world.ops.violations, [])

    def test_the_backstop_asks_the_parked_terms_of_the_slot_the_request_takes(self):
        import serving_prefill_admission as admission
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        with World(environment=PARKED) as world:
            engines = make_set(world)
            engines.build()
            seen = []
            real = __import__('serving_request_factory').dram_backstop

            def backstop(pool, **options):
                seen.append(options.get('parked'))
                options.pop('make_room', None)
                return real(pool, **options)
            with patch('serving_request_factory.dram_backstop', side_effect=backstop):
                first = admit(world, engines, 'first', 300, 4)
                PackedProposalCoordinator()._release_single_user(engines.slots[1].device)
                second = admit(world, engines, 'second', 300, 4)
                engines.unpark(engines.slots[2], 'test')
                third = admit(world, engines, 'third', 300, 4)
            credit = lambda index: engines.ladder_credit(engines.slots[index])   # noqa: E731
            self.assertEqual([None if terms is None else {name: value for name, value in terms.items() if name != 'credit'}
                              for terms in seen],
                             [dict(rebind=admission.PARKED_REBIND_BYTES, single=0),
                              dict(rebind=admission.PARKED_REBIND_BYTES, single=admission.measured_single_capture_bytes()),
                              None])
            self.assertTrue(all('credit' in terms for terms in seen[:2]), 'the ladder\'s credit travels with the terms')
            rebinds = [line for line in world.lines if line.startswith(parked.REBIND_MARKER)]
            self.assertRegex(rebinds[-1], 'req=second slot=1 .* single_rebuilt=1 window=256 widths=')
            self.assertEqual(tuple(second.runtime.drafter.proposal_capture.buckets), (2048,))
            for request in (first, second, third):
                request.close(request.session.request_id)

    def test_the_ledger_reads_the_rebind(self):
        import memory_ledger
        import serving_prefill_admission as admission

        with World(environment=PARKED) as world:
            engines = make_set(world)
            engines.build()
            calls = []
            with patch.object(memory_ledger, 'before', side_effect=lambda op, **kw: calls.append((op, kw)) or 'token'), \
                    patch.object(memory_ledger, 'after', side_effect=lambda token: calls.append(('after', token))):
                request = admit(world, engines, 'ledger', 300, 4)
            self.assertEqual([call[0] for call in calls], ['rebind', 'after'])
            self.assertEqual(calls[0][1]['estimate'], admission.PARKED_REBIND_BYTES)
            self.assertEqual(calls[0][1]['point'], 'req=ledger slot=0')
            request.close('ledger')

    def test_a_failed_rebind_unparks_the_slot_and_the_request_is_served_by_a_cold_build_over_the_same_capture(self):
        with World(environment=PARKED) as world:
            engines = make_set(world)
            engines.build()
            with patch.object(parked, 'rebind_device', side_effect=RuntimeError('device fault')):
                request = admit(world, engines, 'failed', 300, 16)
            capture = world.last_capture
            self.assertEqual(capture.closes, 1, 'the capture has one owner and is closed once')
            self.assertEqual((engines.slots[0].state, engines.unparks, engines.fallbacks), ('unparked', 1, 1))
            self.assertIsNone(getattr(request, 'parked_slot', None), 'a cold build served it')
            self.assertIs(request.runtime.drafter.pool_slot, world.pool.slots[0], 'on the same slot')
            self.assertTrue(any(line.startswith('[PINDIAG] parked rebind failed req=failed slot=0 ') and line.endswith('fallback=build')
                                for line in world.lines))
            run_to_end(request)
            request.close('failed')
            self.assertFalse(world.pool.slots[0].lent)
            self.assertEqual(world.ops.violations, [])
            self.assertEqual(engines.repark_idle(), [0])

    def test_the_slot_pin_is_released_by_from_prefill_on_every_path(self):
        with World(environment=PARKED) as world:
            engines = make_set(world)
            engines.build()
            rebound = admit(world, engines, 'rebound', 300, 8)
            self.assertIsNotNone(rebound.parked_slot)
            self.assertIsNone(world.pool._slot_order, 'a rebind pins nothing and leaves nothing pinned')
            engines.unpark(engines.slots[1], 'test')
            cold = admit(world, engines, 'cold', 300, 8)
            self.assertIsNone(world.pool._slot_order, 'a pinned cold build released its pin by its own exit')
            self.assertIsNone(getattr(cold, 'parked_slot', None))
            run_to_end(rebound)
            rebound.close('rebound')
            run_to_end(cold)
            cold.close('cold')
            self.assertIsNone(world.pool._slot_order)

    def test_a_rebind_refused_on_the_host_unparks_before_any_write_and_a_cold_build_serves_the_slot(self):
        with World(environment=PARKED) as world:
            engines = make_set(world, environ={parked.FAULT_FLAG: 'rebind'})
            engines.build()
            ops = world.ops
            trail_before = len(ops.trail)
            request = admit(world, engines, 'refused', 300, 16)
            self.assertEqual(world.last_capture.closes, 1)
            self.assertEqual((engines.slots[0].state, engines.fallbacks), ('unparked', 1))
            self.assertIs(request.runtime.drafter.pool_slot, world.pool.slots[0])
            refused = [line for line in world.lines if line.startswith(parked.REFUSED_MARKER)]
            self.assertEqual(len(refused), 1)
            self.assertIn('slot=0 reason=%s fallback=build' % parked.REBIND_FAULT_REASON, refused[0])
            self.assertFalse([line for line in world.lines if line.startswith(parked.REBIND_MARKER + 'req=')], 'nothing was rebound')
            run_to_end(request)
            request.close('refused')
            # the hook's detach for a request a cold build served does nothing (other seats decode: a re-park is a full build); the idle
            # moment re-parks the free slot, so the next arrival is rebound, not built again
            builds = len(world.lines)
            self.assertIsNone(parked.release_parked(None, request))
            self.assertEqual(engines.slots[0].state, 'unparked')
            self.assertFalse([line for line in world.lines[builds:] if ' re-parked ms=' in line])
            self.assertEqual(engines.idle()['reparked'], [0])
            self.assertEqual(engines.slots[0].state, 'parked')
            again = admit(world, engines, 'again', 300, 16)
            self.assertIs(again.parked_slot, engines.slots[0])
            again.close('again')
            self.assertEqual(world.ops.violations, [])

    def test_the_cold_build_after_a_fallback_is_admitted_on_todays_terms_and_a_short_reading_refuses_only_that_request(self):
        import serving_prefill_admission as admission
        from serving_request_factory import RequestRefused

        with World(environment=PARKED) as world:
            engines = make_set(world, environ={parked.FAULT_FLAG: 'rebind'})
            engines.build()
            reserve = engines.reserve()
            today = admission.backstop_need(reserve)
            parked_need = admission.parked_backstop_need(reserve, rebind=admission.PARKED_REBIND_BYTES, single=0)
            self.assertGreater(today, parked_need)
            # a reading that holds the parked terms and is short of today's
            reading = dict(free=parked_need + admission.STRANDED_BYTES + 10 ** 6, largest_free=4 * 10 ** 9,
                           trace_largest_free=None, trace_unread='n/a')
            with patch('serving_prefill_admission.dram_reading', return_value=(reading, None)):
                with self.assertRaises(RequestRefused):
                    admit(world, engines, 'short', 300, 16)
            self.assertEqual(world.last_capture.closes, 1, 'the capture is closed on the refusal too')
            self.assertEqual(engines.slots[0].state, 'unparked')
            self.assertFalse(world.pool.slots[0].lent)
            # the cold arrival's backstop ran the release ladder once before it refused (rung 2 released the others' singles, rung 3 unparked the last idle parked slot); the rest live
            self.assertEqual([entry.state for entry in engines.slots[1:]], ['parked', 'parked', 'unparked'], 'the engine and the other slots live')
            self.assertTrue([line for line in world.lines if line.startswith(parked.LADDER_MARKER.split('{')[0])], 'the ladder ran')
            # and the server serves the next arrival: slot 0 is the lowest free slot, so by today's build pinned to it; then the parked ones
            request = admit(world, engines, 'next', 300, 16)
            self.assertIsNone(getattr(request, 'parked_slot', None))
            self.assertIs(request.runtime.drafter.pool_slot, world.pool.slots[0])
            after = admit(world, engines, 'after', 300, 16)
            self.assertIs(after.parked_slot, engines.slots[1])
            after.close('after')
            request.close('next')

    def test_the_slot_taken_is_the_one_the_backstop_priced_or_the_backstop_asks_again(self):
        from serving_request_factory import RequestRefused

        with World(environment=PARKED) as world:
            engines = make_set(world)
            engines.build()
            seen = []
            real = __import__('serving_request_factory').dram_backstop

            def backstop(pool, **options):
                seen.append(options.get('parked'))
                options.pop('make_room', None)
                return real(pool, **options)
            real_peek = engines.peek
            calls = []

            def peek():
                calls.append(1)
                # the placement moves between the backstop's pricing and the take: the first peek says slot 0, the later ones slot 1
                return engines.slots[0] if len(calls) == 1 else engines.slots[1]
            with patch('serving_request_factory.dram_backstop', side_effect=backstop), patch.object(engines, 'peek', peek), \
                    patch.object(engines, 'place', lambda: engines.slots[1]):
                request = admit(world, engines, 'moved', 300, 4, first_block=1)
            self.assertEqual(len(seen), 2, 'asked again, on the terms of the slot actually taken')
            self.assertIs(request.parked_slot, engines.slots[1])
            request.close('moved')


# -----------------------------------------------------------------------------------------------------------------
# The park in two halves, and after it
# -----------------------------------------------------------------------------------------------------------------
class ParkHalvesTests(unittest.TestCase):
    def test_an_engine_that_cannot_park_is_closed_with_its_device(self):
        with World(environment=PARKED) as world:
            engines = make_set(world)
            engines.build()
            entry = engines.slots[0]
            request = admit(world, engines, 'failed', 300, 16)
            request.step('failed', cancelled=lambda: False)
            request.engine.phase = 'failed'
            device = entry.device
            request.close('failed')
            self.assertEqual((entry.state, engines.unparks, entry.engine, entry.device), ('unparked', 1, None, None))
            self.assertTrue(device.closed)
            self.assertFalse(world.pool.slots[0].lent)
            self.assertIn('[PINDIAG] parked slot 0 unparked: engine phase failed', world.lines)

    def test_a_slot_left_unfit_is_unparked_without_parking_its_engine(self):
        with World(environment=PARKED) as world:
            engines = make_set(world)
            engines.build()
            entry = engines.slots[0]
            request = admit(world, engines, 'unfit', 300, 16)
            request.parked_slot.unfit = 'page binding failed'
            with patch.object(type(entry.engine), 'park', side_effect=AssertionError('parked')):
                request.close('unfit')
            self.assertEqual((entry.state, engines.unparks), ('unparked', 1))
            self.assertIn('[PINDIAG] parked slot 0 unparked: page binding failed', world.lines)
            self.assertIsNone(entry.unfit)

    def test_an_engine_with_a_block_in_flight_raises_as_close_does_and_the_slot_stays_serving(self):
        with World(environment=PARKED) as world:
            engines = make_set(world)
            engines.build()
            entry = engines.slots[0]
            request = admit(world, engines, 'busy', 300, 16)
            request.engine.phase = 'verified'
            with self.assertRaisesRegex(ValueError, 'Finish or abort the pending verifier block'):
                request.close('busy')
            self.assertEqual(entry.state, 'serving')
            request.engine.phase = 'idle'
            request.close('busy')
            self.assertEqual(entry.state, 'parked')


class AfterParkTests(unittest.TestCase):
    def released_and_parked(self, world, engines):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        entry = engines.slots[0]
        request = admit(world, engines, 'departing', 300, 8)
        PackedProposalCoordinator()._release_single_user(entry.device)
        request.close('departing')
        self.assertEqual(entry.state, 'parked')
        self.assertTrue(engines.single_released(entry))
        return entry

    def test_the_released_single_is_rebuilt_at_park_when_the_split_allows(self):
        with World(environment=PARKED) as world:
            engines = make_set(world)
            engines.build()
            entry = self.released_and_parked(world, engines)
            coordinator = SimpleNamespace(release_parked=Mock(return_value=dict(quad=0, pairs=[])))
            result = engines.after_park(entry, coordinator)
            coordinator.release_parked.assert_called_once_with(entry.device)
            self.assertEqual(result, dict(released=dict(quad=0, pairs=[]), single=True))
            self.assertEqual(tuple(entry.device.proposal_capture.buckets), (2048,))
            self.assertFalse(entry.device._packed_capture_released)
            self.assertEqual(engines.arrival_terms()['single'], 0)
            self.assertTrue(any(line.startswith('[PINDIAG] parked slot 0 single rebuilt at park ms=')
                                for line in world.lines))
            self.assertEqual(world.ops.violations, [])

    def test_a_short_split_keeps_it_released_and_the_next_rebind_rebuilds_it(self):
        import serving_prefill_admission as admission

        with World(environment=PARKED) as world:
            engines = make_set(world)
            engines.build()
            entry = self.released_and_parked(world, engines)
            short = dict(free=10 ** 9, largest_free=4 * 10 ** 9, trace_largest_free=None, trace_unread='n/a')
            with patch('serving_prefill_admission.dram_reading', return_value=(short, None)):
                self.assertEqual(engines.after_park(entry, None), dict(released=None, single=False))
            kept = [line for line in world.lines if line.startswith('[PINDIAG] parked slot 0 single kept released at park')]
            self.assertEqual(len(kept), 1)
            self.assertIn('short of free', kept[0])
            terms = engines.arrival_terms()
            self.assertEqual((terms['rebind'], terms['single']), (admission.PARKED_REBIND_BYTES,
                                                                   admission.measured_single_capture_bytes()))
            request = admit(world, engines, 'next', 300, 8)
            self.assertIs(request.parked_slot, entry)
            self.assertRegex([line for line in world.lines if line.startswith(parked.REBIND_MARKER)][-1],
                             'single_rebuilt=1')
            request.close('next')

    def test_a_failed_retirement_is_logged_and_the_rebuild_still_runs(self):
        with World(environment=PARKED) as world:
            engines = make_set(world)
            engines.build()
            entry = self.released_and_parked(world, engines)
            coordinator = SimpleNamespace(release_parked=Mock(side_effect=RuntimeError('trace fault')))
            self.assertEqual(engines.after_park(entry, coordinator), dict(released=None, single=True))
            self.assertTrue(any('release parked slot=0 failed (RuntimeError: trace fault)' in line
                                for line in world.lines))

    def test_nothing_happens_for_an_unparked_slot_or_a_request_today_built(self):
        with World(environment=PARKED) as world:
            engines = make_set(world)
            engines.build()
            entry = engines.slots[0]
            engines.unpark(entry, 'test')
            coordinator = SimpleNamespace(release_parked=Mock(side_effect=AssertionError('released')))
            self.assertIsNone(engines.after_park(entry, coordinator))
            # a request today's build served: nothing of the coordinator's, and no re-park at a detach (the idle moment's)
            self.assertIsNone(parked.release_parked(coordinator, SimpleNamespace()))
            self.assertEqual(entry.state, 'unparked')
            self.assertIsNone(parked.release_parked(coordinator, None))
            engines.close()
            self.assertIsNone(parked.release_parked(coordinator, SimpleNamespace()), 'no set: nothing')

    def test_a_failed_capture_at_the_idle_moment_leaves_the_slot_as_it_was_and_the_server_up(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        with World(environment=PARKED) as world:
            engines = make_set(world)
            engines.build()
            PackedProposalCoordinator()._release_single_user(engines.slots[1].device)
            engines.unpark(engines.slots[3], 'test')
            with patch.object(parked, 'build_single_capture', Mock(side_effect=RuntimeError('no room'))),                     patch.object(engines, 'build_slot', Mock(side_effect=RuntimeError('no room either'))):
                self.assertEqual(engines.idle(), dict(singles=[], reparked=[]))
            self.assertTrue(engines.single_released(engines.slots[1]), 'the single stays released')
            self.assertEqual(engines.slots[3].state, 'unparked', 'the slot stays on per-request builds')
            self.assertTrue(any(line.startswith('[PINDIAG] parked slot 1 idle single rebuild failed')
                                for line in world.lines))
            self.assertTrue(any(line.startswith('[PINDIAG] parked slot 3 idle re-park failed') for line in world.lines))
            self.assertEqual(engines.idle(), dict(singles=[1], reparked=[3]), 'tried again at the next idle moment')
            self.assertEqual(world.ops.violations, [])

    def test_a_failed_repark_still_ends_the_resident_engines_residency(self):
        import verifier_engine

        with World(environment=PARKED) as world:
            engines = make_set(world)
            engines.build()
            engines.unpark(engines.slots[3], 'test')
            sentinel = object()
            verifier_engine._resident = sentinel
            try:
                with patch.object(engines.components, 'device', Mock(side_effect=RuntimeError('no room'))):
                    self.assertEqual(engines.repark_idle(), [])
                self.assertIsNone(verifier_engine._resident, 'native slot 0 was overwritten before the build failed')
            finally:
                verifier_engine._resident = None
            self.assertEqual(engines.slots[3].state, 'unparked')
            self.assertEqual(world.ops.violations, [])

    def test_the_idle_moment_rebuilds_released_singles_and_reparks(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        with World(environment=PARKED) as world:
            engines = make_set(world)
            engines.build()
            PackedProposalCoordinator()._release_single_user(engines.slots[1].device)
            engines.unpark(engines.slots[3], 'test')
            self.assertEqual(engines.idle(), dict(singles=[1], reparked=[3]))
            self.assertEqual([entry.state for entry in engines.slots], ['parked'] * 4)
            self.assertFalse(any(engines.single_released(entry) for entry in engines.slots))
            self.assertEqual(engines.idle(), dict(singles=[], reparked=[]), 'nothing left to do')
            self.assertEqual(world.ops.violations, [])


# -----------------------------------------------------------------------------------------------------------------
# The regression: the single bucket across a rebind
# -----------------------------------------------------------------------------------------------------------------
class SingleBucketRegressionTests(unittest.TestCase):
    def test_a_device_released_at_2048_and_rebound_at_300_rebuilds_the_2048_bucket_and_grows_past_512(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        with World(environment=PARKED) as world:
            engines = make_set(world)
            engines.build()
            entry = engines.slots[0]
            device = entry.device
            self.assertEqual((device.position, device.history_rows), (2048, 2048), 'slot 0 was warmed at 2048')
            PackedProposalCoordinator()._release_single_user(device)
            self.assertIsNone(device.proposal_capture)
            request = admit(world, engines, 'short', 300, 256)
            self.assertIs(request.runtime.drafter, device)
            self.assertEqual(tuple(device.proposal_capture.buckets), (2048,))
            self.assertRegex([line for line in world.lines if line.startswith(parked.REBIND_MARKER)][-1],
                             'P=300 .*single_rebuilt=1')
            run_to_end(request)
            self.assertGreater(device.history_rows, 512)
            self.assertEqual(device.history_rows, device.position)
            request.close('short')
            self.assertEqual(world.ops.violations, [])

    def rebuilt_below_2048(self, on):
        """A device rebound at 300 whose single the coordinator released and rebuilt there: its buckets, and what
        growing its history past 512 did (None, or the error)."""
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        with World(environment=PARKED) as world:
            engines = make_set(world)
            engines.build()
            request = admit(world, engines, 'short', 300, 256)
            device = request.runtime.drafter
            coordinator = PackedProposalCoordinator()
            coordinator._release_single_user(device)
            with flag_environment(on):
                self.assertTrue(coordinator._ensure_single_user(device))
            buckets = tuple(device.proposal_capture.buckets)
            failure = None
            try:
                run_to_end(request)
            except ValueError as error:
                failure = str(error)
            rows = device.history_rows
            if failure is None:
                request.close('short')
            return buckets, failure, rows

    def test_the_coordinators_rebuild_below_2048_is_the_2048_bucket_under_the_flag(self):
        buckets, failure, rows = self.rebuilt_below_2048(True)
        self.assertEqual(buckets, (2048,))
        self.assertIsNone(failure)
        self.assertGreater(rows, 512)

    def test_unscoped_it_is_the_512_bucket_and_the_history_outgrows_it(self):
        """The negative control: the rebuild today's coordinator makes, on a device the flag rebinds below 2048."""
        buckets, failure, rows = self.rebuilt_below_2048(False)
        self.assertEqual(buckets, (512,))
        self.assertEqual(failure, 'Committed history exceeds prepared request contexts')
        self.assertGreater(rows, 512)


# -----------------------------------------------------------------------------------------------------------------
# The coordinator's generation (fakes)
# -----------------------------------------------------------------------------------------------------------------
class GenerationTests(unittest.TestCase):
    def setUp(self):
        FakeTrace.instances = []
        for target, value in (('dflash_proposal_trace.PreparedPackedDFlashProposal', FakeTrace),):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.operations = SimpleNamespace(synchronize_device=Mock())
        self.mesh = object()

    def bridges(self, slots):
        return [make_bridge('r%d' % slot, make_device(self.operations, self.mesh, slot=slot), seed=100 + slot)
                for slot in slots]

    def test_a_rebound_member_gets_a_new_pair_trace_only_under_the_flag(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        for on in (False, True):
            with self.subTest(flag=on), flag_environment(on):
                FakeTrace.instances = []
                coordinator = PackedProposalCoordinator()
                bridges = self.bridges((0, 1))
                coordinator.prepare(bridges)
                (first,) = FakeTrace.instances
                bridges[0].request.runtime.drafter.rebind_generation = 1
                coordinator.prepare(bridges)
                if on:
                    self.assertEqual(len(FakeTrace.instances), 2)
                    self.assertTrue(first.closed)
                    self.assertEqual(coordinator.generations[(0, 1)], (1, 0))
                    self.assertEqual(first.prepared, [(100, 101)], 'never prepared for the rebound member')
                else:
                    self.assertEqual(FakeTrace.instances, [first], 'identity and closed decide, as before')
                    self.assertEqual(coordinator.generations, {})
                    self.assertEqual(first.prepared, [(100, 101), (100, 101)])

    def test_release_closed_counts_a_rebound_member_as_closed_under_the_flag(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        for on in (False, True):
            with self.subTest(flag=on), flag_environment(on):
                FakeTrace.instances = []
                coordinator = PackedProposalCoordinator()
                bridges = self.bridges((0, 1, 2, 3))
                coordinator.prepare(bridges)
                bridges[2].request.runtime.drafter.rebind_generation = 3
                result = coordinator.release_closed()
                self.assertEqual(result, dict(quad=0, pairs=[[2, 3]] if on else []))
                self.assertEqual(sorted(coordinator.pairs), [(0, 1)] if on else [(0, 1), (2, 3)])

    def quad_of(self, bridges):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        coordinator = PackedProposalCoordinator()
        devices = [bridge.request.runtime.drafter for bridge in bridges]
        quad = SimpleNamespace(close=Mock(), buckets={1: 1})
        coordinator.quad = (tuple(devices), quad, True)
        coordinator._note_generations('quad', devices)
        return coordinator, devices, quad

    def test_a_rebound_member_retires_the_quad_only_under_the_flag(self):
        """E4's quad backstop, at both places the coordinator retires a quad (_retire_quad in the next draft round and
        release_closed at a detach): the generation is the only guard when release_parked fails, since a parked device
        is neither closed nor a different object."""
        for on in (False, True):
            for place in ('retire', 'release_closed'):
                with self.subTest(flag=on, place=place), flag_environment(on):
                    coordinator, devices, quad = self.quad_of(self.bridges((0, 1, 2, 3)))
                    self.assertEqual(coordinator.generations.get('quad'), (0, 0, 0, 0) if on else None)
                    devices[3].rebind_generation = 2
                    if place == 'retire':
                        coordinator._retire_quad(dict((slot, dict(device=device)) for slot, device in enumerate(devices)))
                    else:
                        self.assertEqual(coordinator.release_closed()['quad'], 1 if on else 0)
                    self.assertEqual(coordinator.quad is None, on)
                    self.assertEqual(quad.close.call_count, 1 if on else 0)
                    self.assertNotIn('quad', coordinator.generations)

    def test_a_built_quad_notes_the_generations_it_was_captured_at(self):
        import quad_draft
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        class Quad:
            def __init__(self, devices):
                self.buckets, self.last_built = {}, False

            def prepare_device(self, seeds):
                return False

            def discard_pending(self):
                pass

        for on in (False, True):
            with self.subTest(flag=on), flag_environment(on),                     patch.object(quad_draft, 'PreparedQuadDFlashProposal', Quad),                     patch.object(quad_draft, 'refusal', Mock(return_value=None)),                     patch('dflash_packed_proposal_coordinator.capture_headroom', Mock(return_value=([], {}))),                     patch('dflash_packed_proposal_coordinator.packable', Mock(return_value=True)):
                bridges = self.bridges((0, 1, 2, 3))
                devices = [bridge.request.runtime.drafter for bridge in bridges]
                for index, device in enumerate(devices):
                    device.rebind_generation = index + 1
                by_slot = dict((slot, dict(device=device, seed=slot, bridge=bridges[slot]))
                               for slot, device in zip(quad_draft.SLOTS, devices))
                coordinator = PackedProposalCoordinator()
                coordinator._prepare_quad([list(group) for group in quad_draft.PAIRS], by_slot, 1, True)
                self.assertIsNotNone(coordinator.quad)
                self.assertEqual(coordinator.generations.get('quad'), (1, 2, 3, 4) if on else None)

    def test_release_parked_retires_every_trace_of_the_device_and_nothing_else(self):
        from dflash_packed_proposal_coordinator import PARKED_RELEASED_LINE, PackedProposalCoordinator
        from test_dflash_proposal_trace import logged

        with flag_environment(True):
            coordinator = PackedProposalCoordinator()
            bridges = self.bridges((0, 1, 2, 3))
            coordinator.prepare(bridges)
            pair_01, pair_23 = FakeTrace.instances
            quad = SimpleNamespace(close=Mock())
            devices = [bridge.request.runtime.drafter for bridge in bridges]
            coordinator.quad = (tuple(devices), quad, True)
            lines, loguru = logged(None)
            with loguru:
                result = coordinator.release_parked(devices[1])
            self.assertEqual(result, dict(quad=1, pairs=[[0, 1]]))
            self.assertTrue(pair_01.closed)
            self.assertFalse(pair_23.closed)
            quad.close.assert_called_once_with()
            self.assertIsNone(coordinator.quad)
            self.assertEqual(sorted(coordinator.pairs), [(2, 3)])
            self.assertNotIn((0, 1), coordinator.generations)
            self.assertEqual(lines, [PARKED_RELEASED_LINE.format(slot=1, quad=1, pairs=[[0, 1]])])

    def test_the_headroom_asks_the_measured_capture_bytes_under_the_flag_only(self):
        import dflash_packed_proposal_coordinator as coordinator_module
        import quad_draft
        import serving_prefill_admission as admission

        for on in (False, True):
            with self.subTest(flag=on), flag_environment(on):
                asked = []
                with patch.object(coordinator_module, 'capture_headroom',
                                  side_effect=lambda device, estimate, reserve=True: asked.append(estimate) or ((), None)):
                    coordinator_module.PackedProposalCoordinator().prepare(self.bridges((0, 1)))
                self.assertEqual(asked, [admission.measured_pair_capture_bytes() if on
                                         else coordinator_module.estimated_pair_capture_bytes()])
                self.assertEqual(coordinator_module.quad_capture_bytes(),
                                 admission.measured_quad_capture_bytes() if on else quad_draft.QUAD_CAPTURE_BYTES_EST)
                with patch.dict(os.environ, {'QWEN_FAST_TP': '4'}):
                    self.assertEqual(coordinator_module.single_capture_bytes(),
                                     206_300_000 if on else coordinator_module.estimated_single_capture_bytes())

    def test_flag_off_every_round_is_the_base_commits_call_for_call(self):
        """The coordinator at the base commit and today's, flag off: pairs forming, a partner leaving (closed),
        release_closed, a single rebuilt, the pair re-forming - the same trace calls, fences and log lines."""
        from test_dflash_proposal_trace import logged
        from test_dflash_packed_proposal_coordinator import FakeSingleUserCapture
        import dflash_packed_proposal_coordinator as today

        base = base_module('dflash_packed_proposal_coordinator')

        def run(module):
            FakeTrace.instances = []
            log = []
            self.operations.synchronize_device = Mock(side_effect=lambda mesh: log.append('sync'))
            with flag_environment(False), patch.dict(os.environ, {'QWEN_FAST_EXTENT_REPLAY': '1',
                                                                  'QWEN_FAST_PACKED_AUDIT': '1'}), \
                    patch('dflash_proposal_trace.PreparedDFlashProposal', FakeSingleUserCapture):
                lines, loguru = logged(None)
                with loguru:
                    coordinator = module.PackedProposalCoordinator()
                    bridges = self.bridges((0, 1, 2, 3))
                    for bridge in bridges:
                        device = bridge.request.runtime.drafter
                        device.prepare_device = Mock(side_effect=lambda seed, slot=device.pool_slot.index:
                                                     log.append(('single', slot, seed)) or True)
                    coordinator.prepare(bridges)
                    bridges[1].request.runtime.drafter.closed = True
                    log.append(coordinator.release_closed())
                    coordinator.prepare([bridges[0], bridges[2], bridges[3]])
                    replacement = self.bridges((1,))[0]
                    coordinator.prepare([bridges[0], replacement, bridges[2], bridges[3]])
                    coordinator.close()
            timings = re.compile(r"propose_ms=[[][^]]*[]]")
            return log, [timings.sub('', line) for line in lines], [(trace.prepared, trace.closed)
                                                                     for trace in FakeTrace.instances]

        self.assertEqual(run(today), run(base))


# -----------------------------------------------------------------------------------------------------------------
# The worker hook
# -----------------------------------------------------------------------------------------------------------------
class HookReleaseTests(unittest.TestCase):
    def hook(self, events):
        coordinator = SimpleNamespace(release_closed=Mock(side_effect=lambda: events.append('release_closed')))
        hook = bare_hook(coordinator)
        request = SimpleNamespace(parked_slot='entry')
        hook.bridges = {'a': SimpleNamespace(request=request, close=Mock(side_effect=lambda: events.append('close'))),
                        'b': SimpleNamespace(request=request, close=Mock())}
        return hook, coordinator, request

    def test_under_the_flag_a_detach_releases_the_parked_device_after_the_dead_traces(self):
        events = []
        hook, coordinator, request = self.hook(events)
        with flag_environment(True), patch.dict(os.environ, {'QWEN_FAST_EXTENT_REPLAY': '1'}), \
                patch.object(parked, 'release_parked',
                             side_effect=lambda owner, departed: events.append(('parked', owner, departed))):
            self.assertEqual(sorted(hook.detach('a')), ['b'])
        self.assertEqual(events, ['close', 'release_closed', ('parked', coordinator, request)])

    def test_with_the_flag_off_a_detach_imports_and_calls_nothing_of_it(self):
        events = []
        hook, _, _ = self.hook(events)
        with flag_environment(False), patch.dict(os.environ, {'QWEN_FAST_EXTENT_REPLAY': '1'}), \
                patch.dict(sys.modules, {'serving_parked_engines': None}):
            self.assertEqual(sorted(hook.detach('a')), ['b'])
        self.assertEqual(events, ['close', 'release_closed'])

    def test_the_detach_is_the_base_commits_with_the_flag_off(self):
        import serving_worker_hook

        base = base_module('serving_worker_hook')

        def run(module):
            events = []
            for extent in ('0', '1'):
                hook = module.FastWorkerHook.__new__(module.FastWorkerHook)
                hook.bridges = {name: SimpleNamespace(request=SimpleNamespace(parked_slot='entry'),
                                                      close=Mock(side_effect=lambda name=name: events.append(name)))
                                for name in 'ab'}
                hook._packed_coordinator = SimpleNamespace(release_closed=Mock(side_effect=lambda: events.append('rc')))
                with flag_environment(False), patch.dict(os.environ, {'QWEN_FAST_EXTENT_REPLAY': extent}):
                    events.append(sorted(hook.detach('a')))
            return events

        self.assertEqual(run(serving_worker_hook), run(base))

    def test_the_flag_is_the_sets(self):
        import serving_worker_hook
        import dflash_packed_proposal_coordinator

        self.assertEqual(serving_worker_hook.PARKED_ENGINES_FLAG, parked.FLAG)
        self.assertEqual(dflash_packed_proposal_coordinator.PARKED_ENGINES_FLAG, parked.FLAG)


# -----------------------------------------------------------------------------------------------------------------
# End to end on the census world
# -----------------------------------------------------------------------------------------------------------------
class Serving:
    """E4's serving loop on the census world: arrivals through from_prefill with the parked set, rounds through the
    real coordinator (pair traces faked by CensusPair) and FastRequest.step, departures through the real
    FastWorkerHook.detach (release_dead_proposals, release_parked), and between waves the hook's close and the
    lifecycle's idle moment."""

    def __init__(self, world, engines):
        self.world, self.engines = world, engines
        self.serial = 0
        self.coordinators = []
        self.open_hook()

    def open_hook(self):
        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        self.coordinator = PackedProposalCoordinator()
        self.coordinators.append(self.coordinator)
        self.hook = bare_hook(self.coordinator)

    def live(self):
        return [bridge.request for bridge in self.hook.bridges.values()]

    def arrive(self, prompt, budget):
        self.serial += 1
        request_id = 'r%d' % self.serial
        request = admit(self.world, self.engines, request_id, prompt, budget)
        self.hook.bridges[request_id] = bridge_of(request)
        return request

    def round(self):
        self.world.replay_block()
        bridges = [bridge for bridge in self.hook.bridges.values() if not bridge.request.session.finished]
        if bridges:
            self.coordinator.prepare(bridges)
        for request_id, bridge in list(self.hook.bridges.items()):
            if not bridge.request.session.finished:
                bridge.request.step(request_id, cancelled=lambda: False)
        # vLLM names a finished request in the next step's output, and the lifecycle detaches it there, before
        # anything else in that step: after every live request consumed this round's drafts.
        for request_id, bridge in list(self.hook.bridges.items()):
            if bridge.request.session.finished:
                self.hook.detach(request_id)

    def leave(self, request_id):
        self.hook.detach(request_id)

    def drain(self):
        """Every user leaves at once: the hook closes (its coordinator first, then every bridge parks), then the idle
        moment the lifecycle finds."""
        self.hook.close()
        self.engines.idle()
        self.open_hook()


class EndToEndTests(unittest.TestCase):
    LENGTHS = (1, 17, 300, 2047, 2048, 2049, 3000, 4000)
    BUDGETS = (2, 3, 5, 16, 40)

    def churn(self, world, engines, rng):
        from dflash_packed_proposal_coordinator import PARKED_RELEASED_LINE

        serving = Serving(world, engines)
        attached = Counter((tensor.label, tuple(tensor.shape)) for tensor in world.ops.live.values())
        # every user leaves at once twice on the way (the hook closes, then the idle moment)
        drains = [7, 14]
        while min(entry.rebinds for entry in engines.slots) < 20:
            while len(serving.hook.bridges) < 4:
                prompt = rng.choice(self.LENGTHS)
                serving.arrive(prompt, min(rng.choice(self.BUDGETS), 68 * 64 - prompt - 1))
            serving.round()
            if serving.hook.bridges and rng.random() < 0.15:
                serving.leave(rng.choice(sorted(serving.hook.bridges)))
            if drains and min(entry.rebinds for entry in engines.slots) >= drains[0]:
                drains.pop(0)
                serving.drain()
            self.assertEqual(world.unowned([], [('parked', engines)] +
                                           [('coordinator', serving.coordinator)] +
                                           [('live%d' % index, request) for index, request in enumerate(serving.live())]),
                             [])
        for request_id in sorted(serving.hook.bridges):
            serving.leave(request_id)
        serving.drain()
        return serving, attached, [line for line in world.lines if line.startswith(PARKED_RELEASED_LINE.split('{')[0])]

    def assert_clean(self, world, engines, attached):
        self.assertEqual(world.ops.violations, [])
        self.assertEqual(engines.unparks, 0)
        self.assertGreaterEqual(min(entry.rebinds for entry in engines.slots), 20)
        self.assertEqual([entry.state for entry in engines.slots], ['parked'] * 4)
        self.assertFalse(any(engines.single_released(entry) for entry in engines.slots), 'every single held at idle')
        for entry in engines.slots:
            # through a closed pair's view the device may still wear: its next rebind unwraps it
            self.assertEqual(tuple(parked.single_capture(entry.device).buckets), (2048,))
        self.assertEqual(Counter((tensor.label, tuple(tensor.shape)) for tensor in world.ops.live.values()), attached,
                         'nothing a request did outlives it; the singles rebuilt are the ones the attach built')
        for trace in CensusPair.instances:
            self.assertTrue(all(served == trace.captured for served in trace.served),
                            'a pair trace was prepared for a rebound member: %r at %r' % (trace.served, trace.captured))
            self.assertTrue(trace.closed)

    def test_four_users_on_parked_engines_through_the_factory_hook_and_coordinator(self):
        import random

        CensusPair.instances = []
        with World(environment=PARKED) as world, \
                patch('dflash_proposal_trace.PreparedPackedDFlashProposal', CensusPair):
            engines = make_set(world)
            engines.build()
            serving, attached, released = self.churn(world, engines, random.Random(23))
            self.assert_clean(world, engines, attached)
            self.assertEqual(len(serving.coordinators), 4, 'two drains on the way and the last: a hook each')
            self.assertGreater(len(CensusPair.instances), 2, 'pairs formed')
            self.assertTrue(any('pairs=[[' in line for line in released), 'a departing member retired its pair')
            self.assertTrue(any(line.startswith('[PINDIAG] parked slot ') and ' single rebuilt at park ' in line
                                for line in world.lines))
            self.assertTrue(any('single_rebuilt=' in line for line in world.lines))

    def test_a_rebound_device_never_replays_an_old_pair_trace_even_when_release_parked_is_skipped(self):
        """The generation is the backstop: with the hook's release_parked skipped, no pair trace is retired when its
        member parks, and none is ever prepared again once a member is rebound (CensusPair records every prepare)."""
        import random
        import serving_worker_hook

        from dflash_packed_proposal_coordinator import PackedProposalCoordinator

        CensusPair.instances = []
        stale = []
        original = PackedProposalCoordinator._rebound

        def rebound(coordinator, key, devices):
            answer = original(coordinator, key, devices)
            if answer:
                stale.append(key)
            return answer
        with World(environment=PARKED) as world, \
                patch('dflash_proposal_trace.PreparedPackedDFlashProposal', CensusPair), \
                patch.object(PackedProposalCoordinator, '_rebound', rebound), \
                patch.object(serving_worker_hook, 'release_parked', Mock(return_value=None)) as skipped:
            engines = make_set(world)
            engines.build()
            serving, attached, released = self.churn(world, engines, random.Random(5))
            self.assertGreater(skipped.call_count, 20)
            self.assertEqual(released, [], 'release_parked never ran')
            self.assertGreater(len(CensusPair.instances), 2)
            self.assertTrue(any(len(trace.served) > 1 for trace in CensusPair.instances), 'a pair did replay')
            self.assertTrue(stale, 'the generation retired a pair trace captured for a rebound member')
            self.assert_clean(world, engines, attached)


# -----------------------------------------------------------------------------------------------------------------
# serving_runtime and the lifecycle
# -----------------------------------------------------------------------------------------------------------------
class AttachTests(unittest.TestCase):
    def test_the_set_is_built_after_the_block_and_closed_before_it(self):
        import test_serving_runtime

        case = test_serving_runtime.RuntimeAttachmentTests()
        for four_as_two in (False, True):
            with self.subTest(four_as_two=four_as_two):
                case.exercise(packed=True, users=4, four_as_two=four_as_two, parked={})
                (options,) = case.parked_calls
                self.assertEqual(options['capture_rows'], 4)
                self.assertEqual(len(options['blocks']), 2 if four_as_two else 1)
                self.assertEqual(sorted(options), ['blocks', 'capture_rows', 'collectives', 'fixtures', 'helpers',
                                                   'model', 'operations', 'pool', 'sampler', 'weights'])
        with self.assertRaisesRegex(RuntimeError, 'request failed'):
            case.exercise(packed=True, users=4, four_as_two=False, parked={}, fail=True)
        with self.assertRaisesRegex(RuntimeError, 'attach failed'):
            case.exercise(packed=True, users=4, four_as_two=False, parked={}, attach_fail=True)

    def test_off_nothing_is_imported_or_built_and_a_bad_value_is_refused(self):
        import test_serving_runtime

        case = test_serving_runtime.RuntimeAttachmentTests()
        for environ in ({}, {'QWEN_FAST_PARKED_ENGINES': '0'}):
            with self.subTest(environ=environ), \
                    patch('serving_parked_engines.ParkedEngineSet', side_effect=AssertionError('built')):
                case.exercise(packed=True, users=4, four_as_two=False, extra_env=environ)
        with self.assertRaisesRegex(ValueError, 'QWEN_FAST_PARKED_ENGINES must be 0 or 1'):
            case.exercise(packed=True, users=4, four_as_two=False, parked={},
                          extra_env={'QWEN_FAST_PARKED_ENGINES': 'yes'},
                          refused_after_blocks="QWEN_FAST_PARKED_ENGINES must be 0 or 1, got 'yes'")


class RuntimeWiringTests(unittest.TestCase):
    def run_attach(self, *, parked_on, request, extra_env=None, fail_binding=False):
        import memory_ledger
        import serving_runtime
        import test_serving_runtime

        seen = {}
        case = test_serving_runtime.RuntimeAttachmentTests()

        def probe(install, diagnostic):
            seen['lifecycle'] = dict(install.call_args.kwargs)
            bridge_factory = install.call_args.kwargs['bridge_factory']
            serving_runtime.ServingCacheOwner.return_value.physical_pages = 100
            state = SimpleNamespace(req_id='request-1', block_ids=([3, 4],), num_computed_tokens=0,
                                    prompt_token_ids=[1] * 300)
            diagnostic.reset_mock()
            binding = patch.object(serving_runtime, 'VerifierPageBinding',
                                   side_effect=ValueError('binding failed') if fail_binding else None)
            with patch.object(serving_runtime, 'from_prefill', return_value=request) as factory, binding, \
                    patch.object(serving_runtime, 'FastRunnerBridge', return_value='bridge'), \
                    patch.object(memory_ledger, 'engine_admitted') as admitted:
                try:
                    seen['bridge'] = bridge_factory(state, 'capture')
                except ValueError as failure:
                    seen['failure'] = str(failure)
            seen['factory'] = dict(factory.call_args.kwargs)
            seen['admitted'] = admitted.call_count
            seen['diag'] = [entry.args for entry in diagnostic.call_args_list]

        case.exercise(packed=True, users=4, four_as_two=False, probe=probe, extra_env=extra_env,
                      **(dict(parked={}) if parked_on else {}))
        seen['sets'] = case.parked_sets
        return seen

    def test_flag_off_the_factory_and_lifecycle_calls_are_todays(self):
        request = SimpleNamespace(engine=object(), close=Mock())
        seen = self.run_attach(parked_on=False, request=request)
        self.assertNotIn('parked', seen['factory'])
        self.assertNotIn('idle', seen['lifecycle'])
        self.assertNotIn('parked_poll', seen['lifecycle'])
        self.assertEqual(seen['admitted'], 1)

    def test_a_rebound_request_logs_no_build_time_walks_no_ledger_and_the_set_is_passed_on(self):
        from test_sticky_sessions import ON

        entry = SimpleNamespace(unfit=None)
        rebound = SimpleNamespace(engine=object(), close=Mock(), parked_slot=entry)
        built = SimpleNamespace(engine=object(), close=Mock())
        for request, expect_marker in ((rebound, False), (built, True)):
            with self.subTest(rebound=request is rebound):
                seen = self.run_attach(parked_on=True, request=request, extra_env=dict(ON))
                (engines,) = seen['sets']
                self.assertIs(seen['factory']['parked'], engines)
                self.assertIs(seen['lifecycle']['idle'], engines.idle)
                self.assertIs(seen['lifecycle']['parked_poll'], engines.poll_off)
                self.assertEqual(seen['bridge'], 'bridge')
                import serving_runtime

                marks = [args for args in seen['diag'] if args[0].startswith(serving_runtime.STICKY_ENGINE_MARKER)]
                # the sticky line stays for every request, with kind= saying which it was (the prefix gate's A8 reads it)
                self.assertEqual(len(marks), 1)
                self.assertTrue(marks[0][0].endswith(' kind={}'))
                self.assertEqual(marks[0][-1], 'build' if expect_marker else 'rebind')
                self.assertEqual(seen['admitted'], 1 if expect_marker else 0)
                self.assertTrue(any(args[0].startswith('[PINDIAG] dram after engine') for args in seen['diag']))

    def test_a_failed_binding_marks_the_parked_slot_unfit_before_the_close(self):
        entry = SimpleNamespace(unfit=None)
        order = []
        rebound = SimpleNamespace(engine=object(), parked_slot=entry,
                                  close=Mock(side_effect=lambda request_id: order.append(entry.unfit)))
        seen = self.run_attach(parked_on=True, request=rebound, fail_binding=True)
        self.assertEqual(seen['failure'], 'binding failed')
        self.assertEqual(order, ['page binding failed'])

    def test_the_dram_registration_gets_the_set_under_the_extent_flag(self):
        import serving_runtime

        calls = []

        def register(pool, **options):
            calls.append(options)
            return lambda: None
        with patch.object(serving_runtime, 'register_dram_admission', side_effect=register):
            for on in (False, True):
                calls.clear()
                with self.subTest(parked=on):
                    case = __import__('test_serving_runtime').RuntimeAttachmentTests()
                    case.exercise(packed=True, users=4, four_as_two=False, admission={},
                                  **(dict(parked={}) if on else {}))
                    self.assertEqual(calls, [dict(parked=case.parked_sets[0])] if on else [{}])


class LifecycleIdleTests(unittest.TestCase):
    def lifecycle(self, idle):
        from serving_lifecycle import FastServingLifecycle
        from test_serving_fast_policy import FastPolicyTests
        from test_serving_lifecycle import LifecycleTests

        fixture = LifecycleTests().fixture()
        lifecycle, worker = fixture[0], fixture[1]
        lifecycle.close()
        rebuilt = FastServingLifecycle(worker, config=FastPolicyTests().fixture(),
                                       capture_factory=lifecycle.capture_factory, bridge_factory=lifecycle.bridge_factory,
                                       eos_ids=(99,), cancelled=lambda: False, idle=idle)
        return (rebuilt, *fixture[1:])

    @staticmethod
    def empty(finished=()):
        return SimpleNamespace(finished_req_ids=set(finished), scheduled_new_reqs=[],
                               scheduled_cached_reqs=SimpleNamespace(req_ids=[]), scheduled_spec_decode_tokens={},
                               num_scheduled_tokens={}, total_num_scheduled_tokens=0)

    def test_idle_runs_only_with_nothing_decoding_nothing_in_prefill_and_nothing_scheduled(self):
        idle = Mock()
        lifecycle, worker, bridge, capture, build, scheduled, decode = self.lifecycle(idle)
        worker.execute_model(self.empty())               # nothing at all: the idle moment
        idle.assert_called_once_with()
        lifecycle.request_id = 'chunked'                  # a prefill in flight between two chunks
        worker.execute_model(self.empty())
        self.assertEqual(idle.call_count, 1)
        lifecycle.request_id = None
        worker.execute_model(scheduled)                  # a prefill step: tokens scheduled
        self.assertEqual(idle.call_count, 1)
        worker.sample_tokens(None)
        self.assertIsNotNone(lifecycle.hook)
        worker.execute_model(self.empty())               # a decoder is live
        self.assertEqual(idle.call_count, 1)

    def test_the_idle_moment_after_the_last_decoder_leaves(self):
        idle = Mock()
        lifecycle, worker, bridge, capture, build, scheduled, decode = self.lifecycle(idle)
        worker.execute_model(scheduled)
        worker.sample_tokens(None)
        self.assertIsNotNone(lifecycle.hook)
        idle.assert_not_called()
        worker.execute_model(self.empty(finished=('request',)))
        self.assertIsNone(lifecycle.hook)
        idle.assert_called_once_with()
        worker.execute_model(self.empty())
        self.assertEqual(idle.call_count, 2, 'every step that schedules nothing while idle')

    def test_a_non_callable_idle_is_refused_and_none_is_todays(self):
        from serving_lifecycle import FastServingLifecycle

        self.assertIsNone(FastServingLifecycle.idle)
        with self.assertRaisesRegex(ValueError, 'explicit serving factories'):
            self.lifecycle('idle')


class ShippingTests(unittest.TestCase):
    def test_every_edited_module_reaches_the_image(self):
        import c2_overlay

        sources = {entry.source for entry in c2_overlay.parse_manifest(
            (ROOT / 'docker' / 'qwen-c2-overlay.txt').read_text(encoding='utf-8'))}
        for name in parked.EDITED_MODULES:
            with self.subTest(module=name):
                self.assertTrue(('scripts/ci/' + name) in sources, '%s is edited by engine reuse and missing from the overlay' % name)

    def test_the_suites_run_in_the_cpu_workflow(self):
        text = CPU_WORKFLOW.read_text(encoding='utf-8')
        for name in ('test_parked_tp4_wiring', 'test_parked_tp4_set', 'test_parked_tp4_engine', 'test_parked_tp4_device',
                     'test_parked_tp4_census'):
            self.assertRegex(text, r'python -B -m unittest [^\n]*\b%s\b' % name)


if __name__ == '__main__':
    unittest.main()
