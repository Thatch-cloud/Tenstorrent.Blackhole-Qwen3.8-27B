"""QWEN_FAST_ADMISSION_LEAN (docs/tp4-sched-1.md section 4): the admission costs that are not the rebind.

A returning turn is a parked-engine rebind (188 ms) wrapped in about 350 ms of host-side accounting with every seat stopped: the post-prefill DRAM
backstop's allocator reading, the GDN slot adoption's conv-slice readback, the 'dram after engine' line and the ledger's rebind bracket. The flag takes the readback at each slot's first
adoption only (the trim's rule), drops the line for a rebound request, leaves the memory ledger's op=rebind bracket (a reading before and after, some 60 ms each) out
for every rebind of a slot after its first, and lets the backstop reuse the reading the same admission's DRAM hold took when it leaves room by a wide margin. Unset, every
call is what it always was.

    nice python3 -B -m unittest test_sched_lean      (from scripts/ci)
"""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / 'speculative-decoding/harness'))

import memory_ledger  # noqa: E402
import serving_prefill_admission as admission  # noqa: E402
import serving_request_factory  # noqa: E402
import test_serving_extent_memory  # noqa: E402  (module import, so that its TestCases are not collected here again)
from test_serving_extent_memory import MB, Pool, TracePool  # noqa: E402

LEAN = {'QWEN_FAST_ADMISSION_LEAN': '1'}
RESERVE = 256 * 2 ** 20


def environ(on, **extra):
    base = {name: value for name, value in os.environ.items() if name not in (admission.LEAN_FLAG, memory_ledger.TRIM_FLAG)}
    if on:
        base.update(LEAN)
    base.update(extra)
    return patch.dict(os.environ, base, clear=True)


class Clock(object):
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class Fresh(unittest.TestCase):
    def setUp(self):
        admission.forget_readings()
        serving_request_factory._ADOPT_VERIFIED.clear()
        serving_request_factory._REBIND_BRACKETED.clear()
        memory_ledger.reset_admission_diag()
        self.addCleanup(admission.forget_readings)
        self.addCleanup(serving_request_factory._ADOPT_VERIFIED.clear)
        self.addCleanup(serving_request_factory._REBIND_BRACKETED.clear)
        self.addCleanup(memory_ledger.reset_admission_diag)


class CountingPool(object):
    """A pool whose DRAM and trace statistics count their calls, the allocator reads the lever removes."""

    def __init__(self, free=4_800 * MB, largest=4_700 * MB, trace=148 * MB, chips=4):
        self.free, self.largest, self.trace, self.chips = free, largest, trace, chips
        self.dram_calls = self.trace_calls = 0

    def dram_statistics(self):
        self.dram_calls += 1
        return [dict(chip=index, free=self.free, largest_free=self.largest, allocated=0, total=self.free) for index in range(self.chips)]

    def trace_statistics(self):
        self.trace_calls += 1
        return [dict(chip=index, largest_free=self.trace, allocated=0) for index in range(self.chips)]


def hold_detail(free=4_800 * MB, largest=4_700 * MB, trace=148 * MB):
    return dict(free=free, largest_free=largest, trace_largest_free=trace, need=1_000 * MB, short=())


class FlagTests(Fresh):
    def test_the_flag_is_off_by_default_and_read_as_one_only(self):
        self.assertIs(admission.lean_enabled({}), False)
        for value in ('0', '', 'true', '2', ' 1', 'yes'):
            self.assertIs(admission.lean_enabled({admission.LEAN_FLAG: value}), False, value)
            self.assertIs(serving_request_factory.lean_enabled({serving_request_factory.LEAN_FLAG: value}), False, value)
        self.assertIs(admission.lean_enabled(LEAN), True)
        self.assertIs(serving_request_factory.lean_enabled(LEAN), True)

    def test_the_two_modules_name_the_same_flag_and_a_malformed_value_is_named(self):
        self.assertEqual(admission.LEAN_FLAG, serving_request_factory.LEAN_FLAG)
        self.assertEqual(admission.lean_problems({}), [])
        self.assertEqual(admission.lean_problems({admission.LEAN_FLAG: '0'}), [])
        self.assertEqual(admission.lean_problems(LEAN), [])
        for value in ('true', '2', '', ' 1'):
            problems = admission.lean_problems({admission.LEAN_FLAG: value})
            self.assertEqual(len(problems), 1, value)
            self.assertIn('must be 0 or 1', problems[0])


class ReusableReadingTests(Fresh):
    def setUp(self):
        super().setUp()
        self.clock = Clock()

    def need(self, parked_single=None):
        return admission.parked_backstop_need(RESERVE, rebind=admission.PARKED_REBIND_BYTES, single=parked_single or 0)

    def test_nothing_is_kept_and_nothing_reused_without_the_flag(self):
        with environ(False):
            admission.note_reading('r', hold_detail(), clock=self.clock)
        with environ(True):
            self.assertIsNone(admission.reusable_reading('r', self.need(), RESERVE, trace_need=0, clock=self.clock), 'never kept')
        with environ(True):
            admission.note_reading('r', hold_detail(), clock=self.clock)
        with environ(False):
            self.assertIsNone(admission.reusable_reading('r', self.need(), RESERVE, trace_need=0, clock=self.clock))

    def test_the_admissions_own_reading_with_room_to_spare_is_reused_once(self):
        with environ(True):
            admission.note_reading('r', hold_detail(), clock=self.clock)
            self.clock.now += 3.5
            kept = admission.reusable_reading('r', self.need(), RESERVE, trace_need=0, clock=self.clock)
            self.assertIsNotNone(kept)
            self.assertAlmostEqual(kept['age_ms'], 3500.0)
            self.assertEqual((kept['free'], kept['largest_free']), (4_800 * MB, 4_700 * MB))
            self.assertIsNone(admission.reusable_reading('r', self.need(), RESERVE, trace_need=0, clock=self.clock), 'single use')

    def test_another_requests_reading_is_not_this_ones(self):
        with environ(True):
            admission.note_reading('a', hold_detail(), clock=self.clock)
            self.assertIsNone(admission.reusable_reading('b', self.need(), RESERVE, trace_need=0, clock=self.clock))
            self.assertIsNotNone(admission.reusable_reading('a', self.need(), RESERVE, trace_need=0, clock=self.clock))

    def test_a_stale_reading_is_read_again(self):
        with environ(True):
            admission.note_reading('r', hold_detail(), clock=self.clock)
            self.clock.now += admission.LEAN_READING_MAX_AGE_S + 0.001
            self.assertIsNone(admission.reusable_reading('r', self.need(), RESERVE, trace_need=0, clock=self.clock))
            admission.note_reading('r', hold_detail(), clock=self.clock)
            self.clock.now += admission.LEAN_READING_MAX_AGE_S
            self.assertIsNotNone(admission.reusable_reading('r', self.need(), RESERVE, trace_need=0, clock=self.clock), 'exactly the bound is fresh')

    def test_the_reading_must_clear_the_need_after_the_allowance_halving_the_largest_block(self):
        need = self.need()
        floor = admission.LEAN_ALLOWANCE_BYTES
        # the free term: free - allowance - stranded >= need
        edge_free = need + admission.STRANDED_BYTES + floor
        # the contiguous term: largest // 2 - allowance >= contiguous_need
        edge_largest = 2 * (admission.contiguous_need(RESERVE) + floor)
        with environ(True):
            for free, largest, expected in ((edge_free, edge_largest, True), (edge_free - 1, edge_largest, False), (edge_free, edge_largest - 2, False),
                                            (edge_free, edge_largest + 2, True)):
                with self.subTest(free=free, largest=largest):
                    admission.note_reading('r', hold_detail(free=free, largest=largest), clock=self.clock)
                    kept = admission.reusable_reading('r', need, RESERVE, trace_need=0, clock=self.clock)
                    self.assertIs(kept is not None, expected)

    def test_a_trace_term_that_applies_needs_a_trace_reading(self):
        with environ(True):
            admission.note_reading('r', hold_detail(trace=None), clock=self.clock)
            self.assertIsNone(admission.reusable_reading('r', self.need(), RESERVE, trace_need=None, clock=self.clock),
                              'the default trace term (an engine build) is unread: read it')
            admission.note_reading('r', hold_detail(trace=None), clock=self.clock)
            self.assertIsNone(admission.reusable_reading('r', self.need(), RESERVE, trace_need=admission.single_trace_bytes(), clock=self.clock))
            admission.note_reading('r', hold_detail(trace=None), clock=self.clock)
            self.assertIsNotNone(admission.reusable_reading('r', self.need(), RESERVE, trace_need=0, clock=self.clock), 'no trace term: nothing to read')
            admission.note_reading('r', hold_detail(trace=10 * MB), clock=self.clock)
            self.assertIsNone(admission.reusable_reading('r', self.need(), RESERVE, trace_need=44 * MB, clock=self.clock), 'a short trace region is short')

    def test_an_incomplete_reading_is_not_kept_and_the_store_is_bounded(self):
        with environ(True):
            for detail in (None, {}, dict(free=None, largest_free=1), dict(free=1, largest_free='x'), 'reading'):
                admission.note_reading('r', detail, clock=self.clock)
            admission.note_reading(None, hold_detail(), clock=self.clock)
            self.assertEqual(admission._LEAN_READINGS, {})
            for index in range(admission.LEAN_READINGS_KEPT + 5):
                admission.note_reading('r%d' % index, hold_detail(), clock=self.clock)
            self.assertEqual(len(admission._LEAN_READINGS), admission.LEAN_READINGS_KEPT)
            self.assertNotIn('r0', admission._LEAN_READINGS)
            self.assertIn('r%d' % (admission.LEAN_READINGS_KEPT + 4), admission._LEAN_READINGS)


class HoldNotesTests(Fresh):
    """dram_hold keeps the reading that let a request through (and only that)."""

    def candidates(self, request_id='r'):
        return [SimpleNamespace(request_id=request_id, num_prompt_tokens=4088, status='WAITING')]

    def hold(self, admits, on, request_id='r'):
        scheduler = SimpleNamespace(waiting=self.candidates(request_id), skipped_waiting=[], _is_blocked_waiting_status=lambda status: False,
                                    finished_req_ids=())
        modules = {admission.DRAM_KEY: SimpleNamespace(admits=admits)}
        with environ(on):
            return admission.dram_hold(scheduler, 3, admission.new_state(), Mock(), modules=modules, candidates=self.candidates(request_id))

    def test_a_fit_is_kept_under_the_flag_only(self):
        for on in (False, True):
            admission.forget_readings()
            held = self.hold(lambda prompt: (True, hold_detail()), on)
            self.assertIs(held, False)
            self.assertEqual(sorted(admission._LEAN_READINGS), ['r'] if on else [], on)

    def test_a_hold_keeps_nothing(self):
        held = self.hold(lambda prompt: (False, dict(hold_detail(), short=('free',))), True)
        self.assertIs(held, True)
        self.assertEqual(admission._LEAN_READINGS, {})

    def test_an_unreadable_predicate_keeps_nothing(self):
        def broken(prompt):
            raise RuntimeError('no reading')
        self.assertIs(self.hold(broken, True), False)
        self.assertEqual(admission._LEAN_READINGS, {})


class BackstopTests(Fresh):
    def backstop(self, pool, on, *, parked=None, kept=None, request_id='r', log=None, clock=None, **options):
        log = Mock() if log is None else log
        with environ(on):
            if kept is not None:
                admission.note_reading(request_id, kept, **({} if clock is None else dict(clock=clock)))
            return serving_request_factory.dram_backstop(pool, request_id=request_id, reserve=RESERVE, log=log, parked=parked, **options), log

    PARKED_INTACT = dict(rebind=admission.PARKED_REBIND_BYTES, single=0)

    def test_flag_off_every_backstop_reads_the_whole_allocator_view(self):
        for parked in (None, self.PARKED_INTACT):
            pool = CountingPool()
            _result, log = self.backstop(pool, False, parked=parked, kept=hold_detail())
            self.assertEqual((pool.dram_calls, pool.trace_calls), (1, 1), parked)
            self.assertFalse([call for call in log.call_args_list if call.args[0] == admission.LEAN_REUSED_LINE])

    def test_lean_the_admissions_reading_with_room_replaces_the_read_and_is_logged(self):
        pool = CountingPool()
        result, log = self.backstop(pool, True, parked=self.PARKED_INTACT, kept=hold_detail())
        self.assertEqual((pool.dram_calls, pool.trace_calls), (0, 0))
        self.assertEqual(result, 4_700 * MB)
        (line,) = [call.args for call in log.call_args_list]
        self.assertEqual(line[0], admission.LEAN_REUSED_LINE)
        self.assertEqual(line[1], 'r')
        self.assertEqual(line[-1], admission.LEAN_ALLOWANCE_BYTES)

    def test_lean_without_a_kept_reading_it_reads_the_whole_allocator_view_as_before(self):
        for parked in (self.PARKED_INTACT, None, dict(rebind=admission.PARKED_REBIND_BYTES, single=admission.measured_single_capture_bytes())):
            pool = CountingPool()
            self.backstop(pool, True, parked=parked)
            self.assertEqual((pool.dram_calls, pool.trace_calls), (1, 1), parked)

    def test_lean_a_tight_reading_is_read_again_and_judged_on_the_real_one(self):
        need = admission.parked_backstop_need(RESERVE, rebind=admission.PARKED_REBIND_BYTES, single=0)
        tight = hold_detail(free=need + admission.STRANDED_BYTES + 10 * MB, largest=admission.contiguous_need(RESERVE) + 10 * MB)
        pool = CountingPool(free=tight['free'], largest=tight['largest_free'])
        _result, _log = self.backstop(pool, True, parked=self.PARKED_INTACT, kept=tight)
        self.assertEqual(pool.dram_calls, 1, 'the kept reading was not good enough: the allocator is read')
        short = CountingPool(free=need, largest=tight['largest_free'])
        with self.assertRaisesRegex(serving_request_factory.RequestRefused, 'DRAM backstop: short of free'):
            self.backstop(short, True, parked=self.PARKED_INTACT, kept=tight)

    def test_lean_a_stale_reading_is_not_reused(self):
        import time

        pool = CountingPool()
        self.backstop(pool, True, parked=self.PARKED_INTACT, kept=hold_detail())
        self.assertEqual(pool.dram_calls, 0, 'a fresh reading')
        aged = lambda: time.monotonic() - admission.LEAN_READING_MAX_AGE_S - 5.0           # noqa: E731
        self.backstop(pool, True, parked=self.PARKED_INTACT, kept=hold_detail(), clock=aged)
        self.assertEqual(pool.dram_calls, 1, 'a reading older than the bound is read again')

    def test_lean_the_make_room_ladder_still_runs_for_a_short_reading(self):
        need = admission.parked_backstop_need(RESERVE, rebind=admission.PARKED_REBIND_BYTES, single=0)
        pool = Pool((need, 9_000 * MB), (9_000 * MB, 9_000 * MB))
        rooms = []
        with self.assertRaises(serving_request_factory.RequestRefused):
            self.backstop(pool, True, parked=self.PARKED_INTACT, make_room=lambda wanted: rooms.append(wanted))
        self.assertEqual(len(rooms), 1)


class AdoptionTests(Fresh):
    def adopt(self, on, slots, **extra):
        helpers = [Mock(spec=['adopt_slot'], **{'adopt_slot.return_value': 2}) for _ in range(3)]
        with environ(on, **extra), patch.object(serving_request_factory, '_log') as log:
            for slot in slots:
                serving_request_factory.adopt_prefill_slot(helpers, SimpleNamespace(prefill_slot=slot), 'request')
        return helpers, log

    def test_flag_off_every_adoption_is_called_as_it_always_was_and_logs_no_lean_line(self):
        helpers, log = self.adopt(False, [1, 1, 2, 1])
        for layer, helper in enumerate(helpers):
            self.assertEqual([tuple(item) for item in helper.adopt_slot.call_args_list], [((slot,), dict(layer=layer)) for slot in (1, 1, 2, 1)], layer)
        self.assertFalse([call for call in log.call_args_list if call.args[0].startswith(serving_request_factory.LEAN_ADOPT_MARKER)])
        self.assertEqual(serving_request_factory._ADOPT_VERIFIED, set())

    def test_lean_the_readback_is_taken_at_each_slots_first_adoption_only(self):
        helpers, log = self.adopt(True, [1, 1, 2, 1, 2, 3])
        for layer, helper in enumerate(helpers):
            first, again = dict(layer=layer), dict(layer=layer, readback=False)
            self.assertEqual([tuple(item) for item in helper.adopt_slot.call_args_list],
                             [((1,), first), ((1,), again), ((2,), first), ((1,), again), ((2,), again), ((3,), first)], layer)
        lines = [call.args for call in log.call_args_list if call.args[0].startswith(serving_request_factory.LEAN_ADOPT_MARKER)]
        self.assertEqual([(line[2], line[3]) for line in lines], [(1, 1), (1, 0), (2, 1), (1, 0), (2, 0), (3, 1)], 'slot, readback taken')

    def test_slot_zero_adopts_nothing_and_marks_nothing(self):
        helpers, _log = self.adopt(True, [0, None])
        for helper in helpers:
            helper.adopt_slot.assert_not_called()
        self.assertEqual(serving_request_factory._ADOPT_VERIFIED, set())

    def test_a_refused_first_adoption_does_not_count_as_verified(self):
        helpers = [Mock(spec=['adopt_slot']) for _ in range(2)]
        helpers[0].adopt_slot.side_effect = ValueError('differs from the row')
        helpers[1].adopt_slot.return_value = 2
        with environ(True), patch.object(serving_request_factory, '_log'):
            with self.assertRaises(ValueError):
                serving_request_factory.adopt_prefill_slot(helpers, SimpleNamespace(prefill_slot=1), 'request')
            self.assertEqual(serving_request_factory._ADOPT_VERIFIED, set())

    def test_the_trim_and_the_lean_flag_share_the_verified_slots(self):
        helpers, _log = self.adopt(False, [1], QWEN_FAST_ADMISSION_DIAG_TRIM='1')
        self.assertEqual(serving_request_factory._ADOPT_VERIFIED, {1})
        with environ(True), patch.object(serving_request_factory, '_log'):
            serving_request_factory.adopt_prefill_slot(helpers, SimpleNamespace(prefill_slot=1), 'request')
        self.assertEqual(helpers[0].adopt_slot.call_args_list[-1], ((1,), dict(layer=0, readback=False)))


class RuntimeLineTests(Fresh):
    """serving_runtime's bridge factory: the 'dram after engine' line, per request kind."""

    def lines(self, on, request, extra=None):
        import test_parked_tp4_wiring
        from test_sticky_sessions import ON

        env = dict(ON, **(extra or {}))
        with environ(on):
            seen = test_parked_tp4_wiring.RuntimeWiringTests().run_attach(parked_on=True, request=request, extra_env=env)
        return [args for args in seen['diag'] if args[0].startswith('[PINDIAG] dram after engine')]

    def test_flag_off_the_line_is_logged_for_a_rebind_and_for_a_build(self):
        rebound = SimpleNamespace(engine=object(), close=Mock(), parked_slot=SimpleNamespace(unfit=None))
        built = SimpleNamespace(engine=object(), close=Mock())
        for request in (rebound, built):
            self.assertEqual(len(self.lines(False, request)), 1)

    def test_lean_a_rebound_request_logs_no_line_and_a_cold_build_still_does(self):
        rebound = SimpleNamespace(engine=object(), close=Mock(), parked_slot=SimpleNamespace(unfit=None))
        built = SimpleNamespace(engine=object(), close=Mock())
        self.assertEqual(self.lines(True, rebound), [])
        self.assertEqual(len(self.lines(True, built)), 1)


class BracketTests(Fresh):
    """The memory ledger's op=rebind bracket (rebind_parked): every rebind brackets it as it always did; under the flag a slot's first rebind does and the later ones do not."""

    def calls(self, on, names):
        import test_parked_tp4_wiring as wiring

        calls = []
        with environ(on), wiring.World(environment=dict(wiring.PARKED, **(LEAN if on else {}))) as world:
            engines = wiring.make_set(world)
            engines.build()
            with patch.object(memory_ledger, 'before', side_effect=lambda op, **kw: calls.append((op, kw['point'])) or 'token'), \
                    patch.object(memory_ledger, 'after', side_effect=lambda token: calls.append(('after', token))):
                for name in names:
                    request = wiring.admit(world, engines, name, 300, 4)
                    wiring.run_to_end(request)
                    request.close(name)
        return calls

    def test_flag_off_every_rebind_is_bracketed(self):
        calls = self.calls(False, ('first', 'second', 'third'))
        self.assertEqual([call[0] for call in calls], ['rebind', 'after'] * 3)
        self.assertEqual(serving_request_factory._REBIND_BRACKETED, set(), 'nothing is recorded without the flag')

    def test_lean_the_first_rebind_of_a_slot_is_bracketed_and_the_later_ones_are_not(self):
        calls = self.calls(True, ('first', 'second', 'third'))
        self.assertEqual([call[0] for call in calls], ['rebind', 'after'])
        self.assertEqual(calls[0][1], 'req=first slot=0')
        self.assertEqual(serving_request_factory._REBIND_BRACKETED, {0})

    def test_the_skip_is_per_slot_and_off_without_the_flag(self):
        skipped = serving_request_factory.lean_bracket_skipped
        with environ(True):
            self.assertEqual([skipped(0), skipped(1), skipped(0), skipped(1), skipped(2)], [False, False, True, True, False])
        serving_request_factory._REBIND_BRACKETED.clear()
        with environ(False):
            self.assertEqual([skipped(0), skipped(0)], [False, False])
        self.assertEqual(serving_request_factory._REBIND_BRACKETED, set())
        with environ(True, QWEN_FAST_ADMISSION_LEAN='0'):
            self.assertFalse(skipped(0))
            self.assertFalse(skipped(0))


class SplitLineTests(Fresh):
    """from_prefill on a parked slot logs the A / B / C split under the flag (the ledger-free rebind_split), and nothing without it."""

    def admitted(self, on):
        import test_parked_tp4_wiring as wiring

        with wiring.World(environment=dict(wiring.PARKED, **(LEAN if on else {}))) as world:
            engines = wiring.make_set(world)
            engines.build()
            request = wiring.admit(world, engines, 'request', 300, 24)
            lines = list(world.lines)
            wiring.run_to_end(request)
            request.close('request')
        return lines

    def test_flag_off_no_lean_line(self):
        with environ(False):
            lines = self.admitted(False)
        self.assertFalse([line for line in lines if line.startswith('[PINDIAG] admission lean')])

    def test_flag_off_the_rebind_logs_no_phases_and_returns_what_it_returned(self):
        with environ(False):
            lines = self.admitted(False)
        self.assertFalse([line for line in lines if 'rebind phases' in line])

    def test_lean_one_phases_line_per_rebind_with_six_phases_that_add_up_to_the_rebind(self):
        import re

        with environ(True):
            lines = self.admitted(True)
        phases = [line for line in lines if line.startswith('[PINDIAG] admission lean rebind phases ')]
        self.assertEqual(len(phases), 1, lines)
        self.assertRegex(phases[0], r'request=request slot=\d+ fence_ms=[0-9.]+ rezero_ms=[0-9.]+ project_ms=[0-9.]+ copy_ms=[0-9.]+ history_ms=[0-9.]+ engine_ms=[0-9.]+$')
        rebinds = [line for line in lines if line.startswith('[PINDIAG] parked rebind req=')]
        total = float(re.search(r' ms=([0-9.]+) ', rebinds[0]).group(1))
        parts = [float(value) for value in re.findall(r'_ms=([0-9.]+)', phases[0])]
        self.assertLessEqual(sum(parts[:5]), total + 0.5, 'the five device phases are inside rebind_device\'s own clock')
        self.assertLessEqual(parts[5], total + 50.0)

    def test_lean_one_split_line_per_rebound_admission(self):
        with environ(True):
            lines = self.admitted(True)
        splits = [line for line in lines if line.startswith(serving_request_factory.LEAN_SPLIT_MARKER)]
        self.assertEqual(len(splits), 1, lines)
        self.assertRegex(splits[0], r'request=request a_ms=[0-9.]+ b_ms=[0-9.]+ c_ms=[0-9.]+ total_ms=[0-9.]+$')
        import re

        a, b, c, total = [float(value) for value in re.findall(r'_ms=([0-9.]+)', splits[0])]
        self.assertAlmostEqual(a + b + c, total, delta=0.2)


class ContractTests(unittest.TestCase):
    """The profile contract: strict 0 or 1, a gate-only profile's, and an inherited value never leaks into a profile that does not name it."""

    def test_the_names_agree(self):
        import serving_c2_contract as contract

        self.assertEqual(contract.LEAN_SWITCH, admission.LEAN_FLAG)
        self.assertEqual(contract.SCHED_NAMES, (admission.LEAN_FLAG,))

    def test_a_gate_only_profile_may_name_it_a_traffic_profile_may_not(self):
        import serving_c2_contract as contract

        self.assertEqual(contract.sched_problems({'name': 'x', 'env': {}}), [])
        self.assertEqual(contract.sched_problems({'name': 'x', 'env': {contract.LEAN_SWITCH: '0'}}), [])
        self.assertEqual(contract.sched_problems({'name': 'x-gate', 'env': {contract.LEAN_SWITCH: '1'}}), [])
        self.assertEqual(contract.sched_problems({'name': 'x', 'gate_only': True, 'env': {contract.LEAN_SWITCH: '1'}}), [])
        (problem,) = contract.sched_problems({'name': 'x', 'env': {contract.LEAN_SWITCH: '1'}})
        self.assertIn('needs a gate-only profile', problem)
        for value in ('true', '2', '', ' 1'):
            (problem,) = contract.sched_problems({'name': 'x-gate', 'env': {contract.LEAN_SWITCH: value}})
            self.assertIn('must be 0 or 1', problem)

    def test_an_inherited_value_is_dropped_under_a_profile_that_does_not_name_it(self):
        import copy
        import json

        import serving_c2_contract as contract

        profiles = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']
        base = profiles['c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-traffic-s4']
        self.assertNotIn(contract.LEAN_SWITCH, base['env'], 'the approved profile does not carry it')
        environ = contract.apply_environment(dict(copy.deepcopy(base), name='x'), {contract.LEAN_SWITCH: '1', 'KEEP': 'me'})
        self.assertNotIn(contract.LEAN_SWITCH, environ)
        self.assertEqual(environ['KEEP'], 'me')
        named = copy.deepcopy(base)
        named['env'][contract.LEAN_SWITCH] = '0'
        environ = contract.apply_environment(dict(named, name='x'), {contract.LEAN_SWITCH: '1'})
        self.assertEqual(environ[contract.LEAN_SWITCH], '0', 'the profile wins over the process')


if __name__ == '__main__':
    unittest.main()
