"""The fast lane's host logic (serving_fast_lane): the request's mark, the config, the admission, the book, the controller, the plan.

No device and no vLLM: fake requests (a session with a request id and a finished flag), the controller driven on a virtual clock. The
whole loop - the hook, the scheduler gate, the runner's persistent batch - is test_fast_lane_hook's."""

import collections
from types import SimpleNamespace
import unittest

import serving_fast_lane as lanes
from serving_fast_lane import (FAST, STANDARD, LaneBook, LaneConfig, LaneController, LaneRefused, LaneRuntime, RateWindow,
                               closed_form_k, lane_admission, lane_of, request_lane)


def params(lane=None, extra=True):
    return SimpleNamespace(extra_args=None if not extra else ({} if lane is None else {'qwen_lane': lane}))


def config(**changes):
    values = dict(LaneConfig.from_environment({})._asdict())
    values.update(changes)
    return LaneConfig(**values)


class Lines:
    def __init__(self):
        self.lines = []

    def __call__(self, template, *values):
        self.lines.append(template.format(*values))


class MarkTests(unittest.TestCase):
    def test_absent_means_standard_and_the_two_names_are_the_only_marks(self):
        self.assertEqual(request_lane(SimpleNamespace()), STANDARD)
        self.assertEqual(request_lane(SimpleNamespace(extra_args=None)), STANDARD)
        self.assertEqual(request_lane(params()), STANDARD)
        self.assertEqual(request_lane(params('fast')), FAST)
        self.assertEqual(request_lane(params('standard')), STANDARD)

    def test_anything_else_is_a_refusal_of_that_request_alone(self):
        for value in ('Fast', 'FAST', 'turbo', '', ' fast', None, 1, True, ['fast'], b'fast'):
            with self.subTest(value=value):
                with self.assertRaisesRegex(LaneRefused, 'qwen_lane must be one of fast/standard'):
                    request_lane(SimpleNamespace(extra_args={'qwen_lane': value}))
        with self.assertRaisesRegex(LaneRefused, 'must be a mapping'):
            request_lane(SimpleNamespace(extra_args=['qwen_lane']))
        self.assertTrue(issubclass(LaneRefused, ValueError), 'the lifecycle quarantines a ValueError of the request\'s own')

    def test_other_extra_args_are_no_business_of_the_lane(self):
        self.assertEqual(request_lane(SimpleNamespace(extra_args={'other': 'x'})), STANDARD)
        self.assertEqual(request_lane(SimpleNamespace(extra_args={'other': 'x', 'qwen_lane': 'fast'})), FAST)

    def test_the_schedulers_reading_is_tolerant(self):
        self.assertEqual(lane_of(SimpleNamespace(sampling_params=params('fast'))), FAST)
        self.assertEqual(lane_of(SimpleNamespace(sampling_params=params('turbo'))), STANDARD)
        self.assertEqual(lane_of(SimpleNamespace()), STANDARD)


class ConfigTests(unittest.TestCase):
    def test_the_defaults(self):
        self.assertEqual(LaneConfig.from_environment({}), LaneConfig(ratio=None, k_max=3, gap_ms=250.0, floor=75.0, margin=0.05,
                                                                     reserve=True, seats=4))

    def test_a_fixed_ratio_is_a_decimal_within_zero_and_k_max(self):
        self.assertEqual(LaneConfig.from_environment({'QWEN_FAST_LANE_RATIO': '1.5'}).ratio, 1.5)
        self.assertEqual(LaneConfig.from_environment({'QWEN_FAST_LANE_RATIO': '0'}).ratio, 0.0)
        self.assertIsNone(LaneConfig.from_environment({'QWEN_FAST_LANE_RATIO': 'auto'}).ratio)
        for bad in ('-0.1', '3.5', 'nan', 'inf', 'x', ''):
            with self.subTest(ratio=bad):
                with self.assertRaises(ValueError):
                    LaneConfig.from_environment({'QWEN_FAST_LANE_RATIO': bad})
        self.assertEqual(LaneConfig.from_environment({'QWEN_FAST_LANE_RATIO': '5', 'QWEN_FAST_LANE_KMAX': '6'}).ratio, 5.0)

    def test_the_other_knobs_are_bounded_and_strict(self):
        cases = {'QWEN_FAST_LANE_KMAX': ('0', '9', '1.5', 'x'), 'QWEN_FAST_LANE_GAP_MS': ('1', '9999', 'x'),
                 'QWEN_FAST_LANE_FLOOR': ('0', '5000', 'x'), 'QWEN_FAST_LANE_MARGIN': ('-1', '0.9', 'x'),
                 'QWEN_FAST_LANE_RESERVE': ('2', '', 'yes')}
        for name, values in cases.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        LaneConfig.from_environment({name: value})
        self.assertFalse(LaneConfig.from_environment({'QWEN_FAST_LANE_RESERVE': '0'}).reserve)

    def test_the_flag_is_strictly_zero_or_one(self):
        self.assertFalse(lanes.lane_requested({}))
        self.assertFalse(lanes.lane_requested({'QWEN_FAST_LANE': '0'}))
        self.assertTrue(lanes.lane_requested({'QWEN_FAST_LANE': '1'}))
        for value in ('2', '', 'true'):
            with self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
                lanes.lane_requested({'QWEN_FAST_LANE': value})


class AdmissionTests(unittest.TestCase):
    ENV = {'QWEN_FAST_LANE': '1', 'QWEN_FAST_ANY_REQUEST': '1'}

    def test_off_reads_nothing(self):
        self.assertIsNone(lane_admission(None, {}, seats=2))
        self.assertIsNone(lane_admission(None, {'QWEN_FAST_LANE': '0', 'QWEN_FAST_LANE_RATIO': 'garbage'}, seats=2))

    def test_the_lane_rides_on_the_solo_lane_c2_any_and_four_seats(self):
        self.assertEqual(lane_admission({'slot': 0}, self.ENV, seats=4).seats, 4)
        log = Lines()
        with self.assertRaises(ValueError) as caught:
            lane_admission(None, {'QWEN_FAST_LANE': '1', 'QWEN_FAST_LANE_RATIO': '9'}, seats=2, log=log)
        text = str(caught.exception)
        for part in ('SOLO_LANE is not 1', 'ANY_REQUEST is not 1', 'not 2', 'QWEN_FAST_LANE_RATIO'):
            self.assertIn(part, text)
        self.assertEqual(len(log.lines), 4)
        self.assertTrue(all(line.startswith('[LANE] refused: ') for line in log.lines))


class BookTests(unittest.TestCase):
    def book(self, **changes):
        self.log = Lines()
        return LaneBook(config(**changes), log=self.log)

    def test_the_first_fast_request_holds_slot_zero_and_standard_requests_take_the_others(self):
        book = self.book()
        fast = book.admit('a', FAST)
        self.assertEqual((fast.granted, fast.slot_order, fast.reason), (FAST, (0,), 'ok'))
        standard = book.admit('b', STANDARD)
        self.assertEqual((standard.granted, standard.slot_order), (STANDARD, (1, 2, 3)))
        self.assertEqual(book.fast_id, 'a')
        self.assertEqual((book.lane('a'), book.lane('b'), book.lane('zz')), (FAST, STANDARD, None))

    def test_a_second_fast_request_is_downgraded_never_queued_or_refused(self):
        book = self.book()
        book.admit('a', FAST)
        second = book.admit('b', FAST)
        self.assertEqual((second.asked, second.granted, second.reason, second.slot_order),
                         (FAST, STANDARD, 'fast-lane-busy', (1, 2, 3)))
        self.assertEqual(book.fast_id, 'a')
        self.assertIn('[LANE-ADMIT] request=b alias=u2 asked=fast granted=standard slots=1,2,3 reason=fast-lane-busy', self.log.lines)

    def test_the_fast_lane_is_free_again_when_its_holder_leaves_and_release_is_idempotent(self):
        book = self.book()
        book.admit('a', FAST)
        book.admit('b', FAST)
        self.assertEqual(book.release('a'), FAST)
        self.assertIsNone(book.fast_id)
        self.assertIsNone(book.release('a'))
        self.assertEqual(book.admit('c', FAST).granted, FAST)
        self.assertEqual(book.release('b'), STANDARD)
        self.assertEqual(book.fast_id, 'c')

    def test_the_reserve_keeps_slot_zero_from_standard_requests_and_lend_gives_it_last(self):
        self.assertEqual(self.book(reserve=True).slot_order(STANDARD), (1, 2, 3))
        self.assertEqual(self.book(reserve=False).slot_order(STANDARD), (1, 2, 3, 0))
        self.assertEqual(self.book(reserve=False).slot_order(STANDARD, alone=True), (0, 1, 2, 3))
        self.assertEqual(self.book(reserve=True).slot_order(STANDARD, alone=True), (1, 2, 3), 'the reserve never lends')
        lend = self.book(reserve=False)
        held = lend.admit('a', FAST, slot0_free=False)
        self.assertEqual((held.granted, held.reason), (STANDARD, 'slot-busy'))
        self.assertIsNone(lend.fast_id)

    def test_under_lend_a_lone_standard_user_takes_slot_zero_and_decodes_on_d0_later_ones_take_the_others(self):
        book = self.book(reserve=False)
        first = book.admit('a', STANDARD)
        self.assertEqual(first.slot_order, (0, 1, 2, 3))
        second = book.admit('b', STANDARD)
        self.assertEqual(second.slot_order, (1, 2, 3, 0), 'not alone any more')
        self.assertIn('[LANE-ADMIT] request=a alias=u1 asked=standard granted=standard slots=0,1,2,3 reason=ok', self.log.lines)
        book.release('a')
        book.release('b')
        self.assertEqual(book.admit('c', STANDARD).slot_order, (0, 1, 2, 3), 'alone again')
        under_reserve = self.book(reserve=True)
        self.assertEqual(under_reserve.admit('a', STANDARD).slot_order, (1, 2, 3), 'the reserve keeps slot 0 for the fast request')

    def test_a_fast_arrival_while_a_lone_standard_user_holds_slot_zero_is_downgraded_under_lend(self):
        book = self.book(reserve=False)
        book.admit('a', STANDARD)
        late = book.admit('b', FAST, slot0_free=False)
        self.assertEqual((late.granted, late.reason), (STANDARD, 'slot-busy'))

    def test_a_request_twice_or_an_unknown_lane_is_refused(self):
        book = self.book()
        book.admit('a', STANDARD)
        with self.assertRaisesRegex(ValueError, 'already in the lane book'):
            book.admit('a', STANDARD)
        with self.assertRaises(LaneRefused):
            book.admit('b', 'turbo')

    def test_every_request_gets_a_stable_short_alias_the_round_lines_use(self):
        book = self.book()
        first, second = book.admit('a', FAST), book.admit('b', STANDARD)
        self.assertEqual((first.alias, second.alias), ('u1', 'u2'))
        self.assertEqual((book.alias('a'), book.alias('zz')), ('u1', 'zz'))
        book.release('a')
        third = book.admit('c', STANDARD)
        self.assertEqual(third.alias, 'u3', 'aliases are never reused within a process')
        self.assertLess(max(len(line) for line in self.log.lines), 130)


class ClosedFormTests(unittest.TestCase):
    """The design's arithmetic (the frame P + kF must fit 1000 tau / floor ms), on its own numbers."""

    def test_the_phase_one_thirty_two_k_cell_at_the_fixture_tau(self):
        # P(4) 88.6 ms, F 55.3 ms, sigma 1 ms, tau 12.1, floor 75: T = 161.3 ms; k = (161.3 - 88.6 - 2) / 55.3 = 1.28
        self.assertAlmostEqual(closed_form_k(88.6, 55.3, 1.0, 12.1, 75.0), 1.278, places=2)

    def test_below_one_the_switch_is_paid_once_per_frame_not_twice_per_solo_round(self):
        # T = 100: T - P = 20 < F + 2 sigma = 52: k = 20 / 52
        self.assertAlmostEqual(closed_form_k(80.0, 50.0, 1.0, 7.5, 75.0), 20.0 / 52.0, places=6)

    def test_zero_when_the_packed_round_alone_misses_the_floor(self):
        self.assertEqual(closed_form_k(170.0, 55.0, 1.0, 12.1, 75.0), 0.0)
        self.assertEqual(closed_form_k(161.4, 55.0, 1.0, 12.1, 75.0), 0.0)
        self.assertGreater(closed_form_k(160.0, 55.0, 1.0, 12.1, 75.0), 0.0)

    def test_unknown_inputs_answer_none(self):
        for args in ((None, 55.0, 1.0, 12.1, 75.0), (88.0, None, 1.0, 12.1, 75.0), (88.0, 55.0, 1.0, None, 75.0),
                     (88.0, 55.0, 1.0, 0.0, 75.0), (88.0, 55.0, 1.0, 12.1, 0.0)):
            self.assertIsNone(closed_form_k(*args))

    def test_the_fast_lane_alone_clears_the_bar_when_f_is_under_six_point_seven_tau(self):
        # F <= 1000 tau / 150: at tau 12.1 that is 80.7 ms
        for f, clears in ((55.3, True), (80.6, True), (81.0, False)):
            self.assertEqual(1000.0 * 12.1 / f >= 150.0, clears, f)


class Feed:
    """Drive a controller on a virtual clock: rounds of a given kind and duration, tokens per member."""

    def __init__(self, controller, *, fast='f', standard=('s1', 's2'), tau_fast=12.0, tau_standard=12.0):
        self.controller, self.fast, self.standard = controller, fast, list(standard)
        self.tau_fast, self.tau_standard = tau_fast, tau_standard
        self.now, self.last_kind = 0.0, None
        self.kinds = []

    def round(self, kind, ms):
        switch = self.last_kind is not None and self.last_kind != kind
        self.now += ms / 1000.0
        members = [self.fast] if kind == 'solo' else [self.fast] + self.standard
        committed = {self.fast: self.tau_fast, **{name: self.tau_standard for name in self.standard}}
        committed = {name: committed[name] for name in members}
        self.controller.note_round(self.now, kind, committed=committed, fast_ids=[self.fast],
                                   standard_ids=[name for name in members if name != self.fast], switch=switch)
        self.last_kind = kind
        self.kinds.append(kind)

    def decide(self, **changes):
        options = dict(fast_live=True, standard_live=True, solo_possible=True)
        options.update(changes)
        return self.controller.decide(self.now, **options)

    def run(self, packed_ms, solo_ms, rounds):
        for _ in range(rounds):
            self.round(self.decide().kind, packed_ms if self.decide().kind == 'packed' else solo_ms)


class ControllerTests(unittest.TestCase):
    def test_a_fixed_ratio_of_one_alternates_and_of_half_runs_a_solo_round_every_other_frame(self):
        for ratio, expected in ((1.0, 'PSPSPSPSPS'), (0.5, 'PPSPPSPPSP'), (2.0, 'PSSPSSPSSP'), (0.0, 'PPPPPPPPPP')):
            with self.subTest(ratio=ratio):
                controller = LaneController(config(ratio=ratio, gap_ms=5000.0))
                feed = Feed(controller)
                feed.run(80.0, 50.0, 10)
                self.assertEqual(''.join('P' if kind == 'packed' else 'S' for kind in feed.kinds), expected)

    def test_a_fractional_k_banks_credit_until_a_solo_round_is_due(self):
        controller = LaneController(config(ratio=0.28, gap_ms=5000.0))
        feed = Feed(controller)
        feed.run(80.0, 50.0, 400)
        solos = feed.kinds.count('solo')
        self.assertAlmostEqual(solos / feed.kinds.count('packed'), 0.28, delta=0.02)

    def test_planning_is_pure_and_only_an_executed_round_moves_the_state(self):
        controller = LaneController(config(ratio=1.0, gap_ms=5000.0))
        feed = Feed(controller)
        feed.round('packed', 80.0)
        before = (controller.credit, controller.solo_run, controller.frame, controller.rounds)
        first = [feed.decide() for _ in range(5)]
        self.assertEqual(len(set(first)), 1)
        self.assertEqual(first[0].kind, 'solo')
        self.assertEqual((controller.credit, controller.solo_run, controller.frame, controller.rounds), before)

    def test_no_standard_user_means_solo_rounds_and_no_fast_user_means_packed_ones(self):
        controller = LaneController(config())
        self.assertEqual(controller.decide(0.0, fast_live=True, standard_live=False, solo_possible=True).kind, 'solo')
        self.assertEqual(controller.decide(0.0, fast_live=False, standard_live=True, solo_possible=True).kind, 'packed')
        self.assertEqual(controller.decide(0.0, fast_live=False, standard_live=False, solo_possible=False).kind, 'packed')
        # a fast user the solo block cannot serve (its slot, its budget, its frontier) is packed with the others
        decision = controller.decide(0.0, fast_live=True, standard_live=True, solo_possible=False)
        self.assertEqual((decision.kind, decision.reason), ('packed', 'no-solo'))
        alone = controller.decide(0.0, fast_live=True, standard_live=False, solo_possible=False)
        self.assertEqual(alone.kind, 'packed')

    def test_never_more_than_k_max_solo_rounds_between_two_packed_rounds(self):
        controller = LaneController(config(ratio=3.0, k_max=2, gap_ms=5000.0))
        controller.credit = 3.0
        feed = Feed(controller)
        feed.round('packed', 80.0)
        run = longest = 0
        for _ in range(60):
            kind = feed.decide().kind
            feed.round(kind, 80.0 if kind == 'packed' else 30.0)
            run = run + 1 if kind == 'solo' else 0
            longest = max(longest, run)
        self.assertEqual(longest, 2)
        self.assertEqual(feed.decide(**{}).kind in ('packed', 'solo'), True)

    def test_a_packed_round_at_least_every_gap_ms_whatever_the_credit(self):
        controller = LaneController(config(ratio=3.0, k_max=3, gap_ms=100.0))
        controller.credit = 3.0
        feed = Feed(controller)
        feed.round('packed', 80.0)
        gaps, last = [], feed.now
        for _ in range(40):
            kind = feed.decide().kind
            feed.round(kind, 80.0 if kind == 'packed' else 60.0)
            if kind == 'packed':
                gaps.append((feed.now - last) * 1000.0)
                last = feed.now
        # the guard fires at the first DECISION past the gap: one more solo round may run, then the packed round itself
        self.assertLessEqual(max(gaps), 100.0 + 60.0 + 80.0)
        self.assertGreater(max(gaps), 100.0, 'and the credit was banked: without the guard the run would be longer')
        decision = LaneController(config(ratio=3.0, gap_ms=100.0))
        decision.credit, decision.last_packed_end = 3.0, 0.0
        self.assertEqual(decision.decide(0.2, fast_live=True, standard_live=True, solo_possible=True).reason, 'gap-bound')

    def test_the_controller_finds_the_closed_form_k_and_holds_the_standard_lane_at_its_target(self):
        controller = LaneController(config())
        feed = Feed(controller, standard=('s1', 's2', 's3'), tau_fast=12.1, tau_standard=12.1)
        feed.run(88.6, 55.3, 600)
        target = 75.0 * 1.05
        k_expected = closed_form_k(88.6 + 0.0, 55.3, 0.0, 12.1, target)
        self.assertAlmostEqual(controller.k, k_expected, delta=0.06)
        self.assertGreaterEqual(controller.std_rate_min(), 74.0)
        self.assertGreaterEqual(controller.fast_rate.rate(), 150.0)

    def test_the_floor_is_out_of_reach_at_any_k_and_the_fast_lane_gets_the_packed_rate(self):
        controller = LaneController(config())
        feed = Feed(controller, standard=('s1',), tau_fast=12.1, tau_standard=4.9)
        feed.run(88.6, 55.3, 200)
        self.assertEqual(controller.k, 0.0)
        self.assertEqual(controller.reason, 'std-below-floor-at-k0')
        self.assertEqual(feed.kinds[-20:].count('solo'), 0)

    def test_a_missed_floor_halves_k_and_clears_the_credit_at_once(self):
        controller = LaneController(config())
        feed = Feed(controller, standard=('s1', 's2'), tau_standard=12.0)
        feed.run(80.0, 50.0, 200)
        k = controller.k
        self.assertGreater(k, 0.3)
        # the measured rate of a standard lane falls under the floor while the EWMAs still say the frame fits
        window = controller.std_rates['s1']
        window.reset()
        for index in range(8):
            window.add(feed.now - 1.4 + index * 0.2, 12)      # 12 tokens per 0.2 s: 60 tok/s
        self.assertLess(controller.std_rate_min(), 75.0)
        controller.note_round(feed.now + 0.08, 'packed', committed={'f': 12, 's1': 12, 's2': 12}, fast_ids=['f'],
                              standard_ids=['s1', 's2'])
        self.assertEqual(controller.reason, 'floor-bound')
        self.assertLessEqual(controller.k, k * 0.5 + 1e-9)
        self.assertAlmostEqual(controller.credit, controller.k, msg='the credit was cleared, then this frame banked its k')

    def test_k_ramps_up_a_quarter_a_frame_from_zero(self):
        controller = LaneController(config())
        feed = Feed(controller, standard=('s1',), tau_standard=12.0)
        seen = []
        for _ in range(60):
            kind = feed.decide().kind
            feed.round(kind, 60.0 if kind == 'packed' else 50.0)
            if kind == 'packed':
                seen.append(controller.k)
        rises = [b - a for a, b in zip(seen, seen[1:]) if b > a]
        self.assertTrue(rises and max(rises) <= lanes.RAMP + 1e-9, max(rises) if rises else None)

    def test_a_stall_teaches_nothing_and_is_counted(self):
        controller = LaneController(config(ratio=1.0))
        feed = Feed(controller)
        feed.round('packed', 80.0)
        feed.round('packed', 80.0)
        packed_before = controller.packed_ms.value
        feed.now += 8.0                      # a cold prefill between two rounds
        feed.round('packed', 80.0)
        self.assertEqual(controller.packed_ms.value, packed_before)
        self.assertAlmostEqual(controller.stall_ms, 8080.0, delta=1.0)
        self.assertEqual((controller.credit, controller.solo_run), (1.0, 0), 'the banked credit is the round after the stall\'s own')

    def test_a_stall_does_not_reach_the_rate_windows_and_so_never_halves_k(self):
        controller = LaneController(config())
        feed = Feed(controller, standard=('s1', 's2'), tau_standard=12.0)
        feed.run(80.0, 50.0, 200)
        k = controller.k
        self.assertGreater(k, 0.3)
        feed.now += 7.0                      # a prefill: 7 s inside every window would read as a lane far under the floor
        feed.round('packed', 80.0)
        self.assertIsNone(controller.std_rate_min(), 'the windows restart after a stall')
        for _ in range(3):
            feed.round(feed.decide().kind, 80.0)
        self.assertNotEqual(controller.reason, 'floor-bound')

    def test_the_first_solo_round_is_a_probe_once_packed_rounds_are_known(self):
        controller = LaneController(config(gap_ms=5000.0))
        feed = Feed(controller)
        kinds = []
        for _ in range(8):
            decision = feed.decide()
            kinds.append((decision.kind, decision.reason))
            feed.round(decision.kind, 80.0 if decision.kind == 'packed' else 50.0)
        self.assertEqual(kinds[0][1], 'probe-packed')
        self.assertIn(('solo', 'probe-solo'), kinds)
        self.assertEqual(kinds.index(('solo', 'probe-solo')), 3, 'after two measured packed cycles (three rounds)')

    def test_the_switch_cost_is_measured_from_the_rounds_that_follow_a_switch(self):
        controller = LaneController(config(ratio=1.0, gap_ms=5000.0))
        feed = Feed(controller)
        feed.round('packed', 80.0)
        feed.round('packed', 80.0)
        feed.round('solo', 50.0)
        feed.round('solo', 50.0)
        feed.round('packed', 80.0)
        for _ in range(6):
            feed.round('solo', 53.0)
            feed.round('packed', 83.0)
        self.assertAlmostEqual(controller.switch_extra.value, 3.0, delta=0.5)

    def test_a_request_that_left_no_longer_binds_the_controller(self):
        controller = LaneController(config())
        feed = Feed(controller, standard=('s1', 's2'))
        for _ in range(8):
            feed.round('packed', 80.0)
        controller.tau_std['s2'].value = 2.0
        self.assertEqual(controller.tau_std_min(), 2.0)
        controller.forget('s2')
        self.assertGreater(controller.tau_std_min(), 2.0)


class ScheduleTests(unittest.TestCase):
    """QWEN_FAST_LANE_SCHEDULE: the gate's ratio sweep on one boot, in frames with both lanes live (never the clock)."""

    def test_the_grammar(self):
        self.assertIsNone(lanes.parse_schedule(None, 3))
        self.assertIsNone(lanes.parse_schedule('', 3))
        self.assertEqual(lanes.parse_schedule('0@30,0.5@30,1@30,2@30,auto', 3),
                         ((0.0, 30), (0.5, 30), (1.0, 30), (2.0, 30), (None, None)))
        self.assertEqual(lanes.parse_schedule('1', 3), ((1.0, None),))
        self.assertEqual(lanes.parse_schedule('auto@5,2@9', 3), ((None, 5), (2.0, 9)))
        for bad in ('0.5,1', '4@3,auto', '-1@3,auto', 'x@3,auto', '1@0,auto', '1@1.5,auto', '1@,auto', '1@3,,auto', 'auto@100001,1'):
            with self.subTest(text=bad):
                with self.assertRaises(ValueError):
                    lanes.parse_schedule(bad, 3)

    def test_the_config_takes_it_and_refuses_a_fixed_ratio_beside_it(self):
        made = LaneConfig.from_environment({'QWEN_FAST_LANE_SCHEDULE': '0@2,auto'})
        self.assertEqual(made.schedule, ((0.0, 2), (None, None)))
        self.assertIsNone(LaneConfig.from_environment({}).schedule)
        with self.assertRaisesRegex(ValueError, 'two answers to one question'):
            LaneConfig.from_environment({'QWEN_FAST_LANE_SCHEDULE': '0@2,auto', 'QWEN_FAST_LANE_RATIO': '1'})
        self.assertEqual(LaneConfig.from_environment({'QWEN_FAST_LANE_SCHEDULE': '0@2,auto', 'QWEN_FAST_LANE_RATIO': 'auto'}).ratio, None)
        with self.assertRaises(ValueError):
            LaneConfig.from_environment({'QWEN_FAST_LANE_SCHEDULE': '5@2,auto'})
        self.assertIn('QWEN_FAST_LANE_SCHEDULE', lanes.FLAG_NAMES)

    def test_each_entry_runs_for_its_frames_with_both_lanes_live_and_the_last_for_good(self):
        controller = LaneController(config(schedule=((0.0, 3), (1.0, 3), (2.0, None)), gap_ms=5000.0))
        feed = Feed(controller)
        labels = []
        for _ in range(60):
            kind = feed.decide().kind
            feed.round(kind, 80.0 if kind == 'packed' else 50.0)
            if kind == 'packed':
                labels.append(controller.ratio_label())
        self.assertEqual(labels[:3], ['0', '0', '0'], 'the packed round labels the frame it opens')
        self.assertEqual(labels[3:6], ['1', '1', '1'])
        self.assertEqual(set(labels[6:]), {'2'})
        # the solo rounds each frame follow the k in force: none in the first stretch, one per frame in the second
        text = ''.join('P' if kind == 'packed' else 'S' for kind in feed.kinds)
        self.assertTrue(text.startswith('PPP'), text)
        self.assertIn('PSPSPS', text)
        self.assertIn('PSSPSSPSS', text)

    def test_frames_without_both_lanes_do_not_advance_the_schedule(self):
        controller = LaneController(config(schedule=((1.0, 2), (None, None))))
        for _ in range(5):        # a packed round of standard users alone (no fast user live)
            controller.note_round(0.1, 'packed', committed={'s1': 5}, fast_ids=[], standard_ids=['s1'])
        self.assertEqual(controller.sched_frames, 0)
        self.assertEqual(controller.ratio_label(), '1')
        for _ in range(2):
            controller.note_round(0.2, 'packed', committed={'f': 5, 's1': 5}, fast_ids=['f'], standard_ids=['s1'])
        self.assertEqual(controller.ratio_label(), '1', 'two frames with both lanes live: the second is still the first entry')
        controller.note_round(0.3, 'packed', committed={'f': 5, 's1': 5}, fast_ids=['f'], standard_ids=['s1'])
        self.assertEqual(controller.ratio_label(), 'auto')

    def test_an_auto_entry_is_the_controllers_own_k_and_a_fixed_one_is_reported_as_sched(self):
        controller = LaneController(config(schedule=((0.5, 1), (None, None))))
        self.assertEqual(controller.target_k(), (0.5, 'sched'))
        controller.note_round(0.1, 'packed', committed={'f': 5, 's1': 5}, fast_ids=['f'], standard_ids=['s1'])
        self.assertEqual(controller.ratio_label(), '0.5', 'the first frame is the first entry')
        controller.note_round(0.2, 'packed', committed={'f': 5, 's1': 5}, fast_ids=['f'], standard_ids=['s1'])
        self.assertEqual(controller.ratio_label(), 'auto')
        self.assertEqual(controller.target_k(), (0.0, 'probe-solo'))

    def test_a_fixed_ratio_is_labelled_and_a_scheduled_round_line_names_its_stretch(self):
        self.assertEqual(LaneController(config(ratio=1.5)).ratio_label(), '1.5')
        self.assertEqual(LaneController(config()).ratio_label(), 'auto')
        lines = Lines()
        runtime = LaneRuntime(config(schedule=((1.0, 1), (None, None))), log=lines, clock=lambda: 0.0)
        runtime.book.admit('f', FAST)
        runtime.book.admit('s', STANDARD)
        runtime.note_round('packed', ['f', 's'], {'f': 9, 's': 9}, now=0.1, block='packed', live=2)
        runtime.note_round('packed', ['f', 's'], {'f': 9, 's': 9}, now=0.2, block='packed', live=2)
        rounds = [line for line in lines.lines if line.startswith('[LANE-ROUND]')]
        self.assertTrue(rounds[0].endswith('ratio=1'), 'the line names the ratio of the frame this round opens')
        self.assertTrue(rounds[1].endswith('ratio=auto'))


class RateWindowTests(unittest.TestCase):
    def test_tokens_after_the_first_event_over_the_span(self):
        window = RateWindow()
        for index in range(6):
            window.add(index * 0.1, 10)
        self.assertAlmostEqual(window.rate(), 50.0 / 0.5)
        self.assertIsNone(RateWindow().rate())
        short = RateWindow()
        for index in range(3):
            short.add(index * 0.1, 10)
        self.assertIsNone(short.rate(), 'under MIN_RATE_EVENTS')


def request(name, finished=False, slot=None):
    return SimpleNamespace(session=SimpleNamespace(request_id=name, finished=finished))


class PlanTests(unittest.TestCase):
    def runtime(self, **changes):
        self.lines = Lines()
        self.now = [0.0]
        runtime = LaneRuntime(config(**changes), log=self.lines, clock=lambda: self.now[0])
        return runtime

    def plan(self, runtime, requests, rows=16, solo=16):
        return runtime.plan(requests, rows_of=lambda live: rows, solo_rows_of=lambda request: solo)

    def test_no_live_request_or_one_the_book_does_not_know_is_not_the_lanes_round(self):
        runtime = self.runtime()
        self.assertIsNone(self.plan(runtime, []))
        self.assertIsNone(self.plan(runtime, [request('a', finished=True)]))
        self.assertIsNone(self.plan(runtime, [request('stranger')]))

    def test_with_no_standard_user_the_round_is_a_solo_round_of_the_fast_user(self):
        runtime = self.runtime()
        runtime.book.admit('f', FAST)
        plan = self.plan(runtime, [request('f')])
        self.assertEqual((plan.kind, plan.members), ('solo', frozenset(['f'])))

    def test_with_no_fast_user_it_is_a_packed_round_of_everyone(self):
        runtime = self.runtime()
        for name in 'abc':
            runtime.book.admit(name, STANDARD)
        plan = self.plan(runtime, [request(name) for name in 'abc'])
        self.assertEqual((plan.kind, plan.members, plan.reason), ('packed', frozenset('abc'), 'no-fast'))

    def test_a_packed_round_the_block_cannot_serve_is_not_the_lanes_and_hides_nobody(self):
        runtime = self.runtime()
        runtime.book.admit('f', FAST)
        runtime.book.admit('s', STANDARD)
        self.assertIsNone(self.plan(runtime, [request('f'), request('s')], rows=None))

    def test_a_fast_user_the_solo_block_cannot_serve_rides_the_packed_round(self):
        runtime = self.runtime()
        runtime.book.admit('f', FAST)
        runtime.book.admit('s', STANDARD)
        plan = self.plan(runtime, [request('f'), request('s')], solo=None)
        self.assertEqual((plan.kind, plan.reason, plan.members), ('packed', 'no-solo', frozenset('fs')))

    def test_a_fast_request_at_its_real_budget_is_not_live_and_no_solo_round_is_planned_for_one_with_under_a_round_left(self):
        runtime = self.runtime()
        runtime.book.admit('f', FAST)
        runtime.book.admit('s', STANDARD)
        live = [request('f'), request('s')]
        budgets = dict(f=0, s=500)
        plan = runtime.plan(live, rows_of=lambda ok: 16, solo_rows_of=lambda r: 16, budget_of=lambda r: budgets[r.session.request_id])
        self.assertEqual((plan.kind, plan.members), ('packed', frozenset('s')))
        budgets['f'] = 5
        for _ in range(3):
            plan = runtime.plan(live, rows_of=lambda ok: 16, solo_rows_of=lambda r: 16,
                                budget_of=lambda r: budgets[r.session.request_id])
            self.assertNotEqual(plan.kind, 'solo', 'a fast request with 5 tokens left cannot fill a 16-row round')

    def test_a_finished_request_is_not_a_member(self):
        runtime = self.runtime()
        runtime.book.admit('f', FAST)
        runtime.book.admit('s', STANDARD)
        runtime.book.admit('t', STANDARD)
        plan = self.plan(runtime, [request('f'), request('s'), request('t', finished=True)])
        self.assertNotIn('t', plan.members)

    def test_planning_twice_is_the_same_plan_and_changes_nothing(self):
        runtime = self.runtime(ratio=1.0)
        runtime.book.admit('f', FAST)
        runtime.book.admit('s', STANDARD)
        live = [request('f'), request('s')]
        runtime.note_round('packed', ['f', 's'], {'f': 12, 's': 12}, now=0.1, block='packed')
        state = (runtime.controller.credit, runtime.controller.solo_run, runtime.controller.frame, runtime.rounds)
        plans = [self.plan(runtime, live) for _ in range(4)]
        self.assertEqual(len(set(plans)), 1)
        self.assertEqual((runtime.controller.credit, runtime.controller.solo_run, runtime.controller.frame, runtime.rounds), state)

    def test_publish_hands_the_members_to_the_gate_and_release_clears_a_plan_naming_the_leaver(self):
        runtime = self.runtime()
        runtime.book.admit('f', FAST)
        runtime.book.admit('s', STANDARD)
        runtime.publish(self.plan(runtime, [request('f')], rows=16))
        self.assertEqual(runtime.gate.members, frozenset(['f']))
        runtime.release('s')
        self.assertEqual(runtime.gate.members, frozenset(['f']), 'a leaver the plan did not name changes nothing')
        runtime.release('f')
        self.assertIsNone(runtime.gate.members)
        self.assertIsNone(runtime.gate.fast_id)
        runtime.publish(None)
        self.assertIsNone(runtime.gate.members)

    def test_the_gate_is_one_module_under_the_scheduler_sides_key(self):
        import serving_fast_lane_scheduler
        import sys

        runtime = self.runtime()
        self.assertEqual(lanes.GATE_KEY, serving_fast_lane_scheduler.GATE_KEY)
        self.assertIs(sys.modules[lanes.GATE_KEY], runtime.gate)
        self.assertIs(lanes.gate(), runtime.gate)
        self.assertEqual(serving_fast_lane_scheduler.FAST, lanes.FAST)
        self.assertEqual(serving_fast_lane_scheduler.SCHED_MARKER, lanes.SCHED_MARKER)


class TelemetryTests(unittest.TestCase):
    ROUND = ('[LANE-ROUND] round=%d kind=%s block=%s members=%s live=%d ms=%s committed=%s switch=%d')

    def test_the_round_and_frame_lines_carry_what_a_gate_reads_and_stay_short(self):
        lines = Lines()
        runtime = LaneRuntime(config(), log=lines, clock=lambda: 0.0)
        fast_id, std_id = 'cmpl-' + 'a' * 32 + '-0-fast0001', 'cmpl-' + 'b' * 32 + '-0-std00002'
        runtime.book.admit(fast_id, FAST)
        runtime.book.admit(std_id, STANDARD)
        admits = [line for line in lines.lines if line.startswith('[LANE-ADMIT]')]
        self.assertEqual(admits[0], '[LANE-ADMIT] request=%s alias=u1 asked=fast granted=fast slots=0 reason=ok' % fast_id)
        self.assertEqual(len(fast_id), 48, 'the 48-character prefix a gate maps to a stream id')
        runtime.note_round('packed', [fast_id, std_id], {fast_id: 13, std_id: 9}, now=0.1, block='packed', live=2)
        runtime.note_round('solo', [fast_id], {fast_id: 12}, now=0.15, block='solo', live=2)
        rounds = [line for line in lines.lines if line.startswith('[LANE-ROUND]')]
        self.assertEqual(rounds[0], '[LANE-ROUND] round=1 kind=packed block=packed members=u1,u2 live=2 ms=- committed=13,9 switch=0 ratio=auto')
        self.assertEqual(rounds[1], '[LANE-ROUND] round=2 kind=solo block=solo members=u1 live=2 ms=50.0 committed=12 switch=1 ratio=auto')
        frames = [line for line in lines.lines if line.startswith('[LANE-FRAME]')]
        self.assertEqual(len(frames), 1, 'one frame line per packed round')
        for name in ('frame=', 'k=', 'F_ms=', 'P_ms=', 'sigma_ms=', 'tau_fast=', 'tau_std_min=', 'fast_rate=',
                     'std_rate_min=', 'floor=', 'reason=', 'ratio='):
            self.assertIn(' ' + name, frames[0].replace('[LANE-FRAME] ', '[LANE-FRAME]  '))
        self.assertTrue(all(len(line) < 250 for line in lines.lines), max(len(line) for line in lines.lines))
        self.assertTrue(all(len(line) < 180 for line in lines.lines if not line.startswith('[LANE-FRAME]')))

    def test_a_round_neither_block_served_is_logged_and_teaches_the_controller_nothing(self):
        lines = Lines()
        runtime = LaneRuntime(config(), log=lines, clock=lambda: 0.0)
        runtime.book.admit('a', STANDARD)
        runtime.note_other('sequential', ['a'], {'a': 4}, now=1.0, live=1)
        self.assertEqual(lines.lines[-1], '[LANE-ROUND] round=1 kind=sequential block=- members=u1 live=1 ms=- committed=4 switch=0 ratio=auto')
        self.assertEqual(runtime.controller.rounds, 0)
        self.assertEqual(runtime.controller.last_end, 1.0)


if __name__ == '__main__':
    unittest.main()
