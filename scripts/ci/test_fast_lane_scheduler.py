"""The lane gate on the scheduler (serving_fast_lane_scheduler): hide the decodes a round does not serve, restore them in order, hold the seat.

On lanes_fakes.FakeTTScheduler, a class shaped like the pinned plugin's TTScheduler (schedule / _schedule_decode_only /
_schedule_prefill_only / _has_pending_prefill over `running`, `waiting` and `skipped_waiting`), with fake requests that carry the
fields the gate reads. The gate is the module the worker parks under sys.modules[GATE_KEY]; here the tests park it themselves."""

import sys
from types import SimpleNamespace
import unittest

import serving_fast_lane as lanes
import serving_fast_lane_scheduler as gate
from lanes_fakes import FakeQueue, FakeTTScheduler, SchedRequest


def request(name, lane=None, chunk=False):
    params = SimpleNamespace(extra_args=None if lane is None else {'qwen_lane': lane}, max_tokens=256)
    state = SimpleNamespace(output_token_ids=[7])
    made = SchedRequest(name, params, 100, state, chunk=chunk)
    made.spec_token_ids = [1, 2, 3] if not chunk else []
    return made


class Lines:
    def __init__(self):
        self.lines = []

    def __call__(self, template, *values):
        self.lines.append(template.format(*values))


class GateCase(unittest.TestCase):
    def setUp(self):
        sys.modules.pop(lanes.GATE_KEY, None)
        self.addCleanup(sys.modules.pop, lanes.GATE_KEY, None)
        self.holder = lanes.gate()
        self.log = Lines()
        self.cls = type('Scheduler', (FakeTTScheduler,), {})
        self.installed = gate.install(SimpleNamespace(scheduler_config=SimpleNamespace(scheduler_cls=self.cls)),
                                      log=self.log, queue_factory=lambda scheduler: FakeQueue())
        self.scheduler = self.cls()

    def running(self, *names):
        self.scheduler.running = [request(name) for name in names]
        return self.scheduler.running

    def scheduled(self):
        return list(self.scheduler.schedule().scheduled_cached_reqs.req_ids)


class HideTests(GateCase):
    def test_a_decode_step_serves_exactly_the_members_and_puts_the_rest_back_in_order(self):
        running = self.running('a', 'b', 'c', 'd')
        self.holder.members = frozenset(['b', 'd'])
        self.assertEqual(self.scheduled(), ['b', 'd'])
        self.assertEqual([r.request_id for r in self.scheduler.running], ['a', 'b', 'c', 'd'])
        self.assertTrue(all(r is s for r, s in zip(running, self.scheduler.running)), 'the same request objects')
        # the hidden ones were not scheduled, so their proposals were not consumed: they are still theirs
        self.assertEqual([r.spec_token_ids for r in running], [[1, 2, 3], [], [1, 2, 3], []])

    def test_no_plan_and_no_gate_module_hide_nothing(self):
        self.running('a', 'b')
        self.assertEqual(self.scheduled(), ['a', 'b'])
        self.holder.members = frozenset(['a'])
        sys.modules.pop(lanes.GATE_KEY)
        self.assertEqual(self.scheduled(), ['a', 'b'], 'the gate module is gone: nothing is hidden')

    def test_a_solo_round_hides_every_standard_user(self):
        self.running('fast', 's1', 's2', 's3')
        self.holder.members = frozenset(['fast'])
        self.assertEqual(self.scheduled(), ['fast'])
        self.assertEqual(len(self.scheduler.running), 4)

    def test_a_plan_that_names_no_running_decode_makes_an_empty_step_then_opens(self):
        """Its members finished after the draft, and the decodes still running were not drafted for the round: an empty step, twice,
        then the gate opens rather than starve the engine."""
        self.running('a', 'b')
        self.holder.members = frozenset(['gone'])
        self.assertEqual(self.scheduled(), [])
        self.assertEqual(self.scheduled(), [])
        self.assertEqual(self.scheduled(), ['a', 'b'])
        self.assertEqual(self.scheduled(), ['a', 'b'])
        self.assertEqual([r.request_id for r in self.scheduler.running], ['a', 'b'], 'nothing is lost, order kept')
        self.assertEqual(len([line for line in self.log.lines if 'an empty step' in line]), 1)
        self.assertEqual(len([line for line in self.log.lines if 'nothing hidden' in line]), 1)

    def test_a_served_step_resets_the_empty_step_count(self):
        self.running('a', 'b')
        self.holder.members = frozenset(['gone'])
        self.assertEqual(self.scheduled(), [])
        self.holder.members = frozenset(['a'])
        self.assertEqual(self.scheduled(), ['a'])
        self.holder.members = frozenset(['gone'])
        self.assertEqual(self.scheduled(), [])
        self.assertEqual(self.scheduled(), [])

    def test_a_partial_prefill_is_never_hidden_or_scheduled_as_a_decode(self):
        self.scheduler.running = [request('a'), request('chunk', chunk=True), request('b')]
        self.holder.members = frozenset(['a'])
        # a partial prefill is pending work: the prefill path runs (nothing waits, so it schedules nothing) and the plugin falls back
        # to its decode-only step - which the gate also wraps
        self.assertEqual(self.scheduled(), ['a'])
        self.assertEqual(sorted(r.request_id for r in self.scheduler.running), ['a', 'b', 'chunk'])
        self.assertEqual([r.request_id for r in self.scheduler.running if not r.is_prefill_chunk], ['a', 'b'],
                         'the decodes keep their order (the plugin prefill path puts the partials first)')

    def test_a_prefill_step_is_left_alone(self):
        self.running('a', 'b')
        self.scheduler.waiting.add_request(request('new'))
        self.holder.members = frozenset(['a'])
        output = self.scheduler.schedule()
        self.assertEqual(output.scheduled_cached_reqs.req_ids, [])
        self.assertEqual([r.request_id for r in output.scheduled_new_reqs], ['new'])
        self.assertEqual([r.request_id for r in self.scheduler.running], ['a', 'b'])
        self.assertNotIn('base', self.scheduler.calls)

    def test_the_decode_fallback_of_a_blocked_prefill_is_gated_too(self):
        """Waiting work the plugin cannot admit (no free seat): its default mode falls back to _schedule_decode_only."""
        self.running('a', 'b', 'c', 'd')
        self.scheduler.waiting.add_request(request('new'))
        self.holder.members = frozenset(['c'])
        self.assertEqual(self.scheduled(), ['c'])
        self.assertIn('decode-only', self.scheduler.calls)
        self.assertEqual([r.request_id for r in self.scheduler.running], ['a', 'b', 'c', 'd'])

    def test_a_forced_prefill_mode_is_not_a_decode_step(self):
        self.running('a', 'b')
        self.scheduler._forced_mode = SimpleNamespace(name='PREFILL_ONLY')
        self.holder.members = frozenset(['a'])
        self.assertEqual(self.scheduled(), [])
        self.assertEqual([r.request_id for r in self.scheduler.running], ['a', 'b'])
        self.assertNotIn('base', self.scheduler.calls)

    def test_a_request_the_base_call_removes_stays_out_and_the_rest_keep_their_order(self):
        running = self.running('a', 'b', 'c')
        self.holder.members = frozenset(['a', 'c'])
        base = self.cls._decode_step

        def preempting(scheduler):
            output = base(scheduler)
            scheduler.running = [r for r in scheduler.running if r.request_id != 'c']     # 'c' was preempted by this call
            return output

        self.cls._decode_step = preempting
        self.assertEqual(self.scheduled(), ['a', 'c'])
        self.assertEqual([r.request_id for r in self.scheduler.running], ['a', 'b'], 'the preempted one is not put back as running')
        self.assertIs(self.scheduler.running[1], running[1])

    def test_an_exception_in_the_base_call_still_restores_the_hidden_requests(self):
        running = self.running('a', 'b')
        self.holder.members = frozenset(['a'])

        def failing(scheduler):
            raise RuntimeError('KV pressure')

        self.cls._decode_step = failing
        with self.assertRaisesRegex(RuntimeError, 'KV pressure'):
            self.scheduler.schedule()
        self.assertEqual([r.request_id for r in self.scheduler.running], ['a', 'b'])
        self.assertFalse(getattr(self.scheduler, gate.HIDING))
        self.assertTrue(all(r is s for r, s in zip(running, self.scheduler.running)))

    def test_one_line_per_distinct_hidden_state_with_the_books_aliases(self):
        self.holder.aliases = {'s1': 'u2', 's2': 'u3', 'fast': 'u1'}
        for _ in range(3):
            self.running('fast', 's1', 's2')
            self.holder.members = frozenset(['fast'])
            self.scheduled()
        self.running('fast', 's1', 's2')
        self.holder.members = frozenset(['fast', 's1'])
        self.scheduled()
        lines = [line for line in self.log.lines if line.startswith(gate.SCHED_MARKER)]
        self.assertEqual(lines, ['[LANE-SCHED] round=0 members=1 hidden=u2,u3', '[LANE-SCHED] round=0 members=2 hidden=u3'])

    def test_the_log_is_bounded_by_the_number_of_distinct_states(self):
        for index in range(gate.LOG_STATES + 20):
            self.scheduler.running = [request('keep'), request('x%d' % index)]
            self.holder.members = frozenset(['keep'])
            self.scheduled()
        self.assertEqual(len([line for line in self.log.lines if line.startswith(gate.SCHED_MARKER)]), gate.LOG_STATES)


class InstallTests(GateCase):
    def test_installed_once_per_class_and_a_subclass_inherits_it(self):
        self.assertEqual(self.installed, 'test_fast_lane_scheduler.Scheduler')
        for name in ('schedule', '_schedule_decode_only', '_schedule_prefill_only'):
            self.assertTrue(getattr(getattr(self.cls, name), gate.WRAPPED), name)
        again = gate.install(SimpleNamespace(scheduler_config=SimpleNamespace(scheduler_cls=self.cls)), log=self.log,
                             queue_factory=lambda scheduler: FakeQueue())
        self.assertEqual(again, self.installed)
        self.assertEqual(len([line for line in self.log.lines if line.startswith(gate.INSTALLED)]), 1)
        sub = type('Sub', (self.cls,), {})
        self.assertIs(sub.schedule, self.cls.schedule)
        self.assertEqual(gate.install(SimpleNamespace(scheduler_config=SimpleNamespace(scheduler_cls=sub)), log=self.log,
                                      queue_factory=lambda s: FakeQueue()).rsplit('.', 1)[-1], 'Sub')
        self.assertEqual(len([line for line in self.log.lines if line.startswith(gate.INSTALLED)]), 1, 'not wrapped twice')

    def test_a_class_without_schedule_is_refused_by_name(self):
        class Bare:
            pass

        with self.assertRaisesRegex(ValueError, 'has no schedule'):
            gate.install(SimpleNamespace(scheduler_config=SimpleNamespace(scheduler_cls=Bare)), log=self.log,
                         queue_factory=lambda scheduler: FakeQueue())

    def test_the_gate_wraps_beside_the_prefill_admission_cap_and_composes_with_it(self):
        """serving_prefill_admission wraps _schedule_prefill_only first; the gate's seat hold wraps that wrapper."""
        calls = []

        class Plugin(FakeTTScheduler):
            pass

        original = Plugin._schedule_prefill_only

        def capped(self):
            calls.append('cap')
            return original(self)

        capped._qwen_one_fresh_prefill = True
        Plugin._schedule_prefill_only = capped
        gate.install(SimpleNamespace(scheduler_config=SimpleNamespace(scheduler_cls=Plugin)), log=self.log,
                     queue_factory=lambda scheduler: FakeQueue())
        scheduler = Plugin()
        scheduler.running = [request('a')]
        scheduler.waiting.add_request(request('new'))
        scheduler.schedule()
        self.assertEqual(calls, ['cap'])
        self.assertIs(Plugin._schedule_prefill_only.__wrapped__, capped)


class SeatHoldTests(GateCase):
    """QWEN_FAST_LANE_RESERVE=1: standard users fill at most seats - 1; a fast request takes the last seat."""

    def setUp(self):
        super().setUp()
        self.holder.seats, self.holder.reserve, self.holder.fast_id = 4, True, None
        self.admitted = []
        original = self.cls._schedule_prefill_only.__wrapped__

        def recording(scheduler):
            self.admitted.append([r.request_id for r in scheduler.waiting])
            return original(scheduler)

        # the wrapper under test wraps whatever is there; put the recorder BELOW it
        self.cls._schedule_prefill_only = gate.wrap_admission(recording, queue_factory=lambda s: FakeQueue(),
                                                              notes=gate.Notes(self.log))

    def waiting(self, *requests):
        for value in requests:
            self.scheduler.waiting.add_request(value)

    def test_three_standard_users_running_hold_the_fourth_standard_arrival(self):
        self.scheduler.running = [request(name) for name in ('s1', 's2', 's3')]
        self.waiting(request('s4'))
        self.scheduler.schedule()
        self.assertEqual(self.admitted, [[]], 'the base scheduler saw an empty waiting queue')
        self.assertEqual([r.request_id for r in self.scheduler.waiting], ['s4'], 'and s4 is back in its queue')
        self.assertTrue([line for line in self.log.lines if 'seat hold' in line])

    def test_a_fast_arrival_passes_the_hold_and_the_standard_request_ahead_of_it_waits(self):
        self.scheduler.running = [request(name) for name in ('s1', 's2', 's3')]
        self.waiting(request('s4'), request('f', lane='fast'), request('s5'))
        self.scheduler.schedule()
        self.assertEqual(self.admitted, [['f']])
        self.assertEqual(sorted(r.request_id for r in self.scheduler.waiting), ['s4', 's5'])

    def test_a_second_fast_arrival_is_standard_and_waits_like_one(self):
        self.holder.fast_id = 'f1'
        self.scheduler.running = [request('f1', lane='fast'), request('s1'), request('s2')]
        self.waiting(request('f2', lane='fast'))
        self.scheduler.schedule()
        self.assertEqual(self.admitted, [['f2']], 'f1 + two standard = three seats; the fourth is open to a standard request')
        self.scheduler.running = [request('f1', lane='fast'), request('s1'), request('s2'), request('s3')]
        self.admitted.clear()
        self.scheduler.waiting = FakeQueue([request('f3', lane='fast')])
        self.scheduler.schedule()
        self.assertEqual(self.admitted, [[]])

    def test_the_hold_counts_only_standard_users_and_a_fast_prefill_in_flight_holds_the_fast_seat(self):
        self.scheduler.running = [request('s1'), request('s2'), request('fchunk', lane='fast', chunk=True)]
        self.waiting(request('s3'), request('f2', lane='fast'))
        self.scheduler.schedule()
        self.assertEqual(self.admitted, [['s3', 'f2']], 'two standard users and the fast seat: a third standard may come')

    def test_below_the_cap_nothing_is_held(self):
        self.scheduler.running = [request('s1'), request('s2')]
        self.waiting(request('s3'))
        self.scheduler.schedule()
        self.assertEqual(self.admitted, [['s3']])

    def test_with_the_reserve_off_nothing_is_ever_held(self):
        self.holder.reserve = False
        self.scheduler.running = [request(name) for name in ('s1', 's2', 's3')]
        self.waiting(request('s4'))
        self.scheduler.schedule()
        self.assertEqual(self.admitted, [['s4']])

    def test_an_absent_gate_module_holds_nothing(self):
        sys.modules.pop(lanes.GATE_KEY)
        self.scheduler.running = [request(name) for name in ('s1', 's2', 's3')]
        self.waiting(request('s4'))
        self.scheduler.schedule()
        self.assertEqual(self.admitted, [['s4']])

    def test_the_skipped_queue_is_hidden_while_standard_requests_are_held_and_restored_after(self):
        self.scheduler.running = [request(name) for name in ('s1', 's2', 's3')]
        skipped = FakeQueue([request('grammar')])
        self.scheduler.skipped_waiting = skipped
        self.waiting(request('s4'))
        self.scheduler.schedule()
        self.assertIs(self.scheduler.skipped_waiting, skipped)
        self.assertEqual([r.request_id for r in skipped], ['grammar'])


class ModuleTests(unittest.TestCase):
    def test_the_key_and_markers_are_the_workers(self):
        self.assertEqual(gate.GATE_KEY, lanes.GATE_KEY)
        self.assertEqual(gate.SCHED_MARKER, lanes.SCHED_MARKER)
        self.assertEqual(gate.FAST, lanes.FAST)

    def test_it_imports_no_worker_module_at_import_time(self):
        import ast
        from pathlib import Path

        tree = ast.parse(Path(gate.__file__).read_text(encoding='utf-8'))
        top = {alias.name.split('.')[0] for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
        top |= {node.module.split('.')[0] for node in tree.body if isinstance(node, ast.ImportFrom)}
        self.assertEqual(top, {'importlib', 'sys'}, 'the scheduler side must import nothing of the worker\'s at module level')


if __name__ == '__main__':
    unittest.main()
