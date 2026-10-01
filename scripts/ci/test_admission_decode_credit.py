"""QWEN_FAST_DECODE_STEPS_PER_ADMISSION (F1 of the four-card freeze plan): the decode credit after each admission.

serving_prefill_admission admits one fresh prompt per prefill step, and each admission's prefill and engine build
holds the gate for seconds, so while four arrivals are admitted the users already admitted do not decode. With R > 0
the wrapper answers the next R prefill calls after an admission with allowed = 0 and both queues hidden, so the plugin
runs R decode-only passes in between (lever N's alternation). Default 0: nothing is wrapped beyond what was.

The scheduler here is self-contained: vLLM's schedule() reduced to its running and waiting loops (and the finished-id
hand-off), under the plugin's default-mode dispatch (TTScheduler.schedule: prefer prefill when work is pending, fall
back to a decode-only pass when it schedules nothing and a decode runs, else decode naturally). It reads no file.

    py -3.11 -B -m unittest test_admission_decode_credit      (from scripts/ci)
"""

import os
import sys
import types
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import serving_prefill_admission as admission

R_ENV = {'QWEN_FAST_DECODE_STEPS_PER_ADMISSION': '1'}


class Queue(list):
    def prepend_requests(self, requests):
        self[:0] = list(requests)


class Request(object):
    def __init__(self, name, prompt_tokens=4096, partial=False):
        self.request_id, self.prompt_tokens, self.is_prefill_chunk = name, prompt_tokens, partial


class Base(object):
    """vLLM's Scheduler.schedule: every running request is scheduled (one token); the waiting loop admits whole
    prompts while a seat is free; the finished ids go out with the output and the set starts again."""

    def __init__(self, seats=4):
        self.running, self.waiting, self.skipped_waiting = [], Queue(), Queue()
        self.policy, self.max_num_running_reqs, self.finished_req_ids = 'fcfs', seats, set()

    def schedule(self):
        counts, new = {}, []
        for request in self.running:
            counts[request.request_id] = 1
        while (self.waiting or self.skipped_waiting) and len(self.running) < self.max_num_running_reqs:
            request = (self.skipped_waiting or self.waiting).pop(0)
            self.running.append(request)
            counts[request.request_id] = request.prompt_tokens
            new.append(request.request_id)
        finished, self.finished_req_ids = self.finished_req_ids, set()
        return SimpleNamespace(new=new, total_num_scheduled_tokens=sum(counts.values()), finished_req_ids=finished,
                               decodes=sorted(name for name in counts if name not in new))


class Plugin(Base):
    """TTScheduler's default-mode dispatch, over the same two helpers the plugin has."""

    def _has_pending_prefill(self):
        return bool(self.waiting) or bool(self.skipped_waiting) or any(r.is_prefill_chunk for r in self.running)

    def schedule(self):
        decoding = any(not request.is_prefill_chunk for request in self.running)
        if self._has_pending_prefill():
            result = self._schedule_prefill_only()
            if result.total_num_scheduled_tokens == 0 and decoding:
                return self._schedule_decode_only()
            return result
        return super().schedule()

    def _schedule_prefill_only(self):
        pure = [r for r in self.running if not r.is_prefill_chunk]
        partial = [r for r in self.running if r.is_prefill_chunk]
        saved = self.max_num_running_reqs
        self.running = partial
        self.max_num_running_reqs = max(0, saved - len(pure))
        try:
            return super().schedule()
        finally:
            self.running.extend(pure)
            self.max_num_running_reqs = saved

    def _schedule_decode_only(self):
        saved_waiting, saved_skipped = self.waiting, self.skipped_waiting
        self.waiting, self.skipped_waiting = Queue(), Queue()
        partial = [r for r in self.running if r.is_prefill_chunk]
        self.running = [r for r in self.running if not r.is_prefill_chunk]
        try:
            return super().schedule()
        finally:
            self.waiting, self.skipped_waiting = saved_waiting, saved_skipped
            self.running.extend(partial)


class Case(unittest.TestCase):
    def setUp(self):
        self.saved = sys.modules.pop(admission.GATE_KEY, None)
        self.addCleanup(self.restore)
        self.lines = []

    def restore(self):
        sys.modules.pop(admission.GATE_KEY, None)
        if self.saved is not None:
            sys.modules[admission.GATE_KEY] = self.saved

    def hold_gate(self, request):
        holder = types.ModuleType(admission.GATE_KEY)
        holder.held = request
        sys.modules[admission.GATE_KEY] = holder

    def release_gate(self):
        sys.modules.pop(admission.GATE_KEY, None)

    def scheduler(self, steps, seats=4, running=(), waiting=()):
        """A fresh Plugin subclass with the admission wrapper installed under R = steps (0: the flag unset)."""
        cls = type('Scheduler', (Plugin,), {})
        environ = {name: value for name, value in os.environ.items() if name != admission.STEPS_FLAG}
        if steps:
            environ[admission.STEPS_FLAG] = str(steps)
        config = SimpleNamespace(scheduler_config=SimpleNamespace(scheduler_cls=cls))
        with patch.dict(os.environ, environ, clear=True):
            admission.install(config, log=lambda message, *values: self.lines.append(message.format(*values)),
                              queue_factory=lambda scheduler: Queue())
        scheduler = cls(seats)
        scheduler.running.extend(Request(name) for name in running)
        scheduler.waiting.extend(Request(name) for name in waiting)
        return scheduler

    @staticmethod
    def step(scheduler):
        """One schedule() call, as 'admit:<names>' or 'decode' (a pass with no new request)."""
        result = scheduler.schedule()
        return 'admit:' + ','.join(result.new) if result.new else 'decode'

    def sequence(self, scheduler, count):
        return [self.step(scheduler) for _ in range(count)]


class CreditSequenceTests(Case):
    def test_off_is_exactly_the_wrapper_that_already_existed(self):
        """R unset: schedule() is the class's own and the sequence is the one-prompt-per-step sequence, no decode
        pass between admissions (four seats, one decode running, four waiting)."""
        scheduler = self.scheduler(0, running=['d'], waiting=['a', 'b', 'c', 'e'])
        self.assertIs(type(scheduler).schedule, Plugin.schedule, 'schedule() is not wrapped with the credit off')
        self.assertEqual(self.sequence(scheduler, 5), ['admit:a', 'admit:b', 'admit:c', 'decode', 'decode'])
        self.assertEqual([line for line in self.lines if 'decode credit' in line], [])

    def test_one_decode_pass_after_each_admission_with_r_1(self):
        scheduler = self.scheduler(1, running=['d'], waiting=['a', 'b', 'c'])
        self.assertEqual(self.sequence(scheduler, 6),
                         ['admit:a', 'decode', 'admit:b', 'decode', 'admit:c', 'decode'])

    def test_the_first_admission_of_a_burst_is_followed_by_a_decode_pass_too(self):
        """With nothing decoding before the burst the first admitted user is the decode the credit serves."""
        scheduler = self.scheduler(1, waiting=['a', 'b', 'c', 'e'])
        self.assertEqual(self.sequence(scheduler, 8), ['admit:a', 'decode', 'admit:b', 'decode', 'admit:c', 'decode',
                                                         'admit:e', 'decode'])

    def test_r_decode_passes_after_each_admission(self):
        for steps in (2, 3):
            with self.subTest(steps=steps):
                scheduler = self.scheduler(steps, running=['d'], waiting=['a', 'b'])
                self.assertEqual(self.sequence(scheduler, 2 * (steps + 1)),
                                 ['admit:a'] + ['decode'] * steps + ['admit:b'] + ['decode'] * (steps - 1) + ['decode'])
                self.assertEqual(self.sequence(scheduler, 1), ['decode'])

    def test_the_held_calls_hide_both_queues_and_the_decodes_are_the_ones_that_were_running(self):
        scheduler = self.scheduler(1, running=['d'], waiting=['a', 'b'])
        scheduler.skipped_waiting.append(Request('s'))
        first = scheduler.schedule()
        self.assertEqual(first.new, ['s'], 'the skipped queue is promoted first, as the base loop does')
        held = scheduler.schedule()
        self.assertEqual((held.new, held.decodes), ([], ['d', 's']))
        self.assertEqual([request.request_id for request in scheduler.waiting], ['a', 'b'], 'the queues are back')

    def test_every_seat_decoding_leaves_the_sequence_as_it_was(self):
        """Four decodes and a waiting prompt: nothing can be admitted; the credit adds no pass the scheduler would not run."""
        for steps in (0, 1):
            scheduler = self.scheduler(steps, running=['d1', 'd2', 'd3', 'd4'], waiting=['a'])
            self.assertEqual(self.sequence(scheduler, 3), ['decode'] * 3, steps)


class CreditLifetimeTests(Case):
    def test_a_decode_only_schedule_with_nothing_pending_drops_the_credit(self):
        """The last of a burst arms a credit nothing pending can spend; the decode pass that follows drops it, so a later
        arrival is admitted at once."""
        scheduler = self.scheduler(2, running=['d'], waiting=['a'])
        self.assertEqual(self.step(scheduler), 'admit:a')
        self.assertEqual(self.step(scheduler), 'decode', 'nothing waits: the scheduler decodes naturally')
        scheduler.waiting.append(Request('b'))
        self.assertEqual(self.step(scheduler), 'admit:b', 'no stale credit holds the next arrival')

    def test_a_credit_is_not_carried_past_a_stretch_of_decoding(self):
        scheduler = self.scheduler(3, running=['d'], waiting=['a', 'b'])
        self.assertEqual(self.step(scheduler), 'admit:a')       # arms 3
        self.assertEqual(self.step(scheduler), 'decode')        # left 2
        scheduler.waiting.clear()                               # the request was cancelled
        self.assertEqual(self.step(scheduler), 'decode')        # nothing pending: drops the credit
        scheduler.waiting.append(Request('c'))
        self.assertEqual(self.step(scheduler), 'admit:c')

    def test_no_decode_running_drops_the_credit_and_admits(self):
        scheduler = self.scheduler(2, waiting=['a'])
        self.assertEqual(self.step(scheduler), 'admit:a')
        scheduler.running.clear()                               # the admitted request finished
        scheduler.waiting.append(Request('b'))
        self.assertEqual(self.step(scheduler), 'admit:b')
        self.assertEqual(scheduler.running[0].request_id, 'b')

    def test_a_gate_held_pass_pays_a_step(self):
        scheduler = self.scheduler(1, running=['d'], waiting=['a', 'b'])
        self.assertEqual(self.step(scheduler), 'admit:a')
        self.hold_gate('a')
        self.assertEqual(self.step(scheduler), 'decode', 'the held gate runs a decode pass')
        self.release_gate()
        self.assertEqual(self.step(scheduler), 'admit:b', 'which paid the credit: no second decode pass')

    def test_a_partial_prefill_in_flight_is_never_held_and_drops_the_credit(self):
        scheduler = self.scheduler(1, running=['d'], waiting=['a', 'b'])
        self.assertEqual(self.step(scheduler), 'admit:a')       # arms 1
        scheduler.running.append(Request('p', partial=True))
        result = scheduler.schedule()                           # the partial's continuation, queues hidden
        self.assertEqual((result.new, result.decodes), ([], ['p']), 'only the partial is scheduled, and not held')
        self.assertEqual([request.request_id for request in scheduler.waiting], ['b'])
        scheduler.running.remove(next(r for r in scheduler.running if r.is_prefill_chunk))
        self.assertEqual(self.step(scheduler), 'admit:b', 'the credit went with the partial')

    def test_a_held_pass_puts_the_finished_ids_back_for_the_decode_pass(self):
        scheduler = self.scheduler(1, running=['d', 'x'], waiting=['a', 'b'])
        self.assertEqual(self.step(scheduler), 'admit:a')
        scheduler.running = [request for request in scheduler.running if request.request_id != 'x']
        scheduler.finished_req_ids.add('x')                     # vLLM names it in the next output
        output = scheduler.schedule()                           # the held pass: its output is discarded
        self.assertEqual((output.new, output.finished_req_ids), ([], {'x'}), 'the decode pass names it exactly once')
        self.assertEqual(scheduler.finished_req_ids, set())
        self.assertTrue(any(line.startswith('[PINDIAG] decode credit carried finished=') for line in self.lines))

    def test_the_lines_are_bounded_and_say_what_happened(self):
        scheduler = self.scheduler(1, running=['d'], waiting=['a', 'b'])
        self.sequence(scheduler, 5)
        credit = [line for line in self.lines if line.startswith('[PINDIAG] decode credit')]
        self.assertEqual(credit[0].split(' installed on ')[0], '[PINDIAG] decode credit')
        self.assertIn('steps=1', credit[0])
        self.assertEqual([line for line in credit if ' armed ' in line],
                         ['[PINDIAG] decode credit armed steps=1 decodes=1 waiting=1',
                          '[PINDIAG] decode credit armed steps=1 decodes=2 waiting=0'])
        self.assertEqual([line for line in credit if ' hold ' in line],
                         ['[PINDIAG] decode credit hold left=0 decodes=2'], 'the last admission leaves nothing to hold')


class FlagTests(Case):
    def test_the_steps_parse_strictly_and_default_to_off(self):
        self.assertEqual(admission.decode_steps_per_admission({}), 0)
        for value, expected in (('0', 0), ('1', 1), ('2', 2), ('64', 64)):
            self.assertEqual(admission.decode_steps_per_admission({admission.STEPS_FLAG: value}), expected, value)
        for value in ('', '-1', '1.5', ' 1', 'one', '65', '١'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'whole number of steps'):
                admission.decode_steps_per_admission({admission.STEPS_FLAG: value})

    def test_install_wraps_schedule_only_with_the_credit_on_and_only_once(self):
        off = self.scheduler(0)
        self.assertIs(type(off).schedule, Plugin.schedule)
        on = self.scheduler(1)
        self.assertIsNot(type(on).schedule, Plugin.schedule)
        self.assertTrue(getattr(type(on).schedule, admission.WRAPPED))
        self.assertTrue(any(line.endswith(': steps=1') and 'decode credit installed on' in line for line in self.lines))
        wrapped = type(on).schedule
        with patch.dict(os.environ, R_ENV):
            admission.install(SimpleNamespace(scheduler_config=SimpleNamespace(scheduler_cls=type(on))),
                              log=lambda *args: None, queue_factory=lambda scheduler: Queue())
        self.assertIs(type(on).schedule, wrapped, 'a second install wraps nothing again')

    def test_a_bad_value_refuses_the_install(self):
        cls = type('Scheduler', (Plugin,), {})
        with patch.dict(os.environ, {admission.STEPS_FLAG: 'two'}), self.assertRaisesRegex(ValueError, 'whole number'):
            admission.install(SimpleNamespace(scheduler_config=SimpleNamespace(scheduler_cls=cls)), log=lambda *args: None,
                              queue_factory=lambda scheduler: Queue())
        self.assertIs(cls._schedule_prefill_only, Plugin._schedule_prefill_only, 'nothing was wrapped')


if __name__ == '__main__':
    unittest.main()
