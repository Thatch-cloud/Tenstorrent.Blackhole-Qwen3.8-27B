"""The one-in-flight cap, checked without a scheduler.

The real proof is probe_prefill_serialisation running the subclass against the
plugin's own scheduler. What is checked here is the arithmetic that makes the cap
correct, because it is easy to get wrong in a way that still looks plausible: the
plugin subtracts the resident decodes from whatever it reads, so a naive cap would
starve prefill entirely whenever any request was decoding.
"""

from types import SimpleNamespace
import collections
import unittest

from serving_one_in_flight import (one_in_flight_scheduler, allowed_prefills, effective_capacity, install,
                                   one_in_flight_scheduler, waiting_headroom)


class CapacityArithmeticTests(unittest.TestCase):
    def test_at_most_one_fresh_prompt_and_never_more_than_configured(self):
        """Two invariants, not one. The cap never leaves room for a second fresh
        prompt, and it never invents capacity the configuration does not have -
        at five decodes and three partials, a max_num_seqs of eight is genuinely
        exhausted and the partials, not a new prompt, are what it holds."""
        for configured in (1, 2, 8):
            for decodes in range(0, 6):
                for partials in range(0, 4):
                    capacity = effective_capacity(configured, decodes, partials)
                    where = 'configured=%d decodes=%d partials=%d' % (configured, decodes, partials)
                    self.assertLessEqual(capacity, allowed_prefills(partials), where)
                    self.assertEqual(capacity,
                                     min(allowed_prefills(partials), max(0, configured - decodes)), where)

    def test_one_slot_survives_whenever_the_configuration_can_afford_it(self):
        for decodes in range(0, 6):
            for partials in range(0, 4):
                self.assertEqual(effective_capacity(decodes + allowed_prefills(partials), decodes, partials),
                                 allowed_prefills(partials),
                                 'decodes=%d partials=%d' % (decodes, partials))

    def test_a_naive_cap_would_starve_prefill_and_this_one_does_not(self):
        """Capping at partials + 1 WITHOUT adding the decodes back gives
        max(0, partials + 1 - decodes), which is zero as soon as two requests
        decode. That is the mistake this arithmetic exists to avoid."""
        decodes, partials = 2, 0
        naive = max(0, min(8, partials + 1) - decodes)
        self.assertEqual(naive, 0, 'the naive cap admits nothing')
        self.assertEqual(effective_capacity(8, decodes, partials), 1)

    def test_a_smaller_configured_capacity_still_wins(self):
        self.assertEqual(effective_capacity(1, 0, 0), 1)
        self.assertEqual(effective_capacity(2, 2, 0), 0)

    def test_negative_or_non_integer_capacities_are_refused(self):
        for arguments in ((8, -1, 0), (8, 0, -1), (-1, 0, 0), (8.0, 0, 0), (8, True, 0)):
            with self.assertRaises(ValueError):
                effective_capacity(*arguments)


class SubclassTests(unittest.TestCase):
    def base(self):
        calls = []

        class FakeScheduler:
            def __init__(self):
                self.max_num_running_reqs = 8
                self.running = []

            def _schedule_prefill_only(self):
                decodes = [r for r in self.running if not r.is_prefill_chunk]
                saved = self.max_num_running_reqs
                self.max_num_running_reqs = max(0, saved - len(decodes))
                calls.append(self.max_num_running_reqs)
                self.max_num_running_reqs = saved
                return 'scheduled'

        return FakeScheduler, calls

    def request(self, prefill_chunk):
        return SimpleNamespace(is_prefill_chunk=prefill_chunk)

    def test_the_plugin_sees_exactly_one_slot_for_a_fresh_prompt(self):
        FakeScheduler, calls = self.base()
        scheduler = one_in_flight_scheduler(FakeScheduler)()
        scheduler.running = [self.request(False), self.request(False)]
        self.assertEqual(scheduler._schedule_prefill_only(), 'scheduled')
        self.assertEqual(calls, [1], 'two decodes resident, still one prefill slot')

    def test_a_partial_prefill_keeps_its_slot_and_gains_one(self):
        FakeScheduler, calls = self.base()
        scheduler = one_in_flight_scheduler(FakeScheduler)()
        scheduler.running = [self.request(True), self.request(False)]
        scheduler._schedule_prefill_only()
        # CHANGED 2026-09-22, build plan step 8. This used to be [2]: the partial
        # continued and one fresh prompt could join it. It may not any more. The pinned
        # plugin replaces self.running with the partials and then admits from waiting up
        # to max_num_running_reqs - len(self.running), so a cap of `partials` leaves
        # zero headroom while a partial is in flight - which is the only place that
        # interleaving can be stopped, the GDN prefill scratch being single-occupancy
        # and the model-side cursor being legitimately reset by a fresh start == 0.
        # Inert on every arm today: is_prefill_chunk requires chunked prefill, which is
        # off unless M3NATIVE_PREFILL_CHUNK_TOKENS is set, so partials is always 0 and
        # allowed_prefills(0) is 1 exactly as before.
        self.assertEqual(calls, [1], 'the partial continues alone; no fresh prompt joins')

    def test_the_configured_capacity_is_restored_even_on_failure(self):
        FakeScheduler, calls = self.base()

        class Failing(FakeScheduler):
            def _schedule_prefill_only(self):
                raise RuntimeError('boom')

        scheduler = one_in_flight_scheduler(Failing)()
        scheduler.running = [self.request(False)]
        with self.assertRaises(RuntimeError):
            scheduler._schedule_prefill_only()
        self.assertEqual(scheduler.max_num_running_reqs, 8)


class InstallTests(unittest.TestCase):
    def test_install_points_the_config_at_the_class(self):
        config = SimpleNamespace(scheduler_config=SimpleNamespace(scheduler_cls=None))
        chosen = install(config, scheduler='marker')
        self.assertEqual(chosen, 'marker')
        self.assertEqual(config.scheduler_config.scheduler_cls, 'marker')

    def test_the_default_is_a_dotted_path_so_nothing_imports_the_plugin(self):
        import serving_one_in_flight

        config = SimpleNamespace(scheduler_config=SimpleNamespace(scheduler_cls=None))
        self.assertEqual(install(config), 'serving_one_in_flight.OneInFlightScheduler')
        self.assertEqual(serving_one_in_flight.SCHEDULER_PATH,
                         config.scheduler_config.scheduler_cls)
        with self.assertRaises(AttributeError):
            serving_one_in_flight.something_else

    def test_install_refuses_to_replace_an_existing_choice(self):
        config = SimpleNamespace(scheduler_config=SimpleNamespace(scheduler_cls='someone_elses'))
        with self.assertRaises(ValueError):
            install(config, scheduler='marker')


class WaitingHeadroomTests(unittest.TestCase):
    """The property the cap exists for, stated in the units that matter: how many FRESH
    prompts the base scheduler's waiting loop can admit.

    Derived from the pinned plugin source captured by probe_plugin_scheduler_sources
    (cpu-probe 35665853903, scheduler.py sha256 a1bd6257d3a14c90:154-173), which
    replaces self.running with the partials before calling super().schedule(), so the
    waiting loop's budget is max_num_running_reqs - partials.
    """

    def test_one_fresh_prompt_with_nothing_in_flight(self):
        for decodes in range(0, 4):
            with self.subTest(decodes=decodes):
                self.assertEqual(waiting_headroom(8, decodes, 0), 1)

    def test_no_fresh_prompt_while_a_partial_prefill_is_in_flight(self):
        """The corruption this prevents: a fresh prompt between two continuations either
        re-zeroes the single-occupancy GDN scratch on its own start == 0 or advances it
        with foreign tokens, and the suspended prompt resumes on another's recurrence."""
        for decodes in range(0, 4):
            for partials in range(1, 4):
                with self.subTest(decodes=decodes, partials=partials):
                    self.assertEqual(waiting_headroom(8, decodes, partials), 0)

    def test_headroom_never_goes_negative_when_capacity_is_exhausted(self):
        self.assertEqual(waiting_headroom(1, 4, 2), 0)
        self.assertEqual(waiting_headroom(0, 0, 0), 0)

    def test_allowed_prefills_rejects_nonsense(self):
        for bad in (-1, 1.0, None, True):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                allowed_prefills(bad)


if __name__ == '__main__':
    unittest.main()


class SchedulerClassTests(unittest.TestCase):
    """What the class DOES, which no test checked until run 35689293766."""

    def build(self, running, waiting=('w1', 'w2'), skipped=('s1',), max_running=4):
        seen = {}

        class Base(object):
            def __init__(self):
                self.running = list(running)
                self.waiting = collections.deque(waiting)
                self.skipped_waiting = collections.deque(skipped)
                self.max_num_running_reqs = max_running

            def _schedule_prefill_only(self):
                # What the base scheduler would see, captured mid-call.
                seen['waiting'] = list(self.waiting)
                seen['skipped_waiting'] = list(self.skipped_waiting)
                seen['max_num_running_reqs'] = self.max_num_running_reqs
                seen['waiting_type'] = type(self.waiting)
                return 'scheduled'

        return one_in_flight_scheduler(Base)(), seen

    @staticmethod
    def request(is_chunk):
        return type('R', (), {'is_prefill_chunk': is_chunk})()

    def test_a_partial_in_flight_hides_both_queues_from_the_base_scheduler(self):
        """The contract's mechanism, docs/lever-n-plugin-contract-2026-09-19.md:123.
        Capping capacity was not enough - v49 computed headroom zero and the waiting
        loop admitted a second prompt anyway."""
        scheduler, seen = self.build([self.request(True)])
        self.assertEqual(scheduler._schedule_prefill_only(), 'scheduled')
        self.assertEqual(seen['waiting'], [], 'nothing new may join a partial')
        self.assertEqual(seen['skipped_waiting'], [])

    def test_the_queues_are_restored_afterwards(self):
        scheduler, _ = self.build([self.request(True)])
        scheduler._schedule_prefill_only()
        self.assertEqual(list(scheduler.waiting), ['w1', 'w2'])
        self.assertEqual(list(scheduler.skipped_waiting), ['s1'])
        self.assertEqual(scheduler.max_num_running_reqs, 4)

    def test_the_queues_are_restored_even_when_scheduling_raises(self):
        scheduler, _ = self.build([self.request(True)])
        def boom():
            raise RuntimeError('scheduling failed')
        type(scheduler).__mro__[1]._schedule_prefill_only = lambda self: boom()
        with self.assertRaises(RuntimeError):
            scheduler._schedule_prefill_only()
        self.assertEqual(list(scheduler.waiting), ['w1', 'w2'])
        self.assertEqual(scheduler.max_num_running_reqs, 4)

    def test_the_queue_type_is_preserved(self):
        """The base scheduler popleft()s its waiting queue, so a list would break it."""
        scheduler, seen = self.build([self.request(True)])
        scheduler._schedule_prefill_only()
        self.assertIs(seen['waiting_type'], collections.deque)

    def test_with_no_partial_the_queues_stay_visible_and_capacity_caps_to_one(self):
        """Hiding alone would admit nothing ever. With no partial in flight the fast
        path serves exactly one fresh prompt, which is the cap's job."""
        scheduler, seen = self.build([])
        scheduler._schedule_prefill_only()
        self.assertEqual(seen['waiting'], ['w1', 'w2'], 'a fresh prompt must be visible')
        self.assertEqual(seen['max_num_running_reqs'], 1, 'exactly one, not four')

    def test_decodes_are_added_back_so_the_plugin_subtraction_nets_out(self):
        """The plugin computes max(0, max_num_running_reqs - decodes), so the value
        written here has to carry the decodes it will remove."""
        scheduler, seen = self.build([self.request(False), self.request(False)])
        scheduler._schedule_prefill_only()
        self.assertEqual(seen['max_num_running_reqs'], 3, 'allowed 1 plus 2 decodes')
        self.assertEqual(seen['waiting'], ['w1', 'w2'], 'decodes are not partials')

    def test_a_partial_alongside_decodes_still_hides_the_queues(self):
        scheduler, seen = self.build([self.request(True), self.request(False)])
        scheduler._schedule_prefill_only()
        self.assertEqual(seen['waiting'], [])
        self.assertEqual(seen['max_num_running_reqs'], 2, 'allowed 1 plus 1 decode')

    def test_a_base_without_skipped_waiting_is_tolerated(self):
        """skipped_waiting is not guaranteed on every plugin version; a missing
        attribute must not turn a policy into an AttributeError."""
        class Base(object):
            def __init__(self):
                self.running = [SchedulerClassTests.request(True)]
                self.waiting = collections.deque(['w1'])
                self.max_num_running_reqs = 4
            def _schedule_prefill_only(self):
                return 'scheduled'
        scheduler = one_in_flight_scheduler(Base)()
        self.assertEqual(scheduler._schedule_prefill_only(), 'scheduled')
        self.assertEqual(list(scheduler.waiting), ['w1'])

