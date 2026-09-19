"""The one-in-flight cap, checked without a scheduler.

The real proof is probe_prefill_serialisation running the subclass against the
plugin's own scheduler. What is checked here is the arithmetic that makes the cap
correct, because it is easy to get wrong in a way that still looks plausible: the
plugin subtracts the resident decodes from whatever it reads, so a naive cap would
starve prefill entirely whenever any request was decoding.
"""

from types import SimpleNamespace
import unittest

from serving_one_in_flight import effective_capacity, install, one_in_flight_scheduler


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
                    self.assertLessEqual(capacity, partials + 1, where)
                    self.assertEqual(capacity, min(partials + 1, max(0, configured - decodes)), where)

    def test_one_slot_survives_whenever_the_configuration_can_afford_it(self):
        for decodes in range(0, 6):
            for partials in range(0, 4):
                self.assertEqual(effective_capacity(decodes + partials + 1, decodes, partials),
                                 partials + 1, 'decodes=%d partials=%d' % (decodes, partials))

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
        self.assertEqual(calls, [2], 'the partial continues and one fresh prompt may join')

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

    def test_install_refuses_to_replace_an_existing_choice(self):
        config = SimpleNamespace(scheduler_config=SimpleNamespace(scheduler_cls='someone_elses'))
        with self.assertRaises(ValueError):
            install(config, scheduler='marker')


if __name__ == '__main__':
    unittest.main()
