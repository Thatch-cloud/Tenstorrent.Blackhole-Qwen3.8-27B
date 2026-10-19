"""The scheduler simulation (sched_sim.py, docs/tp4-sched-1.md section 6): it drives the shipped Alternator, its baseline reproduces the production evidence it is calibrated
to, and the comparisons the note prints keep their direction. Seeded, so every number here is a fixed number; the bounds are loose enough that a reasonable change of a model
constant does not trip them and tight enough that a policy that stopped doing what the note says does.

    nice python3 -B -m unittest test_sched_sim      (from scripts/ci)
"""

import unittest

import levern_policy as policy
import sched_sim as sim

SEEDS, HOURS = 3, 1.0


def run(label_env=None, section='production', **changes):
    env = dict(sim.PRODUCTION)
    env.update(label_env or {})
    return sim.run(env, sim.BASES[section].replace(**changes), seeds=SEEDS, hours=HOURS)


class MachineryTests(unittest.TestCase):
    def test_the_shipped_alternator_decides(self):
        machine = sim.Sim(sim.PRODUCTION, sim.CALIBRATED, seed=1, hours=0.01)
        self.assertIsInstance(machine.alt, policy.Alternator)
        self.assertEqual((machine.cfg.step, machine.cfg.solo, machine.cfg.share), (2048, 16384, 0.5))
        self.assertEqual((machine.merged.ttft_s, machine.merged.max_gap_s, machine.merged.park), (180, 8, 'host'))

    def test_the_new_flags_reach_the_alternator_through_the_parsers(self):
        env = dict(sim.PRODUCTION, QWEN_FAST_LEVERN_OWED_CAP_MS='1500', QWEN_FAST_LEVERN_SHORT_SHARE='0.65', QWEN_FAST_LEVERN_SHORT_WAIT_S='5', QWEN_FAST_LEVERN_ADAPTIVE='1')
        machine = sim.Sim(env, sim.CALIBRATED, seed=1, hours=0.01)
        self.assertEqual(machine.alt.sched, policy.Sched(1500, 0.65, 5))
        self.assertIsNotNone(machine.alt.adaptive)
        with self.assertRaises(ValueError):
            sim.Sim(dict(sim.PRODUCTION, QWEN_FAST_LEVERN_SHORT_SHARE='1.5'), sim.CALIBRATED)

    def test_a_run_is_deterministic_and_the_device_time_adds_up(self):
        one = sim.Sim(sim.PRODUCTION, sim.CALIBRATED, seed=3, hours=0.5).run()
        two = sim.Sim(sim.PRODUCTION, sim.CALIBRATED, seed=3, hours=0.5).run()
        self.assertEqual(one, two)
        self.assertAlmostEqual(one['prefill_share'] + one['decode_share'] + one['other_share'], 1.0, places=6)
        other_seed = sim.Sim(sim.PRODUCTION, sim.CALIBRATED, seed=4, hours=0.5).run()
        self.assertNotEqual((one['turns'], one['ttft_p50']), (other_seed['turns'], other_seed['ttft_p50']), 'a different seed is a different run')

    def test_every_section_of_the_comparison_runs(self):
        results = sim.compare(seeds=1, hours=0.25)
        self.assertEqual(set(results), {'production', 'quiet', 'lone'})
        for rows in results.values():
            for label, row in rows:
                self.assertGreater(row['turns'], 0, label)
        text = sim.tables(results)
        self.assertIn('| production (ship-4 pacing) |', text)
        self.assertIn('LEAN admission', text)
        self.assertIn('**Calibration of the baseline**', '**Calibration of the baseline**' + sim.calibration_table(results['production'][0][1]))


class CalibrationTests(unittest.TestCase):
    """The baseline reproduces the production evidence of the task within the tolerances the note states."""

    def test_the_wall_time_shares_are_the_production_ones(self):
        base = run()
        self.assertAlmostEqual(base['prefill_share'], 0.58, delta=0.05)
        self.assertAlmostEqual(base['decode_share'], 0.36, delta=0.06)

    def test_the_per_seat_rates_at_five_to_seven_live_users_are_the_production_ones(self):
        base = run()
        for live in (5, 6, 7):
            self.assertIn(live, base['per_seat'])
            self.assertAlmostEqual(base['per_seat'][live], 15.5, delta=6.0, msg=str(live))

    def test_the_returning_turn_waits_a_few_seconds(self):
        base = run()
        self.assertGreater(base['ttft_p50'], 1.5)
        self.assertLess(base['ttft_p50'], 4.0)
        self.assertLess(base['ttft_p90'], 10.0)


class DirectionTests(unittest.TestCase):
    """What each lever does in the model: the claims of the note's section 6, as inequalities."""

    @classmethod
    def setUpClass(cls):
        cls.base = run()

    def test_lean_admission_is_the_one_that_returns_device_time(self):
        lean = run(lean_saves_ms=sim.LEAN_SAVES_MS)
        self.assertLess(lean['prefill_share'], self.base['prefill_share'] - 0.02)
        self.assertGreater(lean['seat_rate'], self.base['seat_rate'] * 1.04)
        self.assertGreater(lean['turns_per_min'], self.base['turns_per_min'] * 1.04, 'work leaves the device, so more turns finish')
        self.assertLess(lean['ttft_p90'], self.base['ttft_p90'])

    def test_the_short_share_buys_decoder_rate_with_the_returning_turns_wait(self):
        shared = run({'QWEN_FAST_LEVERN_SHORT_SHARE': '0.65'})
        high = lambda row: sum(row['per_seat'][k] for k in (5, 6, 7)) / 3          # noqa: E731
        self.assertGreater(high(shared), high(self.base) * 1.10)
        self.assertGreater(shared['ttft_p90'], self.base['ttft_p90'] * 1.25)
        self.assertLess(shared['turns_per_min'], self.base['turns_per_min'], 'a work-conserving system does not make delay disappear: fewer turns finish')

    def test_the_guard_takes_back_part_of_that_wait(self):
        shared = run({'QWEN_FAST_LEVERN_SHORT_SHARE': '0.65'})
        guarded = run({'QWEN_FAST_LEVERN_SHORT_SHARE': '0.65', 'QWEN_FAST_LEVERN_SHORT_WAIT_S': '5'})
        self.assertLess(guarded['ttft_p90'], shared['ttft_p90'] * 0.95)
        self.assertGreater(guarded['seat_rate'], self.base['seat_rate'] * 1.0)

    def test_the_owed_cap_costs_the_returning_turns_nothing(self):
        capped = run({'QWEN_FAST_LEVERN_OWED_CAP_MS': '1500'})
        self.assertLess(capped['ttft_p90'], self.base['ttft_p90'] * 1.1)
        self.assertGreater(capped['seat_rate'], self.base['seat_rate'] * 0.98)

    def test_adaptive_shortens_the_cold_prompt_of_the_quiet_hours(self):
        quiet = run(section='quiet')
        adaptive = run({'QWEN_FAST_LEVERN_ADAPTIVE': '1'}, section='quiet')
        self.assertLess(adaptive['cold_ttft_p50'], quiet['cold_ttft_p50'] * 0.85, 'the estimate is -31..-44% for a cold long prompt with at most two decoders')
        self.assertLess(adaptive['decode_share'], quiet['decode_share'], 'and the decoders pay for it')

    def test_the_padded_round_doubles_a_lone_users_rate(self):
        lone = run(section='lone')
        padded = run(section='lone', np1=True)
        self.assertGreater(padded['seat_rate'], lone['seat_rate'] * 1.5)

    def test_the_solo_step_changes_nothing_that_matters_in_the_production_mix(self):
        solo = run({'QWEN_FAST_LEVERN_SOLO_STEP_TOKENS': '8192'})
        self.assertAlmostEqual(solo['seat_rate'], self.base['seat_rate'], delta=self.base['seat_rate'] * 0.04)
        self.assertAlmostEqual(solo['turns_per_min'], self.base['turns_per_min'], delta=self.base['turns_per_min'] * 0.04)


if __name__ == '__main__':
    unittest.main()
