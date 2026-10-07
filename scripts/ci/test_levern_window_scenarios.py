"""The merged route's card drills (prefix_replay.scenario_levern_hit and scenario_levern_faults) and their judging (c2_prefix_gate.levern_findings),
on the sticky fake engine of test_prefix_replay: the sequence of requests, the in-container kill-switch flag, the timed scenario's place in the
plan table, and what the arm's judge says of the drills' events and the server log. The fake has no Lever N: the park and kill lines are fed by hand.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import c2_prefix_gate as pg  # noqa: E402
import c2_serving_job  # noqa: E402
import c2_smoke_check  # noqa: E402
import levern_policy  # noqa: E402
import prefix_replay as replay  # noqa: E402
import test_prefix_replay as base  # noqa: E402


class RecordingContainer(base.FakeContainer):
    """The fake container, with the Lever N drill flag kept apart from the prefix kill switch."""

    def __init__(self, engine):
        super().__init__(engine)
        self.flags = {}

    def exec_shell(self, script, timeout=60):
        self.scripts.append(script)
        for path in (replay.LEVERN_DRILL_OFF_PATH,):
            if path in script and 'echo' in script:
                self.flags[path] = script.split('echo ')[1].split(' >')[0]
                return 0, ''
            if path in script and 'rm -f' in script:
                if self.flags.get(path) == replay.KILL_SWITCH_OWNER:
                    self.flags.pop(path)
                return 0, ''
        return super().exec_shell(script, timeout)


class DrillTests(unittest.TestCase):
    def setUp(self):
        self.engine = None
        patcher = base.burst_aware(lambda: self.engine)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name, value in (('LEVERN_COLD_TOKENS', 16000), ('LEVERN_WHOLE_TOKENS', 9000), ('LEVERN_HIT_INPUT_TOKENS', 800),
                            ('LEVERN_HIT_AFTER_S', 0.0), ('LEVERN_OFF_AFTER_S', 0.0), ('LEVERN_ABORT_AFTER_S', 30.0)):
            patcher = mock.patch.object(replay, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def driver(self, arm):
        self.engine = base.sticky_engine()
        container = RecordingContainer(self.engine)
        driver = replay.Driver(self.engine, arm, base.CORPUS, log=base.FakeLog(self.engine), container=container, seed=1,
                               sleep=lambda seconds: None, say=lambda text: None, pods=lambda: 3, strict=True, sticky=True,
                               batch_invariant=True, prompt_limit=32768)
        return driver, container

    def test_the_timed_scenario_reads_a_hit_alone_then_a_hit_during_a_cold_long_prompt(self):
        driver, _ = self.driver('levern-hit')
        replay.scenario_levern_hit(driver)
        roles = [(record['role'], record['case']) for record in driver.records]
        self.assertIn(('hit', 'levern-hit-solo'), roles)
        self.assertIn(('hit', 'levern-hit-mid-cold'), roles)
        self.assertIn(('unsalted', 'levern-cold'), roles)
        event = driver.events['levern-hit']
        self.assertIsNotNone(event['solo_ttft_s'])
        self.assertIsNotNone(event['mid_ttft_s'])
        self.assertEqual(driver.pairs[0]['case'], 'levern-hit-first')       # the priming pair only: the timed reads are single requests
        self.assertEqual(len(driver.pairs), 1)

    def test_the_timed_hit_runs_beside_decoders_so_the_policys_step_is_the_2048_one_the_bound_describes(self):
        # without a decoder the policy takes its 16,384-token solo step and a hit arriving mid-step waits a whole one: the "solo + 1.5 s" pass line is the
        # 2,048-token step's, so the shape must carry the decoders (and its solo read is taken beside them)
        driver, _ = self.driver('levern-hit')
        replay.scenario_levern_hit(driver)
        decoders = [record for record in driver.records if record['case'] == 'levern-decoder']
        self.assertEqual(len(decoders), replay.LEVERN_HIT_DECODERS)
        self.assertEqual(replay.LEVERN_HIT_DECODERS, 6, 'six decoders, the cold arrival and the hit are the eight seats')
        self.assertEqual(driver.events['levern-hit']['decoders'], 6)
        source = Path(replay.__file__).read_text(encoding='utf-8')
        self.assertNotIn('its first step is 2,048 tokens, a few seconds', source, 'the comment that described the decoder-free shape as 2,048-token steps is gone')

    def test_the_fault_scenario_runs_the_three_drills_and_removes_its_flag_inside_the_container(self):
        driver, container = self.driver('levern-faults')
        replay.scenario_levern_faults(driver)
        for name in ('levern-hit-mid-cold', 'levern-park-abort', 'levern-off'):
            self.assertIn(name, driver.events)
        off = driver.events['levern-off']
        self.assertTrue(off['written'])
        self.assertTrue(off['removed'])
        self.assertEqual(container.flags, {}, 'the flag is gone')
        self.assertTrue(any(replay.LEVERN_DRILL_OFF_PATH in script for script in container.scripts))
        self.assertFalse(any('/models' in script and 'levern' in script for script in container.scripts), 'never on the hub mount')
        cases = [pair['case'] for pair in driver.pairs]
        for case in ('levern-hit-mid-cold', 'levern-hit-before-abort', 'levern-hit-after-abort', 'levern-hit-latched'):
            self.assertIn(case, cases)

    def test_the_flag_is_removed_even_when_a_drill_step_fails(self):
        driver, container = self.driver('levern-faults')
        original = replay.levern_cold

        def failing(driver_, label, tokens=None):
            if label == 'levern-cold-d':
                raise replay.HarnessError('boom')
            return original(driver_, label, tokens)

        with mock.patch.object(replay, 'levern_cold', failing):
            with self.assertRaises(replay.HarnessError):
                replay.scenario_levern_faults(driver)
        self.assertEqual(container.flags, {})

    def test_the_drills_are_registered_and_the_hit_scenario_is_timed(self):
        self.assertIn('levern_faults', replay.SCENARIOS)
        self.assertIn('levern_hit', replay.SCENARIOS)
        self.assertIn('levern_hit', pg.TIMED_SCENARIOS)
        self.assertNotIn('levern_faults', pg.TIMED_SCENARIOS)
        self.assertEqual(pg.LEVERN_DRILL_SCENARIOS, ('levern_faults',))
        for plan in ('levern-faults', 'levern-hit'):
            self.assertIn(plan, c2_serving_job.PREFIX_PLANS)
            self.assertIn(plan, pg.PLAN_ARMS)
            self.assertIn(plan, pg.PLANS)

    def test_the_drill_flag_lives_in_the_container_the_derived_profile_points_at(self):
        self.assertEqual(pg.DERIVED['leverndrill']['env']['QWEN_FAST_LEVERN_OFF_PATH'], replay.LEVERN_DRILL_OFF_PATH)
        self.assertTrue(replay.LEVERN_DRILL_OFF_PATH.startswith('/tmp/'))
        self.assertNotEqual(replay.LEVERN_DRILL_OFF_PATH, levern_policy.OFF_PATH)
        self.assertEqual(replay.LEVERN_OFF_POLL_S, levern_policy.OFF_POLL_S)

    def test_a_derived_drill_arm_is_planned_on_a_sticky_profile(self):
        import json

        document = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))
        arms = pg.plan_arms('levern-faults', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit', None, document)
        self.assertEqual([arm['arm'] for arm in arms], ['levern-faults'])
        self.assertEqual(arms[0]['derived']['profiles'][arms[0]['served']]['env']['QWEN_FAST_LEVERN_OFF_PATH'], replay.LEVERN_DRILL_OFF_PATH)
        timed = pg.plan_arms('levern-hit', 'c2-packed-tp4-8x262k-ship-prefix', None, document)
        self.assertEqual([arm['arm'] for arm in timed], ['levern-hit'])
        self.assertEqual(timed[0]['env'], (), 'a timed arm carries no extent audit')


class FindingsTests(unittest.TestCase):
    EVENTS = {'levern-hit-mid-cold': dict(cold_ok=True), 'levern-park-abort': dict(aborted='after 40s', abort_after_s=40.0),
              'levern-off': dict(written=True, removed=True, inflight_ok=True, whole_ok=True)}
    LOG = 'x\n[PINDIAG] lever N park out req=a at=2048 ms=1.0 bytes=1 parked_now=1\n' + levern_policy.KILL_LINE.format('/tmp/qwen-levern.off') + '\n'

    def test_a_clean_run_has_no_findings(self):
        problems, missing, lines = pg.levern_findings(dict(self.EVENTS), self.LOG)
        self.assertEqual((problems, missing), ([], []))
        self.assertEqual(len(lines), 1)

    def test_a_missing_drill_is_not_exercised_and_a_missing_park_is_too(self):
        events = dict(self.EVENTS)
        del events['levern-off']
        problems, missing, _ = pg.levern_findings(events, 'x\n')
        self.assertEqual(problems, [])
        self.assertTrue(any('levern-off' in text for text in missing))
        self.assertTrue(any('park' in text for text in missing))

    def test_a_flag_that_never_reached_the_server_or_a_prompt_that_did_not_finish_fails(self):
        events = dict(self.EVENTS)
        events['levern-off'] = dict(written=True, removed=False, inflight_ok=False, whole_ok=False)
        problems, _, _ = pg.levern_findings(events, 'x\n')
        text = ' '.join(problems)
        for word in ('not removed', 'did not finish', 'failed', 'kill-switch line'):
            self.assertIn(word, text)

    def test_the_engagement_rule_allows_the_kill_line_only_in_the_drill_arm(self):
        env = {'QWEN_FAST_LEVER_N': '1'}
        quiet = c2_smoke_check.lever_engagement_problems(env, self.LOG, drill=True)
        self.assertFalse([text for text in quiet if 'kill switch line' in text])
        loud = c2_smoke_check.lever_engagement_problems(env, self.LOG, drill=False)
        self.assertTrue([text for text in loud if 'kill switch line' in text])
        self.assertEqual(c2_smoke_check.lever_engagement_problems({}, self.LOG), [], 'a profile without the lever is not judged for it')


if __name__ == '__main__':
    unittest.main()
