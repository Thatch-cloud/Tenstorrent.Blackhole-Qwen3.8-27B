"""serving/tp4-s2: the code the four-card serving window needs beyond its job templates, held on CPU.

E3  the churn gate at four cards: every four-card profile runs QWEN_FAST_QUAD_DRAFT=0, so no quad ever forms and the departures
    are judged on PAIR releases (lever_n_m3native_gate.proposal_releases, c2_serving_gate.run_churn); the quad rule is unchanged.
E4  the platform replay on all four cards (c2_platform_replay --cards quad, the workflow's replay step, the job parser).
E5  the two streamed smoke tests (stream_tool_call, stream_reasoning) and the smoke check's traffic-profile rules."""

import io
import json
import os
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_platform_replay as replay  # noqa: E402
import c2_serving_gate as driver  # noqa: E402
import c2_serving_job as job  # noqa: E402
import c2_smoke_check as smoke_check  # noqa: E402
import lever_n_m3native_gate as gate  # noqa: E402
import test_c2_platform_replay as replay_fixture  # noqa: E402
import test_c2_serving_gate_s2 as s2  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)
QUAD_PROFILE, PAIR_PROFILE = 'c2-packed-tp4', 'c2-packed'


def pair_round(number, pairs=((0, 1), (2, 3))):
    return 'INFO [PACKED-PROPOSE] round=%d pairs=%s propose_ms=%s' % (
        number, [list(pair) for pair in pairs], ["1.0"] * len(pairs))


def singles_round(number, slots=(2,)):
    return "INFO [PACKED-PROPOSE] singles round=%d slots=%s reasons=%s" % (number, list(slots), ["x"] * len(slots))


def quad_off_log(pair_departures=8, release=((0, 1), (2, 3)), others=3, singles=False, rounds=True):
    """A churn arm's log at four cards: `pair_departures` departures each after a pair round (the pairs formed), the detach's
    release line naming `release` (pairs=[...], quad=0), then `others` departures with no pair formed."""
    lines = [s2.prefill_point(110000, 1500.0)] + s2.engine_points()
    for index in range(pair_departures):
        lines += [s2.mask_pooled_line((0, 1), '1x1x32x2080'), s2.mask_pooled_line((2, 3), '1x1x32x2080'),
                  s2.pooled_outputs_line((0, 1), 32, 32), s2.pooled_outputs_line((2, 3), 32, 32)]
        if rounds:
            lines.append(pair_round(10 * index + 1))
        if singles:
            lines.append(singles_round(10 * index + 2))
        lines += [s2.hold_line(120000, 700e6, 1568.4e6, s2.engine_id(index + 4), 4), s2.decision_line(4)]
        pairs = release[index] if isinstance(release, list) else release
        if pairs is not None:
            lines.append(s2.released_line(0, pairs))
        lines.append(s2.execute_line(100.0 + index, 4, finished=[s2.engine_id(index)]))
        lines.append(s2.released_hold_line(120000, 1900e6, 1568.4e6, s2.engine_id(index + 4)))
    for index in range(pair_departures, pair_departures + others):
        lines.append(s2.released_line(0))
        lines.append(s2.execute_line(200.0 + index, 3, finished=[s2.engine_id(index)]))
    return s2.s2_log(extra=lines)


class ChurnAtFourCardsTests(unittest.TestCase):
    def test_a_four_card_profile_is_judged_on_pairs_and_the_pair_profile_on_quads(self):
        self.assertTrue(driver.quad_draft_off(PROFILES, QUAD_PROFILE))
        self.assertTrue(driver.quad_draft_off(PROFILES, 'c2-packed-tp4-gate'))
        self.assertFalse(driver.quad_draft_off(PROFILES, PAIR_PROFILE))

    def releases(self, **kwargs):
        return gate.proposal_releases(quad_off_log(**kwargs))

    def test_every_departure_after_a_pair_round_that_releases_a_pair_passes_the_count(self):
        releases = self.releases()
        self.assertEqual((releases['pair_rounds'], releases['pair_departures'], releases['pair_unreleased'],
                          releases['quad_rounds'], releases['departures']), (8, 8, 0, 0, 11))
        self.assertEqual(releases['pair_ambiguous'], 0)

    def test_a_departure_whose_release_names_no_pair_is_unreleased(self):
        both = ((0, 1), (2, 3))
        releases = self.releases(release=[both, both, None, both, both, both, both, both])
        self.assertEqual((releases['pair_unreleased'], releases['pair_departures']), (1, 8))
        self.assertIn('no release line', releases['pair_unreleased_steps'][0])
        stale = self.releases(release=())
        self.assertEqual(stale['pair_unreleased'], 11, 'a release line with no pair named frees nothing: every pair stays formed')
        self.assertIn('released no pair', stale['pair_unreleased_steps'][0])

    def test_a_round_with_a_single_makes_the_departure_ambiguous_never_a_pass_or_a_fail(self):
        releases = self.releases(singles=True, release=())
        self.assertEqual((releases['pair_departures'], releases['pair_unreleased']), (0, 0))
        self.assertGreaterEqual(releases['pair_ambiguous'], 8)

    def test_no_pair_round_means_nothing_to_judge(self):
        releases = self.releases(rounds=False, release=())
        self.assertEqual((releases['pair_rounds'], releases['pair_departures']), (0, 0))

    def churn(self, log_text, profile=QUAD_PROFILE, forced=None):
        lengths = list(driver.CHURN_LENGTHS)

        def report(n):
            one = s2.s2_arm_report(log_text, PAIR_PROFILE, env=((s2.AUDIT, '1'),), users=len(lengths), lengths=lengths,
                                   max_tokens=2304, finish='length')
            for stream, budget in zip(one['streams'], driver.CHURN_MAX_TOKENS):
                stream['completion_tokens'] = budget
            one['alive'] = True
            return one
        # The fixture's arm is the pair profile's report; what changes is which rule the gate applies to its log.
        patched = mock.patch.object(driver, 'quad_draft_off', return_value=forced if forced is not None else
                                    driver.quad_draft_off(PROFILES, profile))
        with patched:
            return s2.LifecycleMemoryChurnTests('run_plan').run_plan('churn', {'churn': report}, {'churn': log_text})

    def test_churn_at_four_cards_passes_on_pair_releases_and_prints_them(self):
        passed = self.churn(quad_off_log())
        self.assertEqual(passed['verdict'], 'PASS', passed.get('lines'))
        self.assertIn('pair', passed['lines'][0])
        self.assertEqual(passed['facts']['releases']['pair_departures'], 8)

    def test_churn_at_four_cards_fails_an_unreleased_pair_and_a_quad_round_on_a_profile_without_quads(self):
        failed = self.churn(quad_off_log(release=()))
        self.assertEqual(failed['verdict'], 'FAIL')
        self.assertIn('11 of 11 departures while a pair was formed', ' '.join(failed['s2_problems']))
        quads = quad_off_log().replace('INFO [PACKED-PROPOSE] round=1 ', 'INFO ' + s2.quad_line(1, built=1) + '\nINFO [PACKED-PROPOSE] round=1 ')
        self.assertEqual(self.churn(quads)['verdict'], 'FAIL', 'a quad round with the quad draft off')

    def test_churn_at_four_cards_with_no_pair_round_or_no_pair_departure_is_not_exercised_never_a_pass(self):
        self.assertEqual(self.churn(quad_off_log(pair_departures=0, others=11, rounds=False))['verdict'], 'NOT_EXERCISED')
        self.assertEqual(self.churn(quad_off_log(singles=True, release=()))['verdict'], 'NOT_EXERCISED',
                         'pair rounds formed, but every departure met a single: ambiguous, not exercised')
        self.assertEqual(self.churn(quad_off_log(pair_departures=4, others=1))['verdict'], 'NOT_EXERCISED',
                         'five departures for eight replacements: the seats did not churn')

    def test_the_quad_rule_is_unchanged_on_a_profile_with_the_quad_draft_on(self):
        ours = s2.LifecycleMemoryChurnTests('run_plan')
        lengths = list(driver.CHURN_LENGTHS)
        text = ours.churn_log()

        def report(n):
            one = s2.s2_arm_report(text, PAIR_PROFILE, env=((s2.AUDIT, '1'),), users=len(lengths), lengths=lengths,
                                   max_tokens=2304, finish='length')
            for stream, budget in zip(one['streams'], driver.CHURN_MAX_TOKENS):
                stream['completion_tokens'] = budget
            one['alive'] = True
            return one
        result = ours.run_plan('churn', {'churn': report}, {'churn': text})
        self.assertEqual(result['verdict'], 'PASS')
        self.assertIn('quad', result['lines'][0])
        self.assertEqual(result['facts']['releases']['quad_departures'], 8)


class PlatformReplayAtFourCardsTests(unittest.TestCase):
    BOARDS = ['/dev/tenstorrent/by-id/blackhole-%s' % name for name in ('AAAAAAAA', 'BBBBBBBB', 'CCCCCCCC', 'DDDDDDDD')]

    def test_quad_is_refused_without_a_profile_or_with_a_pair_profile_and_pair_refuses_a_four_card_one(self):
        self.assertIn('needs --profile', replay.cards_problem('quad', None))
        self.assertIn('opens the (1, 2) pair', replay.cards_problem('quad', PAIR_PROFILE))
        self.assertIn('opens the four-card (1, 4) mesh', replay.cards_problem('pair', QUAD_PROFILE))
        self.assertIsNone(replay.cards_problem('quad', QUAD_PROFILE))
        self.assertIsNone(replay.cards_problem('quad', 'general-tp4'))
        self.assertIsNone(replay.cards_problem('pair', PAIR_PROFILE))
        self.assertIsNone(replay.cards_problem('pair', None), 'the pair default is unchanged')
        self.assertIn('not in', replay.cards_problem('quad', 'no-such-profile'))
        self.assertIn('--cards must be', replay.cards_problem('triple', QUAD_PROFILE))

    def test_the_quad_copy_gets_four_device_nodes_and_the_profile_and_names_no_board(self):
        info, image = replay.inspect_source(replay_fixture.AGENT)
        with open(replay_fixture.AGENT, encoding='utf-8') as handle:
            recorded = json.load(handle)['argv']
        nodes = ['/dev/tenstorrent/%d' % index for index in range(4)]
        arguments = replay.run_arguments(info, 'img', 'copy', 8011, QUAD_PROFILE, devices=nodes,
                                         image_env=(), extra_env=replay.DEFAULT_ENV)
        devices = [arguments[index + 1] for index, token in enumerate(arguments) if token == '--device']
        self.assertEqual([device for device in devices if device.startswith('/dev/tenstorrent/')], nodes)
        self.assertEqual(sum(1 for device in devices if 'tenstorrent' in device), 4)
        self.assertIn('QWEN_C2_PROFILE=%s' % QUAD_PROFILE, arguments)
        self.assertNotIn('by-id', ' '.join(arguments), 'the four boards are resolved to nodes at run time, never named')
        self.assertTrue(recorded, 'the tracked pair record is the source')

    def test_the_quad_devices_are_the_gates_card_set_and_exactly_four(self):
        with tempfile.TemporaryDirectory() as root:
            for name in ('AAAAAAAA', 'BBBBBBBB', 'CCCCCCCC'):
                open(os.path.join(root, 'blackhole-%s' % name), 'w').close()
            with self.assertRaises(RuntimeError):
                replay.quad_devices(root)

    def test_the_startup_wait_defaults_to_the_agents_ceiling_at_four_cards(self):
        self.assertEqual((replay.PAIR_STARTUP_S, replay.QUAD_STARTUP_S), (600, 1020))

    @staticmethod
    def quad_log():
        return '\n'.join(['[QWEN-C2] mesh device P150x4 [1, 4]',
                          '[PINDIAG] extent replay engaged segments=4 flags=[0x23] mask=narrow',
                          '[PINDIAG] packed-any admission passed: K64j 152951c1 x1; CB1 1 CB2a 2 CB2b 3',
                          'parser M armed: stream re-chunking on'])

    def test_the_runtime_log_of_a_quad_replay_needs_the_mesh_the_extent_the_admission_and_parser_m(self):
        entry = PROFILES['profiles'][QUAD_PROFILE]
        steps = ['start', 'load']
        banner = 'serving Qwen/Qwen3.8-27B:tt on http://0.0.0.0:8000\nmodel reload: x\n'
        good = replay.runtime_log_verdict(banner + self.quad_log(), steps, entry)
        for problem in good['problems']:
            self.assertNotIn('mesh', problem)
            self.assertNotIn('extent', problem)
            self.assertNotIn('admission', problem)
        silent = replay.runtime_log_verdict(banner, steps, entry)
        text = ' '.join(silent['problems'])
        self.assertFalse(silent['ok'])
        self.assertIn('P150x4', text)
        self.assertIn('extent replay engaged', text)
        self.assertIn('packed-any admission passed:', text)
        self.assertIn('parser M armed', text)
        waived = replay.runtime_log_verdict(
            banner + self.quad_log() + '\n[PINDIAG] packed-any admission passed UNQUALIFIED: K64j', steps, entry)
        self.assertIn('UNQUALIFIED', ' '.join(waived['problems']))

    def test_a_pair_replays_runtime_log_rules_are_unchanged(self):
        steps = ['start', 'load']
        banner = 'serving Qwen/Qwen3.8-27B:tt on http://0.0.0.0:8000\nmodel reload: x\n'
        self.assertEqual(replay.runtime_log_verdict(banner, steps)['problems'], [])

    def test_main_refuses_a_pair_profile_under_quad_before_any_container(self):
        commands = []

        def run(command, timeout=None, check=True):
            commands.append(command)
            if command[:3] == ['docker', 'image', 'inspect']:
                return mock.Mock(returncode=0, stdout='[{"Config": {"Env": []}}]', stderr='')
            return mock.Mock(returncode=0, stdout='', stderr='')

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(replay, 'run', side_effect=run), \
                mock.patch.object(sys, 'argv', ['replay', '--source', replay_fixture.AGENT, '--image', 'zot/new@sha256:b',
                                                '--results', directory, '--cards', 'quad', '--profile', PAIR_PROFILE]), \
                redirect_stdout(io.StringIO()) as out:
            code = replay.main()
            with open(os.path.join(directory, 'platform-replay.json'), encoding='utf-8') as handle:
                recorded = json.load(handle)
        self.assertEqual(code, 1)
        self.assertFalse(any(command[:2] == ['docker', 'run'] for command in commands))
        self.assertFalse(recorded['passed'])
        self.assertIn('refused', out.getvalue())
        self.assertFalse(recorded['steps']['cards']['ok'])

    def test_main_under_quad_opens_the_resolved_nodes_with_the_quad_wait(self):
        ran, waits = [], []

        def run(command, timeout=None, check=True):
            ran.append(command)
            if command[:3] == ['docker', 'image', 'inspect']:
                return mock.Mock(returncode=0, stdout='[{"Config": {"Env": []}}]', stderr='')
            return mock.Mock(returncode=0, stdout='', stderr='')

        def wait(port, name, seconds):
            waits.append(seconds)
            return 'container exited'

        nodes = ['/dev/tenstorrent/%d' % index for index in range(4)]
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(replay, 'run', side_effect=run), \
                mock.patch.object(replay, 'quad_devices', return_value=nodes), mock.patch.object(replay.subprocess, 'run'), \
                mock.patch.object(replay, 'wait_http', side_effect=wait), \
                mock.patch.object(sys, 'argv', ['replay', '--source', replay_fixture.AGENT, '--image', 'zot/new@sha256:b',
                                                '--results', directory, '--cards', 'quad', '--profile', QUAD_PROFILE]), \
                redirect_stdout(io.StringIO()):
            replay.main()
            with open(os.path.join(directory, 'platform-replay.json'), encoding='utf-8') as handle:
                recorded = json.load(handle)
        docker_run = next(command for command in ran if command[:2] == ['docker', 'run'])
        self.assertEqual([docker_run[i + 1] for i, token in enumerate(docker_run) if token == '--device'
                          and 'tenstorrent' in docker_run[i + 1]], nodes)
        self.assertEqual(waits, [1020])
        self.assertEqual((recorded['steps']['cards']['devices'], recorded['steps']['cards']['startup_wait_s']), (4, 1020))


class JobAndWorkflowTests(unittest.TestCase):
    def values(self, **changes):
        base = dict(C2_CARDS='quad', C2_ACTIONS='reset replay', C2_IMAGE_TAG='tp4-serve-1', C2_PLATFORM_IMAGE='thin-layer-image-ref',
                    C2_REPLAY_PROFILE=QUAD_PROFILE)
        base.update(changes)
        return {key: value for key, value in base.items() if value is not None}

    def test_a_quad_job_may_replay_with_a_four_card_profile(self):
        outputs = job.read_job(self.values(), sorted(PROFILES['profiles']))
        self.assertEqual((outputs['cards'], outputs['actions'], outputs['replay_profile']), ('quad', 'reset replay', QUAD_PROFILE))
        self.assertIn('replay', job.QUAD_ACTIONS)

    def test_a_quad_replay_without_a_profile_or_with_a_pair_profile_is_refused(self):
        with self.assertRaises(job.JobError):
            job.read_job(self.values(C2_REPLAY_PROFILE=None), sorted(PROFILES['profiles']))
        with self.assertRaises(job.JobError):
            job.read_job(self.values(C2_REPLAY_PROFILE=PAIR_PROFILE), sorted(PROFILES['profiles']))

    def test_a_pair_job_may_not_replay_a_four_card_profile_and_cardm_and_priority_stay_pair_only(self):
        with self.assertRaises(job.JobError):
            job.read_job(self.values(C2_CARDS='pair'), sorted(PROFILES['profiles']))
        for action in ('cardm', 'priority'):
            self.assertNotIn(action, job.QUAD_ACTIONS)

    def test_the_workflow_replay_step_resolves_the_four_cards_and_passes_the_card_set(self):
        with open(os.path.join(HERE, '..', '..', '.github', 'workflows', 'qwen-c2-serving.yml'), encoding='utf-8') as handle:
            text = handle.read()
        step = text[text.index('python3 scripts/ci/c2_platform_replay.py') - 2200: text.index('python3 scripts/ci/c2_platform_replay.py') + 400]
        self.assertIn('CARDS: ${{ steps.job.outputs.cards }}', step)
        self.assertIn('. scripts/ci/card_set.sh', step)
        self.assertIn('card_set_nodes 60', step)
        self.assertIn('card_set_unheld', step)
        self.assertIn('--cards "$CARDS"', step)
        self.assertIn('agent-container-quad.json', step, 'the four-card record is used when one is tracked')
        self.assertIn('agent-container-36104200953.json', step, 'and the pair record stays the fallback')


class StreamedSmokeTests(unittest.TestCase):
    def test_the_two_streamed_tests_are_opt_in_and_named_in_the_smoke(self):
        with open(os.path.join(HERE, 'c2_serving_smoke.py'), encoding='utf-8') as handle:
            text = handle.read()
        self.assertIn("if ONLY and 'stream_tool_call' in ONLY:", text)
        self.assertIn("if ONLY and 'stream_reasoning' in ONLY:", text)
        self.assertEqual(smoke_check.STREAM_PARSER_TESTS, ('stream_tool_call', 'stream_reasoning'))

    def test_the_smoke_check_fails_a_streamed_test_that_is_not_ok_and_passes_one_that_is(self):
        ok = dict(ok=True, problems=[])
        self.assertEqual(smoke_check.smoke_problems(dict(stream_tool_call=ok, stream_reasoning=ok)), [])
        bad = smoke_check.smoke_problems(dict(stream_tool_call=dict(ok=False, problems=['<tool_call> leaked into the content']),
                                              stream_reasoning=dict(error='boom')))
        self.assertIn('stream_tool_call: <tool_call> leaked into the content', bad)
        self.assertIn('stream_reasoning: boom', bad)


class TrafficProfileSmokeCheckTests(unittest.TestCase):
    ENTRY = PROFILES['profiles'][QUAD_PROFILE]

    def test_the_traffic_profile_needs_the_admission_passed_and_parser_m_armed(self):
        good = '[PINDIAG] packed-any admission passed: K64j\nparser M armed: on\n'
        self.assertEqual(smoke_check.traffic_problems(good, self.ENTRY), [])
        none = smoke_check.traffic_problems('', self.ENTRY)
        self.assertEqual(len(none), 2)
        waived = smoke_check.traffic_problems(good + '[PINDIAG] packed-any admission passed UNQUALIFIED: x', self.ENTRY)
        self.assertEqual(len(waived), 1)
        self.assertIn('UNQUALIFIED', waived[0])
        refused = smoke_check.traffic_problems(good + '[PINDIAG] packed-any admission refused (1/2): x', self.ENTRY)
        self.assertEqual(len(refused), 1)

    def test_a_gate_only_profile_and_a_profile_without_the_extent_are_not_judged(self):
        self.assertEqual(smoke_check.traffic_problems('', PROFILES['profiles']['c2-packed-tp4-gate']), [])
        self.assertEqual(smoke_check.traffic_problems('', PROFILES['profiles']['general-tp4']), [])
        self.assertEqual(smoke_check.traffic_problems('', PROFILES['profiles']['general']), [])


if __name__ == '__main__':
    unittest.main()
