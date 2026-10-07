"""stage1_judge_dry_run: re-judging archived artifacts with the committed rules (the short-window plan's tooling item 6).

The artifacts are synthetic (a directory of a few small files); the smoke check and the prefix arm judge are patched where a real artifact would be needed, so the tests
hold the dry run's own logic: how it finds a run's files, which profile it names, how the recorded and the re-judged outcomes classify, what it prints and returns.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_prefix_gate as prefix_gate  # noqa: E402
import c2_smoke_check  # noqa: E402
import stage1_judge_dry_run as dry  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = 'c2-packed-tp4-8x262k-'
PLAIN = R + 'ship-prefix'
CONTAINER = '[QWEN-C2] profile %s: vLLM argv ["--served-model-name"]\nother line\n'


def profiles():
    with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as handle:
        return json.load(handle)


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        handle.write(text)


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def smoke_run(self, name='run', profile=PLAIN, nested=False):
        root = os.path.join(self.tmp, name, 'c2-serving-1') if nested else os.path.join(self.tmp, name)
        write(os.path.join(root, 'smoke.log'), 'warmup {"value": 200}\n')
        write(os.path.join(root, 'container.log'), CONTAINER % profile)
        return os.path.join(self.tmp, name) if nested else root

    def prefix_run(self, name='prefix-run', plan='agent-turns', arm='agent-turns-prefix', verdict='FAIL', profile=PLAIN):
        root = os.path.join(self.tmp, name)
        write(os.path.join(root, 'prefix', 'c2-prefix-summary.json'), json.dumps(dict(
            image='img', profile=profile, baseline='none', plans=[plan], results={plan: dict(verdict=verdict, arms={arm: dict(verdict=verdict)})})))
        directory = os.path.join(root, 'prefix', arm)
        write(os.path.join(directory, 'arm.json'), json.dumps(dict(result=dict(verdict=verdict, stats=None, error=None), markers={})))
        write(os.path.join(directory, 'records.jsonl'), '')
        write(os.path.join(directory, 'pairs.json'), '[]')
        write(os.path.join(directory, 'events.json'), json.dumps(dict(events=[], phases={})))
        write(os.path.join(directory, 'server-final.log'), 'one line\ntwo lines\n')
        return root


class ParseTests(Case):
    def test_a_spec_is_label_target_and_an_optional_recorded_outcome(self):
        self.assertEqual(dry.parse_spec('v578=/tmp/x:FAIL'), ('v578', '/tmp/x', 'FAIL'))
        self.assertEqual(dry.parse_spec('v579=/tmp/x'), ('v579', '/tmp/x', None))
        self.assertEqual(dry.parse_spec('v1=D:/runs/37:PASS'), ('v1', 'D:/runs/37', 'PASS'))
        self.assertEqual(dry.parse_spec('v1=D:/runs/37'), ('v1', 'D:/runs/37', None))
        for bad in ('v578', '=x', 'v578='):
            with self.assertRaises(dry.DryRunError):
                dry.parse_spec(bad)

    def test_the_profile_is_the_one_the_container_log_names(self):
        self.assertEqual(dry.profile_of_container_log(CONTAINER % PLAIN), PLAIN)
        self.assertEqual(dry.profile_of_container_log('[QWEN-C2] profile %s: parser M armed\n[QWEN-C2] profile other: x' % (R + 'w1')), R + 'w1')
        self.assertIsNone(dry.profile_of_container_log('no contract line'))

    def test_classification(self):
        self.assertEqual(dry.classify('FAIL', 'PASS'), 'rule_only_failure')
        self.assertEqual(dry.classify('PASS', 'FAIL'), 'rule_regression')
        self.assertEqual(dry.classify('FAIL', 'FAIL'), 'still_failing')
        self.assertEqual(dry.classify('PASS', 'PASS'), 'agrees_pass')
        self.assertEqual(dry.classify(None, 'PASS'), 'unrecorded')


class RootTests(Case):
    def test_a_run_directory_is_found_directly_or_one_level_down(self):
        direct = self.smoke_run('a')
        self.assertEqual(dry.run_root(direct), direct)
        nested = self.smoke_run('b', nested=True)
        self.assertEqual(dry.run_root(nested), os.path.join(nested, 'c2-serving-1'))
        self.assertEqual(dry.kind_of(dry.run_root(nested)), 'smoke')
        self.assertEqual(dry.kind_of(dry.run_root(self.prefix_run())), 'prefix')

    def test_an_empty_or_ambiguous_directory_is_refused(self):
        os.makedirs(os.path.join(self.tmp, 'empty'))
        with self.assertRaises(dry.DryRunError):
            dry.run_root(os.path.join(self.tmp, 'empty'))
        for name in ('c2-serving-1', 'c2-serving-2'):
            os.makedirs(os.path.join(self.tmp, 'two', name))
        with self.assertRaises(dry.DryRunError):
            dry.run_root(os.path.join(self.tmp, 'two'))
        os.makedirs(os.path.join(self.tmp, 'half'))
        write(os.path.join(self.tmp, 'half', 'smoke.log'), 'x')
        with self.assertRaises(dry.DryRunError):
            dry.kind_of(os.path.join(self.tmp, 'half'))


class SmokeTests(Case):
    def test_the_smoke_check_runs_with_the_profile_the_container_names_and_its_problems_are_reported(self):
        root = self.smoke_run()
        seen = []

        def fake_main(argv):
            seen.append(argv)
            print('SMOKE_CHECK profile=x slide=on {}')
            print('SMOKE_CHECK FAILED: a rule that does not apply')
            return 1

        with mock.patch.object(c2_smoke_check, 'main', fake_main):
            result = dry.judge_smoke(root)
        self.assertEqual(seen[0][seen[0].index('--profile') + 1], PLAIN)
        self.assertEqual((result['rejudged'], result['problems'], result['profile']), ('FAIL', ['a rule that does not apply'], PLAIN))
        with mock.patch.object(c2_smoke_check, 'main', lambda argv: 0):
            self.assertEqual(dry.judge_smoke(root, profile=R + 'w1')['rejudged'], 'PASS')

    def test_an_unreadable_input_and_a_missing_profile_are_refused(self):
        root = self.smoke_run()
        with mock.patch.object(c2_smoke_check, 'main', lambda argv: 2):
            with self.assertRaises(dry.DryRunError):
                dry.judge_smoke(root)
        write(os.path.join(root, 'container.log'), 'no contract line\n')
        with self.assertRaises(dry.DryRunError):
            dry.judge_smoke(root)

    def test_the_real_smoke_check_rejects_an_empty_smoke_and_the_dry_run_reports_it_as_a_failure(self):
        root = self.smoke_run()
        result = dry.judge_run('v1', root, recorded='PASS')
        self.assertEqual((result['rejudged'], result['outcome']), ('FAIL', 'rule_regression'))
        self.assertTrue(result['problems'])


class PrefixTests(Case):
    def test_the_arm_spec_is_rebuilt_from_the_summary_and_judged_with_its_own_files(self):
        root = self.prefix_run()
        calls = []

        def fake_judge(arm, driver, scanned, stats, error, log_text=None):
            calls.append((arm['arm'], arm['served'], len(driver.records), driver.pairs, error, log_text))
            return dict(verdict='PASS', problems=[], not_exercised=[])

        with mock.patch.object(prefix_gate, 'judge_arm', fake_judge):
            result = dry.judge_prefix(root)
        self.assertEqual(calls, [('agent-turns-prefix', PLAIN, 0, [], None, 'one line\ntwo lines')])
        self.assertEqual((result['kind'], result['rejudged'], result['arms']['agent-turns-prefix']['recorded']), ('prefix', 'PASS', 'FAIL'))
        outcome = dry.judge_run('v580', root)
        self.assertEqual(outcome['recorded'], 'FAIL', 'a prefix run is recorded by its own summary')

    def test_a_still_failing_arm_keeps_its_problems_and_an_unknown_arm_is_refused(self):
        root = self.prefix_run()
        with mock.patch.object(prefix_gate, 'judge_arm', lambda *a, **k: dict(verdict='FAIL', problems=['a real problem'], not_exercised=[])):
            result = dry.judge_prefix(root)
        self.assertEqual((result['rejudged'], result['problems']), ('FAIL', ['agent-turns-prefix: a real problem']))
        write(os.path.join(root, 'prefix', 'stranger', 'arm.json'), '{}')
        with self.assertRaises(dry.DryRunError):
            dry.judge_prefix(root)

    def test_every_plan_of_a_summary_has_its_arms_rebuilt(self):
        specs = dry.arm_specs(dict(profile=PLAIN, baseline='none', plans=['agent-turns-prefix', 'levern-hit']), profiles())
        self.assertEqual(sorted(specs), ['agent-turns-prefix', 'levern-hit'])
        self.assertEqual([spec['served'] for spec in specs.values()], [PLAIN, PLAIN])


class RealChainTests(Case):
    """The prefix path with NOTHING patched: a fake-engine gate run writes its arm directories, and the dry run re-judges them with the real scan, resolve and judge_arm."""

    def gate_run(self):
        import test_c2_prefix_gate as tpg
        root = os.path.join(self.tmp, 'real')
        profile_file = os.path.join(self.tmp, 'profiles.json')
        write(profile_file, json.dumps(tpg.profiles()))
        harness = tpg.Harness()

        def factory(*args, **kwargs):
            runner = harness.runner(args[1])
            runner.log = kwargs.get('log', runner.log)
            return runner

        with tpg.fakes.burst_aware(lambda: tpg.CURRENT['engine']):
            code = prefix_gate.main(['--image', 'img', '--profiles', profile_file, '--results', os.path.join(root, 'prefix'), '--plan', 'bringup'],
                                    devices=['/a', '/b'], log=lambda text: None, runner_factory=factory, anchor=tpg.GOOD_ANCHOR)
        self.assertEqual(code, 0)
        return root, profile_file

    def test_the_real_scan_resolve_and_judge_chain_agrees_with_the_run_it_judges(self):
        root, profile_file = self.gate_run()
        result = dry.judge_prefix(root, profiles_path=profile_file)
        self.assertEqual(sorted(result['arms']), ['bringup-prefix', 'bringup-reference'])
        for name, arm in result['arms'].items():
            self.assertEqual(arm['rejudged'], arm['recorded'], name)
            self.assertEqual(arm['recorded_problems'], [], name)
        self.assertEqual(result['rejudged'], 'PASS')

    def test_a_recorded_failure_is_shown_with_its_recorded_problems(self):
        root, profile_file = self.gate_run()
        path = os.path.join(root, 'prefix', 'c2-prefix-summary.json')
        with open(path, encoding='utf-8') as handle:
            summary = json.load(handle)
        summary['results']['bringup']['arms']['bringup-prefix'].update(verdict='FAIL', problems=['an old rule that is gone'])
        write(path, json.dumps(summary))
        result = dry.judge_run('v580', root, profiles_path=profile_file)
        self.assertEqual(result['arms']['bringup-prefix']['recorded_problems'], ['an old rule that is gone'])
        self.assertEqual((result['recorded'], result['rejudged'], result['outcome']), ('FAIL', 'PASS', 'rule_only_failure'))


class RecordedAndPairTests(Case):
    def test_the_smoke_check_failed_lines_of_the_run_log_are_the_recorded_problems(self):
        log = '2026-10-07T01:02:03Z SMOKE_CHECK FAILED: ramp commit 51.7 ms, above 50 ms\n2026-10-07T01:02:04Z other line\n2026-10-07T01:02:05Z SMOKE_CHECK FAILED: HOSTGAP_LOG is not set\n'
        self.assertEqual(dry.recorded_problems_of(log), ['ramp commit 51.7 ms, above 50 ms', 'HOSTGAP_LOG is not set'])
        root = self.smoke_run()
        write(os.path.join(root, 'run.log'), log)
        with mock.patch.object(c2_smoke_check, 'main', lambda argv: 0):
            result = dry.judge_smoke(root)
        self.assertEqual(result['recorded_problems'], ['ramp commit 51.7 ms, above 50 ms', 'HOSTGAP_LOG is not set'])
        other = self.smoke_run('nolog')
        with mock.patch.object(c2_smoke_check, 'main', lambda argv: 0):
            self.assertIsNone(dry.judge_smoke(other)['recorded_problems'])

    def test_one_arm_of_a_pair_is_never_re_judged_alone(self):
        a, b = self.smoke_run('a'), self.smoke_run('b')
        lines = []
        with mock.patch.object(c2_smoke_check, 'main', lambda argv: 0):
            self.assertEqual(dry.main(['--run', 'A=%s:FAIL' % a, '--pair', 'A,B'], out=lines.append), 2)
            self.assertIn('one arm of an A/B pair is never re-judged alone', lines[-1])
            lines.clear()
            self.assertEqual(dry.main(['--run', 'A=%s:FAIL' % a, '--run', 'B=%s:FAIL' % b, '--pair', 'A,B'], out=lines.append), 0)
        pair = [line for line in lines if line.startswith('STAGE1_JUDGE_PAIR')]
        self.assertEqual(len(pair), 1)
        document = json.loads(pair[0][len('STAGE1_JUDGE_PAIR '):])
        self.assertEqual(document['outcomes'], ['rule_only_failure', 'rule_only_failure'])
        self.assertIn('commit', document['rules'])

    def test_the_rules_state_names_the_commit_and_the_judge_files_with_uncommitted_changes(self):
        state = dry.rules_state()
        self.assertEqual(set(state), {'commit', 'dirty'})
        self.assertIsNone(dry.rules_state(self.tmp)['commit'], 'a folder that is no repository is reported as unknown')


class MainTests(Case):
    def run_main(self, *argv):
        lines = []
        code = dry.main(list(argv), out=lines.append)
        return code, lines

    def test_a_run_prints_one_line_and_the_summary_names_the_rule_only_failures(self):
        failing, passing = self.smoke_run('a'), self.smoke_run('b')
        verdicts = iter([0, 0])
        with mock.patch.object(c2_smoke_check, 'main', lambda argv: next(verdicts)):
            code, lines = self.run_main('--run', 'v578=%s:FAIL' % failing, '--run', 'v579=%s:PASS' % passing, '--json', os.path.join(self.tmp, 'out.json'))
        self.assertEqual(code, 0)
        self.assertTrue(lines[0].startswith('STAGE1_JUDGE {'))
        self.assertEqual(json.loads(lines[0][len('STAGE1_JUDGE '):])['outcome'], 'rule_only_failure')
        self.assertEqual(lines[-1], 'STAGE1_JUDGE_SUMMARY runs=2 rule_only_failure=v578 rule_regression=none unreadable=0')
        with open(os.path.join(self.tmp, 'out.json'), encoding='utf-8') as handle:
            self.assertEqual(json.load(handle)['rule_only_failure'], ['v578'])

    def test_strict_exits_one_on_a_rule_regression_only(self):
        root = self.smoke_run()
        with mock.patch.object(c2_smoke_check, 'main', lambda argv: 1):
            self.assertEqual(self.run_main('--run', 'v1=%s:PASS' % root)[0], 0)
            self.assertEqual(self.run_main('--run', 'v1=%s:PASS' % root, '--strict')[0], 1)
            self.assertEqual(self.run_main('--run', 'v1=%s:FAIL' % root, '--strict')[0], 0)

    def test_an_unreadable_run_is_reported_and_exits_two(self):
        code, lines = self.run_main('--run', 'v1=%s' % os.path.join(self.tmp, 'missing'))
        self.assertEqual(code, 2)
        self.assertIn('"error"', lines[0])
        self.assertEqual(self.run_main()[0], 2)
        self.assertEqual(self.run_main('--run', 'nolabel')[0], 2)

    def test_download_calls_gh_run_download_into_a_directory_per_run(self):
        calls = []

        def fake_run(argv, **kwargs):
            calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, b'', b'')

        with mock.patch.object(subprocess, 'run', fake_run):
            target = dry.download(37238391881, self.tmp)
        self.assertEqual(calls[0][:4], ['gh', 'run', 'download', '37238391881'])
        self.assertEqual(target, os.path.join(self.tmp, '37238391881'))
        with mock.patch.object(subprocess, 'run', lambda argv, **k: subprocess.CompletedProcess(argv, 1, b'', b'no such run')):
            with self.assertRaises(dry.DryRunError):
                dry.download(1, self.tmp)


class HygieneTests(unittest.TestCase):
    def test_the_module_names_no_private_string_and_is_lf(self):
        import re

        banned = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|hom' + 'e/|zo' + r't\.|pass' + 'word|pl' + 'ink')
        for name in ('stage1_judge_dry_run.py', 'test_stage1_judge_dry_run.py'):
            with open(os.path.join(HERE, name), encoding='utf-8', newline='') as handle:
                text = handle.read()
            self.assertIsNone(banned.search(text), name)
            self.assertNotIn('\r', text, name)

    def test_the_test_is_allowlisted_in_the_cpu_workflow(self):
        root = os.path.dirname(os.path.dirname(HERE))
        with open(os.path.join(root, '.github', 'workflows', 'qwen-integration-cpu.yml'), encoding='utf-8') as handle:
            self.assertIn('test_stage1_judge_dry_run', handle.read())


if __name__ == '__main__':
    unittest.main()
