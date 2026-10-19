"""prefix_rejudge: a finished prefix-gate arm judged again from its saved directory, on CPU.

The directories come from the gate's own Runner against the fake sticky engine (test_c2_prefix_gate.run_sticky): server-final.log, records.jsonl, pairs.json, events.json and arm.json
exactly as a card run leaves them. What is held: an unchanged rule reproduces the saved verdict line for line, a rule's verdict follows the code (a damaged log fails, a stale
saved verdict is replaced), the arm is rebuilt from the plan and the profile the log names, the checkpoint pair comes back out of the log, the salt key of the run is never opened, and
the command line says what it did."""

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_prefix_gate as gate  # noqa: E402
import prefix_rejudge as rejudge  # noqa: E402
import test_c2_prefix_gate as gate_tests  # noqa: E402
import test_prefix_replay as fakes  # noqa: E402


class RejudgeTests(unittest.TestCase):
    PLAN = 'exactness-eager'

    def setUp(self):
        self.results = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.results, True)
        for name, value in (('CHAIN_HITS', (4200, 9000, 16500)), ('CHAIN_HITS_LONG', (24000,)), ('EVICT_LENGTHS', (12000, 24000)),
                            ('SHARED_AGENT_TARGETS', gate_tests.StickyRunnerTests.SHARED), ('SHARED_AGENT_MAX_TOKENS', 64)):
            patcher = mock.patch.object(gate_tests.replay, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = fakes.burst_aware(lambda: gate_tests.CURRENT['engine'])
        patcher.start()
        self.addCleanup(patcher.stop)
        self.document = gate_tests.sticky_profiles()
        result, _, _ = gate_tests.run_sticky(self.PLAN, self.results)
        self.saved = result['arms'][self.PLAN]
        self.arm_dir = os.path.join(self.results, self.PLAN)

    def test_the_gate_leaves_what_the_replay_reads(self):
        self.assertEqual(self.saved['verdict'], 'PASS', self.saved['lines'])
        for name in ('server-final.log', 'records.jsonl', 'pairs.json', 'events.json', 'arm.json', 'docker-run.json'):
            self.assertTrue(os.path.isfile(os.path.join(self.arm_dir, name)), name)

    def test_an_unchanged_rule_reproduces_the_saved_verdict_line_for_line(self):
        result, arm, saved = rejudge.rejudge(self.arm_dir, self.document, profile=None)
        self.assertEqual(result['verdict'], self.saved['verdict'])
        self.assertEqual(result['lines'], self.saved['lines'])
        self.assertEqual((result['problems'], result['not_exercised']), (self.saved['problems'], self.saved['not_exercised']))
        self.assertEqual((arm['arm'], arm['served'], arm['scenario']), (self.PLAN, 'c2-packed-prefix', 'exactness_eager'))
        self.assertEqual(saved['original']['verdict'], 'PASS')

    def test_the_profile_the_log_launched_is_the_one_rebuilt(self):
        saved = rejudge.read_arm_dir(self.arm_dir)
        scanned = rejudge.markers.scan(saved['lines'])
        self.assertEqual(rejudge.profile_of(scanned), 'c2-packed-prefix')
        self.assertIsNone(rejudge.profile_of({}))
        self.assertEqual(rejudge.find_plan(self.PLAN), self.PLAN)
        self.assertEqual(rejudge.find_plan('exact-resume'), 'exact-resume')
        self.assertEqual(rejudge.find_plan('exactness-eager', plan='exactness'), 'exactness')
        with self.assertRaises(rejudge.RejudgeError):
            rejudge.find_plan('no-such-arm')
        with self.assertRaises(rejudge.RejudgeError):
            rejudge.find_plan(self.PLAN, plan='no-such-plan')
        with self.assertRaises(rejudge.RejudgeError):
            rejudge.rebuild_arm(self.document, 'exactness', 'exactness-eager', 'not-a-profile')
        with self.assertRaises(rejudge.RejudgeError):
            rejudge.rebuild_arm(self.document, 'exactness', 'no-such-arm', 'c2-packed-prefix')

    def test_a_damaged_log_fails_the_replay_the_way_it_fails_the_gate(self):
        path = os.path.join(self.arm_dir, 'server-final.log')
        with open(path, encoding='utf-8') as handle:
            lines = handle.read().split('\n')
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(line for line in lines if 'install scheduler=' not in line))
        result, _, saved = rejudge.rejudge(self.arm_dir, self.document)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any('no "[PINDIAG] prefix: install" line' in problem for problem in result['problems']), result['problems'])
        self.assertEqual(saved['original']['verdict'], 'PASS', 'the saved verdict is only reported beside the new one')

    def test_the_log_is_split_on_newlines_alone(self):
        """str.splitlines also splits on form feeds and a few other characters, which would shift every line index of the records' log windows."""
        path = os.path.join(self.arm_dir, 'server.log')
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write('first\x0cstill the first\nsecond\n')
        self.assertEqual(rejudge.read_arm_dir(self.arm_dir)['lines'], ['first\x0cstill the first', 'second'])

    def test_the_checkpoint_pair_comes_back_out_of_the_log(self):
        revision = '0123456789abcdef0123456789abcdef01234567'
        lines = ['noise', '[QWEN-C2] checkpoint some/Repo-1.5@%s key=some--Repo-1.5--%s snapshot=/models/x' % (revision, revision), '[QWEN-C2] checkpoint other/Repo@%s ' % ('c' * 40)]
        self.assertEqual(rejudge.checkpoint_of(lines), ('some/Repo-1.5', revision))
        self.assertIsNone(rejudge.checkpoint_of(['[QWEN-C2] checkpoint some/Repo@short key=x', 'nothing']))
        seen = []
        with mock.patch.object(gate, 'judge_arm', side_effect=lambda *a, **k: seen.append(k) or dict(verdict='PASS', lines=[], problems=[], not_exercised=[])):
            rejudge.rejudge(self.arm_dir, self.document, model=('given/Repo', 'f' * 40))
            rejudge.rejudge(self.arm_dir, self.document)
        self.assertEqual(seen[0]['model'], ('given/Repo', 'f' * 40), 'an explicit pair wins')
        self.assertNotIn('model', seen[1], 'a log with no checkpoint line asks for none')

    def test_the_salt_key_of_the_run_is_never_opened(self):
        with open(os.path.join(self.arm_dir, 'salt.key'), 'w') as handle:
            handle.write('not for the replay\n')
        opened = []
        real_open = open

        def spy(path, *args, **kwargs):
            opened.append(str(path))
            return real_open(path, *args, **kwargs)

        with mock.patch('builtins.open', spy):
            rejudge.rejudge(self.arm_dir, self.document)
        self.assertTrue(opened)
        self.assertFalse([path for path in opened if path.endswith('salt.key')], opened)

    def test_a_directory_that_cannot_be_replayed_says_why(self):
        empty = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, empty, True)
        with self.assertRaises(rejudge.RejudgeError) as raised:
            rejudge.rejudge(empty, self.document)
        self.assertIn('neither server.log nor server-final.log', str(raised.exception))
        shutil.copy(os.path.join(self.arm_dir, 'server-final.log'), os.path.join(empty, 'server-final.log'))
        with self.assertRaises((rejudge.RejudgeError, OSError)):
            rejudge.rejudge(empty, self.document)         # no records.jsonl

    def run_main(self, *arguments):
        out = io.StringIO()
        with redirect_stdout(out):
            code = rejudge.main(list(arguments))
        return code, out.getvalue()

    def profiles_file(self):
        path = os.path.join(self.results, 'profiles.json')
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(self.document, handle)
        return path

    def test_the_command_line(self):
        out_json = os.path.join(self.results, 'verdict.json')
        code, text = self.run_main('--arm-dir', self.arm_dir, '--profiles', self.profiles_file(), '--json', out_json)
        self.assertEqual(code, 0, text)
        self.assertIn('[REJUDGE] saved verdict: PASS', text)
        self.assertIn('[REJUDGE] arm %s: PASS (0 problem(s), 0 not exercised)' % self.PLAN, text)
        self.assertIn('[REJUDGE] %s pairs: ' % self.PLAN, text)
        with open(out_json, encoding='utf-8') as handle:
            self.assertEqual(json.load(handle)['verdict'], 'PASS')
        code, text = self.run_main('--arm-dir', self.results + '-nowhere', '--profiles', self.profiles_file())
        self.assertEqual(code, 2)
        self.assertIn('cannot replay', text)
        code, text = self.run_main('--arm-dir', self.arm_dir, '--profiles', self.profiles_file(), '--profile', 'not-a-profile')
        self.assertEqual(code, 2, text)
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            self.run_main('--arm-dir', self.arm_dir, '--profiles', self.profiles_file(), '--model', 'no-revision')

    def test_a_failing_replay_exits_one_and_prints_its_problems(self):
        path = os.path.join(self.arm_dir, 'server-final.log')
        with open(path, encoding='utf-8') as handle:
            text = handle.read()
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(line for line in text.split('\n') if 'install scheduler=' not in line))
        code, text = self.run_main('--arm-dir', self.arm_dir, '--profiles', self.profiles_file())
        self.assertEqual(code, 1)
        self.assertIn('problem: no "[PINDIAG] prefix: install" line', text)
        self.assertIn(': FAIL (', text.splitlines()[-1])


if __name__ == '__main__':
    unittest.main()
