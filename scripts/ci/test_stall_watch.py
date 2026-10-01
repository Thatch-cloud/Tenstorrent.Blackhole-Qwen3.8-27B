"""stall_watch: the watch's flags, the triage runner, the sidecar's order of events (stacks first, then triage, only then the end of
the process) and the in-process scope; one test runs the real sidecar process against a real sleeping 'serving process'."""

import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stall_watch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


class DeadlineTests(unittest.TestCase):
    def test_off_without_the_flag(self):
        self.assertFalse(stall_watch.enabled({}))
        for kind in stall_watch.KINDS:
            self.assertIsNone(stall_watch.deadline(kind, {}))
            self.assertIsNone(stall_watch.deadline(kind, {stall_watch.DEADLINE_FLAG: ''}))

    def test_a_step_takes_the_flag_and_a_build_and_a_prefill_their_own_defaults_and_overrides(self):
        environ = {stall_watch.DEADLINE_FLAG: '120'}
        self.assertEqual([stall_watch.deadline(kind, environ) for kind in stall_watch.KINDS],
                         [120.0, stall_watch.DEFAULT_BUILD_S, stall_watch.DEFAULT_PREFILL_S])
        environ.update({stall_watch.BUILD_FLAG: '30', stall_watch.PREFILL_FLAG: '45.5'})
        self.assertEqual([stall_watch.deadline(kind, environ) for kind in stall_watch.KINDS], [120.0, 30.0, 45.5])
        self.assertTrue(stall_watch.enabled(environ))

    def test_bad_values_are_configuration_errors(self):
        for bad in ('0', '-3', 'soon', 'inf', 'nan'):
            with self.assertRaises(ValueError, msg=bad):
                stall_watch.deadline('step', {stall_watch.DEADLINE_FLAG: bad})
            with self.assertRaises(ValueError, msg=bad):
                stall_watch.deadline('build', {stall_watch.DEADLINE_FLAG: '5', stall_watch.BUILD_FLAG: bad})
        with self.assertRaises(ValueError):
            stall_watch.deadline('round', {stall_watch.DEADLINE_FLAG: '5'})


class TriageTests(unittest.TestCase):
    def test_the_six_pinned_tools_in_order(self):
        self.assertEqual(stall_watch.TOOLS, ('dump_running_operations', 'dump_callstacks', 'check_binary_integrity',
                                             'check_noc_status', 'dump_fast_dispatch', 'check_eth_status'))
        self.assertEqual(stall_watch.triage_tools({}), stall_watch.TOOLS)
        self.assertEqual(stall_watch.triage_tools({stall_watch.TOOLS_FLAG: 'a, b'}), ('a', 'b'))

    def test_two_command_lines_are_tried_by_default_and_a_template_replaces_them(self):
        commands = stall_watch.triage_commands('dump_callstacks', {}, python='py')
        self.assertEqual(commands, [['py', '/opt/tt-metal/tools/triage/dump_callstacks.py'],
                                    ['py', '/opt/tt-metal/tools/triage/triage.py', '--run=dump_callstacks']])
        environ = {stall_watch.ROOT_FLAG: '/r', stall_watch.COMMAND_FLAG: '{python} {root}/x.py {tool}'}
        self.assertEqual(stall_watch.triage_commands('t', environ, python='py'), [['py', '/r/x.py', 't']])

    def runner(self, outcomes, calls):
        def run(argv, **kwargs):
            calls.append((argv, kwargs.get('timeout')))
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return SimpleNamespace(returncode=outcome[0], stdout=outcome[1])
        return run

    def test_a_tool_that_fails_the_first_way_is_tried_the_second_and_its_output_is_logged_with_its_name(self):
        lines, calls = [], []
        ok = stall_watch.run_tool('dump_callstacks', {stall_watch.TIMEOUT_FLAG: '7'},
                                  self.runner([(2, 'usage: error\n'), (0, 'core 1\ncore 2\n')], calls), lines.append)
        self.assertTrue(ok)
        self.assertEqual([timeout for _, timeout in calls], [7.0, 7.0])
        self.assertIn('[TRIAGE dump_callstacks] usage: error', lines)
        self.assertIn('[TRIAGE dump_callstacks] core 2', lines)
        self.assertEqual(lines[-1], '[TRIAGE dump_callstacks] exit 0')

    def test_a_timeout_a_missing_tool_and_a_second_failure_are_all_logged_and_the_tool_is_failed(self):
        lines = []
        runner = self.runner([subprocess.TimeoutExpired('x', 1, output='partial\n'), FileNotFoundError('no python')], [])
        self.assertFalse(stall_watch.run_tool('t', {}, runner, lines.append))
        self.assertTrue(any('timed out after' in line for line in lines))
        self.assertIn('[TRIAGE t] partial', lines)
        self.assertTrue(any('could not run: FileNotFoundError' in line for line in lines))

    def test_long_output_is_cut_and_says_so(self):
        lines = []
        stall_watch.run_tool('t', {stall_watch.COMMAND_FLAG: '{python} {tool}'},
                             self.runner([(0, ''.join('row %d\n' % row for row in range(500)))], []), lines.append)
        self.assertEqual(len([line for line in lines if ' row ' in line]), stall_watch.OUTPUT_LINES)
        self.assertTrue(any('340 more lines' in line for line in lines))

    def test_every_tool_runs_even_when_one_blows_up_and_nothing_raises(self):
        seen = []

        def runner(argv, **kwargs):
            seen.append(argv[1])
            if 'check_noc_status' in argv[1]:
                raise RuntimeError('boom')
            return SimpleNamespace(returncode=0, stdout='')

        results = stall_watch.run_triage({stall_watch.COMMAND_FLAG: '{python} {tool}'}, runner, lambda line: None)
        self.assertEqual(list(results), list(stall_watch.TOOLS))
        self.assertEqual([tool for tool, ok in results.items() if not ok], ['check_noc_status'])
        self.assertEqual(seen, list(stall_watch.TOOLS))


class TriageReadinessTests(unittest.TestCase):
    def test_a_missing_directory_tool_or_ttexalens_is_named(self):
        import tempfile
        have, lack = (lambda name: object()), (lambda name: None)
        with tempfile.TemporaryDirectory() as root:
            environ = {stall_watch.ROOT_FLAG: root}
            problems = stall_watch.triage_problems(environ, lack)
            self.assertEqual(len(problems), 2)
            self.assertIn('missing from', problems[0])
            self.assertIn('ttexalens', problems[1])
            with open(os.path.join(root, 'triage.py'), 'w') as handle:
                handle.write('')
            self.assertEqual(stall_watch.triage_problems(environ, have), [])
            self.assertEqual(len(stall_watch.triage_problems(environ, lack)), 1)
        self.assertIn('does not exist', stall_watch.triage_problems({stall_watch.ROOT_FLAG: root}, have)[0])

    def test_a_stall_says_plainly_that_the_triage_is_unavailable(self):
        log = []
        runner = lambda argv, **kwargs: SimpleNamespace(returncode=1, stdout='')
        stall_watch.stalled((0.0, 'packed round'), 1, {stall_watch.ROOT_FLAG: os.path.join(os.sep, 'no', 'such', 'dir'),
                                                       stall_watch.KILL_FLAG: '0'}, runner, log.append, lambda pid: None)
        line = [entry for entry in log if entry.startswith('[TRIAGE-CHECK]')]
        self.assertEqual(len(line), 1)
        self.assertIn('UNAVAILABLE', line[0])
        self.assertLess(log.index(line[0]), [i for i, entry in enumerate(log) if 'triage finished' in entry][0])


class SidecarTests(unittest.TestCase):
    def feed(self, *lines, then=None):
        """A line source that yields `lines` then blocks until `then` is set (or ends at once without it)."""
        def generate():
            for line in lines:
                yield line + '\n'
            if then is not None:
                then.wait(5)
        return generate()

    def events(self, lines, environ, release=None, expect='stalled'):
        log, order = [], []

        def runner(argv, **kwargs):
            order.append(('triage', argv[1]))
            return SimpleNamespace(returncode=0, stdout='')

        def logger(message):
            log.append(message)
            if message.startswith('[STALL]'):
                order.append(('stall', message[:40]))

        killed = []
        result = stall_watch.sidecar(self.feed(*lines, then=release), 4242, dict({stall_watch.COMMAND_FLAG: '{python} {tool}'}, **environ),
                                     runner, logger, lambda pid: (killed.append(pid), order.append(('kill', pid))))
        self.assertEqual(result, expect)
        return log, order, killed

    def test_a_stall_logs_then_triages_every_tool_and_only_then_ends_the_process(self):
        release = threading.Event()
        self.addCleanup(release.set)
        log, order, killed = self.events(['arm %.3f packed round users=4' % (time.time() + 0.15)], {}, release)
        kinds = [kind for kind, _ in order]
        self.assertEqual(kinds.index('stall'), 0)
        triages = [index for index, kind in enumerate(kinds) if kind == 'triage']
        self.assertEqual(len(triages), len(stall_watch.TOOLS))
        self.assertEqual(kinds[-1], 'kill', 'the process is ended after every triage tool has run')
        self.assertLess(max(triages), kinds.index('kill'))
        self.assertEqual(killed, [4242])
        self.assertIn('packed round users=4', log[0])
        self.assertTrue(any('ending the serving process' in line for line in log))

    def test_kill_off_leaves_the_process_alone_after_the_triage(self):
        release = threading.Event()
        self.addCleanup(release.set)
        log, order, killed = self.events(['arm %.3f build' % (time.time() + 0.1)], {stall_watch.KILL_FLAG: '0'}, release)
        self.assertEqual(killed, [])
        self.assertTrue(any('left hung for inspection' in line for line in log))
        self.assertEqual(len([1 for kind, _ in order if kind == 'triage']), len(stall_watch.TOOLS))

    def test_a_disarm_before_the_deadline_is_no_stall_and_the_end_of_input_ends_the_sidecar(self):
        log, order, killed = self.events(['arm %.3f step' % (time.time() + 0.2), 'disarm'], {}, None, expect='eof')
        self.assertEqual((log, order, killed), ([], [], []))

    def test_quit_ends_it_and_a_bad_arm_line_is_ignored(self):
        log, order, killed = self.events(['arm soon x', 'quit'], {}, None, expect='quit')
        self.assertTrue(any('bad arm line' in line for line in log))
        self.assertEqual((order, killed), ([], []))


class FakeProcess:
    def __init__(self):
        self.stdin = SimpleNamespace(writes=[], write=lambda text: self.stdin.writes.append(text), flush=lambda: None,
                                     close=lambda: self.stdin.writes.append('<closed>'))


class ScopeTests(unittest.TestCase):
    def watch(self):
        made = []

        def spawn():
            made.append(FakeProcess())
            return made[-1]

        return stall_watch.Watch(spawn, clock=lambda: 1000.0), made

    def test_without_the_flag_nothing_is_spawned_and_the_scope_is_a_nullcontext(self):
        with patch.dict(os.environ):
            os.environ.pop(stall_watch.DEADLINE_FLAG, None)
            with patch.object(stall_watch.WATCH, 'start') as start, stall_watch.scope('step', 'x'):
                pass
        start.assert_not_called()

    def test_the_outer_scope_arms_the_sidecar_and_faulthandler_without_exit_and_the_inner_one_does_nothing(self):
        watch, made = self.watch()
        environ = {stall_watch.DEADLINE_FLAG: '120'}
        with patch('faulthandler.dump_traceback_later') as dump, patch('faulthandler.cancel_dump_traceback_later') as cancel:
            with watch.scope('step', 'packed round users=4', environ):
                with watch.scope('build', 'engine request=r', environ):
                    pass
                writes = list(made[0].stdin.writes)
            self.assertEqual(writes, ['arm 1120.000 step packed round users=4\n'])
            self.assertEqual(made[0].stdin.writes, ['arm 1120.000 step packed round users=4\n', 'disarm\n'])
        dump.assert_called_once()
        self.assertEqual(dump.call_args[0], (120.0,))
        self.assertIs(dump.call_args[1]['exit'], False, 'stacks are dumped without ending the process')
        cancel.assert_called_once()
        self.assertEqual(len(made), 1)
        self.assertEqual(watch.depth, 0)

    def test_the_build_scope_uses_its_own_deadline(self):
        watch, made = self.watch()
        with patch('faulthandler.dump_traceback_later') as dump, patch('faulthandler.cancel_dump_traceback_later'):
            with watch.scope('build', 'engine', {stall_watch.DEADLINE_FLAG: '120', stall_watch.BUILD_FLAG: '33'}):
                pass
        self.assertEqual(dump.call_args[0], (33.0,))
        self.assertEqual(made[0].stdin.writes[0], 'arm 1033.000 build engine\n')

    def test_a_scope_that_raises_still_disarms(self):
        watch, made = self.watch()
        with patch('faulthandler.dump_traceback_later'), patch('faulthandler.cancel_dump_traceback_later') as cancel:
            with self.assertRaises(KeyError):
                with watch.scope('step', 'x', {stall_watch.DEADLINE_FLAG: '5'}):
                    raise KeyError('x')
        self.assertEqual(made[0].stdin.writes[-1], 'disarm\n')
        cancel.assert_called_once()
        self.assertEqual(watch.depth, 0)

    def test_a_sidecar_that_cannot_start_leaves_the_stacks_and_the_work_running(self):
        def refuse():
            raise OSError('no fork')

        watch = stall_watch.Watch(refuse)
        with patch('faulthandler.dump_traceback_later') as dump, patch('faulthandler.cancel_dump_traceback_later'), \
                patch.object(stall_watch, 'say') as say:
            with watch.scope('step', 'x', {stall_watch.DEADLINE_FLAG: '5'}):
                result = 'work ran'
        self.assertEqual(result, 'work ran')
        dump.assert_called_once()
        self.assertTrue(any('sidecar unavailable' in call[0][0] for call in say.call_args_list))

    def test_a_dead_sidecar_pipe_is_dropped_quietly(self):
        watch, made = self.watch()

        def broken(text):
            raise BrokenPipeError

        with patch('faulthandler.dump_traceback_later'), patch('faulthandler.cancel_dump_traceback_later'):
            with watch.scope('step', 'x', {stall_watch.DEADLINE_FLAG: '5'}):
                made[0].stdin.write = broken
        self.assertFalse(watch.pipe)


class RealSidecarTests(unittest.TestCase):
    @unittest.skipIf(os.name == 'nt', 'os.kill on a Windows pid is refused for a Store-python stub; the serving process is Linux')
    def test_a_real_sidecar_triages_and_then_ends_a_real_stalled_process(self):
        victim = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
        self.addCleanup(lambda: victim.poll() is None and victim.kill())
        environ = dict(os.environ, **{stall_watch.COMMAND_FLAG: '{python} -c pass', stall_watch.TIMEOUT_FLAG: '30',
                                      stall_watch.TOOLS_FLAG: 'dump_callstacks,check_noc_status'})
        sidecar = subprocess.Popen([sys.executable, os.path.join(HERE, 'stall_watch.py'), '--sidecar', str(victim.pid)],
                                   stdin=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, env=environ)
        sidecar.stdin.write('arm %.3f engine build\n' % (time.time() + 0.5))
        sidecar.stdin.flush()
        victim.wait(timeout=60)
        sidecar.stdin.close()
        sidecar.wait(timeout=60)
        error = sidecar.stderr.read()
        sidecar.stderr.close()
        self.assertIsNotNone(victim.returncode)
        lines = error.splitlines()
        stall = next(index for index, line in enumerate(lines) if line.startswith('[STALL] engine build did not finish'))
        triage = [index for index, line in enumerate(lines) if line.startswith('[TRIAGE ') and line.endswith('exit 0')]
        ending = next(index for index, line in enumerate(lines) if 'ending the serving process' in line)
        self.assertEqual(len(triage), 2, error)
        self.assertTrue(stall < min(triage) and max(triage) < ending, error)


if __name__ == '__main__':
    unittest.main()
