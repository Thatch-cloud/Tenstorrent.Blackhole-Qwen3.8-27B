"""The sequential step: correct, ordered, and honest about what it costs."""

import os
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from serving_sequential_step import describe, sequential_packed_step, step_request
import trace_census


def entry(request_id, stepped):
    ticket = SimpleNamespace(request_id=request_id)

    def step(name, *, cancelled):
        stepped.append((name, cancelled()))
        return SimpleNamespace(request_id=name, token_ids=[1, 2])

    return dict(request_id=request_id, ticket=ticket,
                request=SimpleNamespace(step=step))


class SequentialStepTests(unittest.TestCase):
    def test_every_request_steps_in_the_schedulers_order(self):
        stepped = []
        outputs = sequential_packed_step([entry('B', stepped), entry('A', stepped)],
                                         cancelled=lambda: False)
        self.assertEqual([name for name, _ in stepped], ['B', 'A'],
                         'entry order is the scheduler order, not creation order')
        self.assertEqual([output.request_id for output in outputs], ['B', 'A'])

    def test_cancellation_reaches_every_request(self):
        stepped = []
        sequential_packed_step([entry('A', stepped), entry('B', stepped)],
                               cancelled=lambda: True)
        self.assertEqual([flag for _, flag in stepped], [True, True])

    def test_a_ticket_for_another_request_is_refused(self):
        stepped = []
        broken = entry('A', stepped)
        broken['ticket'] = SimpleNamespace(request_id='someone-else')
        with self.assertRaises(ValueError):
            sequential_packed_step([broken], cancelled=lambda: False)
        self.assertEqual(stepped, [], 'refused before touching the device')

    def test_an_output_for_the_wrong_request_is_refused(self):
        stepped = []
        wrong = entry('A', stepped)
        wrong['request'] = SimpleNamespace(
            step=lambda name, *, cancelled: SimpleNamespace(request_id='B', token_ids=[1]))
        with self.assertRaises(ValueError):
            sequential_packed_step([wrong], cancelled=lambda: False)

    def test_an_empty_block_is_refused(self):
        with self.assertRaises(ValueError):
            sequential_packed_step([], cancelled=lambda: False)

    def test_it_reports_that_it_is_not_the_batched_verifier(self):
        """A benchmark that reads this must not mistake it for the goal."""
        cost = describe()
        self.assertIs(cost['batched'], False)
        self.assertEqual(cost['weight_passes_per_round'], 'one per user')


HERE = os.path.dirname(os.path.abspath(__file__))
DIAGNOSTICS = (trace_census.SEQ_DEADLINE_FLAG, trace_census.STAGE_LOG_FLAG, trace_census.CCL_LOG_FLAG)


class WatchedStepTests(unittest.TestCase):
    """QWEN_FAST_SEQ_DEADLINE_S and QWEN_FAST_SEQ_STAGE_LOG (trace_census) around step_request."""

    def setUp(self):
        stack = patch.dict(os.environ)
        stack.start()
        self.addCleanup(stack.stop)
        for name in DIAGNOSTICS:
            os.environ.pop(name, None)
        self.lines = []
        logger = patch.object(trace_census, 'log', self.lines.append)
        logger.start()
        self.addCleanup(logger.stop)

    def one(self, rows=4):
        item = entry('request-A', [])
        item['ticket'].tokens = list(range(rows))
        return item

    def test_with_the_flags_unset_faulthandler_is_never_called_and_nothing_is_logged(self):
        item = self.one()
        with patch('faulthandler.dump_traceback_later') as arm, patch('faulthandler.cancel_dump_traceback_later') as cancel:
            output = step_request(item['request'], item['ticket'], lambda: False)
        self.assertEqual(output.request_id, 'request-A')
        arm.assert_not_called()
        cancel.assert_not_called()
        self.assertEqual(self.lines, [])

    def test_the_deadline_arms_a_process_exiting_timer_around_the_step_and_cancels_it(self):
        os.environ[trace_census.SEQ_DEADLINE_FLAG] = '120'
        item = self.one()
        calls = []
        item['request'].step = lambda name, *, cancelled: calls.append(('step',)) or SimpleNamespace(request_id=name)
        with patch('faulthandler.dump_traceback_later', side_effect=lambda *a, **k: calls.append(('arm', a, k))), \
                patch('faulthandler.cancel_dump_traceback_later', side_effect=lambda: calls.append(('cancel',))):
            step_request(item['request'], item['ticket'], lambda: False)
        self.assertEqual([call[0] for call in calls], ['arm', 'step', 'cancel'])
        self.assertEqual(calls[0][1], (120.0,))
        self.assertIs(calls[0][2]['exit'], True)
        self.assertIs(calls[0][2]['file'], sys.stderr)

    def test_a_step_that_raises_still_cancels_the_timer(self):
        os.environ[trace_census.SEQ_DEADLINE_FLAG] = '5'
        item = self.one()
        item['request'].step = lambda name, *, cancelled: (_ for _ in ()).throw(RuntimeError('refused'))
        with patch('faulthandler.dump_traceback_later'), patch('faulthandler.cancel_dump_traceback_later') as cancel:
            with self.assertRaises(RuntimeError):
                step_request(item['request'], item['ticket'], lambda: False)
        cancel.assert_called_once()

    def test_a_deadline_that_is_not_a_positive_number_is_refused(self):
        for text in ('0', '-3', 'soon', 'inf'):
            with self.subTest(text=text):
                os.environ[trace_census.SEQ_DEADLINE_FLAG] = text
                item = self.one()
                with self.assertRaises(ValueError):
                    step_request(item['request'], item['ticket'], lambda: False)

    def test_the_stage_log_brackets_the_step(self):
        os.environ[trace_census.STAGE_LOG_FLAG] = '1'
        item = self.one(rows=16)
        step_request(item['request'], item['ticket'], lambda: False)
        self.assertEqual(self.lines, ['[SEQ-STAGE] request=request-A rows=16 stage=step begin',
                                      '[SEQ-STAGE] request=request-A rows=16 stage=step end'])

    def test_the_collective_handle_state_rides_every_stage_line(self):
        os.environ[trace_census.STAGE_LOG_FLAG] = '1'
        os.environ[trace_census.CCL_LOG_FLAG] = '1'
        collectives = SimpleNamespace(gather_idx=3, barrier_idx=1, handles=[0, 2], name='ccl')
        with patch.dict(trace_census.COLLECTIVES, {'shared': collectives, 'model': SimpleNamespace(reduce_idx=5)}):
            trace_census.stage('r', 4, 'validate')
        self.assertEqual(self.lines, ['[SEQ-STAGE] request=r rows=4 stage=validate begin '
                                      'ccl{shared{gather_idx=3 barrier_idx=1 handles=0,2} model{reduce_idx=5}}'])

    def test_collective_state_too_long_for_the_stage_line_follows_it_one_object_per_line(self):
        os.environ[trace_census.STAGE_LOG_FLAG] = '1'
        os.environ[trace_census.CCL_LOG_FLAG] = '1'
        wide = lambda: SimpleNamespace(**{'handle_index_%d' % index: index for index in range(6)})
        with patch.dict(trace_census.COLLECTIVES, {'shared': wide(), 'model': wide(), 'sampler': wide()}):
            trace_census.stage('r', 4, 'validate')
        self.assertEqual(len(self.lines), 4)
        self.assertEqual(self.lines[0], '[SEQ-STAGE] request=r rows=4 stage=validate begin')
        for line, name in zip(self.lines[1:], ('shared', 'model', 'sampler')):
            self.assertTrue(line.startswith('[SEQ-STAGE] request=r rows=4 stage=validate begin ccl{%s{handle_index_0=0' % name), line)

    def run_script(self, body, **flags):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=HERE, **flags)
        started = time.perf_counter()
        done = subprocess.run([sys.executable, '-B', '-c', textwrap.dedent(body)], cwd=HERE, env=env,
                              capture_output=True, text=True, timeout=60)
        return done, time.perf_counter() - started

    def test_a_stalled_step_exits_the_process_naming_the_blocked_frame_and_the_last_stage(self):
        done, elapsed = self.run_script("""
            import re
            from types import SimpleNamespace
            import serving_sequential_step

            def stalled_in_step(name, *, cancelled):
                re.match('(a+)+$', 'a' * 60 + 'b')      # a C call that holds the GIL, as a wedged driver call does

            request = SimpleNamespace(step=stalled_in_step)
            ticket = SimpleNamespace(request_id='request-stall', tokens=[1, 2, 3, 4])
            serving_sequential_step.step_request(request, ticket, lambda: False)
            print('RETURNED')
        """, QWEN_FAST_SEQ_DEADLINE_S='1', QWEN_FAST_SEQ_STAGE_LOG='1')
        output = done.stdout + done.stderr
        self.assertNotEqual(done.returncode, 0)
        self.assertNotIn('RETURNED', output)
        self.assertLess(elapsed, 30)
        self.assertIn('stalled_in_step', output)
        stage_lines = [line for line in output.splitlines() if '[SEQ-STAGE]' in line]
        self.assertTrue(stage_lines[-1].endswith('request=request-stall rows=4 stage=step begin'), stage_lines)

    def test_a_step_that_returns_in_time_is_left_alone_after_twice_the_deadline(self):
        done, _ = self.run_script("""
            import time
            from types import SimpleNamespace
            import serving_sequential_step

            request = SimpleNamespace(step=lambda name, *, cancelled: SimpleNamespace(request_id=name))
            ticket = SimpleNamespace(request_id='request-fast', tokens=[1])
            serving_sequential_step.step_request(request, ticket, lambda: False)
            time.sleep(2.2)
            print('SURVIVED')
        """, QWEN_FAST_SEQ_DEADLINE_S='1')
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn('SURVIVED', done.stdout)


class PublicationStageTests(unittest.TestCase):
    def test_each_publication_stage_logs_its_begin_line_under_the_flag_only(self):
        import serving_packed_step

        stages = serving_packed_step.PUBLISH_STAGES
        for flag, expected in (('1', [(name, 'begin') for name in stages]), (None, [])):
            with self.subTest(flag=flag), patch.dict(os.environ):
                os.environ.pop(trace_census.STAGE_LOG_FLAG, None)
                if flag:
                    os.environ[trace_census.STAGE_LOG_FLAG] = flag
                lines, sink = [], {}
                runtime = SimpleNamespace(session=SimpleNamespace(request_id='request-P',
                                                                  pending=SimpleNamespace(tokens=[1] * 16)))
                with patch.object(trace_census, 'log', lines.append):
                    restore = serving_packed_step.install_stage_timer(runtime, sink)
                    for name in stages:
                        with runtime.publication_stage(name, 4):
                            pass
                    restore()
                self.assertEqual(sorted(sink), sorted(stages))
                self.assertEqual([(line.split()[-2][len('stage='):], line.split()[-1]) for line in lines], expected)
                self.assertTrue(all('request=request-P rows=16 ' in line for line in lines))


if __name__ == '__main__':
    unittest.main()
