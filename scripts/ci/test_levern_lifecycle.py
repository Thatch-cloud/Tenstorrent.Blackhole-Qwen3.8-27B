"""Lever N at TP4: the worker lifecycle's part (serving_lifecycle under QWEN_FAST_LEVER_N=1 or QWEN_FAST_LEVERN_AUDIT=1).

The lifecycle announces each prefill step to the model route (levern_route.announce) for the one execute_model call and withdraws it in a finally, keeps
the resident engine resident across a step that wrote no decode slot, releases a split prompt's scratch owner when the request finishes or is aborted
between two steps, and refuses to attach a Lever N profile whose scheduler cap could not be installed. With neither flag set none of it exists:
`levern` is None and every path is the one test_serving_lifecycle holds. The fixtures are test_serving_lifecycle's."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import levern_route as route
from serving_lifecycle import FastServingLifecycle
import test_serving_lifecycle as base

ROUTE = {'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_ANY_REQUEST': '1'}
AUDIT = {'QWEN_FAST_LEVERN_AUDIT': '1'}


def helper():
    return base.LifecycleTests('test_a_first_chunk_short_of_the_prompt_is_admitted')


def clean_environ(extra):
    environ = {key: value for key, value in os.environ.items() if not key.startswith(('QWEN_FAST_LEVER', 'QWEN_FAST_ANY'))}
    environ.update(extra)
    return patch.dict(os.environ, environ, clear=True)


def built(extra, chunked=True, prompt=4096, chunk=2048):
    """(lifecycle, worker, capture, scheduled, model, continuation(chunk, computed)) with the environment `extra` in force at construction."""
    owner = helper()
    with clean_environ(extra), patch.object(FastServingLifecycle, '_install_admission', return_value='TTScheduler'):
        if chunked:
            lifecycle, worker, bridge, capture, build, scheduled, decode = owner.chunked_fixture(chunk=chunk, prompt=prompt)
        else:
            lifecycle, worker, bridge, capture, build, scheduled, decode = owner.fixture()
    model = SimpleNamespace()
    lifecycle.runner.model = SimpleNamespace(model=[model])
    capture.position = prompt

    def continuation(size, computed):
        step = owner.continuation(scheduled, size, computed)
        step.scheduled_cached_reqs.num_computed_tokens = [computed]
        return step

    return lifecycle, worker, capture, scheduled, model, continuation


class OffTests(unittest.TestCase):
    def test_without_the_flags_there_is_no_route_and_nothing_is_announced(self):
        lifecycle, worker, capture, scheduled, model, continuation = built({})
        self.assertIsNone(lifecycle.levern)
        worker.execute_model(scheduled)
        worker.execute_model(continuation(2048, 2048))
        self.assertEqual(vars(model), {})

    def test_the_flag_set_to_zero_is_off(self):
        lifecycle, *_ = built({'QWEN_FAST_LEVER_N': '0', 'QWEN_FAST_LEVERN_AUDIT': '0'})
        self.assertIsNone(lifecycle.levern)

    def test_a_malformed_flag_is_refused_at_construction(self):
        with self.assertRaises(ValueError):
            built({'QWEN_FAST_LEVER_N': 'yes'})
        with self.assertRaises(ValueError):
            built({'QWEN_FAST_LEVERN_AUDIT': '2'})

    def test_the_whole_prompt_path_is_unchanged_with_no_flag(self):
        lifecycle, worker, capture, scheduled, model, _ = built({}, chunked=False)
        with patch('serving_lifecycle.note_prefill') as displace:
            worker.execute_model(scheduled)
        displace.assert_called_once_with()
        capture.capture.assert_called_once()


class AnnouncementTests(unittest.TestCase):
    def test_the_first_step_of_a_split_prompt_is_announced_and_displaces_nothing(self):
        lifecycle, worker, capture, scheduled, model, continuation = built(ROUTE)
        seen = []
        capture.segment.side_effect, original = None, capture.segment.side_effect

        def segment():
            seen.append(vars(model).get(route.STEP_ATTR))
            return original()

        capture.segment.side_effect = segment
        with patch('serving_lifecycle.note_prefill') as displace, patch('serving_lifecycle.note_fixture_writer') as epoch:
            worker.execute_model(scheduled)
        self.assertEqual(seen, [route.Step('request', 0, 2048, 4096)])
        displace.assert_not_called()
        epoch.assert_called_once_with('prefill')
        self.assertNotIn(route.STEP_ATTR, vars(model), 'withdrawn after the call')

    def test_a_continuation_is_announced_with_its_start_end_and_the_prompt(self):
        lifecycle, worker, capture, scheduled, model, continuation = built(ROUTE, prompt=8192)
        worker.execute_model(scheduled)
        seen = []
        original = capture.segment.side_effect

        def segment():
            seen.append(vars(model).get(route.STEP_ATTR))
            return original()

        capture.segment.side_effect = segment
        worker.execute_model(continuation(2048, 2048))
        worker.execute_model(continuation(4096, 4096))
        self.assertEqual(seen, [route.Step('request', 2048, 4096, 8192), route.Step('request', 4096, 8192, 8192)])
        self.assertNotIn(route.STEP_ATTR, vars(model))

    def test_a_whole_prompt_is_announced_too_and_displaces_as_before(self):
        lifecycle, worker, capture, scheduled, model, _ = built(ROUTE, chunked=False)
        seen = []
        original = capture.capture.side_effect

        def entered():
            seen.append(vars(model).get(route.STEP_ATTR))
            return original()

        capture.capture.side_effect = entered
        with patch('serving_lifecycle.note_prefill') as displace:
            worker.execute_model(scheduled)
        self.assertEqual(seen, [route.Step('request', 0, 4096, 4096)])
        displace.assert_called_once_with()

    def test_the_announcement_is_withdrawn_when_the_step_fails(self):
        lifecycle, worker, capture, scheduled, model, continuation = built(ROUTE)
        worker.model_runner.execute_model.side_effect = RuntimeError('device went away')
        with self.assertRaises(RuntimeError):
            worker.execute_model(scheduled)
        self.assertNotIn(route.STEP_ATTR, vars(model))
        self.assertTrue(lifecycle.failed)

    def test_the_audit_alone_announces_the_whole_prompt_for_the_digests(self):
        lifecycle, worker, capture, scheduled, model, _ = built(AUDIT, chunked=False)
        self.assertIsNotNone(lifecycle.levern)
        seen = []
        original = capture.capture.side_effect

        def entered():
            seen.append(vars(model).get(route.STEP_ATTR))
            return original()

        capture.capture.side_effect = entered
        worker.execute_model(scheduled)
        self.assertEqual(seen, [route.Step('request', 0, 4096, 4096)])


class DisplacementTests(unittest.TestCase):
    def chunks(self, wrote):
        lifecycle, worker, capture, scheduled, model, continuation = built(ROUTE, prompt=6144)
        capture.finish_at = 3
        capture.segment_slot = None
        capture.segment_wrote_slot = None
        original = capture.segment.side_effect
        from contextlib import contextmanager

        @contextmanager
        def segment():
            capture.segment_slot = None
            capture.segment_wrote_slot = None
            with original() as value:
                capture.segment_slot = 0
                capture.segment_wrote_slot = wrote[capture.segments - 1]
                yield value

        capture.segment.side_effect = segment
        return lifecycle, worker, capture, scheduled, continuation

    def test_an_intermediate_step_that_wrote_no_slot_leaves_the_resident_engine_alone(self):
        lifecycle, worker, capture, scheduled, continuation = self.chunks([False, False, True])
        displaced = []
        with patch('serving_lifecycle.note_prefill', side_effect=lambda: displaced.append(capture.segments)):
            worker.execute_model(scheduled)
            worker.execute_model(continuation(2048, 2048))
            self.assertEqual(displaced, [], 'two steps that wrote nothing displaced nothing, though both nominally name slot 0')
            worker.execute_model(continuation(2048, 4096))
        self.assertEqual(displaced, [3], 'the final step wrote slot 0 and displaces once')
        self.assertTrue(lifecycle.prefill_pending)

    def test_a_step_that_reports_a_write_or_nothing_displaces_as_before(self):
        for reported in (True, None):
            with self.subTest(reported=reported):
                lifecycle, worker, capture, scheduled, continuation = self.chunks([reported] * 3)
                displaced = []
                with patch('serving_lifecycle.note_prefill', side_effect=lambda: displaced.append(1)):
                    worker.execute_model(scheduled)
                    worker.execute_model(continuation(2048, 2048))
                self.assertEqual(len(displaced), 1, 'admission does not displace under Lever N; the slot-0 continuation does')


class ReleaseTests(unittest.TestCase):
    def test_a_prompt_aborted_between_two_steps_releases_its_scratch_owner(self):
        lifecycle, worker, capture, scheduled, model, continuation = built(ROUTE, prompt=8192)
        worker.execute_model(scheduled)
        vars(model)[route.OWNER_ATTR] = ('request', 2048)
        gone = continuation(2048, 2048)
        gone.finished_req_ids = {'request'}
        gone.total_num_scheduled_tokens = 0
        worker.execute_model(gone)
        self.assertNotIn(route.OWNER_ATTR, vars(model))
        self.assertIsNone(lifecycle.request_id)
        capture.close.assert_called()

    def test_another_requests_finish_leaves_the_owner(self):
        lifecycle, worker, capture, scheduled, model, continuation = built(ROUTE, prompt=8192)
        worker.execute_model(scheduled)
        vars(model)[route.OWNER_ATTR] = ('request', 2048)
        other = continuation(2048, 2048)
        other.finished_req_ids = {'someone-else'}
        worker.execute_model(other)
        self.assertEqual(vars(model)[route.OWNER_ATTR], ('request', 2048))


class AttachTests(unittest.TestCase):
    def test_a_lever_n_profile_whose_scheduler_cap_could_not_be_installed_is_refused(self):
        owner = helper()
        with clean_environ(ROUTE), patch.object(FastServingLifecycle, '_install_admission', return_value=None):
            with self.assertRaises(ValueError) as caught:
                owner.fixture()
        self.assertIn('Lever N needs the one-fresh-prefill cap', str(caught.exception))

    def test_the_audit_alone_needs_no_chunk_cap(self):
        owner = helper()
        with clean_environ(AUDIT), patch.object(FastServingLifecycle, '_install_admission', return_value=None):
            lifecycle, *_ = owner.fixture()
        self.assertIsNotNone(lifecycle.levern)


if __name__ == '__main__':
    unittest.main()
