"""Lever N with prefix reuse: the worker lifecycle's part of the merged route (serving_lifecycle with QWEN_FAST_LEVER_N=1 beside QWEN_PREFIX_REUSE=1 and
QWEN_FAST_STICKY_SESSIONS=1; docs/lever-n-prefix-merged-route.md sections 2, 4 (R3), 8 and 9).

What is held:
  R3         the sticky admission accepts a granted hit whose step is a step of the plan from R (levern_policy.valid_step), still refuses everything else
             fatally (no grant, a stale grant, a start off the boundary, above the window), and without the merged route still demands the whole rest;
  announce   every step names where its state comes from (COLD, CHECKPOINT for a hit's first step, SCRATCH for a continuation, PARKED for a resumed
             parked prefill) and whether it takes the scratch from a suspended one;
  capture    a split hit's capture counts from R, runs its steps as segments, and records the prefix route's slot;
  epoch      under the route scope an intermediate step that wrote no slot and left the batched buffers bound as before moves no epoch (a disjoint writer);
             the final step, a step that wrote a slot, one the route did not vouch for, and every step under the global scope bump as before;
  park       admission v2: a short prompt arriving while a long prefill is suspended parks its lifecycle state and is announced as taking the scratch;
             the long resumes through source PARKED, is released when it finishes or is aborted while parked, and with the flag off the old refusal stands."""

import os
import sys
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock
from unittest.mock import Mock, patch

import levern_policy
import levern_route as route
import qwen_prefix_registry as prefix_registry
import serving_lifecycle
import test_sticky_sessions as sticky
from serving_lifecycle import FastServingLifecycle

MERGED = {'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_ANY_REQUEST': '1', 'QWEN_PREFIX_REUSE': '1', 'QWEN_FAST_STICKY_SESSIONS': '1'}
PARK = dict(MERGED, QWEN_FAST_LEVERN_PARK='host')
ROUTE_SCOPE = dict(MERGED, QWEN_FAST_LEVERN_EPOCH_SCOPE='route')


def environment(extra):
    environ = {key: value for key, value in os.environ.items() if not key.startswith(('QWEN_FAST_LEVER', 'QWEN_FAST_ANY', 'QWEN_FAST_STICKY', 'QWEN_PREFIX'))}
    environ.update(extra)
    return patch.dict(os.environ, environ, clear=True)


class Built(object):
    """A lifecycle under `extra` with a merged fixture: the capture, the registry with one granted hit, the model the route announces to."""

    def __init__(self, case, extra=MERGED, prompt=8192, computed=0, chunk=None, grant=None, captures=1):
        self.case = case
        self.registry = prefix_registry.PrefixRegistry(budget_bytes=1 << 30)
        self.tokens = [1] * prompt
        if grant is not None:
            sticky.committed(self.registry, 'request', grant, self.tokens)
        with environment(extra), patch.object(FastServingLifecycle, '_install_admission', return_value='TTScheduler'):
            self.fix = sticky.LifecycleFixture().fixture(prompt=prompt, computed=computed, chunk=chunk)
        self.lifecycle, self.worker = self.fix.lifecycle, self.fix.worker
        self.model = SimpleNamespace()
        self.lifecycle.runner.model = SimpleNamespace(model=[self.model])
        self.capture = self.fix.capture
        self.capture.position = prompt
        self.capture.cursor = computed
        self.capture.complete = False
        self.capture.segments = 0
        self.captures = [self.capture]
        for _ in range(captures - 1):
            other = SimpleNamespace(capture=Mock(side_effect=lambda: nullcontext()), close=Mock(), segment=Mock(side_effect=lambda: nullcontext()),
                                    complete=False, records_prefix_route=True, position=0, cursor=0, segments=0)
            self.captures.append(other)
        for capture in self.captures:
            capture.finish_at = 10 ** 9
            capture.segment = Mock(side_effect=self.segment_of(capture))
        if captures > 1:
            self.fix.capture_factory.side_effect = list(self.captures)
        self.seen = []
        self.hook_announcements()
        case.addCleanup(sys.modules.pop, serving_lifecycle.PREFILL_GATE_KEY, None)

    @staticmethod
    def segment_of(capture):
        """PrefillWindowCapture.segment as far as the lifecycle sees it: a segment counts itself and completes the capture when its cursor reaches the end."""
        from contextlib import contextmanager

        @contextmanager
        def segment():
            capture.segments += 1
            yield capture
            if capture.segments >= capture.finish_at:
                capture.complete = True
        return segment

    def hook_announcements(self):
        """Record the announcement each segment/capture runs under (what the route would see for that execute_model call)."""
        for capture in self.captures:
            for name in ('segment', 'capture'):
                original = getattr(capture, name).side_effect

                def entered(original=original, capture=capture):
                    self.seen.append(vars(self.model).get(route.STEP_ATTR))
                    return original()

                getattr(capture, name).side_effect = entered

    def execute(self, scheduled=None):
        with sticky.registry_holder(sticky.holder_with(self.registry)):
            return self.worker.execute_model(self.fix.scheduled if scheduled is None else scheduled)

    def continuation(self, request_id, chunk, computed):
        return SimpleNamespace(finished_req_ids=set(), scheduled_new_reqs=[],
                               scheduled_cached_reqs=SimpleNamespace(req_ids=[request_id], num_computed_tokens=[computed]),
                               scheduled_spec_decode_tokens={}, num_scheduled_tokens={request_id: chunk}, total_num_scheduled_tokens=chunk)

    def new_request(self, request_id, prompt, chunk, computed=0):
        new = SimpleNamespace(req_id=request_id, prompt_token_ids=[2] * prompt, num_computed_tokens=computed, mm_features=[], prompt_embeds=None,
                              lora_request=None, sampling_params=self.fix.new.sampling_params)
        return SimpleNamespace(finished_req_ids=set(), scheduled_new_reqs=[new], scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
                               scheduled_spec_decode_tokens={}, num_scheduled_tokens={request_id: chunk}, total_num_scheduled_tokens=chunk)


class AdmissionTests(unittest.TestCase):
    """R3."""

    def fault(self, built, words):
        with self.assertRaisesRegex(ValueError, words):
            built.execute()
        self.assertTrue(built.lifecycle.failed)

    def test_a_split_hit_is_admitted_as_a_step_of_the_plan_and_announced_as_a_checkpoint_resume(self):
        built = Built(self, prompt=8192, computed=4096, chunk=2048, grant=4096)
        self.assertTrue(built.lifecycle.merged)
        self.assertIsNone(built.execute())
        built.fix.capture_factory.assert_called_once_with(8192, start=4096)
        built.capture.segment.assert_called_once_with()
        built.capture.capture.assert_not_called()
        self.assertEqual(built.seen, [route.Step('request', 4096, 6144, 8192, 'CHECKPOINT', False)])
        self.assertFalse(built.lifecycle.prefill_pending)
        self.assertEqual(built.lifecycle.request_id, 'request')

    def test_a_hit_that_takes_the_whole_rest_is_a_whole_step_and_is_announced_too(self):
        built = Built(self, prompt=8192, computed=4096, chunk=4096, grant=4096)
        built.execute()
        built.capture.capture.assert_called_once_with()
        self.assertEqual(built.seen, [route.Step('request', 4096, 8192, 8192, 'CHECKPOINT', False)])
        self.assertTrue(built.lifecycle.prefill_pending)

    def test_a_cold_first_step_is_announced_cold(self):
        built = Built(self, prompt=8192, computed=0, chunk=2048)
        built.execute()
        self.assertEqual(built.seen, [route.Step('request', 0, 2048, 8192, 'COLD', False)])

    def test_a_hit_whose_step_ends_past_s_last_is_an_engine_fault(self):
        # P=10241, S_last=8192: a step [6144, 8192 + 2048) would leave a one-token final step
        built = Built(self, prompt=10241, computed=6144, chunk=4096, grant=6144)
        self.fault(built, 'not a step of the plan')
        built.fix.runner_execute.assert_not_called()

    def test_a_hit_whose_step_does_not_end_on_the_boundary_is_an_engine_fault(self):
        built = Built(self, prompt=10241, computed=4096, chunk=1000, grant=4096)
        self.fault(built, 'not a step of the plan')

    def test_the_other_grant_clauses_stay_fatal(self):
        self.fault(Built(self, prompt=8192, computed=4096, chunk=2048, grant=None), 'no committed prefix grant')
        self.fault(Built(self, prompt=8192, computed=4096, chunk=2048, grant=2048), 'the committed grant is at Q=2048')
        self.fault(Built(self, prompt=8000, computed=6144, chunk=1856, grant=6144), 'R is above the prompt minus 2048')
        self.fault(Built(self, prompt=8192, computed=4032, chunk=4160, grant=None), 'R is not a 2048-token boundary')

    def test_without_the_merged_route_a_split_tail_is_still_the_old_fault(self):
        built = Built(self, extra={'QWEN_FAST_ANY_REQUEST': '1', 'QWEN_PREFIX_REUSE': '1', 'QWEN_FAST_STICKY_SESSIONS': '1'}, prompt=8192,
                      computed=4096, chunk=2048, grant=4096)
        self.assertFalse(built.lifecycle.merged)
        self.fault(built, 'the step carries 2048 of the 4096 tokens after R')

    def test_lever_n_without_prefix_reuse_is_not_merged_and_announces_without_a_source(self):
        built = Built(self, extra={'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_ANY_REQUEST': '1'}, prompt=8192, computed=0, chunk=2048)
        self.assertFalse(built.lifecycle.merged)
        built.execute()
        self.assertEqual(built.seen, [route.Step('request', 0, 2048, 8192)])
        self.assertIsNone(built.seen[0].source)

    def test_the_merged_flags_are_read_once(self):
        built = Built(self, extra=dict(PARK, QWEN_FAST_LEVERN_PARK_SLOTS='2', QWEN_FAST_LEVERN_EPOCH_SCOPE='route'))
        self.assertEqual(built.lifecycle.merged_cfg.park_slots, 2)
        self.assertEqual(built.lifecycle.merged_cfg.park, 'host')
        self.assertEqual(built.lifecycle.merged_cfg.epoch_scope, 'route')

    def test_prefix_reuse_without_sticky_sessions_is_refused_at_construction(self):
        with self.assertRaisesRegex(ValueError, 'QWEN_FAST_STICKY_SESSIONS'):
            Built(self, extra={'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_ANY_REQUEST': '1', 'QWEN_PREFIX_REUSE': '1'})


class ContinuationTests(unittest.TestCase):
    def test_a_continuation_is_announced_as_a_scratch_step_and_the_whole_chain_runs_as_segments(self):
        built = Built(self, prompt=8192, computed=4096, chunk=2048, grant=4096)
        built.capture.finish_at = 2
        built.execute()
        built.execute(built.continuation('request', 2048, 6144))
        self.assertEqual(built.seen, [route.Step('request', 4096, 6144, 8192, 'CHECKPOINT', False),
                                      route.Step('request', 6144, 8192, 8192, 'SCRATCH', False)])
        self.assertNotIn(route.STEP_ATTR, vars(built.model))

    def test_an_abort_between_two_steps_of_a_hit_releases_its_owner(self):
        built = Built(self, prompt=8192, computed=4096, chunk=2048, grant=4096)
        built.execute()
        vars(built.model)[route.OWNER_ATTR] = ('request', 6144)
        gone = built.continuation('request', 2048, 6144)
        gone.finished_req_ids, gone.total_num_scheduled_tokens = {'request'}, 0
        built.execute(gone)
        self.assertNotIn(route.OWNER_ATTR, vars(built.model))
        self.assertIsNone(built.lifecycle.request_id)
        built.capture.close.assert_called()


class PreemptionTests(unittest.TestCase):
    def test_s6_a_prefill_in_flight_that_vllm_preempted_releases_its_owner(self):
        built = Built(self, prompt=8192, computed=4096, chunk=2048, grant=4096)
        built.execute()
        vars(built.model)[route.OWNER_ATTR] = ('request', 6144)
        gone = built.continuation('request', 2048, 6144)
        gone.finished_req_ids, gone.preempted_req_ids, gone.total_num_scheduled_tokens = set(), {'request'}, 0
        built.execute(gone)
        self.assertNotIn(route.OWNER_ATTR, vars(built.model))
        self.assertIsNone(built.lifecycle.request_id)
        built.capture.close.assert_called()


class EpochTests(unittest.TestCase):
    """Section 8: the writer class an intermediate step takes."""

    def run_steps(self, extra, wrote, identity=True, engaged=True, final_at=3):
        built = Built(self, extra=extra, prompt=8192, computed=0, chunk=2048)
        capture = built.capture
        capture.finish_at = final_at
        capture.segment_wrote_slot = None
        capture.segment_slot = 0
        original = capture.segment.side_effect

        def segment():
            from contextlib import contextmanager

            @contextmanager
            def inner():
                with original() as value:
                    capture.segment_wrote_slot = wrote[capture.segments - 1]
                    yield value
            return inner()

        capture.segment.side_effect = segment
        vars(built.model)[route.IDENTITY_ATTR] = identity
        bumps, disjoint = [], []
        with patch('serving_lifecycle.note_fixture_writer', side_effect=bumps.append), \
                patch('serving_lifecycle.note_prefill', side_effect=lambda: bumps.append('displace')), \
                patch('verify_prestage.disjoint_engaged', return_value=engaged), \
                patch('verify_prestage.note_disjoint', side_effect=disjoint.append):
            built.execute()
            built.execute(built.continuation('request', 2048, 2048))
            built.execute(built.continuation('request', 4096, 4096))
        return bumps, disjoint

    def test_the_route_scope_charges_nothing_for_steps_that_wrote_no_slot_and_one_bump_for_the_prompt(self):
        bumps, disjoint = self.run_steps(ROUTE_SCOPE, [False, False, True])
        self.assertEqual(disjoint, ['prefill', 'prefill-chunk'])
        self.assertEqual([entry for entry in bumps if entry != 'displace'], ['prefill-chunk'], 'the final step bumps once')

    def test_the_global_scope_bumps_before_every_step_as_before(self):
        bumps, disjoint = self.run_steps(MERGED, [False, False, True])
        self.assertEqual(disjoint, [])
        self.assertEqual([entry for entry in bumps if entry != 'displace'], ['prefill', 'prefill-chunk', 'prefill-chunk'])

    def test_a_step_the_route_did_not_vouch_for_bumps(self):
        bumps, disjoint = self.run_steps(ROUTE_SCOPE, [False, False, True], identity=False)
        self.assertEqual(disjoint, [])
        self.assertEqual(len([entry for entry in bumps if entry != 'displace']), 3)

    def test_a_step_that_wrote_a_slot_bumps(self):
        bumps, disjoint = self.run_steps(ROUTE_SCOPE, [True, True, True])
        self.assertEqual(disjoint, [])
        self.assertEqual(len([entry for entry in bumps if entry != 'displace']), 3)

    def test_the_scope_is_global_while_the_class_is_not_engaged(self):
        bumps, disjoint = self.run_steps(ROUTE_SCOPE, [False, False, True], engaged=False)
        self.assertEqual(disjoint, [])
        self.assertEqual([entry for entry in bumps if entry != 'displace'], ['prefill', 'prefill-chunk', 'prefill-chunk'])

    def test_the_hooks_pass_through_charges_nothing_while_the_class_is_engaged(self):
        import serving_worker_hook as hook

        with patch('verify_prestage.disjoint_engaged', return_value=True), patch.object(hook, 'note_fixture_writer') as bump:
            hook.note_prefill_pass_through('prefill')
        bump.assert_not_called()
        with patch('verify_prestage.disjoint_engaged', return_value=False), patch.object(hook, 'note_fixture_writer') as bump:
            hook.note_prefill_pass_through('prefill')
        bump.assert_called_once_with('prefill')


class ParkTests(unittest.TestCase):
    """Admission v2 in the lifecycle."""

    def long_in_flight(self, extra=PARK):
        built = Built(self, extra=extra, prompt=12289, computed=0, chunk=2048, captures=2)
        built.execute()
        self.assertEqual(built.lifecycle.request_id, 'request')
        return built

    def test_a_short_prompt_parks_the_suspended_long_one_and_is_announced_as_taking_the_scratch(self):
        built = self.long_in_flight()
        built.captures[1].position = 5000
        short = built.new_request('short', 5000, 2048)
        built.execute(short)
        self.assertEqual(sorted(built.lifecycle.parked), ['request'])
        self.assertEqual(built.lifecycle.parked['request']['capture'], built.captures[0])
        self.assertEqual(built.lifecycle.request_id, 'short')
        self.assertEqual(built.seen[-1], route.Step('short', 0, 2048, 5000, 'COLD', True))
        self.assertEqual(serving_lifecycle.prefill_gate().held, 'short')

    def test_the_long_one_resumes_through_the_parked_source_when_the_short_one_is_done(self):
        built = self.long_in_flight()
        built.captures[1].position = 5000
        built.captures[1].finish_at = 1
        built.execute(built.new_request('short', 5000, 5000))
        # the short prompt's seed is sampled and bridged: the prefill phase is over for it
        built.lifecycle.prefill_pending = False
        built.lifecycle.request_id = built.lifecycle.capture = None
        self.assertIsNone(built.lifecycle.request_id)
        self.assertEqual(sorted(built.lifecycle.parked), ['request'])
        built.captures[0].cursor = 2048
        built.execute(built.continuation('request', 2048, 2048))
        self.assertEqual(built.seen[-1], route.Step('request', 2048, 4096, 12289, 'PARKED', False))
        self.assertEqual(built.lifecycle.parked, {})
        self.assertEqual(built.lifecycle.request_id, 'request')
        self.assertIs(built.lifecycle.capture, built.captures[0])
        # and its next step is an ordinary scratch continuation again
        built.execute(built.continuation('request', 2048, 4096))
        self.assertEqual(built.seen[-1].source, 'SCRATCH')

    def test_a_new_prompt_with_the_flag_off_is_refused_as_always(self):
        built = Built(self, extra=MERGED, prompt=12289, computed=0, chunk=2048, captures=2)
        built.execute()
        with self.assertRaisesRegex(ValueError, 'one complete fresh prefill'):
            built.execute(built.new_request('short', 5000, 2048))

    def test_the_slot_count_caps_the_parks(self):
        built = self.long_in_flight()
        built.captures[1].position = 5000
        built.execute(built.new_request('short', 5000, 2048))
        with self.assertRaisesRegex(ValueError, 'one complete fresh prefill'):
            built.execute(built.new_request('third', 4000, 2048))

    def test_a_parked_prefill_that_finishes_or_is_aborted_is_released(self):
        built = self.long_in_flight()
        built.captures[1].position = 5000
        built.execute(built.new_request('short', 5000, 2048))
        vars(built.model)[route.HANDLE_ATTR] = None
        gone = SimpleNamespace(finished_req_ids={'request'}, scheduled_new_reqs=[], scheduled_cached_reqs=SimpleNamespace(req_ids=['short'],
                               num_computed_tokens=[2048]), scheduled_spec_decode_tokens={}, num_scheduled_tokens={'short': 2048},
                               total_num_scheduled_tokens=2048)
        built.execute(gone)
        built.captures[0].close.assert_called()
        self.assertEqual(built.lifecycle.parked, {})
        self.assertEqual(built.lifecycle.request_id, 'short')

    def test_s6_a_parked_prefill_vllm_preempted_is_released_like_a_finished_one(self):
        built = self.long_in_flight()
        built.captures[1].position = 5000
        built.execute(built.new_request('short', 5000, 2048))
        vars(built.model)[route.HANDLE_ATTR] = None
        preempted = SimpleNamespace(finished_req_ids=set(), preempted_req_ids={'request'}, scheduled_new_reqs=[],
                                    scheduled_cached_reqs=SimpleNamespace(req_ids=['short'], num_computed_tokens=[2048]),
                                    scheduled_spec_decode_tokens={}, num_scheduled_tokens={'short': 2048}, total_num_scheduled_tokens=2048)
        built.execute(preempted)
        built.captures[0].close.assert_called()
        self.assertEqual(built.lifecycle.parked, {}, 'its re-admission starts clean, not announced as PARKED')

    def test_close_releases_every_parked_prefill(self):
        built = self.long_in_flight()
        built.captures[1].position = 5000
        built.execute(built.new_request('short', 5000, 2048))
        built.lifecycle.close()
        built.captures[0].close.assert_called()
        self.assertEqual(built.lifecycle.parked, {})

    def test_a_parked_continuation_while_another_prefill_is_active_is_an_engine_fault(self):
        built = self.long_in_flight()
        built.captures[1].position = 5000
        built.execute(built.new_request('short', 5000, 2048))
        with self.assertRaisesRegex(ValueError, 'one prefill at a time owns the scratch'):
            built.execute(built.continuation('request', 2048, 2048))

    def test_only_a_clean_single_new_request_parks(self):
        built = self.long_in_flight()
        mixed = built.new_request('short', 5000, 2048)
        mixed.scheduled_cached_reqs = SimpleNamespace(req_ids=['other'])
        self.assertFalse(built.lifecycle._park_current(mixed))
        self.assertEqual(built.lifecycle.parked, {})
        self.assertEqual(built.lifecycle.request_id, 'request')


if __name__ == '__main__':
    unittest.main()
