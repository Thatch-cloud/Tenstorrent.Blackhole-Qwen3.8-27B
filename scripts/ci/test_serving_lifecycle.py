from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from serving_lifecycle import FastServingLifecycle
from test_serving_fast_policy import FastPolicyTests
from test_serving_request_factory import RequestFactoryTests
from test_serving_worker_hook import WorkerHookTests


class LifecycleTests(unittest.TestCase):
    def fixture(self):
        worker, bridge, _, decode = WorkerHookTests().fixture()
        worker.model_runner.execute_model.return_value = None
        worker.model_runner.sample_tokens.return_value = SimpleNamespace(req_ids=['request'], sampled_token_ids=[[10]])
        _, _, _, arguments = RequestFactoryTests().fixture()
        new = SimpleNamespace(req_id='request', prompt_token_ids=[1] * 4096,
            num_computed_tokens=0, mm_features=[], prompt_embeds=None, lora_request=None,
            sampling_params=arguments['state'].sampling_params)
        scheduled = SimpleNamespace(finished_req_ids=set(), scheduled_new_reqs=[new],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[]), scheduled_spec_decode_tokens={},
            num_scheduled_tokens={'request': 4096}, total_num_scheduled_tokens=4096)
        capture = SimpleNamespace(capture=Mock(side_effect=lambda: nullcontext()), close=Mock())

        def factory(state, features):
            self.assertIs(state, bridge.state)
            self.assertIs(features, capture)
            features.close()
            return bridge

        build = Mock(side_effect=factory)
        lifecycle = FastServingLifecycle(worker, config=FastPolicyTests().fixture(),
            capture_factory=Mock(return_value=capture), bridge_factory=build,
            eos_ids=(99,), cancelled=lambda: False)
        return lifecycle, worker, bridge, capture, build, scheduled, decode

    def chunked_fixture(self, chunk=2048, prompt=4096):
        """The same fixture, but the step carries only the first chunk and the capture
        knows about suspension - a `segment` context and a `complete` flag."""
        lifecycle, worker, bridge, capture, build, scheduled, decode = self.fixture()
        capture.complete = False
        capture.segments = 0
        # complete is set at a segment's EXIT, when the cursor has reached position -
        # never before it. finish_at says which segment gets there.
        capture.finish_at = 10 ** 9

        @contextmanager
        def segment():
            capture.segments += 1
            yield capture
            if capture.segments >= capture.finish_at:
                capture.complete = True

        capture.segment = Mock(side_effect=segment)
        scheduled.num_scheduled_tokens = {'request': chunk}
        scheduled.total_num_scheduled_tokens = chunk
        scheduled.scheduled_new_reqs[0].prompt_token_ids = [1] * prompt
        return lifecycle, worker, bridge, capture, build, scheduled, decode

    def continuation(self, scheduled, chunk, computed):
        """How a continuation actually arrives: a CACHED request, no new ones."""
        return SimpleNamespace(finished_req_ids=set(), scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=['request']),
            scheduled_spec_decode_tokens={}, num_scheduled_tokens={'request': chunk},
            total_num_scheduled_tokens=chunk)

    def test_a_first_chunk_short_of_the_prompt_is_admitted(self):
        """The admission contract used to demand the whole prompt in one step. That was
        the only clause relaxed; text-only, uncached and single-request all still hold."""
        lifecycle, worker, _, capture, _, scheduled, _ = self.chunked_fixture()
        self.assertIsNone(worker.execute_model(scheduled))
        capture.segment.assert_called_once()
        capture.capture.assert_not_called()
        self.assertEqual(lifecycle.request_id, 'request')
        self.assertFalse(lifecycle.prefill_pending, 'mid-prompt has no token to sample')

    def test_a_whole_prompt_step_still_uses_the_stricter_capture_api(self):
        """The unchunked path must not acquire a dependency on suspension support."""
        lifecycle, worker, _, capture, _, scheduled, _ = self.fixture()
        self.assertIsNone(worker.execute_model(scheduled))
        capture.capture.assert_called_once()
        self.assertTrue(lifecycle.prefill_pending)

    def test_a_continuation_is_routed_to_the_open_capture(self):
        lifecycle, worker, _, capture, _, scheduled, _ = self.chunked_fixture()
        worker.execute_model(scheduled)
        self.assertIsNone(worker.execute_model(self.continuation(scheduled, 2048, 2048)))
        self.assertEqual(capture.segments, 2)
        self.assertFalse(lifecycle.prefill_pending)

    def test_the_final_chunk_completes_the_prefill_and_arms_the_sampler(self):
        lifecycle, worker, _, capture, _, scheduled, _ = self.chunked_fixture()
        capture.finish_at = 2                        # the second segment reaches position
        worker.execute_model(scheduled)
        self.assertFalse(capture.complete)
        self.assertIsNone(worker.execute_model(self.continuation(scheduled, 2048, 2048)))
        self.assertTrue(capture.complete)
        self.assertTrue(lifecycle.prefill_pending, 'the final chunk produces the token')
        self.assertEqual(lifecycle.chunk_deferred, [True, True])

    def test_a_sampler_behaviour_change_mid_prompt_is_refused(self):
        """Whether the plugin defers on an intermediate chunk is not determinable from
        this repo, so both are accepted - but not a change of mind mid-prompt, which
        would mean one of the chunks was not the shape it appeared to be."""
        lifecycle, worker, _, capture, _, scheduled, _ = self.chunked_fixture()
        worker.execute_model(scheduled)
        self.assertEqual(lifecycle.chunk_deferred, [True])
        worker.model_runner.execute_model.return_value = SimpleNamespace(req_ids=['request'])
        with self.assertRaisesRegex(ValueError, 'changed sampler behaviour mid-prompt'):
            worker.execute_model(self.continuation(scheduled, 2048, 2048))

    def test_a_mid_prompt_chunk_sampling_step_delegates_instead_of_refusing(self):
        """Run 35687608717, the first arm to reach the sampler with chunking really on.

        vLLM calls model_executor.sample_tokens on EVERY step (v1/engine/core.py:499),
        not only when a token is due, so _sample is entered after a mid-prompt chunk.
        prefill_pending is correctly still clear there, and the native-sampling guard
        took the engine down with 'Only an unstructured pending prefill may use native
        sampling'. The step has to delegate to the plugin's own sampler instead.
        """
        lifecycle, worker, _, capture, _, scheduled, _ = self.chunked_fixture()
        worker.execute_model(scheduled)
        self.assertTrue(lifecycle.chunk_in_flight)
        self.assertFalse(lifecycle.prefill_pending)

        sentinel = SimpleNamespace(req_ids=[], sampled_token_ids=[])
        worker.model_runner.sample_tokens.return_value = sentinel
        self.assertIs(worker.sample_tokens(None), sentinel,
                      'the mid-prompt step must return the original sampler result')
        self.assertFalse(lifecycle.failed, 'delegating must not fail the lifecycle')

    def test_the_mid_prompt_sampling_step_delegates_even_with_grammar_output(self):
        """The guard refuses structured output because native sampling cannot honour
        it. A mid-prompt chunk is not native sampling at all, so it delegates rather
        than refusing - the original sampler is the one that decides."""
        lifecycle, worker, _, capture, _, scheduled, _ = self.chunked_fixture()
        worker.execute_model(scheduled)
        sentinel = SimpleNamespace(req_ids=[], sampled_token_ids=[])
        worker.model_runner.sample_tokens.return_value = sentinel
        self.assertIs(worker.sample_tokens(object()), sentinel)

    def test_the_final_chunk_clears_the_in_flight_flag_and_arms_native_sampling(self):
        lifecycle, worker, _, capture, _, scheduled, _ = self.chunked_fixture()
        capture.finish_at = 2
        worker.execute_model(scheduled)
        self.assertTrue(lifecycle.chunk_in_flight)
        worker.execute_model(self.continuation(scheduled, 2048, 2048))
        self.assertFalse(lifecycle.chunk_in_flight, 'the seed is due, not deferred')
        self.assertTrue(lifecycle.prefill_pending)

    def test_the_unchunked_path_never_sets_the_in_flight_flag(self):
        """A whole-prompt step arms native sampling directly; nothing about the
        unchunked path may start depending on the chunk flag."""
        lifecycle, worker, _, capture, _, scheduled, _ = self.fixture()
        worker.execute_model(scheduled)
        self.assertFalse(lifecycle.chunk_in_flight)
        self.assertTrue(lifecycle.prefill_pending)

    def test_a_four_chunk_prompt_interleaves_execute_and_sample_like_the_engine(self):
        """The shape the rig actually runs, which no test drove before v45.

        vLLM alternates execute_model and sample_tokens every step. The suite tested
        those calls separately and never in sequence, so the mid-prompt sampling step
        - the one that killed run 35687608717 - had no coverage at all. This drives
        four chunks with a sampling step after each, and asserts the flags move the
        way the engine needs them to.
        """
        lifecycle, worker, _, capture, _, scheduled, _ = self.chunked_fixture()
        capture.finish_at = 4
        deferred = SimpleNamespace(req_ids=[], sampled_token_ids=[])
        worker.model_runner.sample_tokens.return_value = deferred

        worker.execute_model(scheduled)
        for index in range(2, 4):
            self.assertTrue(lifecycle.chunk_in_flight, 'chunk %d is mid-prompt' % (index - 1))
            self.assertFalse(lifecycle.prefill_pending)
            self.assertIs(worker.sample_tokens(None), deferred)
            worker.execute_model(self.continuation(scheduled, 2048 * (index - 1), 2048))

        self.assertTrue(lifecycle.chunk_in_flight, 'chunk 3 is still mid-prompt')
        self.assertIs(worker.sample_tokens(None), deferred)
        worker.execute_model(self.continuation(scheduled, 2048 * 3, 2048))

        self.assertTrue(capture.complete, 'the fourth chunk finishes the prompt')
        self.assertFalse(lifecycle.chunk_in_flight)
        self.assertTrue(lifecycle.prefill_pending, 'the seed is due on the final chunk')
        self.assertEqual(lifecycle.chunk_deferred, [True, True, True, True])
        self.assertFalse(lifecycle.failed)

    def test_chunk_state_is_per_request_not_per_process(self):
        """chunk_deferred was initialised once and never cleared, so a new prompt's
        first chunk was compared against the PREVIOUS request's last one. Harmless at
        one request in flight and wrong the moment there are two, which is the point
        of the whole programme.

        Driven by poisoning the residue rather than running two full requests: without
        the reset on admission, the mid-prompt check sees deferred=True against a
        stale False and raises 'changed sampler behaviour mid-prompt'.
        """
        lifecycle, worker, _, capture, _, scheduled, _ = self.chunked_fixture()
        lifecycle.chunk_deferred = [False]
        lifecycle.chunk_in_flight = True
        worker.execute_model(scheduled)
        self.assertEqual(lifecycle.chunk_deferred, [True],
                         'admission must start this prompt its own chunk history')

    def test_a_continuation_for_another_request_is_not_routed_to_this_capture(self):
        """It used to RAISE. Under concurrency a cached step naming another request is
        that request's own decode, not an error, so it now reaches the stock path -
        run 35718626867 died exactly there, on a decode step judged by the admission
        contract while a prefill was in flight.

        The guarantee this test exists for is unchanged and is now asserted directly
        rather than inferred from an exception: the capture must not see it, and the
        in-flight prefill must be left alone.
        """
        lifecycle, worker, _, capture, _, scheduled, _ = self.chunked_fixture()
        worker.execute_model(scheduled)
        segments_before = capture.segment.call_count
        held_before = lifecycle.request_id
        other = self.continuation(scheduled, 2048, 2048)
        other.scheduled_cached_reqs = SimpleNamespace(req_ids=['someone-else'])
        # original_execute is the worker's bound execute_model, a plain function and
        # not a Mock, so the delegation is observed by substituting a recorder rather
        # than by asserting on a mock that does not exist.
        delegated = []
        lifecycle.original_execute = delegated.append

        worker.execute_model(other)

        self.assertEqual(capture.segment.call_count, segments_before,
                         'another request must never be fed into this capture')
        self.assertEqual(lifecycle.request_id, held_before,
                         'the in-flight prefill must be undisturbed')
        self.assertEqual(delegated, [other], 'it belongs on the stock path')

    def test_a_chunk_larger_than_the_prompt_is_refused(self):
        lifecycle, worker, _, _, _, scheduled, _ = self.chunked_fixture(chunk=8192, prompt=4096)
        with self.assertRaisesRegex(ValueError, 'whole or first chunk, required'):
            worker.execute_model(scheduled)

    def test_a_step_scheduling_a_second_request_is_still_refused(self):
        lifecycle, worker, _, _, _, scheduled, _ = self.chunked_fixture()
        scheduled.num_scheduled_tokens = {'request': 2048, 'other': 2048}
        with self.assertRaisesRegex(ValueError, 'whole or first chunk, required'):
            worker.execute_model(scheduled)

    def test_second_prefill_during_decode_reaches_the_prefill_path(self):
        """A new request arriving while another decodes is a PREFILL-only step.

        TTScheduler never mixes prefill and decode in one batch - probe 35435453374
        measured stock vllm producing new=['B'] cached=['A'] where the plugin gives
        new=['B'] cached=[] - so this step belongs on the prefill path. It used to be
        delegated to the hook, which sent it to admit_scheduler_output and was refused
        for carrying a new request.

        It must now reach the prefill path and fail on the HOOK instead, which is the
        real boundary: one FastWorkerHook binds one request to the worker.
        """
        lifecycle, worker, bridge, capture, build, prefill, decode = self.fixture()
        self.assertIsNone(worker.execute_model(prefill))
        worker.sample_tokens(None)
        self.assertIsNotNone(lifecycle.hook)
        self.assertEqual(lifecycle.decoding_id, 'request')
        self.assertIsNone(lifecycle.request_id, 'the prefill slot must be free again')

        second = SimpleNamespace(req_id='second', prompt_token_ids=[1] * 4096,
            num_computed_tokens=0, mm_features=[], prompt_embeds=None, lora_request=None,
            sampling_params=prefill.scheduled_new_reqs[0].sampling_params)
        step = SimpleNamespace(finished_req_ids=set(), scheduled_new_reqs=[second],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[]), scheduled_spec_decode_tokens={},
            num_scheduled_tokens={'second': 4096}, total_num_scheduled_tokens=4096)

        # the prefill path runs: a capture is taken for the new request, and the
        # step is NOT routed into the decode contract
        self.assertIsNone(worker.execute_model(step))
        self.assertEqual(lifecycle.request_id, 'second')
        self.assertEqual(lifecycle.decoding_id, 'request',
                         'the first request keeps decoding')
        self.assertTrue(lifecycle.prefill_pending)
        self.assertTrue(lifecycle.hook is not None, 'the first hook survives')

    def test_hook_passes_a_prefill_step_through_to_the_runner(self):
        """The hook replaces runner.execute_model, so it decides this, not the lifecycle.

        It used to send every step to bridge.execute_decode, and
        admit_scheduler_output refuses any step carrying a new request. A step with
        new requests is someone else's prefill - TTScheduler never mixes - so it must
        reach the runner instead.
        """
        lifecycle, worker, bridge, capture, build, prefill, decode = self.fixture()
        worker.execute_model(prefill)
        worker.sample_tokens(None)
        hook = lifecycle.hook
        self.assertIsNotNone(hook)

        seen = []
        hook.original_execute = lambda scheduled: seen.append(scheduled) or 'passed-through'
        step = SimpleNamespace(scheduled_new_reqs=[SimpleNamespace(req_id='second')],
                               scheduled_cached_reqs=SimpleNamespace(req_ids=[]))
        self.assertEqual(hook._execute(worker.model_runner, step), 'passed-through')
        self.assertEqual(len(seen), 1, 'the prefill step reached the runner')

    def test_a_second_prefill_attaches_to_the_hook_instead_of_needing_another(self):
        """This used to raise. One hook WAS the binding of one request to the
        worker, so a second arrival could not decode at all - run 35436193682's
        named boundary. Probe 35436807668 then showed the scheduler putting every
        live decode in one step, so the worker serves them together or not at all,
        and the hook holds a registry.
        """
        lifecycle, worker, bridge, capture, build, prefill, decode = self.fixture()
        lifecycle.packed_step = lambda entries, *, cancelled: []
        sampler = worker.model_runner.sample_tokens
        worker.execute_model(prefill)
        worker.sample_tokens(None)
        first = lifecycle.hook
        self.assertIsNotNone(first)
        self.assertEqual(lifecycle.decoding_ids, ['request'])

        second = SimpleNamespace(req_id='second', prompt_token_ids=[1] * 4096,
            num_computed_tokens=0, mm_features=[], prompt_embeds=None, lora_request=None,
            sampling_params=prefill.scheduled_new_reqs[0].sampling_params)
        step = SimpleNamespace(finished_req_ids=set(), scheduled_new_reqs=[second],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[]), scheduled_spec_decode_tokens={},
            num_scheduled_tokens={'second': 4096}, total_num_scheduled_tokens=4096)
        sampler.return_value = SimpleNamespace(req_ids=['second'], sampled_token_ids=[[11]])
        worker.model_runner.requests['second'] = SimpleNamespace(output_token_ids=[11])

        def second_bridge(state, features):
            features.close()
            return SimpleNamespace(runner=worker.model_runner, failed=False,
                request=SimpleNamespace(session=SimpleNamespace(request_id='second')),
                close=lambda: None)

        lifecycle.bridge_factory = second_bridge
        worker.execute_model(step)
        worker.sample_tokens(None)

        self.assertIs(lifecycle.hook, first, 'the same hook serves both')
        self.assertEqual(sorted(first.bridges), ['request', 'second'])
        self.assertEqual(lifecycle.decoding_ids, ['request', 'second'])

    def test_one_request_finishing_leaves_the_other_decoding(self):
        """Detaching must not tear down the hook the other request is using."""
        lifecycle, worker, bridge, capture, build, prefill, decode = self.fixture()
        lifecycle.packed_step = lambda entries, *, cancelled: []
        worker.execute_model(prefill)
        worker.sample_tokens(None)
        hook = lifecycle.hook
        other = Mock()
        other.request.session.request_id = 'second'
        other.runner = worker.model_runner
        hook.attach(other)
        lifecycle.decoding_ids.append('second')

        # a real finish step still carries the scheduler's request lists
        finished = SimpleNamespace(finished_req_ids={'request'}, total_num_scheduled_tokens=0,
            scheduled_new_reqs=[], scheduled_cached_reqs=SimpleNamespace(req_ids=['second']))
        worker.execute_model(finished)
        self.assertIs(lifecycle.hook, hook, 'the hook survives')
        self.assertEqual(list(hook.bridges), ['second'])
        self.assertEqual(lifecycle.decoding_ids, ['second'])
        self.assertEqual(lifecycle.decoding_id, 'second')

    def test_prefill_to_committed_decode_to_finished_cleanup(self):
        lifecycle, worker, bridge, capture, build, prefill, decode = self.fixture()
        self.assertIsNone(worker.execute_model(prefill))
        build.assert_not_called()
        seed = worker.sample_tokens(None)
        self.assertEqual(seed.sampled_token_ids, [[10]])
        build.assert_called_once()
        outputs = ModuleType('vllm.v1.outputs')
        outputs.ModelRunnerOutput = outputs.DraftTokenIds = SimpleNamespace
        with patch.dict('sys.modules', {'vllm.v1.outputs': outputs}):
            self.assertEqual(worker.take_draft_token_ids().draft_token_ids, [list(range(11, 26))])
            result = worker.execute_model(decode)
        self.assertEqual(result.sampled_token_ids, [list(range(11, 27))])
        finished = SimpleNamespace(finished_req_ids={'request'}, total_num_scheduled_tokens=0)
        worker.execute_model(finished)
        self.assertTrue(bridge.request.closed)
        self.assertIsNone(lifecycle.hook)
        self.assertIsNone(lifecycle.request_id)
        self.assertIsNone(worker.take_draft_token_ids())
        capture.close.assert_called_once()
        lifecycle.close()
        self.assertFalse(hasattr(worker, '_qwen_fast_lifecycle'))

    def test_terminal_seed_skips_factory_and_releases_features(self):
        lifecycle, worker, bridge, capture, build, prefill, _ = self.fixture()
        bridge.state.output_token_ids[:] = [99]
        worker.model_runner.sample_tokens.return_value.sampled_token_ids = [[99]]
        worker.execute_model(prefill)
        worker.sample_tokens(None)
        build.assert_not_called()
        capture.close.assert_called_once()
        self.assertIsNone(lifecycle.hook)
        self.assertIsNone(lifecycle.request_id,
                          'a request finishing at its first token must free the prefill slot')
        lifecycle.close()

    def test_a_terminal_first_token_does_not_block_the_next_request(self):
        """Run 35441524535 refused the second request with the prefill slot still
        naming the first, which had finished at its first token. Invisible with one
        user, because that user is done; fatal with two."""
        lifecycle, worker, bridge, capture, build, prefill, _ = self.fixture()
        bridge.state.output_token_ids[:] = [99]
        worker.model_runner.sample_tokens.return_value.sampled_token_ids = [[99]]
        worker.execute_model(prefill)
        worker.sample_tokens(None)

        second = SimpleNamespace(req_id='second', prompt_token_ids=[1] * 4096,
            num_computed_tokens=0, mm_features=[], prompt_embeds=None, lora_request=None,
            sampling_params=prefill.scheduled_new_reqs[0].sampling_params)
        step = SimpleNamespace(finished_req_ids=set(), scheduled_new_reqs=[second],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[]), scheduled_spec_decode_tokens={},
            num_scheduled_tokens={'second': 4096}, total_num_scheduled_tokens=4096)
        self.assertIsNone(worker.execute_model(step))
        self.assertEqual(lifecycle.request_id, 'second')
        self.assertTrue(lifecycle.prefill_pending)

    def test_a_terminal_first_token_under_ignore_eos_still_builds_a_bridge(self):
        """Run 35441818361: both users prefilled, then the decode step arrived with
        no hook, because both had taken the terminal branch. Under ignore_eos the
        request does NOT stop - vLLM keeps scheduling it - so short-circuiting
        leaves it with nothing to decode on. The branch is only correct when EOS
        is actually honoured.
        """
        lifecycle, worker, bridge, capture, build, prefill, _ = self.fixture()
        prefill.scheduled_new_reqs[0].sampling_params.ignore_eos = True
        bridge.state.output_token_ids[:] = [99]
        worker.model_runner.sample_tokens.return_value.sampled_token_ids = [[99]]
        worker.execute_model(prefill)
        worker.sample_tokens(None)
        build.assert_called_once()
        self.assertIsNotNone(lifecycle.hook, 'the request keeps decoding, so it needs a bridge')
        self.assertEqual(lifecycle.decoding_ids, ['request'])
        lifecycle.close()

    def test_a_prefill_displaces_the_resident_gdn_engine(self):
        """A native prefill rewrites GDN slot 0, so the engine that was resident no
        longer is; the next verify must restore its own carry (verifier_engine)."""
        lifecycle, worker, bridge, capture, build, prefill, _ = self.fixture()
        with patch('serving_lifecycle.note_prefill') as displace:
            worker.execute_model(prefill)
        displace.assert_called_once_with()

    def test_partial_prefill_rejected_without_device_execution(self):
        lifecycle, worker, _, capture, build, prefill, _ = self.fixture()
        prefill.total_num_scheduled_tokens = 2048
        with self.assertRaises(ValueError):
            worker.execute_model(prefill)
        lifecycle.original_execute.__self__.model_runner.execute_model.assert_not_called()
        capture.capture.assert_not_called()
        build.assert_not_called()
        self.assertTrue(lifecycle.failed)
        lifecycle.close()

    def test_factory_failure_poisons_lifecycle_and_keeps_features_for_cleanup(self):
        lifecycle, worker, _, capture, build, prefill, _ = self.fixture()
        build.side_effect = RuntimeError('trace setup failed')
        worker.execute_model(prefill)
        with self.assertRaises(RuntimeError):
            worker.sample_tokens(None)
        with self.assertRaises(ValueError):
            worker.take_draft_token_ids()
        lifecycle.close()
        capture.close.assert_called_once()
