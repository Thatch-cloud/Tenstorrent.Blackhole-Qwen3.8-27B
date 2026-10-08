"""Lever N at TP4: the scheduler side (levern_scheduler + serving_prefill_admission under QWEN_FAST_LEVER_N=1).

The plugin's TTScheduler is fixtures/plugin_scheduler.py (bf77cd63's class body) executed over ChunkingVllmScheduler, a reduced model of vLLM
0.25.1's Scheduler.schedule that CHUNKS: it starts from token_budget = self.max_num_scheduled_tokens, advances num_computed_tokens by what it
schedules, marks a request mid-prompt with is_prefill_chunk and keeps it in `running`. So the plugin's own prefill/decode split,
_schedule_prefill_only, _schedule_decode_only and decode fallback run unmodified under the wrappers. The tests check:
- flag off: nothing is wrapped that was not, and a long prompt is admitted whole (ByteIdentityTests);
- the cap: every non-final step ends on a 2,048 boundary, the final step coalesces the last full chunk and the tail, the budget is put
  back after the call and on an exception, a scheduler without max_num_scheduled_tokens is capped through long_prefill_token_threshold
  (CapTests);
- the alternation, end to end on the plugin scheduler: static R, the time share on a fake clock, no yield without a decoder, a forced mode
  untouched, one prefill in flight, the finished ids reaching the decode step (AlternationTests);
- the final-step DRAM gate and its negative control (FinalGateTests);
- the step lines, the install refusals and the logs (InstallTests).
The real vLLM scheduler's version of the cap (the attribute name and the split) is test_levern_scheduler_vllm, which runs where vLLM is installed."""

import os
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import levern_policy
import levern_scheduler
import serving_prefill_admission as admission
from test_serving_prefill_admission import GateFreeCase, Queue, configured, create_request_queue, plugin_class

CHUNK = levern_policy.CHUNK


class Request(object):
    def __init__(self, request_id, prompt_tokens):
        self.request_id, self.prompt_tokens, self.num_prompt_tokens = request_id, prompt_tokens, prompt_tokens
        self.num_computed_tokens = 0
        self.is_prefill_chunk = False
        self.status = 'waiting'
        self.sampling_params = None


class ChunkingVllmScheduler(object):
    """vLLM's Scheduler.schedule with chunked prefill, as far as a prefill cap touches it."""

    def __init__(self, max_num_seqs=8, budget=262144):
        self.running, self.waiting, self.skipped_waiting = [], Queue(), Queue()
        self.policy = 'fcfs'
        self.max_num_running_reqs = max_num_seqs
        self.max_num_scheduled_tokens = budget
        self.requests = {}
        self.finished_req_ids = set()
        self.steps = []

    def add_request(self, request):
        self.requests[request.request_id] = request
        self.waiting.append(request)

    def add_decoder(self, request_id, prompt=4096):
        request = Request(request_id, prompt)
        request.num_computed_tokens = prompt
        self.requests[request_id] = request
        self.running.append(request)
        return request

    def finish(self, request_id):
        request = self.requests.pop(request_id)
        self.running.remove(request)
        self.finished_req_ids.add(request_id)

    def schedule(self):
        budget, counts, new, cached = self.max_num_scheduled_tokens, {}, [], []
        for request in list(self.running):
            if budget <= 0:
                break
            if request.num_computed_tokens < request.prompt_tokens:     # a partial prefill: its next chunk
                n = min(request.prompt_tokens - request.num_computed_tokens, budget)
            else:
                n = 1
            counts[request.request_id] = n
            budget -= n
            cached.append(request.request_id)
        while (self.waiting or self.skipped_waiting) and budget > 0:
            if len(self.running) >= self.max_num_running_reqs:
                break
            queue = self.skipped_waiting or self.waiting
            request = queue[0]
            queue.pop(0)
            n = min(request.prompt_tokens - request.num_computed_tokens, budget)
            self.running.append(request)
            request.start_of_step = request.num_computed_tokens
            counts[request.request_id] = n
            budget -= n
            new.append(request)
        assert len(self.running) <= self.max_num_running_reqs
        for request in new:
            request.num_computed_tokens = counts[request.request_id]
            request.is_prefill_chunk = request.num_computed_tokens < request.prompt_tokens
        for request_id in cached:
            request = self.requests[request_id]
            if request.num_computed_tokens < request.prompt_tokens:
                request.num_computed_tokens += counts[request_id]
                request.is_prefill_chunk = request.num_computed_tokens < request.prompt_tokens
        finished, self.finished_req_ids = self.finished_req_ids, set()
        output = SimpleNamespace(
            scheduled_new_reqs=[SimpleNamespace(req_id=request.request_id, prompt_token_ids=[1] * request.prompt_tokens,
                                                num_computed_tokens=0, mm_features=[], prompt_embeds=None, lora_request=None,
                                                sampling_params=None) for request in new],
            scheduled_cached_reqs=SimpleNamespace(req_ids=cached), num_scheduled_tokens=counts,
            total_num_scheduled_tokens=sum(counts.values()), finished_req_ids=finished, scheduled_spec_decode_tokens={})
        self.steps.append(output)
        return output


class Clock(object):
    def __init__(self):
        self.now = 5000.0

    def __call__(self):
        return self.now

    def advance_ms(self, ms):
        self.now += ms / 1000.0


def install_levern(environ=None, clock=None, runtime_log=None, fault=None, cls=None):
    """The plugin's TTScheduler over the chunking fake, with the two wrappers applied the way install() applies them, over a runtime
    on `clock` (a fake one by default: on the real clock a step that took a few microseconds is owed that long again, and a runner that is slow
    between two calls adds decode steps a test counting steps does not expect; run 37856694214 failed on exactly that). Returns (class, runtime, log mock)."""
    cls = plugin_class(ChunkingVllmScheduler) if cls is None else cls
    clock = Clock() if clock is None else clock
    log = Mock() if runtime_log is None else runtime_log
    cfg = levern_policy.config(environ or {})
    runtime = levern_scheduler.LevernRuntime(cfg, log=log, clock=clock, fault=fault)
    original_prefill = cls._schedule_prefill_only
    original_schedule = cls.schedule
    cls._schedule_prefill_only = admission.wrap(original_prefill, queue_factory=lambda scheduler: create_request_queue(None),
                                                log=log, steps=0, levern=runtime)
    cls.schedule = levern_scheduler.wrap_schedule(original_schedule, runtime)
    return cls, runtime, log


def new_scheduler(cls, **kwargs):
    scheduler = cls.__new__(cls)
    ChunkingVllmScheduler.__init__(scheduler, **kwargs)
    scheduler._forced_mode = plugin_mode('DEFAULT')
    return scheduler


def plugin_mode(name):
    import test_serving_prefill_admission as base

    return getattr(base.TTSchedulingMode, name)


def run(scheduler, steps, clock=None, chunk_ms=700.0, round_ms=245.0, until=None):
    """Drive `steps` schedule() calls; the clock advances by the step's kind. Returns [(kind, request, tokens, start)] in order."""
    out = []
    for _ in range(steps):
        before = {request.request_id: request.num_computed_tokens for request in scheduler.running}
        result = scheduler.schedule()
        new = [request.req_id for request in result.scheduled_new_reqs]
        prefill = bool(new) or any(request_id in before and before[request_id] < scheduler.requests[request_id].prompt_tokens
                                   and scheduler.requests[request_id].num_computed_tokens > before[request_id] and
                                   result.num_scheduled_tokens[request_id] > 1
                                   for request_id in result.scheduled_cached_reqs.req_ids)
        if new:
            who = new[0]
            start = 0
        else:
            cached = [request_id for request_id in result.scheduled_cached_reqs.req_ids if result.num_scheduled_tokens[request_id] > 1]
            who = cached[0] if cached else None
            start = before.get(who)
        if clock is not None:
            clock.advance_ms(chunk_ms if prefill else round_ms)
        out.append(('prefill' if prefill else 'decode', who, sum(result.num_scheduled_tokens.values()) if prefill else len(
            result.scheduled_cached_reqs.req_ids), start))
        if until is not None and until(scheduler):
            break
    return out


class ByteIdentityTests(GateFreeCase):
    def test_flag_off_wraps_the_cap_only_and_schedule_stays_the_plugins(self):
        cls = plugin_class(ChunkingVllmScheduler)
        original_schedule = cls.schedule
        with patch.dict(os.environ, {'QWEN_FAST_ANY_REQUEST': '1'}, clear=False):
            os.environ.pop('QWEN_FAST_LEVER_N', None)
            os.environ.pop('QWEN_FAST_DECODE_STEPS_PER_ADMISSION', None)
            os.environ.pop('QWEN_FAST_KV_RESERVATION', None)
            admission.install(configured(cls), log=Mock())
        self.assertIs(cls.schedule, original_schedule)
        self.assertTrue(getattr(cls._schedule_prefill_only, admission.WRAPPED))
        self.assertFalse(hasattr(cls.schedule, levern_scheduler.WRAPPED))

    def test_flag_off_a_long_prompt_is_admitted_whole(self):
        cls = plugin_class(ChunkingVllmScheduler)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('QWEN_FAST_LEVER_N', None)
            os.environ.pop('QWEN_FAST_DECODE_STEPS_PER_ADMISSION', None)
            os.environ.pop('QWEN_FAST_KV_RESERVATION', None)
            admission.install(configured(cls), log=Mock())
        scheduler = new_scheduler(cls)
        scheduler.add_decoder('d0')
        scheduler.add_request(Request('cold', 60000))
        result = scheduler.schedule()
        self.assertEqual(result.num_scheduled_tokens, {'cold': 60000})

    def test_a_wrapper_with_no_runtime_caps_nothing(self):
        cls = plugin_class(ChunkingVllmScheduler)
        cls._schedule_prefill_only = admission.wrap(cls._schedule_prefill_only, queue_factory=lambda s: Queue(), log=Mock())
        scheduler = new_scheduler(cls)
        scheduler.add_request(Request('cold', 60000))
        self.assertEqual(scheduler.schedule().num_scheduled_tokens, {'cold': 60000})
        self.assertEqual(scheduler.max_num_scheduled_tokens, 262144)


class CapTests(GateFreeCase):
    def test_the_cold_prompt_is_split_at_model_chunk_boundaries_with_static_rounds(self):
        cls, runtime, log = install_levern({'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_LEVERN_ROUNDS': '1'})
        scheduler = new_scheduler(cls)
        for index in range(7):
            scheduler.add_decoder('d%d' % index)
        scheduler.add_request(Request('cold', 10000))
        steps = run(scheduler, 12, until=lambda s: not s.waiting and not any(r.is_prefill_chunk for r in s.running))
        prefill = [(who, tokens, start) for kind, who, tokens, start in steps if kind == 'prefill']
        self.assertEqual(prefill, [('cold', 2048, 0), ('cold', 2048, 2048), ('cold', 2048, 4096), ('cold', 3856, 6144)])
        # alternating with one decode step between prefill steps, every decode step serving all seven seats
        self.assertEqual([kind for kind, *_ in steps], ['prefill', 'decode', 'prefill', 'decode', 'prefill', 'decode', 'prefill'])
        self.assertTrue(all(tokens == 7 for kind, _, tokens, _ in steps if kind == 'decode'))
        # the final step's start is the last full chunk's start and it carries the tail
        self.assertEqual(levern_policy.final_start(10000), 6144)
        self.assertEqual(scheduler.max_num_scheduled_tokens, 262144, 'the token budget is put back')

    def test_every_non_final_end_is_aligned_for_any_prompt(self):
        for prompt in (2047, 2048, 4096, 4097, 6145, 8191, 12288, 20001):
            with self.subTest(prompt=prompt):
                cls, runtime, log = install_levern({'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_LEVERN_ROUNDS': '1'})
                scheduler = new_scheduler(cls)
                scheduler.add_decoder('d0')
                scheduler.add_request(Request('cold', prompt))
                steps = run(scheduler, 40, until=lambda s: not s.waiting and not any(r.is_prefill_chunk for r in s.running))
                ends, start = [], 0
                for kind, who, tokens, begun in steps:
                    if kind == 'prefill':
                        self.assertEqual(begun, start)
                        start += tokens
                        ends.append(start)
                self.assertEqual(ends[-1], prompt)
                self.assertTrue(all(end % CHUNK == 0 for end in ends[:-1]), ends)
                self.assertEqual(ends, [end for _, end in levern_policy.plan(prompt, decoding=True)])

    def test_with_no_decoder_the_solo_step_is_used_and_bounded(self):
        cls, runtime, log = install_levern({'QWEN_FAST_LEVER_N': '1'})
        scheduler = new_scheduler(cls)
        scheduler.add_request(Request('cold', 100000))
        steps = run(scheduler, 20, until=lambda s: not s.waiting and not any(r.is_prefill_chunk for r in s.running))
        self.assertEqual(steps[0][2], levern_policy.DEFAULT_SOLO)
        self.assertTrue(all(kind == 'prefill' for kind, *_ in steps), 'nothing to yield to')
        self.assertTrue(all(tokens <= levern_policy.DEFAULT_SOLO + 2 * CHUNK for _, _, tokens, _ in steps))

    def test_another_waiting_prompt_shortens_the_solo_step(self):
        cls, runtime, log = install_levern({'QWEN_FAST_LEVER_N': '1'})
        scheduler = new_scheduler(cls)
        scheduler.add_request(Request('cold', 100000))
        scheduler.add_request(Request('later', 5000))
        result = scheduler.schedule()
        self.assertEqual(result.num_scheduled_tokens, {'cold': CHUNK})

    def test_a_short_prompt_is_one_step(self):
        cls, runtime, log = install_levern({'QWEN_FAST_LEVER_N': '1'})
        scheduler = new_scheduler(cls)
        scheduler.add_decoder('d0')
        scheduler.add_request(Request('short', 3000))
        self.assertEqual(scheduler.schedule().num_scheduled_tokens, {'short': 3000})

    def test_the_budget_is_restored_after_an_exception(self):
        cls, runtime, log = install_levern({'QWEN_FAST_LEVER_N': '1'})
        scheduler = new_scheduler(cls)
        scheduler.add_decoder('d0')
        scheduler.add_request(Request('cold', 50000))
        original = ChunkingVllmScheduler.schedule

        def boom(self):
            self.seen = self.max_num_scheduled_tokens
            raise RuntimeError('base scheduler failed')

        with patch.object(ChunkingVllmScheduler, 'schedule', boom):
            with self.assertRaises(RuntimeError):
                scheduler.schedule()
        self.assertEqual(scheduler.seen, CHUNK)
        self.assertEqual(scheduler.max_num_scheduled_tokens, 262144)

    def test_a_scheduler_without_the_budget_attribute_is_capped_through_the_threshold(self):
        cls, runtime, log = install_levern({'QWEN_FAST_LEVER_N': '1'})
        scheduler = new_scheduler(cls)
        del scheduler.max_num_scheduled_tokens
        scheduler.scheduler_config = SimpleNamespace(long_prefill_token_threshold=0)
        seen = {}

        def base(self):
            seen.setdefault('threshold', self.scheduler_config.long_prefill_token_threshold)
            return SimpleNamespace(total_num_scheduled_tokens=0, scheduled_new_reqs=[],
                                   scheduled_cached_reqs=SimpleNamespace(req_ids=[]), finished_req_ids=set())

        scheduler.add_decoder('d0')
        scheduler.add_request(Request('cold', 50000))
        with patch.object(ChunkingVllmScheduler, 'schedule', base):
            scheduler.schedule()
        self.assertEqual(seen['threshold'], CHUNK)
        self.assertEqual(scheduler.scheduler_config.long_prefill_token_threshold, 0)
        self.assertTrue(any('long_prefill_token_threshold' in str(call) for call in log.call_args_list))

    def test_a_scheduler_with_neither_way_to_cap_refuses(self):
        cls, runtime, log = install_levern({'QWEN_FAST_LEVER_N': '1'})
        scheduler = new_scheduler(cls)
        del scheduler.max_num_scheduled_tokens
        scheduler.add_request(Request('cold', 50000))
        with self.assertRaises(ValueError):
            scheduler.schedule()

    def test_an_unreadable_prompt_runs_uncapped_and_says_so(self):
        cls, runtime, log = install_levern({'QWEN_FAST_LEVER_N': '1'})
        scheduler = new_scheduler(cls)
        scheduler.add_decoder('d0')
        request = Request('odd', 50000)
        request.num_prompt_tokens = None
        scheduler.add_request(request)
        scheduler.schedule()
        self.assertTrue(any(levern_policy.REFUSED_LINE in str(call) for call in log.call_args_list))

    def test_two_partial_prefills_are_refused(self):
        cls, runtime, log = install_levern({'QWEN_FAST_LEVER_N': '1'})
        scheduler = new_scheduler(cls)
        for name in ('a', 'b'):
            request = scheduler.add_decoder(name, prompt=50000)
            request.num_computed_tokens = 4096
            request.is_prefill_chunk = True
        with self.assertRaises(ValueError):
            scheduler.schedule()


class AlternationTests(GateFreeCase):
    def test_the_share_alternation_on_the_plugin_scheduler(self):
        clock = Clock()
        cls, runtime, log = install_levern({'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_LEVERN_PREFILL_SHARE': '0.5',
                                            'QWEN_FAST_LEVERN_MAX_ROUNDS': '64'}, clock=clock)
        scheduler = new_scheduler(cls)
        for index in range(7):
            scheduler.add_decoder('d%d' % index)
        scheduler.add_request(Request('cold', 40000))
        steps = run(scheduler, 400, clock=clock, chunk_ms=1000.0, round_ms=250.0,
                    until=lambda s: not s.waiting and not any(r.is_prefill_chunk for r in s.running))
        prefill = sum(1 for kind, *_ in steps if kind == 'prefill')
        decode = sum(1 for kind, *_ in steps if kind == 'decode')
        self.assertEqual(prefill, len(levern_policy.plan(40000)))
        # f = 0.5: the decoders get about as much wall time as the prefill
        self.assertAlmostEqual(250.0 * decode / (1000.0 * prefill), 1.0, delta=0.3)
        # never two prefill steps back to back while a decoder runs
        kinds = [kind for kind, *_ in steps]
        self.assertFalse(any(a == b == 'prefill' for a, b in zip(kinds, kinds[1:])))

    def test_the_prefill_resumes_after_the_decode_step_and_is_never_dropped(self):
        cls, runtime, log = install_levern({'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_LEVERN_ROUNDS': '2'})
        scheduler = new_scheduler(cls)
        scheduler.add_decoder('d0')
        scheduler.add_request(Request('cold', 9000))
        steps = run(scheduler, 20, until=lambda s: not s.waiting and not any(r.is_prefill_chunk for r in s.running))
        self.assertEqual(scheduler.requests['cold'].num_computed_tokens, 9000)
        self.assertEqual([kind for kind, *_ in steps], ['prefill', 'decode', 'decode', 'prefill', 'decode', 'decode', 'prefill', 'decode', 'decode', 'prefill'])

    def test_a_decode_step_hides_the_partial_and_the_waiting_queue(self):
        cls, runtime, log = install_levern({'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_LEVERN_ROUNDS': '1'})
        scheduler = new_scheduler(cls)
        scheduler.add_decoder('d0')
        scheduler.add_request(Request('cold', 9000))
        scheduler.schedule()
        scheduler.add_request(Request('later', 3000))
        result = scheduler.schedule()
        self.assertEqual(list(result.scheduled_cached_reqs.req_ids), ['d0'])
        self.assertEqual(result.scheduled_new_reqs, [])
        self.assertEqual([request.request_id for request in scheduler.waiting], ['later'])
        self.assertIn('cold', [request.request_id for request in scheduler.running])

    def test_one_prefill_in_flight_the_second_prompt_waits_behind_the_first(self):
        cls, runtime, log = install_levern({'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_LEVERN_PREFILL_SHARE': '1'})
        scheduler = new_scheduler(cls)
        scheduler.add_decoder('d0')
        scheduler.add_request(Request('cold', 9000))
        scheduler.add_request(Request('later', 3000))
        order = []
        for _ in range(8):
            result = scheduler.schedule()
            order += [(request_id, tokens) for request_id, tokens in result.num_scheduled_tokens.items() if tokens > 1]
            if not any(r.is_prefill_chunk for r in scheduler.running) and not scheduler.waiting:
                break
        self.assertEqual([name for name, _ in order], ['cold', 'cold', 'cold', 'cold', 'later'])

    def test_finished_ids_reach_the_decode_step_that_replaces_a_yielded_pass(self):
        cls, runtime, log = install_levern({'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_LEVERN_ROUNDS': '1'})
        scheduler = new_scheduler(cls)
        scheduler.add_decoder('d0')
        scheduler.add_decoder('d1')
        scheduler.add_request(Request('cold', 9000))
        scheduler.schedule()
        scheduler.finish('d1')
        result = scheduler.schedule()           # the yielded decode step carries the finished id
        self.assertEqual(result.finished_req_ids, {'d1'})
        self.assertEqual(list(result.scheduled_cached_reqs.req_ids), ['d0'])

    def test_the_last_decoder_finishing_mid_prefill_stops_the_yielding(self):
        cls, runtime, log = install_levern({'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_LEVERN_ROUNDS': '3'})
        scheduler = new_scheduler(cls)
        scheduler.add_decoder('d0')
        scheduler.add_request(Request('cold', 20000))
        scheduler.schedule()
        scheduler.finish('d0')
        kinds = []
        while any(r.is_prefill_chunk for r in scheduler.running):
            result = scheduler.schedule()
            kinds.append(max(result.num_scheduled_tokens.values()) > 1)
        self.assertTrue(all(kinds), 'with no decoder every step is the prefill')

    def test_a_forced_scheduling_mode_is_delegated_untouched(self):
        cls, runtime, log = install_levern({'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_LEVERN_ROUNDS': '1'})
        scheduler = new_scheduler(cls)
        scheduler.add_decoder('d0')
        scheduler.add_request(Request('cold', 9000))
        scheduler._forced_mode = plugin_mode('PREFILL_ONLY')
        for _ in range(3):
            result = scheduler.schedule()
            self.assertGreater(max(result.num_scheduled_tokens.values()), 1)
        self.assertEqual(runtime.steps, 0, 'a forced mode logs no step and runs no alternation')

    def test_no_pending_prefill_is_a_plain_decode_and_logs_nothing(self):
        cls, runtime, log = install_levern({'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_LEVERN_ROUNDS': '1'})
        scheduler = new_scheduler(cls)
        scheduler.add_decoder('d0')
        result = scheduler.schedule()
        self.assertEqual(result.num_scheduled_tokens, {'d0': 1})
        log.assert_not_called()


class StepLineTests(GateFreeCase):
    def lines(self, log):
        return [call.args for call in log.call_args_list if call.args and call.args[0] == levern_policy.STEP_LINE]

    def test_each_step_while_a_prefill_is_pending_logs_kind_seats_tokens_and_the_prompt(self):
        cls, runtime, log = install_levern({'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_LEVERN_ROUNDS': '1'})
        scheduler = new_scheduler(cls)
        for index in range(3):
            scheduler.add_decoder('d%d' % index)
        scheduler.add_request(Request('cold', 5000))
        run(scheduler, 5, until=lambda s: not s.waiting and not any(r.is_prefill_chunk for r in s.running))
        lines = self.lines(log)
        fields = [dict(zip(('template', 'n', 'kind', 'seats', 'req', 'start', 'tokens', 'end', 'prompt', 'final', 'reason', 'prev_kind',
                            'prev_ms', 'owed_ms', 'owed_rounds'), args)) for args in lines]
        self.assertEqual([f['kind'] for f in fields], ['prefill', 'decode', 'prefill'])
        self.assertEqual([f['n'] for f in fields], [1, 2, 3])
        self.assertEqual((fields[0]['req'], fields[0]['start'], fields[0]['tokens'], fields[0]['end'], fields[0]['prompt'],
                          fields[0]['final']), ('cold', 0, 2048, 2048, 5000, 0))
        self.assertEqual(fields[1]['seats'], 3)
        self.assertEqual((fields[2]['start'], fields[2]['end'], fields[2]['final']), (2048, 5000, 1))
        self.assertEqual(fields[1]['reason'], 'owed')
        # the line formats with its own template
        levern_policy.STEP_LINE.format(*lines[0][1:])

    def test_a_plain_decode_after_the_prefill_is_not_logged(self):
        cls, runtime, log = install_levern({'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_LEVERN_ROUNDS': '1'})
        scheduler = new_scheduler(cls)
        scheduler.add_decoder('d0')
        scheduler.add_request(Request('short', 1000))
        scheduler.schedule()
        count = len(self.lines(log))
        scheduler.schedule()
        scheduler.schedule()
        self.assertEqual(len(self.lines(log)), count)


class BlockedWaiterTests(GateFreeCase):
    def test_a_prompt_waiting_for_a_seat_does_not_log_a_line_a_round(self):
        cls, runtime, log = install_levern({'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_LEVERN_ROUNDS': '1'})
        scheduler = new_scheduler(cls, max_num_seqs=2)
        scheduler.add_decoder('d0')
        scheduler.add_decoder('d1')
        scheduler.add_request(Request('waiting', 9000))
        for _ in range(5):
            result = scheduler.schedule()
            self.assertEqual(sorted(result.scheduled_cached_reqs.req_ids), ['d0', 'd1'], 'the decoders keep decoding')
        lines = [c.args for c in log.call_args_list if c.args and c.args[0] == levern_policy.STEP_LINE]
        self.assertEqual(lines, [])
        scheduler.finish('d1')
        scheduler.schedule()                       # a seat is free: the prompt is admitted, and that is a logged prefill step
        lines = [c.args for c in log.call_args_list if c.args and c.args[0] == levern_policy.STEP_LINE]
        self.assertEqual([args[2] for args in lines], ['prefill'])


class FinalGateTests(GateFreeCase):
    """The admission's own DRAM hold asks the predicate once, for the fresh prompt, and the final-step gate asks it again for the same
    prompt before the step that builds the engine: so each test lets the admission pass and then toggles the reading."""

    def predicate(self):
        state = SimpleNamespace(ok=True, asked=[])

        def admits(prompt):
            state.asked.append(prompt)
            return state.ok, dict(largest_free=900, need=800, free=2000, trace_largest_free=60, short=() if state.ok else ('free',))

        return admits, state

    def setUp(self):
        super().setUp()
        self.saved = admission.sys.modules.pop(admission.DRAM_KEY, None)
        self.addCleanup(self.restore)

    def restore(self):
        admission.sys.modules.pop(admission.DRAM_KEY, None)
        if self.saved is not None:
            admission.sys.modules[admission.DRAM_KEY] = self.saved

    def started(self, prompt, decoders=1, environ=None):
        admits, state = self.predicate()
        admission.register_dram_predicate(admits)
        flags = {'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_LEVERN_PREFILL_SHARE': '1'}
        flags.update(environ or {})
        cls, runtime, log = install_levern(flags, fault=flags.get('QWEN_FAST_LEVERN_FAULT'))
        scheduler = new_scheduler(cls)
        for index in range(decoders):
            scheduler.add_decoder('d%d' % index)
        scheduler.add_request(Request('cold', prompt))
        scheduler.schedule()                       # the admission: [0, 2048), asked once and passed
        self.assertEqual(scheduler.requests['cold'].num_computed_tokens, CHUNK)
        state.asked.clear()
        return scheduler, state, log

    def lines(self, log, template):
        return [c.args for c in log.call_args_list if c.args and c.args[0] == template]

    def test_the_final_step_waits_for_dram_while_a_decoder_runs_and_goes_when_it_fits(self):
        scheduler, state, log = self.started(6000)         # S_last = 2048: the next step is the final one
        state.ok = False
        for _ in range(3):
            held = scheduler.schedule()
            self.assertEqual(list(held.scheduled_cached_reqs.req_ids), ['d0'], 'a held final step is a decode step')
        self.assertEqual(scheduler.requests['cold'].num_computed_tokens, CHUNK)
        self.assertEqual(len(self.lines(log, levern_policy.FINAL_HOLD_LINE)), 1, 'one line per hold state')
        state.ok = True
        final = scheduler.schedule()
        self.assertEqual(final.num_scheduled_tokens, {'cold': 6000 - CHUNK})
        self.assertTrue(set(state.asked) == {6000})

    def test_with_no_decoder_the_hold_lifts_and_the_final_step_runs(self):
        scheduler, state, log = self.started(6000, decoders=0)
        state.ok = False
        result = scheduler.schedule()
        self.assertEqual(result.num_scheduled_tokens, {'cold': 6000 - CHUNK})
        self.assertEqual(state.asked, [], 'with no decoder to free memory the final step is never held, so nothing is asked')
        self.assertEqual(self.lines(log, levern_policy.FINAL_HOLD_LINE), [])

    def test_a_chunk_before_the_final_step_never_asks_the_predicate(self):
        scheduler, state, log = self.started(20000)
        state.ok = False
        scheduler.schedule()
        scheduler.schedule()
        self.assertEqual(state.asked, [])
        self.assertEqual(self.lines(log, levern_policy.FINAL_HOLD_LINE), [])

    def test_no_predicate_holds_nothing(self):
        cls, runtime, log = install_levern({'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_LEVERN_PREFILL_SHARE': '1'})
        scheduler = new_scheduler(cls)
        scheduler.add_decoder('d0')
        scheduler.add_request(Request('cold', 6000))
        scheduler.schedule()
        self.assertEqual(scheduler.schedule().num_scheduled_tokens, {'cold': 6000 - CHUNK})

    def test_the_fault_forces_one_hold_then_lets_the_final_step_through(self):
        scheduler, state, log = self.started(6000, environ={'QWEN_FAST_LEVERN_FAULT': 'final-hold'})
        held = scheduler.schedule()
        self.assertEqual(list(held.scheduled_cached_reqs.req_ids), ['d0'], 'the held step is a decode step')
        final = scheduler.schedule()
        self.assertEqual(final.num_scheduled_tokens, {'cold': 6000 - CHUNK})
        lines = self.lines(log, levern_policy.FINAL_HOLD_LINE)
        self.assertEqual(len(lines), 1)
        self.assertIn('fault:final-hold', lines[0])

    def test_an_unavailable_reading_or_a_raising_predicate_holds_nothing(self):
        for predicate in (lambda prompt: (True, dict(largest_free=None, need=1, unavailable='no stats')),
                          Mock(side_effect=RuntimeError('boom'))):
            admission.register_dram_predicate(predicate)
            cls, runtime, log = install_levern({'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_LEVERN_PREFILL_SHARE': '1'})
            scheduler = new_scheduler(cls)
            scheduler.add_decoder('d0')
            scheduler.add_request(Request('cold', 6000))
            scheduler.schedule()
            self.assertEqual(scheduler.schedule().num_scheduled_tokens, {'cold': 6000 - CHUNK})


class InstallTests(GateFreeCase):
    def install(self, **environ):
        cls = plugin_class(ChunkingVllmScheduler)
        log = Mock()
        with patch.dict(os.environ, environ, clear=False):
            for name in levern_policy.ALL_FLAGS + (admission.STEPS_FLAG, 'QWEN_FAST_KV_RESERVATION'):
                if name not in environ:
                    os.environ.pop(name, None)
            admission.install(configured(cls), log=log)
        return cls, log

    def test_the_flag_wraps_schedule_and_logs_the_install_once(self):
        cls, log = self.install(QWEN_FAST_LEVER_N='1', QWEN_FAST_LEVERN_ROUNDS='1')
        self.assertTrue(getattr(cls.schedule, levern_scheduler.WRAPPED))
        installed = [c.args for c in log.call_args_list if c.args and c.args[0] == levern_policy.INSTALLED_LINE]
        self.assertEqual(len(installed), 1)
        self.assertEqual(installed[0][2:], (2048, 16384, 0.5, 1, 8))

    def test_install_is_idempotent(self):
        cls, log = self.install(QWEN_FAST_LEVER_N='1')
        wrapped = cls.schedule
        with patch.dict(os.environ, {'QWEN_FAST_LEVER_N': '1'}):
            admission.install(configured(cls), log=log)
        self.assertIs(cls.schedule, wrapped)

    def test_the_decode_credit_and_lever_n_are_mutually_exclusive(self):
        with self.assertRaises(ValueError) as caught:
            self.install(QWEN_FAST_LEVER_N='1', QWEN_FAST_DECODE_STEPS_PER_ADMISSION='2')
        self.assertIn('mutually exclusive', str(caught.exception))

    def test_a_malformed_flag_is_refused_not_defaulted(self):
        for name, value in (('QWEN_FAST_LEVER_N', 'yes'),):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.install(**{name: value})
        with self.assertRaises(ValueError):
            self.install(QWEN_FAST_LEVER_N='1', QWEN_FAST_LEVERN_STEP_TOKENS='1000')

    def test_a_class_without_the_decode_only_step_is_refused(self):
        cls = plugin_class(ChunkingVllmScheduler)
        del cls._schedule_decode_only
        with patch.dict(os.environ, {'QWEN_FAST_LEVER_N': '1'}), self.assertRaises(ValueError) as caught:
            for name in (admission.STEPS_FLAG, 'QWEN_FAST_KV_RESERVATION'):
                os.environ.pop(name, None)
            admission.install(configured(cls), log=Mock())
        self.assertIn('_schedule_decode_only', str(caught.exception))
        # and the class was left unwrapped
        self.assertFalse(hasattr(cls._schedule_prefill_only, admission.WRAPPED))


class VerifyTests(GateFreeCase):
    """The pass the cap shaped is checked before it runs: it must be the request the budget was computed for and end on the model's boundary."""

    def rig(self, *waiting):
        cls, runtime, log = install_levern({'QWEN_FAST_LEVER_N': '1'})
        scheduler = new_scheduler(cls)
        scheduler.add_decoder('d0')
        for request in waiting:
            scheduler.add_request(request)
        return scheduler, runtime, log

    def refused(self, log):
        return [call.args for call in log.call_args_list if call.args and call.args[0] == levern_policy.REFUSED_LINE]

    def test_a_clean_pass_is_left_alone(self):
        scheduler, runtime, log = self.rig(Request('cold', 10000))
        self.assertEqual(scheduler.schedule().num_scheduled_tokens, {'cold': CHUNK})
        self.assertEqual(self.refused(log), [])

    def test_a_pass_that_admits_another_request_than_the_capped_one_fails_closed(self):
        scheduler, runtime, log = self.rig(Request('cold', 60000))
        decoy = Request('decoy', 60000)
        with patch.object(runtime, 'target', return_value=(decoy, True)):
            with self.assertRaises(ValueError) as caught:
                scheduler.schedule()
        self.assertIn("cap was computed for 'decoy' and the pass admitted 'cold'", str(caught.exception))
        self.assertTrue(self.refused(log))
        self.assertEqual(scheduler.max_num_scheduled_tokens, 262144, 'the token budget is put back even then')

    def test_a_misaligned_non_final_end_fails_closed(self):
        scheduler, runtime, log = self.rig(Request('cold', 60000))
        with patch.object(runtime, 'budget', return_value=3000):
            with self.assertRaises(ValueError) as caught:
                scheduler.schedule()
        self.assertIn('would end a non-final step at 3000 of 60000', str(caught.exception))

    def test_a_prompt_under_4096_tokens_is_never_split(self):
        scheduler, runtime, log = self.rig(Request('short', 3000))
        with patch.object(runtime, 'budget', return_value=CHUNK):
            with self.assertRaises(ValueError):
                scheduler.schedule()
        scheduler, runtime, log = self.rig(Request('short', 3000))
        self.assertEqual(scheduler.schedule().num_scheduled_tokens, {'short': 3000})

    def test_a_partial_continuing_on_the_boundary_passes_to_its_final_step(self):
        scheduler, runtime, log = self.rig(Request('cold', 5000))
        ends = []
        while scheduler.waiting or any(r.is_prefill_chunk for r in scheduler.running):
            result = scheduler.schedule()
            ends.append(result.num_scheduled_tokens.get('cold'))
            if len(ends) > 6:
                break
        self.assertEqual(self.refused(log), [])
        self.assertEqual(sum(value for value in ends if value and value > 1), 5000)


if __name__ == '__main__':
    unittest.main()
