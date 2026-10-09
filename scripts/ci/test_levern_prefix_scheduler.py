"""Lever N with prefix reuse: the scheduler side of the merged route (levern_scheduler.LevernRuntime.merged, serving_prefill_admission's wrapper),
on the plugin's own TTScheduler (fixtures/plugin_scheduler.py, bf77cd63) over a reduced model of vLLM's Scheduler.schedule that CHUNKS and applies the
G1 trim: a waiting request admitted with `hit` cached tokens starts at min(hit, the graft's peek), exactly as the in-pass get_computed_blocks does
(docs/lever-n-prefix-merged-route.md sections 6, 9, 10 and 12.1).

What is held:
  cap        a fresh admission's step budget is computed from (P, Q): a hit at S_last takes the whole rest in one step; the same hit with the cap
             computed from (P, 0) (the negative control, defect D1) is refused by verify() as a step past S_last; verify also refuses a start the plan
             cannot have; the budget is put back;
  single     the pass sees ONE candidate: the waiting queues hold only the target, the other waiting requests keep their order, the admitted one leaves
             the queue, and a parked partial is out of `running` for the call and back after it;
  short lane a request with at most QWEN_FAST_LEVERN_SHORT_TOKENS left after its hit is admitted ahead of a long prefill in flight (shortest first, FIFO
             on ties), the long resumes when no short can be admitted, a short never parks a short, the slot cap, the park age and the deadline
             governor each close the lane, a refused KV reservation leaves the long running, and a hit turn arriving during a cold long prefill waits one
             boundary instead of the whole prefill;
  quarantine a partial whose suspended state is gone, or beyond the in-flight limit, is ended through the D2 consumer BEFORE the pass (never scheduled,
             nothing allocated), and with no consumer the refusal stays fatal;
  kill       the flag file stops splitting new prompts and closes the short lane, and in-flight prefills finish;
  governor   the alternation runs at the share the deadline governor asks for and the step line says so."""

import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import c2_smoke_check
import levern_policy
import levern_scheduler
import serving_prefill_admission as admission
import serving_request_quarantine as quarantine
from test_levern_scheduler import ChunkingVllmScheduler, Clock, Request, plugin_mode
from test_serving_prefill_admission import GateFreeCase, Queue, create_request_queue, plugin_class

CHUNK = levern_policy.CHUNK
MERGED_ENV = {'QWEN_FAST_LEVER_N': '1', 'QWEN_PREFIX_REUSE': '1', 'QWEN_FAST_STICKY_SESSIONS': '1', 'QWEN_FAST_LEVERN_ROUNDS': '1',
              'QWEN_FAST_LEVERN_TTFT_TARGET_S': '0', 'QWEN_FAST_LEVERN_PARK': 'host', 'QWEN_FAST_LEVERN_SHORT_TOKENS': '16384'}


class HitRequest(Request):
    def __init__(self, request_id, prompt_tokens, hit=0, arrival=None):
        super(HitRequest, self).__init__(request_id, prompt_tokens)
        self.hit, self.arrival_time = hit, arrival


class FakeGraft(object):
    """What the scheduler sees of the G1 graft: peek(request) -> the Q the trim will land on (memoized per schedule() call), counted."""

    def __init__(self, ceiling_error=0):
        self.calls = 0
        self.memo = {}
        self.moved = {}

    def peek(self, request):
        if request.request_id not in self.memo:
            self.calls += 1
            self.memo[request.request_id] = request.hit
        return self.memo[request.request_id]


class TrimmingScheduler(ChunkingVllmScheduler):
    """ChunkingVllmScheduler whose waiting loop applies the trim: a request starts at its hit (capped by the peek when the graft memoized one)."""

    def schedule(self):
        graft = self.__dict__.get('_qwen_prefix')
        budget, counts, new, cached, before = self.max_num_scheduled_tokens, {}, [], [], {}
        for request in list(self.running):
            if budget <= 0:
                break
            if request.num_computed_tokens < request.prompt_tokens:
                n = min(request.prompt_tokens - request.num_computed_tokens, budget)
            else:
                n = 1
            counts[request.request_id] = n
            before[request.request_id] = request.num_computed_tokens
            budget -= n
            cached.append(request.request_id)
        while (self.waiting or self.skipped_waiting) and budget > 0:
            if len(self.running) >= self.max_num_running_reqs:
                break
            queue = self.skipped_waiting or self.waiting
            request = queue[0]
            queue.pop(0)
            start = request.hit
            if graft is not None and request.request_id in graft.memo:
                start = min(start, graft.memo[request.request_id])
            request.num_computed_tokens = start
            request.start_of_step = start
            n = min(request.prompt_tokens - start, budget)
            self.running.append(request)
            counts[request.request_id] = n
            budget -= n
            new.append(request)
        assert len(self.running) <= self.max_num_running_reqs
        for request in new:
            request.num_computed_tokens = request.start_of_step + counts[request.request_id]
            request.is_prefill_chunk = request.num_computed_tokens < request.prompt_tokens
        for request_id in cached:
            request = self.requests[request_id]
            if request.num_computed_tokens < request.prompt_tokens:
                request.num_computed_tokens += counts[request_id]
                request.is_prefill_chunk = request.num_computed_tokens < request.prompt_tokens
        finished, self.finished_req_ids = self.finished_req_ids, set()
        output = SimpleNamespace(
            scheduled_new_reqs=[SimpleNamespace(req_id=request.request_id, prompt_token_ids=[1] * request.prompt_tokens,
                                                num_computed_tokens=request.start_of_step, mm_features=[], prompt_embeds=None, lora_request=None,
                                                sampling_params=None) for request in new],
            scheduled_cached_reqs=SimpleNamespace(req_ids=cached, num_computed_tokens=[before[request_id] for request_id in cached],
                                                  resumed_req_ids=set()),
            num_scheduled_tokens=counts, total_num_scheduled_tokens=sum(counts.values()), finished_req_ids=finished,
            scheduled_spec_decode_tokens={})
        self.steps.append(output)
        return output


def install(environ=None, *, kv=None, off_exists=None):
    """The plugin's TTScheduler over TrimmingScheduler with the merged runtime's two wrappers, as install() applies them."""
    cls = plugin_class(TrimmingScheduler)
    log = Mock()
    clock, wall = Clock(), Clock()
    wall.now = 1.7e9
    env = dict(MERGED_ENV)
    env.update(environ or {})
    env = {name: value for name, value in env.items() if value is not None}
    runtime = levern_scheduler.LevernRuntime(levern_policy.config(env), log=log, clock=clock, wall=wall, merged=levern_policy.merged_config(env),
                                             off_path='/nonexistent/levern.off')
    if off_exists is not None:
        runtime.off.exists = off_exists
    original_prefill, original_schedule = cls._schedule_prefill_only, cls.schedule
    cls._schedule_prefill_only = admission.wrap(original_prefill, queue_factory=lambda scheduler: create_request_queue(None), log=log, steps=0,
                                                kv=kv, levern=runtime)
    cls.schedule = levern_scheduler.wrap_schedule(original_schedule, runtime)
    return cls, runtime, log, clock, wall


def new_scheduler(cls, seats=8, graft=True):
    scheduler = cls.__new__(cls)
    TrimmingScheduler.__init__(scheduler, max_num_seqs=seats)
    scheduler._forced_mode = plugin_mode('DEFAULT')
    if graft:
        scheduler.__dict__['_qwen_prefix'] = FakeGraft()
    return scheduler


class Rig(object):
    """A scheduler, its runtime and a driver that runs schedule() calls and emulates what the lifecycle and the route do after each (the owner and the
    parks in the shared state the scheduler reads)."""

    def __init__(self, environ=None, seats=8, kv=None, decoders=1, off_exists=None, route=True):
        self.cls, self.runtime, self.log, self.clock, self.wall = install(environ, kv=kv, off_exists=off_exists)
        self.scheduler = new_scheduler(self.cls, seats=seats)
        for index in range(decoders):
            self.scheduler.add_decoder('d%d' % index)
        self.events = []
        levern_policy.reset_state()
        self.holder = levern_policy.state_holder()
        self.holder.route = route
        self.quarantined = []

    def add(self, request_id, prompt, hit=0, arrival=None):
        request = HitRequest(request_id, prompt, hit, self.wall.now if arrival is None else arrival)
        self.scheduler.add_request(request)
        return request

    def step(self, prefill_ms=700.0, decode_ms=245.0):
        """One schedule() call; returns (kind, request id, start, tokens) of the prefill it scheduled, or ('decode', None, None, seats)."""
        scheduler = self.scheduler
        self.consume_quarantine()
        graft = scheduler.__dict__.get('_qwen_prefix')
        if graft is not None and not getattr(graft, 'keep', False):
            graft.memo.clear()          # SchedulerGraft.begin_step clears the peek memo at the start of every schedule() call
        before = {request.request_id: request.num_computed_tokens for request in scheduler.running}
        result = scheduler.schedule()
        new = [value.req_id for value in result.scheduled_new_reqs]
        partial = [request_id for request_id in result.scheduled_cached_reqs.req_ids
                   if before.get(request_id, scheduler.requests[request_id].prompt_tokens) < scheduler.requests[request_id].prompt_tokens
                   and result.num_scheduled_tokens[request_id] > 0 and scheduler.requests[request_id].num_computed_tokens > before[request_id]
                   and request_id in [r.request_id for r in scheduler.running if r.request_id not in ('d0',)]
                   and before[request_id] < scheduler.requests[request_id].prompt_tokens]
        if new:
            who, start = new[0], result.scheduled_new_reqs[0].num_computed_tokens
        elif partial:
            who, start = partial[0], before[partial[0]]
        else:
            who = start = None
        if who is not None:
            tokens = result.num_scheduled_tokens[who]
            self.emulate_route(who, start, start + tokens, scheduler.requests[who].prompt_tokens)
            event = ('prefill', who, start, tokens)
            took = prefill_ms(who, start, tokens) if callable(prefill_ms) else prefill_ms
            self.clock.advance_ms(took)
            self.wall.advance_ms(took)
        else:
            event = ('decode', None, None, sum(1 for value in result.num_scheduled_tokens.values() if value == 1))
            self.clock.advance_ms(decode_ms)
            self.wall.advance_ms(decode_ms)
        self.events.append(event)
        self.last = result
        return event

    def consume_quarantine(self):
        """serving_request_quarantine.abort_quarantined, as the consumer wraps update_from_output: the request is finished and leaves the scheduler."""
        shared = quarantine.holder()
        for request_id in list(shared.pending):
            request = self.scheduler.requests.pop(request_id, None)
            if request is not None and request in self.scheduler.running:
                self.scheduler.running.remove(request)
            self.scheduler.finished_req_ids.add(request_id)
            self.quarantined.append((request_id, shared.pending.pop(request_id)))
            if self.holder.owner is not None and self.holder.owner[0] == request_id:
                self.holder.owner = None
            self.holder.parks.pop(request_id, None)

    def prefill_step(self, limit=6):
        """Steps until a prefill step ran (the alternation puts a decode step between two prefill steps while decoders run); returns it."""
        for _ in range(limit):
            event = self.step()
            if event[0] == 'prefill':
                return event
        return event

    def emulate_route(self, who, start, end, total):
        """The route's bookkeeping of the shared state: a step by `who` takes the scratch (parking its owner), the last one gives it up."""
        holder = self.holder
        if holder.owner is not None and holder.owner[0] != who:
            holder.parks[holder.owner[0]] = holder.owner[1]
        holder.parks.pop(who, None)
        holder.owner = None if end >= total else (who, end)

    def run(self, steps, until=None):
        for _ in range(steps):
            self.step()
            if until is not None and until(self):
                break
        return self.events

    def prefills(self):
        return [event for event in self.events if event[0] == 'prefill']

    def idle(self):
        return not self.scheduler.waiting and not any(request.is_prefill_chunk for request in self.scheduler.running)


class CapFromTheHitTests(GateFreeCase):
    def test_a_hit_at_s_last_takes_the_whole_rest_in_one_step(self):
        rig = Rig()
        rig.add('hit', 13000, hit=10240)
        event = rig.prefill_step()
        self.assertEqual(event, ('prefill', 'hit', 10240, 2760))
        self.assertEqual(rig.scheduler.requests['hit'].num_computed_tokens, 13000)
        self.assertFalse(any(levern_policy.REFUSED_LINE in str(call) for call in rig.log.call_args_list))

    def test_the_cap_computed_from_zero_for_that_hit_is_refused_by_verify(self):
        """Defect D1's negative control: with no peek the cap is the step budget from (P, 0), and the pass runs [Q, Q + budget) past S_last."""
        rig = Rig()
        rig.scheduler.__dict__['_qwen_prefix'] = None
        rig.add('hit', 13000, hit=10240)
        with self.assertRaisesRegex(ValueError, 'past S_last=10240'):
            rig.prefill_step()

    def test_a_hit_below_s_last_is_stepped_from_q(self):
        rig = Rig()
        rig.add('hit', 13000, hit=6144)
        rig.run(10, until=lambda r: r.idle())
        self.assertEqual([event[2:] for event in rig.prefills()], [(6144, 2048), (8192, 2048), (10240, 2760)])

    def test_the_peek_is_asked_once_per_request_per_call_and_the_in_pass_trim_never_exceeds_it(self):
        rig = Rig()
        graft = rig.scheduler.__dict__['_qwen_prefix']
        rig.add('hit', 13000, hit=10240)
        rig.prefill_step()
        self.assertEqual(graft.calls, 1)
        # a trim that would land above what the cap was computed from is clamped to the peek by the graft; the fake models that clamp
        rig2 = Rig()
        graft2 = rig2.scheduler.__dict__['_qwen_prefix']
        request = rig2.add('moved', 13000, hit=10240)
        graft2.memo['moved'] = 6144       # the peek said 6144; vLLM's own hit moved up to 10240 between the peek and the pass
        graft2.keep = True
        rig2.prefill_step()
        self.assertEqual(rig2.events[0], ('prefill', 'moved', 6144, 2048))
        del request

    def test_verify_refuses_a_start_the_plan_cannot_have(self):
        runtime = levern_scheduler.LevernRuntime(levern_policy.config(MERGED_ENV), log=Mock(), merged=levern_policy.merged_config(MERGED_ENV))
        request = HitRequest('r', 13000)
        result = SimpleNamespace(total_num_scheduled_tokens=2048, num_scheduled_tokens={'r': 2048},
                                 scheduled_new_reqs=[SimpleNamespace(req_id='r', num_computed_tokens=1000)])
        with self.assertRaisesRegex(ValueError, 'off the 2048-token boundary|not a step of the plan'):
            runtime.verify(result, (request, 0))
        ok = SimpleNamespace(total_num_scheduled_tokens=2048, num_scheduled_tokens={'r': 2048},
                             scheduled_new_reqs=[SimpleNamespace(req_id='r', num_computed_tokens=4096)])
        runtime.verify(ok, (request, 0))

    def test_the_budget_is_put_back(self):
        rig = Rig()
        rig.add('hit', 13000, hit=6144)
        rig.prefill_step()
        self.assertEqual(rig.scheduler.max_num_scheduled_tokens, 262144)


class SinglePassTests(GateFreeCase):
    def test_only_the_target_is_in_the_waiting_queue_during_the_pass_and_the_rest_keep_their_order(self):
        rig = Rig(environ={'QWEN_FAST_LEVERN_PARK': '0'})
        seen = []
        original = TrimmingScheduler.schedule

        def spy(self):
            seen.append([request.request_id for request in self.waiting])
            return original(self)

        for name, prompt in (('a', 50000), ('b', 5000), ('c', 7000)):
            rig.add(name, prompt)
        with patch.object(TrimmingScheduler, 'schedule', spy):
            rig.prefill_step()
        self.assertEqual(seen[-1], ['a'])
        self.assertEqual([request.request_id for request in rig.scheduler.waiting], ['b', 'c'])
        self.assertEqual(rig.events[0][1], 'a')

    def test_a_target_the_pass_did_not_admit_stays_where_it_was(self):
        rig = Rig(environ={'QWEN_FAST_LEVERN_PARK': '0'}, seats=1, decoders=1)
        rig.add('a', 50000)
        rig.add('b', 5000)
        rig.prefill_step()
        self.assertEqual([request.request_id for request in rig.scheduler.waiting], ['a', 'b'])

    def test_a_parked_partial_is_out_of_running_for_the_call_and_back_after_it(self):
        rig = Rig()
        rig.add('long', 100000)
        rig.prefill_step()
        seen = []
        original = TrimmingScheduler.schedule

        def spy(self):
            seen.append(sorted(request.request_id for request in self.running))
            return original(self)

        rig.add('short', 5000)
        with patch.object(TrimmingScheduler, 'schedule', spy):
            rig.prefill_step()
        self.assertEqual(rig.events[-1][:2], ('prefill', 'short'))
        self.assertNotIn('long', seen[-1])
        self.assertIn('long', [request.request_id for request in rig.scheduler.running])
        self.assertTrue(rig.scheduler.requests['long'].is_prefill_chunk)
        self.assertIn('short', [request.request_id for request in rig.scheduler.running])


class ShortLaneTests(GateFreeCase):
    def start_long(self, rig, prompt=100000, steps=1):
        rig.add('long', prompt)
        for _ in range(steps):
            rig.prefill_step()
        self.assertEqual(rig.events[-1][:2], ('prefill', 'long'))

    def prefill_order(self, rig):
        return [event[1] for event in rig.prefills()]

    def test_a_short_prompt_preempts_a_long_prefill_at_the_next_boundary_and_the_long_resumes_after_it(self):
        rig = Rig()
        self.start_long(rig)
        rig.add('short', 6000)
        rig.run(40, until=lambda r: r.idle() or len(r.prefills()) > 8)
        order = self.prefill_order(rig)
        self.assertEqual(order[:4], ['long', 'short', 'short', 'long'])
        short_steps = [event for event in rig.prefills() if event[1] == 'short']
        self.assertEqual([event[2:] for event in short_steps], [(0, 2048), (2048, 3952)])
        long_steps = [event for event in rig.prefills() if event[1] == 'long']
        self.assertEqual(long_steps[1][2], long_steps[0][2] + long_steps[0][3], 'the long resumes exactly where it was parked')
        self.assertTrue(any(levern_policy.SHORT_LINE.split('{}')[0] in str(call) for call in rig.log.call_args_list))

    def test_shortest_remaining_first_among_the_waiting_shorts_and_fifo_on_ties(self):
        rig = Rig()
        self.start_long(rig)
        rig.add('s1', 9000)
        rig.add('s2', 3000)
        rig.add('s3', 3000)
        rig.run(60, until=lambda r: r.idle())
        order = []
        for who in self.prefill_order(rig):
            if not order or order[-1] != who:
                order.append(who)
        self.assertEqual(order[:5], ['long', 's2', 's3', 's1', 'long'])

    def test_remaining_after_the_hit_decides_what_is_short(self):
        rig = Rig()
        self.start_long(rig)
        rig.add('big-hit', 120000, hit=110592)         # 9,408 tokens left after the hit: short
        rig.add('plain', 20000)                         # 20,000 left: long
        rig.prefill_step()
        self.assertEqual(rig.events[-1][:2], ('prefill', 'big-hit'))
        self.assertEqual(rig.runtime.pending_class['big-hit'], 'short', 'fixed from the peek of the admitting call')
        self.assertEqual(rig.runtime.klass_of(rig.scheduler, rig.scheduler.requests['plain']), 'long')
        self.assertNotIn('plain', rig.runtime.klass, 'a request still waiting is not frozen')

    def test_a_short_never_parks_a_short(self):
        rig = Rig()
        rig.add('first', 9000)
        rig.prefill_step()
        rig.add('second', 3000)
        rig.run(20, until=lambda r: r.idle())
        order = []
        for who in self.prefill_order(rig):
            if not order or order[-1] != who:
                order.append(who)
        self.assertEqual(order, ['first', 'second'])

    def test_with_no_prefill_in_flight_the_shortest_waiting_short_goes_before_an_earlier_long(self):
        rig = Rig()
        rig.add('long', 100000)
        rig.add('short', 5000)
        rig.prefill_step()
        self.assertEqual(rig.events[0][:2], ('prefill', 'short'))

    def test_the_slot_cap_keeps_a_second_long_from_being_in_flight(self):
        rig = Rig()
        self.start_long(rig)
        rig.add('short', 6000)
        rig.add('other-long', 90000)
        rig.run(400, until=lambda r: r.idle())
        order = []
        for who in self.prefill_order(rig):
            if not order or order[-1] != who:
                order.append(who)
        self.assertEqual(order[:4], ['long', 'short', 'long', 'other-long'])

    def test_a_second_short_takes_the_same_park_slot_again(self):
        rig = Rig()
        self.start_long(rig)
        rig.add('s1', 5000)
        rig.run(8, until=lambda r: r.events[-1][1:2] == ('s1',) and r.scheduler.requests['s1'].num_computed_tokens >= 5000)
        rig.add('s2', 4000)
        rig.run(8)
        order = []
        for who in self.prefill_order(rig):
            if not order or order[-1] != who:
                order.append(who)
        self.assertEqual(order[:4], ['long', 's1', 's2', 'long'], 'the long is already parked: the second short takes the scratch with no new park')

    def test_a_long_parked_for_the_max_park_time_admits_no_more_shorts(self):
        rig = Rig(environ={'QWEN_FAST_LEVERN_MAX_PARK_S': '5'})
        self.start_long(rig)
        rig.add('s1', 5000)
        rig.prefill_step()
        self.assertEqual(rig.events[-1][1], 's1')
        rig.clock.advance_ms(6000)
        rig.add('s2', 4000)
        rig.run(400, until=lambda r: r.idle())
        order = []
        for who in self.prefill_order(rig):
            if not order or order[-1] != who:
                order.append(who)
        self.assertLess(order.index('long', 2), order.index('s2'), 'the long resumed before the second short: its park time was used up')

    def test_the_lane_is_closed_without_the_park_flag(self):
        rig = Rig(environ={'QWEN_FAST_LEVERN_PARK': '0'})
        self.start_long(rig)
        rig.add('short', 6000)
        rig.run(400, until=lambda r: r.idle())
        order = []
        for who in self.prefill_order(rig):
            if not order or order[-1] != who:
                order.append(who)
        self.assertEqual(order, ['long', 'short'], 'v1: the short waits behind the whole long prefill')

    def test_the_deadline_governor_that_needs_the_long_closes_the_lane(self):
        rig = Rig(environ={'QWEN_FAST_LEVERN_TTFT_TARGET_S': '180', 'QWEN_FAST_LEVERN_ROUNDS': None})
        rig.add('long', 253920, arrival=rig.wall.now - 150.0)
        rig.prefill_step()
        rig.add('short', 6000)
        rig.prefill_step()
        self.assertEqual(rig.events[-1][:2], ('prefill', 'long'), 'the long has 30 s left for 90 s of work: nothing may park it')
        # with plenty of slack the same short preempts
        relaxed = Rig(environ={'QWEN_FAST_LEVERN_TTFT_TARGET_S': '180', 'QWEN_FAST_LEVERN_ROUNDS': None})
        relaxed.add('long', 253920, arrival=relaxed.wall.now)
        relaxed.prefill_step()
        relaxed.add('short', 6000)
        relaxed.prefill_step()
        self.assertEqual(relaxed.events[-1][:2], ('prefill', 'short'))

    def test_a_refused_kv_reservation_leaves_the_long_running_and_the_short_waiting(self):
        kv = SimpleNamespace(hold=Mock(side_effect=lambda scheduler, candidates, decodes, state, log: candidates[-1].request_id == 'short'),
                             CARRIED_LINE='carried {}', INSTALLED_LINE='installed {}')
        rig = Rig(kv=kv)
        self.start_long(rig)
        rig.add('short', 6000)
        rig.prefill_step()
        self.assertEqual(rig.events[-1][:2], ('prefill', 'long'))
        self.assertEqual([request.request_id for request in rig.scheduler.waiting], ['short'])
        self.assertTrue(kv.hold.called)
        asked = kv.hold.call_args_list[-1][0][1]
        self.assertEqual([request.request_id for request in asked], ['short'], 'the reservation is asked about the one request the pass would admit')

    def test_no_seat_means_no_preemption(self):
        rig = Rig(seats=2, decoders=1)
        self.start_long(rig)
        rig.add('short', 6000)
        rig.prefill_step()
        self.assertEqual(rig.events[-1][:2], ('prefill', 'long'))

    def test_a_hit_turn_arriving_during_a_cold_long_prefill_waits_one_boundary_not_the_whole_prefill(self):
        def first_step_of_the_hit(park):
            rig = Rig(environ={'QWEN_FAST_LEVERN_PARK': park})
            rig.add('cold-254k', 253920)
            for _ in range(5):
                rig.prefill_step()
            rig.add('hit-turn', 120000, hit=110592)
            waited = 0
            while rig.prefill_step()[1] != 'hit-turn':
                waited += 1
                if waited > 400:
                    break
            return waited, len([event for event in rig.prefills() if event[1] == 'cold-254k'])

        with_lane, cold_steps_before = first_step_of_the_hit('host')
        without_lane, cold_steps_without = first_step_of_the_hit('0')
        # decode steps between prefill steps are not what is measured: the hit's first step comes right after the next boundary
        self.assertLessEqual(cold_steps_before, 6)
        self.assertGreater(cold_steps_without, 100, 'v1: the hit turn queues behind the whole cold prefill (about 123 steps)')
        self.assertLessEqual(with_lane, 2)
        self.assertGreater(without_lane, with_lane * 20)


class QuarantineTests(GateFreeCase):
    def setUp(self):
        super(QuarantineTests, self).setUp()
        shared = quarantine.holder()
        self.saved = (shared.installed, dict(shared.pending))
        shared.installed, shared.pending = 'a scheduler', {}
        self.addCleanup(lambda: (setattr(shared, 'installed', self.saved[0]), setattr(shared, 'pending', self.saved[1])))

    def test_a_partial_whose_state_is_gone_is_registered_and_never_scheduled(self):
        rig = Rig(decoders=2)
        rig.add('lost', 50000)
        rig.prefill_step()
        rig.holder.owner, rig.holder.parks = ('somebody-else', 4096), {}
        for _ in range(5):
            rig.step()
            if quarantine.holder().pending:
                break
        self.assertIn('lost', quarantine.holder().pending)
        self.assertIn('suspended state', quarantine.holder().pending['lost'])
        self.assertNotIn('lost', rig.last.num_scheduled_tokens, 'nothing is allocated or scheduled for it')
        self.assertEqual(rig.events[-1][0], 'decode')
        self.assertEqual(rig.runtime.quarantined, 1)
        rig.step()
        self.assertEqual([request_id for request_id, _ in rig.quarantined], ['lost'])
        self.assertNotIn('lost', rig.scheduler.requests)
        self.assertTrue(any(levern_policy.QUARANTINE_LINE.split('{}')[0] in str(call) for call in rig.log.call_args_list))

    def test_a_parked_partial_at_the_right_position_is_not_quarantined(self):
        rig = Rig()
        rig.add('long', 100000)
        rig.prefill_step()
        rig.add('short', 5000)
        rig.prefill_step()
        self.assertEqual(quarantine.holder().pending, {})
        self.assertEqual(rig.holder.parks, {'long': 2048})

    def test_more_partials_than_the_limit_are_quarantined_youngest_first(self):
        rig = Rig(environ={'QWEN_FAST_LEVERN_PARK': '0'})
        for name in ('a', 'b', 'c'):
            request = HitRequest(name, 50000)
            request.num_computed_tokens = 4096
            request.is_prefill_chunk = True
            rig.scheduler.requests[name] = request
            rig.scheduler.running.append(request)
        rig.holder.owner = ('a', 4096)
        levern_policy.state_holder().parks = {'b': 4096, 'c': 4096}
        rig.step()
        self.assertEqual(sorted(quarantine.holder().pending), ['b', 'c'])
        self.assertEqual(rig.events[-1][:2], ('prefill', 'a'))

    def test_with_no_consumer_installed_the_refusal_stays_fatal(self):
        quarantine.holder().installed = None
        rig = Rig()
        rig.add('lost', 50000)
        rig.prefill_step()
        rig.holder.owner, rig.holder.parks = ('somebody-else', 4096), {}
        with self.assertRaises(ValueError):
            for _ in range(3):
                rig.step()
        self.assertTrue(any(levern_policy.REFUSED_LINE in str(call) for call in rig.log.call_args_list))

    def test_a_scheduler_with_no_route_state_published_checks_nothing(self):
        rig = Rig(route=False)
        rig.add('long', 50000)
        rig.run(4)
        self.assertEqual(quarantine.holder().pending, {})


class KillSwitchTests(GateFreeCase):
    def test_once_engaged_a_new_prompt_is_not_split_and_the_short_lane_closes(self):
        present = [False]
        rig = Rig(off_exists=lambda path: present[0])
        rig.add('cold', 100000)
        rig.prefill_step()
        self.assertEqual(rig.events[-1][3], 2048, 'split while the switch is off')
        present[0] = True
        rig.clock.advance_ms(2000)
        rig.add('short', 5000)
        rig.add('new-long', 60000)
        rig.run(200, until=lambda r: r.idle())
        order = []
        for who in [event[1] for event in rig.prefills()]:
            if not order or order[-1] != who:
                order.append(who)
        self.assertEqual(order, ['cold', 'short', 'new-long'], 'the in-flight prefill finished first; no short lane while killed')
        new_long = [event for event in rig.prefills() if event[1] == 'new-long']
        self.assertEqual(new_long, [('prefill', 'new-long', 0, 60000)], 'a new prompt takes its whole rest in one step')
        self.assertTrue(any(levern_policy.KILL_LINE.split('{}')[0] in str(call) for call in rig.log.call_args_list))
        self.assertTrue(rig.runtime.killed())


class GovernorIntegrationTests(GateFreeCase):
    def test_the_step_line_names_the_share_the_alternation_ran_at(self):
        rig = Rig(environ={'QWEN_FAST_LEVERN_TTFT_TARGET_S': '180', 'QWEN_FAST_LEVERN_ROUNDS': None})
        rig.add('cold', 253920, arrival=rig.wall.now - 120.0)
        rig.prefill_step()
        lines = [call.args for call in rig.log.call_args_list if call.args and call.args[0] == levern_policy.STEP_LINE_MERGED]
        self.assertTrue(lines)
        f_eff = lines[-1][-3]
        self.assertGreater(f_eff, 0.5, 'a prompt that arrived 120 s ago with 90 s of work left has 60 s: the share is raised')
        self.assertLessEqual(f_eff, 1.0)
        calm = Rig(environ={'QWEN_FAST_LEVERN_TTFT_TARGET_S': '180', 'QWEN_FAST_LEVERN_ROUNDS': None})
        calm.add('cold', 253920, arrival=calm.wall.now)
        calm.prefill_step()
        calm_lines = [call.args for call in calm.log.call_args_list if call.args and call.args[0] == levern_policy.STEP_LINE_MERGED]
        self.assertGreater(calm_lines[-1][-3], 0.5)
        self.assertLess(calm_lines[-1][-3], f_eff, 'a fresh arrival is governed less than one 120 s old')

    def test_the_pending_hint_lists_the_longs_in_service_order_and_leaves_the_shorts_out(self):
        rig = Rig(environ={'QWEN_FAST_LEVERN_TTFT_TARGET_S': '180'})
        now = rig.wall.now
        rig.add('long', 100000, arrival=now - 30.0)
        rig.prefill_step()
        rig.add('short', 5000, arrival=now - 20.0)
        rig.add('long2', 80000, arrival=now - 10.0)
        hint = rig.runtime.pending_hint(rig.scheduler)
        self.assertEqual([round(now - arrival) for arrival, _ in hint], [round(now - rig.scheduler.requests[name].arrival_time) for name in ('long', 'long2')])
        self.assertEqual(len(hint), 2)
        self.assertGreater(hint[0][1], 0)

    def test_the_governor_off_gives_no_hint(self):
        rig = Rig(environ={'QWEN_FAST_LEVERN_TTFT_TARGET_S': '0'})
        rig.add('long', 100000)
        self.assertEqual(rig.runtime.pending_hint(rig.scheduler), [])

    def test_a_measured_prefill_step_teaches_the_step_time_model(self):
        rig = Rig()
        rig.add('cold', 100000)
        rig.step(prefill_ms=1400.0)
        rig.prefill_step()
        rig.prefill_step()
        rig.prefill_step()
        self.assertEqual(rig.runtime.step_times.observed >= 1, True)
        self.assertGreater(rig.runtime.step_times.scale, 1.0)


class ReviewFixTests(GateFreeCase):
    """The review of 5b6c0448: B1 and S1 to S4."""

    def test_b1_a_hit_that_moved_down_from_a_whole_rest_peek_still_runs_a_valid_step(self):
        rig = Rig()
        graft = rig.scheduler.__dict__['_qwen_prefix']
        graft.peek = lambda request: 2048          # P=4500, S_last=2048: the peeked step is the whole rest ...
        rig.add('x', 4500, hit=0)                   # ... but the in-pass trim lands on 0 (the prefix kill switch latched, an eviction)
        event = rig.prefill_step()
        self.assertEqual(event, ('prefill', 'x', 0, 4500), 'the whole prompt in one step, not [0, 2452)')

    def test_b1_the_levern_kill_switch_with_a_peeked_hit_also_takes_the_whole_prompt_from_any_start(self):
        present = [True]
        rig = Rig(off_exists=lambda path: present[0])
        graft = rig.scheduler.__dict__['_qwen_prefix']
        graft.peek = lambda request: 8192
        rig.add('y', 20000, hit=0)
        rig.clock.advance_ms(2000)
        self.assertEqual(rig.prefill_step(), ('prefill', 'y', 0, 20000))

    def test_s1_a_waiting_long_with_no_seat_to_take_is_not_a_governed_prefill(self):
        rig = Rig(environ={'QWEN_FAST_LEVERN_TTFT_TARGET_S': '180'}, seats=2)
        now = rig.wall.now
        rig.add('inflight', 100000, arrival=now - 10.0)
        rig.prefill_step()
        rig.add('queued', 100000, arrival=now - 10.0)
        self.assertEqual(len(rig.runtime.pending_hint(rig.scheduler)), 1, 'one decoder and the prefill in flight hold both seats')
        roomy = Rig(environ={'QWEN_FAST_LEVERN_TTFT_TARGET_S': '180'}, seats=8)
        roomy.add('inflight', 100000, arrival=roomy.wall.now - 10.0)
        roomy.prefill_step()
        roomy.add('queued', 100000, arrival=roomy.wall.now - 10.0)
        self.assertEqual(len(roomy.runtime.pending_hint(roomy.scheduler)), 2)

    def test_s1_a_waiting_long_already_past_the_deadline_does_not_pin_the_share_at_one(self):
        rig = Rig(environ={'QWEN_FAST_LEVERN_TTFT_TARGET_S': '180', 'QWEN_FAST_LEVERN_ROUNDS': None})
        now = rig.wall.now
        rig.add('inflight', 100000, arrival=now - 10.0)
        rig.prefill_step()
        rig.add('late', 100000, arrival=now - 400.0)
        hint = rig.runtime.pending_hint(rig.scheduler)
        self.assertEqual(len(hint), 1)
        self.assertLess(levern_policy.effective_share(0.5, 180, hint, rig.wall()), 1.0)

    def test_s2_only_intermediate_steps_that_continued_their_scratch_teach_the_model(self):
        rig = Rig()
        rig.add('whole', 5000)                 # never split: one final step
        for _ in range(4):
            rig.step(prefill_ms=2900.0)
        self.assertEqual(rig.runtime.step_times.observed, 0, 'a final step carries the build; a new admission may carry a restore')
        rig = Rig()
        rig.add('cold', 100000)
        for _ in range(8):
            rig.step(prefill_ms=700.0)
        self.assertGreaterEqual(rig.runtime.step_times.observed, 1)
        self.assertAlmostEqual(rig.runtime.step_times.scale, 1.0, delta=0.5)

    def test_s3_a_long_that_waited_past_the_park_bound_goes_before_a_newly_arrived_short(self):
        rig = Rig(environ={'QWEN_FAST_LEVERN_MAX_PARK_S': '30'})
        now = rig.wall.now
        rig.add('old-long', 50000, arrival=now - 100.0)
        rig.add('short', 5000, arrival=now)
        self.assertEqual(rig.prefill_step()[1], 'old-long')
        fresh = Rig(environ={'QWEN_FAST_LEVERN_MAX_PARK_S': '30'})
        fresh.add('new-long', 50000, arrival=fresh.wall.now - 5.0)
        fresh.add('short', 5000, arrival=fresh.wall.now)
        self.assertEqual(fresh.prefill_step()[1], 'short', 'inside the bound the short still goes first')

    def test_s6_a_parked_long_keeps_its_place_in_running_so_it_is_not_the_first_preemption_victim(self):
        rig = Rig()
        rig.add('long', 100000)
        rig.prefill_step()
        rig.add('short', 5000)
        rig.prefill_step()
        names = [request.request_id for request in rig.scheduler.running]
        self.assertLess(names.index('long'), names.index('d0'), 'vLLM preempts the LAST running request: the parked long one is not appended behind the decoder')

    def test_s4_a_waiting_request_is_classified_again_until_it_is_admitted(self):
        rig = Rig()
        request = rig.add('x', 100000, hit=95000)
        graft = rig.scheduler.__dict__['_qwen_prefix']
        self.assertEqual(rig.runtime.klass_of(rig.scheduler, request), 'short')
        request.hit = 0                          # the checkpoint was evicted before the request was admitted
        graft.memo.clear()
        self.assertEqual(rig.runtime.klass_of(rig.scheduler, request), 'long', 'a cold 100k prompt does not run as short')


class InstallTests(GateFreeCase):
    def test_install_gives_the_merged_runtime_only_beside_prefix_reuse(self):
        cls = plugin_class(TrimmingScheduler)
        config = SimpleNamespace(scheduler_config=SimpleNamespace(scheduler_cls=cls))
        with patch.dict(os.environ, {'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_ANY_REQUEST': '1', 'QWEN_PREFIX_REUSE': '1', 'QWEN_FAST_STICKY_SESSIONS': '1',
                                     'QWEN_FAST_LEVERN_PARK': 'host'}, clear=False):
            for name in ('QWEN_FAST_DECODE_STEPS_PER_ADMISSION', 'QWEN_FAST_KV_RESERVATION'):
                os.environ.pop(name, None)
            admission.install(config, log=Mock())
        self.assertTrue(getattr(cls.schedule, levern_scheduler.WRAPPED))
        scheduler = new_scheduler(cls)
        scheduler.add_decoder('d0')
        scheduler.add_request(HitRequest('a', 50000))
        scheduler.schedule()
        alone = plugin_class(TrimmingScheduler)
        config = SimpleNamespace(scheduler_config=SimpleNamespace(scheduler_cls=alone))
        with patch.dict(os.environ, {'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_ANY_REQUEST': '1'}, clear=False):
            for name in ('QWEN_PREFIX_REUSE', 'QWEN_FAST_STICKY_SESSIONS', 'QWEN_FAST_DECODE_STEPS_PER_ADMISSION', 'QWEN_FAST_KV_RESERVATION',
                         'QWEN_FAST_LEVERN_PARK'):
                os.environ.pop(name, None)
            admission.install(config, log=Mock())
        self.assertTrue(getattr(alone.schedule, levern_scheduler.WRAPPED))

    def test_the_threshold_fallback_is_refused_beside_the_graft(self):
        rig = Rig()
        del rig.scheduler.max_num_scheduled_tokens
        rig.scheduler.scheduler_config = SimpleNamespace(long_prefill_token_threshold=0)
        rig.add('cold', 50000)
        with self.assertRaisesRegex(ValueError, 'threshold'):
            rig.prefill_step()

    def test_the_graft_marker_names_are_the_wrappers(self):
        import qwen_prefix_scheduler_patch as graft

        self.assertEqual(graft.LEVERN_SCHEDULE_WRAPPED, levern_scheduler.WRAPPED)
        self.assertEqual(graft.ADMISSION_WRAPPED, admission.WRAPPED)
        self.assertEqual(graft.LEVERN_ENV, levern_policy.FLAG)
        self.assertEqual(levern_scheduler.GRAFT_ATTR, '_qwen_prefix')


class SkewTests(GateFreeCase):
    """concurrent8_skew (v584 / v620): two cold prompts of 251,727 and 253,238 tokens arrive together beside four decoders. Their 180 s targets cannot
    both be met even at share 1.0, so the governor pins 1.0; the decoders must still get a round every QWEN_FAST_LEVERN_MAX_DECODE_GAP_S seconds (they
    got none for one continuous 199-207 s stretch spanning both longs), and the second long must still land inside the 240 s client deadline."""
    LONGS = (('long1', 251727), ('long2', 253238))
    ROUND_MS = 167.0

    @staticmethod
    def step_ms(who, start, tokens):
        """0.31-1.23 s per 2,048-token chunk over the context, plus the engine build on the final step."""
        ms = (300.0 + 3.7 * start / 1000.0) * max(1, -(-tokens // CHUNK))
        return ms + (5000.0 if start + tokens >= dict(SkewTests.LONGS)[who] else 0.0)

    def simulate(self, gap):
        rig = Rig(environ={'QWEN_FAST_LEVERN_TTFT_TARGET_S': '180', 'QWEN_FAST_LEVERN_ROUNDS': None, 'QWEN_FAST_LEVERN_PREFILL_SHARE': '0.5',
                           'QWEN_FAST_LEVERN_PARK': '0', 'QWEN_FAST_LEVERN_MAX_DECODE_GAP_S': str(gap)}, seats=8, decoders=4)
        t0 = rig.clock.now
        for name, prompt in self.LONGS:
            rig.add(name, prompt)
        finished, rounds, guard, longest = {}, [], 0, 0.0
        while len(finished) < 2:
            guard += 1
            self.assertLess(guard, 5000)
            event = rig.step(prefill_ms=self.step_ms, decode_ms=self.ROUND_MS)
            if event[0] == 'decode':
                rounds.append(rig.clock.now - t0)
            else:
                longest = max(longest, self.step_ms(event[1], event[2], event[3]) / 1000.0)
                if event[2] + event[3] >= dict(self.LONGS)[event[1]]:
                    finished[event[1]] = rig.clock.now - t0
        lines = [call.args for call in rig.log.call_args_list if call.args and call.args[0] == levern_policy.STEP_LINE_MERGED]
        f_effs = [line[15] for line in lines]
        edges = [0.0] + rounds + [max(finished.values())]
        worst = max(b - a for a, b in zip(edges, edges[1:]))
        return dict(finished=finished, worst=worst, longest=longest, f_effs=f_effs, decode_s=len(rounds) * self.ROUND_MS / 1000.0, wall=rig.clock.now - t0)

    def test_the_decoders_get_a_round_every_g_seconds_through_a_pinned_one(self):
        run = self.simulate(8)
        self.assertIn(1.0, run['f_effs'], 'the shape cannot meet T*: the governor pins 1.0')
        self.assertLessEqual(run['worst'], 8.0 + run['longest'] + 1.0, 'G + the longest step + 1 s, the smoke-rule bound (a step cannot be split); without the floor the stall is a whole long')
        self.assertLess(max(run['finished'].values()), 240.0, 'the second long still lands inside the client deadline')
        off = self.simulate(0)
        self.assertLessEqual((run['decode_s'] - off['decode_s']) / run['wall'], 0.03, 'the floor itself costs at most 3% of the wall time')
        self.assertLessEqual(run['wall'] - off['wall'], 0.03 * off['wall'])

    def test_without_the_floor_the_decoders_stall_for_a_whole_long(self):
        run = self.simulate(0)
        self.assertGreater(run['worst'], 150.0, 'v584 / v620: one stretch of 199 / 207 s without a decode round, across both longs')


class SkewWithShortsTests(GateFreeCase):
    """The whole concurrent8_skew shape (v620): two cold longs and six 4k shorts, the short lane on, the measured step times (a short step 4.4 s with its
    engine build, a long's final step 5 s of build, the first decode round after a seat joins 0.93 s). The smoke rule runs over the scheduler's own step
    lines. The numbers in docs/lever-n-prefix-merged-route.md section 11 come from this replay."""
    LONGS = {'long1': 251727, 'long2': 253238}
    SHORTS = {'s1': 4088, 's2': 4083, 's3': 4062, 's5': 4087, 's6': 4082, 's7': 4087}
    ORDER = ('s1', 'long1', 's3', 's6', 's2', 'long2', 's5', 's7')
    ROUND_MS, FIRST_ROUND_MS = 167.0, 930.0

    def step_ms(self, who, start, tokens):
        if who in self.SHORTS:
            return 4400.0
        prompt = self.LONGS[who]
        return (300.0 + 3.7 * start / 1000.0) * max(1, -(-tokens // CHUNK)) + (5000.0 if start + tokens >= prompt else 0.0)

    def replay(self, gap, build_ms=2500.0):
        prompts = dict(self.LONGS, **self.SHORTS)
        saved = levern_policy.effective_share.__defaults__, levern_policy.governor_need.__defaults__
        # build_ms is the first defaulted parameter; the trailing ones (builds, engine reuse's per-prefill costs) keep theirs.
        levern_policy.effective_share.__defaults__ = (build_ms,) + saved[0][1:]
        levern_policy.governor_need.__defaults__ = (build_ms,) + saved[1][1:]
        try:
            rig = Rig(environ={'QWEN_FAST_LEVERN_TTFT_TARGET_S': '180', 'QWEN_FAST_LEVERN_ROUNDS': None, 'QWEN_FAST_LEVERN_PREFILL_SHARE': '0.5',
                               'QWEN_FAST_LEVERN_PARK': 'host', 'QWEN_FAST_LEVERN_MAX_DECODE_GAP_S': str(gap)}, seats=8, decoders=0)
            t0 = rig.clock.now
            for index, name in enumerate(self.ORDER):
                rig.add(name, prompts[name], arrival=rig.wall.now + 0.05 * index)
            ttft, rounds, joined = {}, [], False
            for _ in range(20000):
                if len(ttft) == len(self.ORDER):
                    break
                event = rig.step(prefill_ms=self.step_ms, decode_ms=self.ROUND_MS)
                if event[0] == 'decode':
                    if joined:
                        rig.clock.advance_ms(self.FIRST_ROUND_MS - self.ROUND_MS)
                        rig.wall.advance_ms(self.FIRST_ROUND_MS - self.ROUND_MS)
                        joined = False
                    rounds.append(rig.clock.now - t0)
                elif event[2] + event[3] >= prompts[event[1]]:
                    ttft[event[1]] = rig.clock.now - t0
                    joined = True
            else:
                self.fail('the replay made no progress')
        finally:
            levern_policy.effective_share.__defaults__, levern_policy.governor_need.__defaults__ = saved
        end = max(ttft.values())
        edges = [0.0] + [r for r in rounds if r <= end] + [end]
        lines = [levern_policy.STEP_LINE_MERGED.format(*call.args[1:]) for call in rig.log.call_args_list
                 if call.args and call.args[0] == levern_policy.STEP_LINE_MERGED]
        return dict(ttft=ttft, worst=max(b - a for a, b in zip(edges, edges[1:])), lines=lines)

    def rule(self, run, gap):
        steps = c2_smoke_check.levern_facts(chr(10).join(run['lines']))['steps']
        return c2_smoke_check.levern_alternation_problems(steps, {'QWEN_FAST_LEVERN_PREFILL_SHARE': '0.5', 'QWEN_FAST_LEVERN_MAX_DECODE_GAP_S': str(gap)})

    def test_the_floor_bounds_the_stall_and_the_slowest_user_stays_inside_the_deadline(self):
        on, off = self.replay(8), self.replay(0)
        self.assertEqual(self.rule(on, 8), [], 'the smoke rule is clean over the scheduler\'s own lines with the floor on')
        self.assertLess(on['worst'], 8.0 + 8.0 + 1.0, 'the smoke rule above is the one bound (G + longest step + 1 s); this is its sanity ceiling')
        self.assertGreater(off['worst'], 150.0)
        self.assertLess(max(on['ttft'].values()), 240.0 - 3.0, 'the slowest user keeps at least 3 s inside the client deadline')
        self.assertLessEqual(on['ttft']['long2'] - off['ttft']['long2'], 6.0, 'the floor costs the second long at most 6 s')
        self.assertLessEqual(max(on['ttft'].values()) - max(off['ttft'].values()), 3.0, 'and the slowest user at most 3 s')

    def test_the_smoke_rule_reads_the_floorless_run_as_one_stall_of_about_two_hundred_seconds(self):
        off = self.replay(0)
        found = self.rule(off, 8)
        gaps = [problem for problem in found if 'decode-gap floor' in problem]
        self.assertEqual(len(gaps), 1, found)
        stall = float(gaps[0].split('decoders went ')[1].split(' s')[0])
        self.assertGreater(stall, 150.0)
        self.assertEqual(len(found), 1, 'the shorts keep their round with the floor off: the stall is the only problem')

    def test_the_shorts_of_the_skew_are_admitted_early_at_the_modelled_build(self):
        on = self.replay(8)
        self.assertEqual(sum(1 for name in self.SHORTS if on['ttft'][name] < 60.0), 5)

    def test_a_five_second_build_closes_the_short_lane_for_one_more_short(self):
        """Why BUILD_MS stays 2500: at 5000 the governor pins f_eff at 1.0 earlier and a short waits behind both longs."""
        late = [name for name in self.SHORTS if self.replay(8, build_ms=5000.0)['ttft'][name] > 200.0]
        self.assertEqual(len(late), 2)


if __name__ == '__main__':
    unittest.main()
