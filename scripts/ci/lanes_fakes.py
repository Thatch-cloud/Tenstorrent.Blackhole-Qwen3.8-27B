"""Fakes for driving the fast lane end to end on a CPU: a vLLM-shaped scheduler, the plugin runner's persistent batch, the packed step's
blocks, and a deterministic target model - all in one small world (LaneWorld) whose clock the blocks advance by the modelled
round times (lanes_sim.ROUND_MS).

What is REAL in the loop the tests drive (test_fast_lane_hook):
  serving_worker_hook.FastWorkerHook       _drafts (the lane plan, members-only drafting) and _lane_round / _lane_bridges
  serving_fast_lane.LaneRuntime            the book, the controller, the plan, the gate the scheduler reads
  serving_fast_lane_scheduler              the installed wrappers, on a class shaped like TTScheduler (waiting / running queues,
                                           _has_pending_prefill, schedule, _schedule_decode_only, _schedule_prefill_only)
  serving_packed_step.PackedStep           the routing (solo block / M3 / sequential), commit_entry, the fixture-epoch bump
  serving_packed_bridge.execute_packed_decode, serving_vllm_packed.admit_packed_scheduler_output, serving_vllm_state
                                           (validate_runner_reservation: the runner's persistent batch must hold EXACTLY the
                                           scheduled requests, on their frontiers)
  serving_runner_bridge.FastRunnerBridge   drafts()
What is a fake: the vLLM scheduler's request bookkeeping, the runner's `_update_states` (the plugin's algorithm: unscheduled requests
leave the persistent batch and keep their state, and return when next scheduled), the blocks' verify, the drafter.

The target model is a pure function of a user's OWN history: the token after position p, whose previous token is t, is
next_token(user, p, t). A round's ticket is (seed, proposals...); the block's predictions are next_token along the ticket; the user commits
the leading proposals the target agrees with plus the target's own token. So whatever frame of rounds serves a user - packed with others,
solo, narrowed, held out for a while - its committed stream is next_token's chain from its prompt: the exactness invariant the tests
assert per lane. A stale proposal, a wrong frontier or a segment mix-up breaks the chain or a state check.
"""

import collections
from itertools import count
import random
import sys
from types import ModuleType, SimpleNamespace

import numpy

from serving_fast_request import CommittedOutput
from test_serving_packed_step import FakeBlock, FakeEngine, FakeRuntime, FakeSession

VOCAB = 100000
FIRST_POSITION_OFFSET = 0


def next_token(user, position, previous):
    """The target's token after `previous` at `position`, for `user`: a function of the user's own history alone."""
    return 10 + (user * 1000003 + position * 7919 + previous * 31 + 17) % (VOCAB - 10)


def true_stream(user, seed, position, count_):
    """The next `count_` tokens after `seed` (the last committed token, at `position`)."""
    tokens, previous = [], seed
    for index in range(count_):
        previous = next_token(user, position + index, previous)
        tokens.append(previous)
    return tokens


def accept_length(user, position, tau, width, spread=1.5):
    """How many leading proposals the drafter gets right for a round at `position`: a pure function of (user, position), mean tau - 1."""
    rng = random.Random(user * 1315423911 + position)
    return max(0, min(width - 1, int(round(rng.gauss(tau - 1.0, spread)))))


def draft(user, position, seed, width, tau):
    """A ticket (seed, proposals...) of `width` rows: the first accept_length proposals are the target's own tokens, the rest are not."""
    truth = true_stream(user, seed, position, width - 1)
    right = accept_length(user, position, tau, width)
    proposals = [token if index < right else token + 1 for index, token in enumerate(truth)]
    return (seed, *proposals)


class Clock:
    """A virtual monotonic clock, in seconds; the blocks advance it by the modelled round time."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, milliseconds):
        self.now += milliseconds / 1000.0


class LaneRequest:
    """serving_fast_request.FastRequest as the lane loop drives it: a session, an engine, a runtime, prepare() and the sequential step."""

    def __init__(self, request_id, user, position, seed, tau, stepped, max_new_tokens=256):
        self.user, self.tau, self.stepped = user, tau, stepped
        self.session = FakeSession(request_id, position)
        self.session.max_new_tokens = max_new_tokens
        self.engine = FakeEngine(self.session)
        self.runtime = FakeRuntime(self.engine)
        self.closed = self.cancelled = self.busy = False
        self.collect_timings = False
        self.seed = seed
        self.rows_stepped = []

    @property
    def last_token(self):
        return self.session.emitted[-1] if self.session.emitted else self.seed

    def prepare(self, request_id, packed_rows=None):
        width = 4 if packed_rows is None else packed_rows
        session = self.session
        session.propose(draft(self.user, session.position, self.last_token, width, self.tau))

    def close(self, request_id):
        self.closed = True

    def step(self, request_id, *, cancelled):
        """The per-request engines' round (the sequential fallback): the pending ticket verified by the target model, committed."""
        flag = cancelled()
        self.stepped.append((request_id, flag))
        ticket = self.session.pending
        self.rows_stepped.append(len(ticket.tokens))
        if not self.engine.serves(ticket):
            raise ValueError('The sequential step reached a ticket its own engine never captured')
        predictions = [next_token(self.user, ticket.position + index, ticket.tokens[index])
                       for index in range(len(ticket.tokens))]
        def publish(prefix):
            self.engine.position += prefix

        decision = self.session.commit(request_id, ticket, predictions, publish)
        return CommittedOutput(request_id, tuple(decision.emitted), self.session.position, False)


class ModelBlock(FakeBlock):
    """A packed block that verifies with the target model, and advances the world's clock by the modelled time of its round."""

    def __init__(self, users, clock, cost, *, extent=True):
        super().__init__(users=users)
        self.clock, self.cost, self.extent = clock, cost, extent
        self.users_by_engine = {}
        self.round_log = []

    def bind(self, engine, segment, user=None):
        super().bind(engine, segment)
        self.users_by_engine[id(engine)] = user

    def admits(self, position):
        return 128 <= position and position + 16 <= 131328

    def accept_limit(self, position):
        return None

    def verify(self, entries):
        entries = list(entries)
        _, metrics = super().verify(entries)
        real = []
        for item in entries:
            request, ticket = item['request'], item['ticket']
            real.append([next_token(request.user, ticket.position + index, ticket.tokens[index])
                         for index in range(len(ticket.tokens))])
        self.clock.advance(self.cost(len(entries)))
        self.round_log.append(tuple(item['request_id'] for item in entries))
        return real, metrics


class PaddedModelBlock(ModelBlock):
    """M3: four segments, padded rounds of two or three live users (packed_verifier.PackedVerifierEngine.pads)."""

    def __init__(self, clock, cost):
        super().__init__(4, clock, cost)

    def pads(self, count):
        return 2 <= count < 4

    def padded_refusal(self, users):
        return None


class World:
    """One engine: a clock, a scheduler class with the lane gate installed, a runner, the packed step and the lane runtime."""

    def __init__(self, *, round_ms, sigma_ms=1.0, config=None, install=True, log=None, reserve_lines=None):
        from serving_fast_lane import LaneConfig, LaneRuntime, GATE_KEY
        import serving_fast_lane_scheduler as gate
        from serving_packed_step import PackedStep
        from serving_worker_hook import FastWorkerHook

        sys.modules.pop(GATE_KEY, None)
        self.clock = Clock()
        self.stepped = []
        self.lines = [] if log is None else log
        self.sigma_ms = sigma_ms
        self.last_kind = None
        solo_ms, packed = round_ms

        def switch_cost(kind):
            extra = self.sigma_ms if self.last_kind is not None and self.last_kind != kind else 0.0
            self.last_kind = kind
            return extra

        self.m3 = PaddedModelBlock(self.clock, lambda live: packed[live] + switch_cost('packed'))
        self.solo = ModelBlock(1, self.clock, lambda live: solo_ms + switch_cost('solo'))
        self.step = PackedStep(self.m3, solo=self.solo)
        config = LaneConfig.from_environment({}) if config is None else config
        self.config = config
        self.lanes = LaneRuntime(config, log=self._log, clock=self.clock)
        self.runner = LaneRunner()
        self.worker = LaneWorker(self.runner)
        self.scheduler_class = type('LaneTTScheduler', (FakeTTScheduler,), {})
        if install:
            gate.install(SimpleNamespace(scheduler_config=SimpleNamespace(scheduler_cls=self.scheduler_class)),
                         log=self._log, queue_factory=lambda scheduler: FakeQueue())
        self.scheduler = self.scheduler_class()
        self.hook = None
        self.users = collections.OrderedDict()      # request id -> LaneRequest
        self.FastWorkerHook = FastWorkerHook
        self.outputs_module = ModuleType('vllm.v1.outputs')
        self.outputs_module.ModelRunnerOutput = SimpleNamespace
        self.outputs_module.DraftTokenIds = SimpleNamespace
        self.commits = collections.defaultdict(list)     # request id -> [(clock, tokens)]
        self.kinds = []
        self.observed = []
        self.steps = 0

    def _log(self, template, *values):
        self.lines.append(template.format(*values))

    def add_user(self, request_id, user, *, lane=None, tau=12.1, prompt_length=4100, max_tokens=256):
        """Admit a user the way the bridge factory does: the lane grant, the slot it names, the engine bound to that slot's segment, the
        runner state and the scheduler request."""
        from serving_runner_bridge import FastRunnerBridge

        params = SimpleNamespace(extra_args=None if lane is None else {'qwen_lane': lane}, max_tokens=max_tokens)
        grant = self.lanes.admit(request_id, params, slot0_free=True)
        used = {segment for segment in self.m3.bound.values()}
        slot = next(index for index in grant.slot_order if index not in used)
        seed = next_token(user, prompt_length - 1, 1)
        state = SimpleNamespace(req_id=request_id, prompt_token_ids=list(range(prompt_length)), output_token_ids=[seed],
                                block_ids=([1, 2],), sampling_params=params, num_computed_tokens=prompt_length)
        self.runner.requests[request_id] = state
        request = LaneRequest(request_id, user, prompt_length, seed, tau, self.stepped, max_new_tokens=max_tokens)
        self.m3.bind(request.engine, slot, user)
        if slot == 0:
            self.solo.bind(request.engine, 0, user)
        binding = SimpleNamespace(engine=request.engine, refresh=lambda *args, **options: None)
        bridge = FastRunnerBridge(self.runner, request, binding)
        bridge.state = state
        self.users[request_id] = request
        scheduled = SchedRequest(request_id, params, prompt_length, state)
        self.scheduler.running.append(scheduled)
        if self.hook is None:
            self.hook = self.FastWorkerHook(self.worker, bridge, cancelled=lambda: False, packed_step=self.step,
                                            lanes=self.lanes)
        else:
            self.hook.attach(bridge)
        return grant, slot, request

    def finish_user(self, request_id):
        self.scheduler.finish(request_id)

    def engine_step(self):
        """One EngineCore step: schedule, execute, update, draft. Returns the SchedulerOutput."""
        from unittest.mock import patch

        scheduler = self.scheduler
        before = {request.request_id: (request.num_computed_tokens, list(request.spec_token_ids))
                  for request in scheduler.running}
        scheduled = scheduler.schedule()
        self.observed.append(dict(scheduled=list(scheduled.scheduled_cached_reqs.req_ids), before=before,
                                  members=getattr(self.lanes.gate, 'members', None)))
        # the lifecycle detaches a request vLLM names finished before the step runs (while others still decode)
        for request_id in sorted(scheduled.finished_req_ids):
            if request_id in self.hook.bridges and len(self.hook.bridges) > 1:
                request = self.users[request_id]
                self.hook.detach(request_id)
                for block in (self.m3, self.solo):
                    block.bound.pop(id(request.engine), None)
        with patch.dict(sys.modules, {'vllm.v1.outputs': self.outputs_module}):
            if scheduled.total_num_scheduled_tokens:
                output = self.runner.execute_model(scheduled)
                self.kinds.append(getattr(self.step, 'route', None))
                scheduler.update_from_output(scheduled, output)
                for request_id, tokens in zip(output.req_ids, output.sampled_token_ids):
                    self.commits[request_id].append((self.clock.now, len(tokens)))
                # vLLM (EngineCore.post_step) takes drafts only after a step that ran the model: an empty step never does.
                drafts = self.worker.take_draft_token_ids()
            else:
                drafts = None
        if drafts is not None:
            scheduler.update_draft_token_ids(drafts)
        self.steps += 1
        return scheduled

    def prime(self):
        """The first drafts, as the tick after the last admission makes them."""
        from unittest.mock import patch

        with patch.dict(sys.modules, {'vllm.v1.outputs': self.outputs_module}):
            drafts = self.worker.take_draft_token_ids()
        if drafts is not None:
            self.scheduler.update_draft_token_ids(drafts)

    def rate(self, request_id, *, after=0.0):
        events = [event for event in self.commits[request_id] if event[0] >= after]
        if len(events) < 2:
            return 0.0
        return sum(tokens for _, tokens in events[1:]) / (events[-1][0] - events[0][0])


class FakeQueue:
    """vLLM's RequestQueue: FCFS, add_request / prepend_requests / iteration / truthiness."""

    def __init__(self, items=()):
        self.items = list(items)

    def add_request(self, request):
        self.items.append(request)

    def prepend_requests(self, other):
        self.items = list(other) + self.items

    def remove(self, request):
        self.items.remove(request)

    def __iter__(self):
        return iter(list(self.items))

    def __len__(self):
        return len(self.items)

    def __bool__(self):
        return bool(self.items)


class SchedRequest:
    """vllm.v1.request.Request, as far as the scheduler and the lane gate read it."""

    def __init__(self, request_id, sampling_params, prompt_length, state, *, chunk=False):
        self.request_id = request_id
        self.sampling_params = sampling_params
        self.is_prefill_chunk = chunk
        self.prompt_length = prompt_length
        self.state = state
        self.num_computed_tokens = prompt_length
        self.spec_token_ids = []
        self.output_token_ids = [state.output_token_ids[0]] if state is not None else []

    @property
    def max_tokens(self):
        return self.sampling_params.max_tokens


class FakeTTScheduler:
    """vllm_tt_plugin.scheduler.TTScheduler (and the base scheduler's decode loop) in the parts the lane gate touches."""

    def __init__(self):
        self.running = []
        self.waiting = FakeQueue()
        self.skipped_waiting = FakeQueue()
        self.finished_req_ids = set()
        self._forced_mode = None
        self.max_num_running_reqs = 4
        self.policy = 'fcfs'
        self.calls = []

    def _has_pending_prefill(self):
        return bool(self.waiting) or bool(self.skipped_waiting) or any(r.is_prefill_chunk for r in self.running)

    def schedule(self, throttle_prefills=False):
        has_pending_prefill = self._has_pending_prefill()
        has_running_decode = any(not r.is_prefill_chunk for r in self.running)
        if getattr(self._forced_mode, 'name', None) == 'PREFILL_ONLY':
            return self._schedule_prefill_only()
        if has_pending_prefill:
            prefill = self._schedule_prefill_only()
            if prefill.total_num_scheduled_tokens == 0 and has_running_decode:
                return self._schedule_decode_only()
            return prefill
        self.calls.append('base')
        return self._decode_step()

    def _schedule_decode_only(self):
        self.calls.append('decode-only')
        saved_waiting, saved_skipped = self.waiting, self.skipped_waiting
        self.waiting, self.skipped_waiting = FakeQueue(), FakeQueue()
        partials = [r for r in self.running if r.is_prefill_chunk]
        if partials:
            self.running = [r for r in self.running if not r.is_prefill_chunk]
        try:
            return self._decode_step()
        finally:
            self.waiting, self.skipped_waiting = saved_waiting, saved_skipped
            if partials:
                self.running.extend(partials)

    def _schedule_prefill_only(self):
        """The plugin's: decodes hidden, the waiting head admitted when a seat is free. Here a step that admits nothing is empty; a
        prefill itself is the lifecycle's business, not this fake's."""
        self.calls.append('prefill-only')
        pure_decodes = [r for r in self.running if not r.is_prefill_chunk]
        partials = [r for r in self.running if r.is_prefill_chunk]
        saved_max = self.max_num_running_reqs
        self.running = partials
        self.max_num_running_reqs = max(0, saved_max - len(pure_decodes))
        try:
            admitted = []
            while self.waiting and len(self.running) + len(admitted) < self.max_num_running_reqs:
                head = next(iter(self.waiting))
                self.waiting.remove(head)
                admitted.append(head)
            self.admitted = admitted
            return SimpleNamespace(scheduled_new_reqs=list(admitted), scheduled_cached_reqs=_cached([], []),
                                   num_scheduled_tokens={r.request_id: r.prompt_length for r in admitted},
                                   total_num_scheduled_tokens=sum(r.prompt_length for r in admitted),
                                   scheduled_spec_decode_tokens={}, finished_req_ids=self._take_finished(),
                                   preempted_req_ids=set())
        finally:
            self.running.extend(pure_decodes)
            self.max_num_running_reqs = saved_max

    def _take_finished(self):
        finished, self.finished_req_ids = self.finished_req_ids, set()
        return finished

    def _decode_step(self):
        """The base scheduler's running loop: every running decode, 1 + its proposals tokens each, proposals consumed."""
        order, positions, tokens, spec = [], [], {}, {}
        for request in self.running:
            if request.is_prefill_chunk:
                continue
            order.append(request.request_id)
            positions.append(request.num_computed_tokens)
            proposals = list(request.spec_token_ids)
            tokens[request.request_id] = 1 + len(proposals)
            if proposals:
                spec[request.request_id] = proposals
            request.spec_token_ids = []
        return SimpleNamespace(scheduled_new_reqs=[], scheduled_cached_reqs=_cached(order, positions),
                               num_scheduled_tokens=tokens, total_num_scheduled_tokens=sum(tokens.values()),
                               scheduled_spec_decode_tokens=spec, finished_req_ids=self._take_finished(),
                               preempted_req_ids=set(), has_structured_output_requests=False, scheduled_encoder_inputs={})

    def update_from_output(self, scheduled, output):
        by_id = dict(zip(output.req_ids, output.sampled_token_ids))
        keep = []
        for request in self.running:
            tokens = by_id.get(request.request_id)
            if not tokens:
                keep.append(request)
                continue
            request.output_token_ids.extend(tokens)
            request.num_computed_tokens += len(tokens)
            del request.output_token_ids[request.max_tokens:]
            if len(request.output_token_ids) >= request.max_tokens:
                self.finished_req_ids.add(request.request_id)
                continue
            keep.append(request)
        self.running = keep

    def update_draft_token_ids(self, drafts):
        by_id = {request.request_id: request for request in self.running}
        for request_id, tokens in zip(drafts.req_ids, drafts.draft_token_ids):
            if request_id in by_id:
                by_id[request_id].spec_token_ids = list(tokens)

    def finish(self, request_id):
        self.running = [r for r in self.running if r.request_id != request_id]
        self.finished_req_ids.add(request_id)


def _cached(order, positions):
    return SimpleNamespace(req_ids=list(order), num_computed_tokens=list(positions), new_block_ids=[None] * len(order),
                           resumed_req_ids=set())


class LaneBatch:
    """The runner's persistent batch (input_batch): rows condensed to 0..n-1, each holding the request's shared output list."""

    def __init__(self, capacity=8, width=16384):
        self.req_id_to_index = {}
        self.req_output_token_ids = [None] * capacity
        self.num_tokens = numpy.zeros(capacity, dtype=numpy.int32)
        self.num_computed_tokens_cpu = numpy.zeros(capacity, dtype=numpy.int32)
        self.token_ids_cpu = numpy.zeros((capacity, width), dtype=numpy.int32)
        self.layout_changes = 0

    @property
    def num_reqs(self):
        return len(self.req_id_to_index)

    def add_request(self, state):
        row = min(set(range(len(self.req_output_token_ids))) - set(self.req_id_to_index.values()))
        self.req_id_to_index[state.req_id] = row
        self.req_output_token_ids[row] = state.output_token_ids
        tokens = list(state.prompt_token_ids) + list(state.output_token_ids)
        self.num_tokens[row] = len(tokens)
        self.token_ids_cpu[row, :len(tokens)] = numpy.array(tokens, dtype=numpy.int32)
        self.num_computed_tokens_cpu[row] = state.num_computed_tokens
        self.layout_changes += 1

    def remove_request(self, request_id):
        row = self.req_id_to_index.pop(request_id, None)
        if row is not None:
            self.req_output_token_ids[row] = None
            self.layout_changes += 1
        return row

    def condense(self):
        """Move the rows above the live count into the holes below it, as vLLM's condense does."""
        live = sorted(self.req_id_to_index.items(), key=lambda item: item[1])
        for target, (request_id, row) in enumerate(live):
            if row == target:
                continue
            self.req_id_to_index[request_id] = target
            self.req_output_token_ids[target], self.req_output_token_ids[row] = self.req_output_token_ids[row], None
            self.num_tokens[target] = self.num_tokens[row]
            self.num_computed_tokens_cpu[target] = self.num_computed_tokens_cpu[row]
            self.token_ids_cpu[target] = self.token_ids_cpu[row]


class LaneRunner:
    """The pinned plugin's TTModelRunner state, in the algorithm of its _update_states: finished requests leave; a running request the
    step does not schedule leaves the persistent batch but KEEPS its state; a scheduled one not in the batch is added again."""

    def __init__(self):
        self.requests = {}
        self.input_batch = LaneBatch()
        self.model_config = SimpleNamespace(max_model_len=16384, get_vocab_size=lambda: VOCAB)
        self.non_dp_async_scheduling = False
        self.tt_data_parallel_size = 1
        self._pending_samples = collections.deque()
        self.updates = []

    def execute_model(self, scheduled):
        raise AssertionError('the stock forward must never run: the hook owns every decode step')

    def sample_tokens(self, grammar_output):
        raise AssertionError('the stock sampler must never run: the hook owns every decode step')

    def _update_states(self, scheduled):
        batch = self.input_batch
        for request_id in scheduled.finished_req_ids:
            self.requests.pop(request_id, None)
            batch.remove_request(request_id)
        scheduled_ids = set(scheduled.num_scheduled_tokens)
        for request_id in [rid for rid in batch.req_id_to_index if rid not in scheduled_ids]:
            batch.remove_request(request_id)
        data = scheduled.scheduled_cached_reqs
        for index, request_id in enumerate(data.req_ids):
            state = self.requests[request_id]
            state.num_computed_tokens = data.num_computed_tokens[index]
            row = batch.req_id_to_index.get(request_id)
            if row is None:
                batch.add_request(state)
            else:
                batch.num_computed_tokens_cpu[row] = data.num_computed_tokens[index]
        batch.condense()
        self.updates.append(tuple(data.req_ids))


class LaneWorker:
    is_driver_worker = True

    def __init__(self, runner):
        self.model_runner = runner
