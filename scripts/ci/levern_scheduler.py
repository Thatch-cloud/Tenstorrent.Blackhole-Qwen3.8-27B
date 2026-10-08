"""Lever N at TP4: the scheduler side (QWEN_FAST_LEVER_N=1), installed on the class the engine runs.

serving_prefill_admission.install wraps TTScheduler._schedule_prefill_only (one fresh prompt per step) and, under
QWEN_FAST_LEVER_N=1, hands this module's LevernRuntime to that wrapper and wraps TTScheduler.schedule with
wrap_schedule below. The plugin's scheduler (bf77cd63) is not edited and not grafted: both are runtime wraps of its
own methods, the pattern serving_prefill_admission already proves (its positive control is the "live in
TTScheduler" marker, and here every decision logs its own step line).

WHAT IT DOES
  cap          A prefill step schedules at most levern_policy.step_budget tokens of the one request it advances:
               the partial prefill in flight, or the one fresh prompt it admits. The cap is vLLM's own token budget
               (Scheduler.max_num_scheduled_tokens, set for the call and restored in a finally), so vLLM does the
               splitting (num_computed_tokens, is_prefill_chunk, no sample for the intermediate row) and every
               non-final end lands on a 2,048-token boundary. The profile keeps max-num-batched-tokens at
               max-model-len: chunking happens only through this cap, so with the flag off nothing splits.
  alternation  schedule() asks the Alternator (levern_policy) before the plugin's default branch. When the
               decoders are owed a step it runs the plugin's own _schedule_decode_only, which hides the waiting
               queues and the partial prefill, so a decode round runs and the prefill resumes on the next call.
               Only the DEFAULT scheduling mode is touched; a forced mode is delegated unchanged.
  final gate   Before the FINAL step of a partial (the one that builds the engine) the registered DRAM predicate is
               asked, with the admission's terms, for the prompt. Short, with a decoder running: a decode step
               instead, logged once per state, so a fully prefilled 254k prompt is not refused by the backstop after
               about 92 s of work. With no decoder left to free memory nothing is held or asked. QWEN_FAST_LEVERN_FAULT=
               final-hold forces one such hold (a gate-only negative control).

WHAT IT LOGS (levern_policy.STEP_LINE), one line per scheduling decision while a prefill is pending or in
flight and none otherwise: kind (prefill or decode), the decode seats served, the request, its start, the tokens
scheduled, the end and the prompt, whether it is the final step, why, the previous step's wall time (the scheduler's own
clock) and what is still owed. c2_smoke_check reads them (alternation visible whenever decoders and a partial coexist).

ONE PREFILL IN FLIGHT (v1). serving_prefill_admission's rule is unchanged: partials hide the queues, so a second
prompt waits behind the first; v2's parking and shortest-remaining-first lift that.

THE MERGED ROUTE (docs/lever-n-prefix-merged-route.md; LevernRuntime.merged, QWEN_PREFIX_REUSE=1 beside Lever N). plan_pass replaces the v1
target rule:
  peek         a fresh admission's step budget is computed from (P, Q), the hit the G1 graft's trim WILL land on (SchedulerGraft.peek),
               not (P, 0): a hit at Q = S_last then takes the whole rest of the prompt in one step instead of splitting the drafter window
               (defect D1). verify() checks the window rule on what the pass really scheduled.
  single       the pass sees ONE candidate: the waiting queues hold only the target and every partial but the active one is hidden from
               `running`, so vLLM cannot admit another request in the pass (verify's "admitted another request" refusal stays, unreachable).
  quarantine   before anything is allocated, a partial whose suspended state is gone (the route's owner and parks, levern_policy.state_holder) or
               that exceeds the in-flight limit is ended as FINISHED_ABORTED through the D2 consumer (serving_request_quarantine): nothing is
               allocated or published for it, so no unwritten block is ever cached. A refusal AFTER allocation is never made per-request.
  short lane   QWEN_FAST_LEVERN_PARK=host: a waiting request whose remaining tokens after its peeked hit fit QWEN_FAST_LEVERN_SHORT_TOKENS is SHORT
               and is admitted ahead of a long prefill in flight (shortest remaining first), which the route parks to the host; the long
               resumes when no short can be admitted. A short never parks a short, a long parked for QWEN_FAST_LEVERN_MAX_PARK_S admits no more
               shorts, and a long the deadline governor needs (f_eff = 1) is never preempted. A short is paced at R = 1.
  governor     the Alternator's share is raised so every pending long finishes within QWEN_FAST_LEVERN_TTFT_TARGET_S (levern_policy.effective_share).
  kill switch  the file levern.off (levern_policy.OffSwitch): new prompts get a whole-prompt cap and the short lane closes; in-flight prefills
               finish through the route.
"""

import time

import levern_policy
import serving_prefill_admission as admission

WRAPPED = '_qwen_levern_schedule'
# The attribute the G1 scheduler graft is bound to (qwen_prefix_scheduler_patch.install: scheduler._qwen_prefix = graft).
GRAFT_ATTR = '_qwen_prefix'


class PassPlan(object):
    """What plan_pass decided for one prefill pass of the merged route.

    active       the partial prefill the pass advances (None: none, or a fresh admission instead)
    fresh        the one waiting request the pass admits as the SINGLE candidate of the waiting loop (None: none)
    hidden       the partial prefills hidden from `running` for the pass (every live partial but the active one, and the quarantined)
    quarantined  the partial prefills ended before the pass (their state is gone, or there are more than the limit)
    short        whether the request the pass advances is a short one (paced at R = 1)
    override_gate  the lifecycle's prefill gate names a partial this pass is parking for a short one: it must not hide the queues
    preempted    the long partial a short fresh admission parks (None when the pass is no preemption); demote() gives it back
    """

    def __init__(self):
        self.active = None
        self.fresh = None
        self.hidden = []
        self.quarantined = []
        self.short = False
        self.override_gate = False
        self.preempted = None
        self.reason = 'continue'

    @property
    def target(self):
        return self.fresh if self.fresh is not None else self.active

    def demote(self):
        """The short admission was held (the KV reservation or the DRAM hold): the long prefill it would have parked simply continues."""
        if self.preempted is not None:
            self.active, self.fresh = self.preempted, None
            self.hidden = [value for value in self.hidden if value is not self.preempted]
            self.preempted, self.override_gate, self.short = None, False, False
            self.reason = 'continue'
        else:
            self.fresh = None


def _dequeue(queue, request):
    """Take `request` out of a request queue (vLLM's RequestQueue has remove_request; the test fakes are lists)."""
    remover = getattr(queue, 'remove_request', None)
    try:
        if callable(remover):
            remover(request)
        else:
            queue.remove(request)
    except (ValueError, KeyError):
        pass


def _requests(queue):
    try:
        return list(queue or ())
    except Exception:
        return []



class LevernRuntime(object):
    """What the two wrappers share: the parsed flags, the Alternator, the step counter and the log."""

    def __init__(self, cfg=None, *, log=None, clock=None, fault=None, merged=None, wall=None, off_path=None):
        self.cfg = levern_policy.config() if cfg is None else cfg
        # The merged route's flags (levern_policy.Merged), None on the stage-1 route: every merged decision below is behind it.
        self.merged = merged
        kwargs = {} if clock is None else dict(clock=clock)
        if wall is not None:
            kwargs['wall'] = wall
        self.alternator = levern_policy.Alternator(self.cfg, merged=merged, **kwargs)
        self.log = admission._log if log is None else log
        self.fault = levern_policy.fault() if fault is None else fault
        self.fault_fired = False
        self.steps = 0
        self.final_held = None
        self.cap_fallback_logged = False
        self.computed = None
        self.step_times = levern_policy.StepTimes()
        self.clock = clock if clock is not None else time.monotonic
        self.wall = wall if wall is not None else time.time
        # request id -> 'short' | 'long' (fixed when the request is first seen), when it was hidden (monotonic) and how long it has been parked in all
        self.klass = {}
        self.pending_class = {}
        self.unmeasured = set()
        self.parked_since = {}
        self.parked_for = {}
        self.pass_short = False
        self.last_prefill = None
        self.quarantined = 0
        self.preemptions = 0
        self.off = levern_policy.OffSwitch(off_path, clock=self.clock) if merged is not None else None
        self.off_logged = False

    # ------------------------------------------------------------------ the merged route

    def graft(self, scheduler):
        """The G1 scheduler graft bound to this scheduler (qwen_prefix_scheduler_patch.SchedulerGraft), or None."""
        return getattr(scheduler, '__dict__', {}).get(GRAFT_ATTR)

    def peek(self, scheduler, request):
        """The Q the trim would land on for this waiting request (0 without a graft or without a hit)."""
        graft = self.graft(scheduler)
        peek = getattr(graft, 'peek', None)
        if not callable(peek):
            return 0
        try:
            value = peek(request)
        except Exception as failure:
            self.log(levern_policy.REFUSED_LINE, 'peek failed for %r (%s: %s): cap computed from Q=0' % (
                getattr(request, 'request_id', None), type(failure).__name__, str(failure)[:160]))
            return 0
        return value if type(value) is int and value >= 0 else 0

    def killed(self):
        """Whether the Lever N kill switch is engaged (polled at most once a second, latched)."""
        if self.off is None:
            return False
        if self.off.poll() and not self.off_logged:
            self.off_logged = True
            self.log(levern_policy.KILL_LINE, self.off.path)
        return self.off.engaged

    def klass_of(self, scheduler, request):
        """'short' or 'long', fixed the first time the request is seen: its remaining tokens (after the peeked hit) against the short class."""
        request_id = getattr(request, 'request_id', None)
        known = self.klass.get(request_id)
        if known is not None:
            return known
        running = getattr(scheduler, 'running', None) or ()
        admitted = any(value is request for value in running) or (getattr(request, 'num_computed_tokens', 0) or 0) > 0
        if admitted and request_id in self.pending_class:
            # The class the admitting pass decided from the hit it was trimmed to (commit_plan): fixed from here on.
            self.klass[request_id] = self.pending_class.pop(request_id)
            return self.klass[request_id]
        value = 'long'
        merged = self.merged
        if merged is not None and merged.park == 'host' and merged.short_tokens:
            prompt = admission.prompt_tokens(request)
            computed = getattr(request, 'num_computed_tokens', None)
            if type(prompt) is int and type(computed) is int:
                start = computed if computed else self.peek(scheduler, request)
                if prompt - start <= merged.short_tokens:
                    value = 'short'
        if admitted:
            self.klass[request_id] = value
        # A request still WAITING is classified afresh on every call (its hit may be evicted before it is admitted, or may appear): the class is
        # fixed at its admission (commit_plan), from the peek of that same call, so a cold 200k+ prompt whose checkpoint went never runs as short.
        return value

    def remaining(self, scheduler, request):
        prompt = admission.prompt_tokens(request)
        computed = getattr(request, 'num_computed_tokens', None)
        if type(prompt) is not int or type(computed) is not int:
            return None
        return prompt - (computed if computed else self.peek(scheduler, request))

    def waiting_in_order(self, scheduler):
        """The waiting requests in the order the base waiting loop takes them (skipped_waiting first), blocked heads left out."""
        check = getattr(scheduler, '_is_blocked_waiting_status', None)
        found = []
        for name in ('skipped_waiting', 'waiting'):
            for request in _requests(getattr(scheduler, name, None)):
                if not admission._blocked(check, request):
                    found.append(request)
        return found

    def prune(self, scheduler):
        """Forget the per-request state of requests the scheduler no longer holds."""
        known = getattr(scheduler, 'requests', None)
        if not isinstance(known, dict):
            return
        for table in (self.klass, self.parked_since, self.parked_for, self.pending_class):
            for request_id in [value for value in table if value not in known]:
                del table[request_id]

    def quarantine(self, request, reason):
        """End one request before the pass allocates anything for it (the module docstring): the D2 consumer finishes it as FINISHED_ABORTED at
        this step's update_from_output and the scheduler frees its blocks; the lifecycle and the route release its capture, owner and park on
        the finished id. A scheduler without a consumer keeps the refusal fatal, as before."""
        request_id = getattr(request, 'request_id', None)
        import serving_request_quarantine

        try:
            serving_request_quarantine.register(request_id, reason)
        except ValueError:
            self.log(levern_policy.REFUSED_LINE, 'request %r cannot continue (%s) and no quarantine consumer is installed' % (request_id, reason))
            raise
        self.quarantined += 1
        self.log(levern_policy.QUARANTINE_LINE, request_id, reason)

    def park_age(self, request_id):
        """How long this long prefill has been parked in all (seconds), the current stretch included."""
        total = self.parked_for.get(request_id, 0.0)
        since = self.parked_since.get(request_id)
        return total + (0.0 if since is None else self.clock() - since)

    def may_preempt(self, scheduler, live, active, short, decodes):
        """Whether the long partial `active` may be parked for the waiting short request `short` at this step boundary (design section 9): a park
        slot free, a seat free, the host able to hold the parked scratch, the long's total park time under QWEN_FAST_LEVERN_MAX_PARK_S, and the
        deadline governor not needing the long."""
        merged = self.merged
        if len(live) > merged.park_slots:
            return False
        seats = getattr(scheduler, 'max_num_running_reqs', None)
        if type(seats) is int and decodes + len(live) + 1 > seats:
            return False
        if self.park_age(active.request_id) >= merged.max_park_s:
            return False
        if merged.ttft_s and self.alternator.govern() >= 1.0 and self.cfg.share < 1.0 and self.cfg.rounds is None:
            return False
        try:
            import levern_route

            if not levern_route.host_memory_ok(levern_policy.PARK_NBYTES):
                return False
        except ImportError:
            pass
        return True

    def starved_long(self, scheduler):
        """The oldest waiting long prompt that must be admitted before the waiting shorts, or None: the governor needs it (f_eff >= 1 with a deadline
        set, as may_preempt reads it) or it has waited longer than QWEN_FAST_LEVERN_MAX_PARK_S."""
        merged = self.merged
        for request in self.waiting_in_order(scheduler):
            if self.klass_of(scheduler, request) != 'long':
                continue
            arrival = getattr(request, 'arrival_time', None)
            waited = self.wall() - arrival if type(arrival) in (int, float) else 0.0
            if waited >= merged.max_park_s:
                return request
            if merged.ttft_s and self.alternator.govern() >= 1.0 and self.cfg.share < 1.0 and self.cfg.rounds is None:
                return request
            return None
        return None

    def plan_pass(self, scheduler, decodes, held):
        """The merged route's decision for one prefill pass (PassPlan): which prefill advances, which are hidden, which are ended. Reads the
        scheduler, registers quarantines, and mutates nothing else; serving_prefill_admission's wrapper applies it."""
        merged = self.merged
        self.prune(scheduler)
        killed = self.killed()
        plan = PassPlan()
        partials = [request for request in scheduler.running if getattr(request, 'is_prefill_chunk', False)]
        parking = merged.park == 'host' and not killed
        limit = 1 + (merged.park_slots if merged.park == 'host' else 0)
        live = []
        for request in partials:
            computed = getattr(request, 'num_computed_tokens', None)
            reason = None
            if type(computed) is int and not levern_policy.state_has(request.request_id, computed):
                reason = ('its suspended state after %d tokens is gone (neither the scratch owner nor a parked state), so it cannot continue'
                          % computed)
            elif len(live) >= limit:
                reason = 'more than %d prefill(s) in flight' % limit
            if reason is not None:
                self.quarantine(request, reason)
                plan.quarantined.append(request)
                plan.hidden.append(request)
            else:
                live.append(request)
        shorts = []
        if parking:
            for index, request in enumerate(self.waiting_in_order(scheduler)):
                if self.klass_of(scheduler, request) == 'short':
                    shorts.append((self.remaining(scheduler, request), index, request))
            shorts.sort(key=lambda item: (item[0] if item[0] is not None else float('inf'), item[1]))
        short_live = sorted((request for request in live if self.klass_of(scheduler, request) == 'short'),
                            key=lambda request: self.remaining(scheduler, request) or 0)
        active = short_live[0] if short_live else (live[0] if live else None)
        if active is not None:
            plan.active = active
            if (parking and shorts and self.klass_of(scheduler, active) == 'long'
                    and self.may_preempt(scheduler, live, active, shorts[0][2], decodes)):
                plan.preempted, plan.active, plan.fresh = active, None, shorts[0][2]
                plan.override_gate = held is not None and held in {request.request_id for request in live}
                plan.reason = 'short-lane'
        elif shorts:
            plan.fresh, plan.reason = shorts[0][2], 'short-first'
            starved = self.starved_long(scheduler)
            if starved is not None:
                # A waiting long that never ran must not starve behind shorts that keep arriving (parking only ages a long that is already parked):
                # the oldest one goes first when the deadline governor needs it, or when it has waited longer than the park bound.
                plan.fresh, plan.reason = starved, 'long-aged'
        else:
            candidates = admission.admission_candidates(scheduler)
            plan.fresh = candidates[-1] if candidates else None
            plan.reason = 'admit'
        plan.hidden += [request for request in live if request is not plan.active]
        target = plan.target
        plan.short = target is not None and self.klass_of(scheduler, target) == 'short'
        return plan

    def commit_plan(self, plan):
        """The plan the pass really runs (after the holds): the park ages of the prefills it hides and resumes."""
        now = self.clock()
        hidden = {getattr(request, 'request_id', None) for request in plan.hidden if request not in plan.quarantined}
        for request_id in hidden:
            self.parked_since.setdefault(request_id, now)
        active = getattr(plan.active, 'request_id', None)
        if active is not None and active in self.parked_since:
            self.parked_for[active] = self.parked_for.get(active, 0.0) + now - self.parked_since.pop(active)
            # The step that resumes a parked prefill also restores it from the host: not a measure of a chunk's device time.
            self.unmeasured.add(active)
        if plan.preempted is not None and plan.fresh is not None:
            self.preemptions += 1
            self.log(levern_policy.SHORT_LINE, plan.fresh.request_id, self.remaining_noted(plan.fresh), sorted(hidden),
                     getattr(plan.preempted, 'request_id', None))
        if plan.fresh is not None and plan.active is None:
            self.pending_class[plan.fresh.request_id] = 'short' if plan.short else 'long'
        self.pass_short = plan.short

    def remaining_noted(self, request):
        prompt = admission.prompt_tokens(request)
        computed = getattr(request, 'num_computed_tokens', None)
        return '-' if type(prompt) is not int or type(computed) is not int else prompt - computed

    def pending_hint(self, scheduler):
        """The deadline governor's input: [(arrival wall time, remaining device ms)] of every pending LONG prefill in service order (the ones in
        flight first, then the waiting). Shorts are paced at R = 1 and never governed."""
        merged = self.merged
        if merged is None or not merged.ttft_s:
            return []
        hint, now = [], self.wall()
        running = [request for request in scheduler.running if getattr(request, 'is_prefill_chunk', False)]
        seats = getattr(scheduler, 'max_num_running_reqs', None)
        free = seats - len(scheduler.running) if type(seats) is int else None
        for request in running + self.waiting_in_order(scheduler):
            if request not in running:
                # A waiting prompt counts only if a seat would be free when it reaches the front (a prompt queued for a seat is not waiting for the
                # prefill), and not once it is already past the deadline: it cannot be helped, and pinning the share at 1.0 for it only stalls the
                # decoders behind the prefill in flight (the in-flight one past its deadline still gets 1.0, effective_share).
                if free is not None:
                    if free <= 0:
                        continue
                    free -= 1
                arrival = getattr(request, 'arrival_time', None)
                if type(arrival) in (int, float) and merged.ttft_s - (now - arrival) <= 0:
                    continue
            if self.klass_of(scheduler, request) != 'long':
                continue
            prompt = admission.prompt_tokens(request)
            computed = getattr(request, 'num_computed_tokens', None)
            if type(prompt) is not int or type(computed) is not int:
                continue
            start = computed if computed else self.peek(scheduler, request)
            arrival = getattr(request, 'arrival_time', None)
            hint.append((arrival if type(arrival) in (int, float) else now, self.step_times.remaining_ms(prompt, start, self.cfg.step)))
        return hint

    def observe(self, decision):
        """Teach the step-time model the wall time of the previous prefill step (the scheduler's own interval between two schedule() calls)."""
        if self.last_prefill is None or decision.prev_kind != 'prefill' or decision.prev_ms is None:
            return
        start, tokens = self.last_prefill
        self.last_prefill = None
        self.step_times.observe(start, tokens, decision.prev_ms)

    # ------------------------------------------------------------------ the cap

    def target(self, scheduler, partials, allowed, hide):
        """(request, fresh) the step advances, or (None, False): the one partial prefill, else the one fresh prompt the
        waiting loop will admit (the last of admission_candidates: the first that is not blocked)."""
        if partials:
            running = [request for request in scheduler.running if getattr(request, 'is_prefill_chunk', False)]
            if len(running) != 1:
                raise ValueError('Lever N v1 runs one prefill in flight, found %d partial prefills' % len(running))
            return running[0], False
        if allowed > 0 and not hide:
            candidates = admission.admission_candidates(scheduler)
            if candidates:
                return candidates[-1], True
        return None, False

    def budget(self, scheduler, request, fresh, decodes):
        """The token cap for this step's request, or None when it cannot be read (the step then runs uncapped: whole
        prompt, the stock path, still exact; logged, so a gate sees the cap was absent)."""
        prompt = admission.prompt_tokens(request)
        computed = getattr(request, 'num_computed_tokens', None)
        if type(prompt) is not int or type(computed) is not int:
            self.log(levern_policy.REFUSED_LINE, 'no cap for %r: prompt=%r computed=%r' % (
                getattr(request, 'request_id', None), prompt, computed))
            return None
        waiting = len(getattr(scheduler, 'waiting', None) or ()) + len(getattr(scheduler, 'skipped_waiting', None) or ())
        others = max(0, waiting - (1 if fresh else 0))
        if fresh and computed == 0:
            # A fresh admission is trimmed to its prefix hit INSIDE the pass (the G1 graft's get_computed_blocks), after this cap is fixed: the budget is
            # computed from the hit the trim will land on (SchedulerGraft.peek), or a hit at Q = S_last would run [Q, floor2048(P)) as a non-final step
            # and split the drafter window across two steps (docs/lever-n-prefix-merged-route.md, D1). Without a graft the peek is 0.
            computed = self.peek(scheduler, request)
            if computed and computed >= levern_policy.final_start(prompt):
                # The peeked step is the whole rest. The trim may still land LOWER (the prefix kill switch latching between the peek and the pass, an
                # eviction): a budget of exactly prompt - Q_peek would then end at an unaligned, non-final point. The budget that takes the whole rest
                # from ANY Q' <= Q_peek is the whole prompt (the step takes min(budget, prompt - Q')), which valid_step accepts.
                return prompt
        if self.merged is not None and self.killed():
            # The kill switch: no new prompt is split; the whole rest in one step (the final step's own window rule holds: Q <= S_last).
            return prompt if fresh else (prompt - computed if computed else None)
        return levern_policy.step_budget(prompt, computed, decoding=decodes > 0, others_waiting=others,
                                         step=self.cfg.step, solo=self.cfg.solo)

    def apply_cap(self, scheduler, budget):
        """Lower the scheduler's token budget to `budget` for one call; returns the function that puts it back.

        vLLM 0.25.1's Scheduler.schedule starts from `token_budget = self.max_num_scheduled_tokens`. A scheduler without
        that integer attribute falls back to scheduler_config.long_prefill_token_threshold, which the same loops apply
        to every request (0 < threshold < num_new_tokens): the only request in a prefill step is the capped one."""
        saved = getattr(scheduler, 'max_num_scheduled_tokens', None)
        if type(saved) is int:
            scheduler.max_num_scheduled_tokens = min(saved, budget)

            def restore():
                scheduler.max_num_scheduled_tokens = saved
            return restore
        if self.graft(scheduler) is not None:
            raise ValueError('Lever N cannot cap a prefill step beside the prefix graft through long_prefill_token_threshold: vLLM would split there '
                             "without the cap's alignment (the graft refuses the threshold absolutely); max_num_scheduled_tokens must be an integer")
        config = getattr(scheduler, 'scheduler_config', None)
        previous = getattr(config, 'long_prefill_token_threshold', None)
        if config is None or type(previous) is not int:
            raise ValueError('Lever N cannot cap a prefill step: the scheduler has neither max_num_scheduled_tokens nor '
                             'scheduler_config.long_prefill_token_threshold')
        if not self.cap_fallback_logged:
            self.cap_fallback_logged = True
            self.log(levern_policy.REFUSED_LINE, 'max_num_scheduled_tokens is not an integer on this scheduler: the cap '
                     'is applied through long_prefill_token_threshold')
        config.long_prefill_token_threshold = budget

        def restore():
            config.long_prefill_token_threshold = previous
        return restore

    def plan_cap(self, scheduler, partials, allowed, hide, decodes):
        """(restore, request, fresh, budget): the cap installed for this step's call, restore None when there is none."""
        request, fresh = self.target(scheduler, partials, allowed, hide)
        if request is None:
            return None, None, False, None
        return self.cap_for(scheduler, request, fresh, decodes)

    def cap_for(self, scheduler, request, fresh, decodes):
        """plan_cap's second half for a target the caller already chose (the merged route's plan_pass): the cap installed for the request's step."""
        budget = self.budget(scheduler, request, fresh, decodes)
        if budget is None:
            return None, request, fresh, None
        self.computed = getattr(request, 'num_computed_tokens', None)
        return self.apply_cap(scheduler, budget), request, fresh, budget

    # ------------------------------------------------------------ what the cap actually scheduled

    def verify(self, result, request):
        """Fail closed when the pass the cap shaped is not the one it was shaped for. The cap is vLLM's token budget for the whole call, and
        the waiting loop may admit a different request than the one the budget was computed for (a blocked head promoted in this very pass), or
        end a step off the model's 2,048-token boundary. Either the route would be handed a step it cannot continue exactly (it raises and the
        engine dies) or a prompt under 4,096 tokens would be split in two (the drafter window straddles two steps, hazard H1). So a pass is
        checked here, before any device work: no new request but the target is admitted, and the target's non-final end is a multiple of
        2,048 inside a prompt of at least 4,096 tokens. Logs a REFUSED line and raises ValueError. `request` is the target plan_cap chose with
        the tokens it had computed before the call; None when no cap was applied (nothing to check)."""
        if request is None or not getattr(result, 'total_num_scheduled_tokens', 0):
            return
        request, computed = request
        target = getattr(request, 'request_id', None)
        counts = getattr(result, 'num_scheduled_tokens', None)
        if not isinstance(counts, dict):
            return
        admitted = [value.req_id for value in getattr(result, 'scheduled_new_reqs', None) or () if value.req_id != target]
        if admitted:
            self.refuse('the cap was computed for %r and the pass admitted %r' % (target, admitted[0]))
        prompt, tokens = admission.prompt_tokens(request), counts.get(target)
        # A fresh admission starts where the trim put it (a prefix hit's Q), which only the pass knows.
        for value in getattr(result, 'scheduled_new_reqs', None) or ():
            if value.req_id == target and type(getattr(value, 'num_computed_tokens', None)) is int:
                computed = value.num_computed_tokens
        if type(prompt) is not int or type(tokens) is not int or type(computed) is not int:
            return
        end = computed + tokens
        if end < prompt and (end % levern_policy.CHUNK or prompt < 2 * levern_policy.CHUNK):
            self.refuse('request %r would end a non-final step at %d of %d: off the %d-token boundary, or a prompt too short to split'
                        % (target, end, prompt, levern_policy.CHUNK))
        if end < prompt and end > levern_policy.final_start(prompt):
            self.refuse('request %r would end a non-final step at %d of %d, past S_last=%d: the drafter window [P - %d, P) would straddle two steps'
                        % (target, end, prompt, levern_policy.final_start(prompt), levern_policy.CHUNK))
        if not levern_policy.valid_step(prompt, computed, tokens):
            self.refuse('request %r step [%d, %d) of %d is not a step of the plan (a hit above P - %d, or a start off the %d-token boundary)'
                        % (target, computed, end, prompt, levern_policy.CHUNK, levern_policy.CHUNK))

    def refuse(self, why):
        self.log(levern_policy.REFUSED_LINE, why)
        raise ValueError('Lever N REFUSED: %s' % why)

    def pick_active(self, scheduler, partials):
        """The partial prefill the next prefill pass advances: the only one, or (admission v2) the shortest short, else the oldest long."""
        if self.merged is None or len(partials) < 2:
            return partials[0]
        shorts = [request for request in partials if self.klass_of(scheduler, request) == 'short']
        if shorts:
            return min(shorts, key=lambda request: self.remaining(scheduler, request) or 0)
        return partials[0]

    # --------------------------------------------------------------- the final-step DRAM gate

    def final_hold(self, scheduler, request, decodes, modules=None):
        """Whether the NEXT step of the partial `request` (its final one: it builds the engine) waits for DRAM, as the
        admission's predicate reads it for the prompt. False when the next step is not the final one, no predicate is
        registered, the prompt fits, or no decoder is left to free memory."""
        prompt = admission.prompt_tokens(request)
        computed = getattr(request, 'num_computed_tokens', None)
        if type(prompt) is not int or type(computed) is not int or computed < levern_policy.final_start(prompt):
            return False
        request_id = getattr(request, 'request_id', None)
        if self.fault == 'final-hold' and not self.fault_fired and decodes:
            self.fault_fired = True
            self.final_held = (request_id, decodes)
            self.log(levern_policy.FINAL_HOLD_LINE, request_id, prompt, decodes, 'fault:final-hold')
            return True
        admits = admission.dram_admits(modules)
        if admits is None:
            return False
        try:
            ok, detail = admits(prompt)
            unavailable = detail.get('largest_free') is None
            short = detail.get('short')
        except Exception:
            return False
        if ok or unavailable:
            self.final_held = None
            return False
        if not decodes:
            self.final_held = None
            return False
        if self.final_held != (request_id, decodes):
            self.final_held = (request_id, decodes)
            self.log(levern_policy.FINAL_HOLD_LINE, request_id, prompt, decodes, admission._terms(short))
        return True


def _default_mode(scheduler):
    mode = getattr(scheduler, '_forced_mode', None)
    return mode is None or getattr(mode, 'name', None) == 'DEFAULT'


def _scheduled_prefill(result, partial_ids):
    """Whether `result` scheduled prefill tokens: a new request, or one of the partials that were in flight."""
    if not getattr(result, 'total_num_scheduled_tokens', 0):
        return False
    if getattr(result, 'scheduled_new_reqs', None):
        return True
    cached = getattr(getattr(result, 'scheduled_cached_reqs', None), 'req_ids', None) or ()
    return any(request_id in partial_ids for request_id in cached)


def _prefill_step(result, partial_ids, before):
    """(request id, start, tokens) of the one prefill step `result` scheduled (a new request at its trimmed start, or a partial from where it was), or None."""
    counts = getattr(result, 'num_scheduled_tokens', None)
    if not isinstance(counts, dict):
        return None
    for value in getattr(result, 'scheduled_new_reqs', None) or ():
        start = getattr(value, 'num_computed_tokens', 0)
        return value.req_id, start if type(start) is int else 0, counts.get(value.req_id, 0)
    for request_id in getattr(getattr(result, 'scheduled_cached_reqs', None), 'req_ids', None) or ():
        if request_id in partial_ids and type(before.get(request_id)) is int:
            return request_id, before[request_id], counts.get(request_id, 0)
    return None


def wrap_schedule(original, runtime):
    """The wrapper installed as <class>.schedule under QWEN_FAST_LEVER_N=1 (the module docstring)."""

    def schedule(self, *args, **kwargs):
        if not _default_mode(self):
            return original(self, *args, **kwargs)
        running = list(self.running)
        partials = [request for request in running if getattr(request, 'is_prefill_chunk', False)]
        decodes = len(running) - len(partials)
        pending = bool(getattr(self, 'waiting', None)) or bool(getattr(self, 'skipped_waiting', None)) or bool(partials)
        merged = runtime.merged is not None
        if merged:
            runtime.pass_short = False
            # Only a call that can start or continue a prefill needs the governor's input (a decode round would peek every waiting long for nothing).
            runtime.alternator.pending_hint = runtime.pending_hint(self) if pending else []
        decision = runtime.alternator.begin(decodes, pending)
        if merged:
            runtime.observe(decision)
        kind, reason = decision.kind, decision.reason
        if kind == 'prefill' and partials and decodes and runtime.final_hold(self, runtime.pick_active(self, partials), decodes):
            kind, reason = 'decode', 'final-dram'
        partial_ids = {request.request_id for request in partials}
        before = {request.request_id: getattr(request, 'num_computed_tokens', None) for request in partials}
        if kind == 'decode':
            result = self._schedule_decode_only()
            finalize = getattr(self, '_finalize_scheduler_output', None)
            if callable(finalize):
                result = finalize(result)
        else:
            result = original(self, *args, **kwargs)
        prefill = _scheduled_prefill(result, partial_ids)
        if merged and prefill:
            step = _prefill_step(result, partial_ids, before)
            runtime.last_prefill = None if step is None else step[1:]
            if step is not None:
                # The step-time model learns only from an INTERMEDIATE step that continued its own scratch: a final step carries the engine
                # build (effective_share adds BUILD_MS itself), a first step may carry a checkpoint restore or a park, a resumed one an unpark.
                known = getattr(self, 'requests', None) or {}
                total = admission.prompt_tokens(known.get(step[0])) if known.get(step[0]) is not None else None
                skip = step[0] in runtime.unmeasured
                runtime.unmeasured.discard(step[0])
                if skip or step[0] not in partial_ids or type(total) is not int or step[1] + step[2] >= total:
                    runtime.last_prefill = None
        runtime.alternator.end('prefill' if prefill else 'decode', short=runtime.pass_short and prefill)
        # One line per decision that is part of a prefill's life: a prefill step, a yield, a step while a partial is in flight. A plain decode step
        # beside a prompt that is merely waiting for a seat (every seat busy) is not, or the log would grow a line a round for as long as it waits.
        if pending and (partials or prefill or kind == 'decode'):
            _log_step(self, runtime, decision, kind, reason, prefill, result, partial_ids, before, decodes)
        return result

    setattr(schedule, WRAPPED, True)
    schedule.__wrapped__ = original
    return schedule


def _log_step(scheduler, runtime, decision, wanted, reason, prefill, result, partial_ids, before, decodes):
    """One STEP_LINE for a scheduling decision made while a prefill was pending."""
    runtime.steps += 1
    request_id, start, tokens, prompt = '-', '-', getattr(result, 'total_num_scheduled_tokens', 0), '-'
    end, final = '-', '-'
    if prefill:
        new = list(getattr(result, 'scheduled_new_reqs', None) or ())
        if new:
            request_id, start = new[0].req_id, getattr(new[0], 'num_computed_tokens', 0)
            prompt = len(getattr(new[0], 'prompt_token_ids', None) or ()) or '-'
        else:
            cached = [value for value in result.scheduled_cached_reqs.req_ids if value in partial_ids]
            request_id = cached[0] if cached else '-'
            start = before.get(request_id, '-')
            known = getattr(scheduler, 'requests', None) or {}
            prompt = admission.prompt_tokens(known.get(request_id)) if known.get(request_id) is not None else '-'
        if isinstance(start, int) and isinstance(tokens, int):
            end = start + tokens
        if isinstance(end, int) and isinstance(prompt, int):
            final = int(end == prompt)
        seats = decodes
    else:
        seats = len(getattr(getattr(result, 'scheduled_cached_reqs', None), 'req_ids', None) or ())
    prev_ms = '-' if decision.prev_ms is None else '%.1f' % decision.prev_ms
    values = (runtime.steps, 'prefill' if prefill else 'decode', seats, request_id, start, tokens, end, prompt, final, reason,
              decision.prev_kind or '-', prev_ms, '%.0f' % decision.owed_ms, decision.owed_rounds)
    if runtime.merged is not None:
        alternator = runtime.alternator
        runtime.log(levern_policy.STEP_LINE_MERGED, *(values + (alternator.share, alternator.need, alternator.gap_ms())))
    else:
        runtime.log(levern_policy.STEP_LINE, *values)
