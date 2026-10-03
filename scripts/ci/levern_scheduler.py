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
"""

import levern_policy
import serving_prefill_admission as admission

WRAPPED = '_qwen_levern_schedule'


class LevernRuntime(object):
    """What the two wrappers share: the parsed flags, the Alternator, the step counter and the log."""

    def __init__(self, cfg=None, *, log=None, clock=None, fault=None):
        self.cfg = levern_policy.config() if cfg is None else cfg
        kwargs = {} if clock is None else dict(clock=clock)
        self.alternator = levern_policy.Alternator(self.cfg, **kwargs)
        self.log = admission._log if log is None else log
        self.fault = levern_policy.fault() if fault is None else fault
        self.fault_fired = False
        self.steps = 0
        self.final_held = None
        self.cap_fallback_logged = False

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
        budget = self.budget(scheduler, request, fresh, decodes)
        if budget is None:
            return None, request, fresh, None
        return self.apply_cap(scheduler, budget), request, fresh, budget

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


def wrap_schedule(original, runtime):
    """The wrapper installed as <class>.schedule under QWEN_FAST_LEVER_N=1 (the module docstring)."""

    def schedule(self, *args, **kwargs):
        if not _default_mode(self):
            return original(self, *args, **kwargs)
        running = list(self.running)
        partials = [request for request in running if getattr(request, 'is_prefill_chunk', False)]
        decodes = len(running) - len(partials)
        pending = bool(getattr(self, 'waiting', None)) or bool(getattr(self, 'skipped_waiting', None)) or bool(partials)
        decision = runtime.alternator.begin(decodes, pending)
        kind, reason = decision.kind, decision.reason
        if kind == 'prefill' and partials and decodes and runtime.final_hold(self, partials[0], decodes):
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
        runtime.alternator.end('prefill' if prefill else 'decode')
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
    runtime.log(levern_policy.STEP_LINE, runtime.steps, 'prefill' if prefill else 'decode', seats, request_id, start,
                tokens, end, prompt, final, reason,
                decision.prev_kind or '-', prev_ms, '%.0f' % decision.owed_ms, decision.owed_rounds)
