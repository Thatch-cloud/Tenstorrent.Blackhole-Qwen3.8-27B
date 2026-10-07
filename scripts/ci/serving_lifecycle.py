"""Opt-in synchronous prefill-to-fast-decode worker lifecycle."""

from types import MethodType

from serving_fast_policy import (any_request_enabled, sticky_sessions_enabled, validate_fast_config,
                                  validate_request_sampling)
from serving_worker_hook import FastWorkerHook, note_fixture_writer

# The scheduler cannot see the lifecycle: one is the mounted plugin, the other the
# baked evidence tree. They do share the EngineCore process, so a module parked
# under a fixed sys.modules key reaches both without either importing the other by
# path. Readers treat absent-or-unset as "no prefill held", which is the behaviour
# before this existed.
PREFILL_GATE_KEY = '_qwen_prefill_gate'
# Sticky sessions: the prefix-reuse registry the scheduler graft commits grants into, parked
# under this key the same way (qwen_prefix_registry.REGISTRY_KEY). Read only under
# QWEN_FAST_STICKY_SESSIONS=1.
PREFIX_REGISTRY_KEY = '_qwen_prefix_registry'
# A granted hit resumes on a 2048-token prefill chunk boundary (qwen_prefix_registry.CHUNK), and
# at most at the prompt minus one chunk, so the draft window is this prefill's own.
RESUME_CHUNK = 2048


def committed_grant(request_id):
    """The prefix-reuse registry's committed grant for request_id in this step, or None (no
    registry in this process, or no grant)."""
    import sys
    registry = getattr(sys.modules.get(PREFIX_REGISTRY_KEY), 'registry', None)
    grant_for = getattr(registry, 'grant_for', None)
    return grant_for(request_id) if callable(grant_for) else None


def prefill_gate():
    """The shared holder, created on first use by whichever side runs first."""
    import sys
    import types
    gate = sys.modules.get(PREFILL_GATE_KEY)
    if gate is None:
        gate = types.ModuleType(PREFILL_GATE_KEY)
        gate.held = None
        sys.modules[PREFILL_GATE_KEY] = gate
    return gate


def note_prefill():
    # Round-fence plan H1a: a prefill may write what a pre-staged packed verify relies on, so
    # it bumps the fixture write epoch (verify_prestage.bump; host only, and absent from a tree
    # without the module).
    note_fixture_writer('prefill')
    # verifier_engine is only importable where the target model is; elsewhere a
    # prefill has no resident engine to displace.
    try:
        from verifier_engine import note_prefill as displace
    except ImportError:
        return
    displace()


class FastServingLifecycle:
    # The C2-any state (QWEN_FAST_ANY_REQUEST, set per instance in __init__), defaulted on the
    # class so an instance built without __init__ behaves exactly as before the flag existed.
    any_request = False
    quarantine = None
    admission = None
    refused = None
    lanes = None            # QWEN_FAST_LANE's runtime (per instance in __init__); a lifecycle built without __init__ has none
    lane_gate = None
    max_model_len = None
    # Sticky sessions (QWEN_FAST_STICKY_SESSIONS, set per instance in __init__), defaulted on the
    # class for the same reason: off, a new request with num_computed_tokens != 0 is refused as
    # it always was.
    sticky = False
    # Lever N at TP4 (QWEN_FAST_LEVER_N=1, or the digest instrument QWEN_FAST_LEVERN_AUDIT=1; set per instance in __init__): levern_route, or None.
    # Defaulted on the class so a lifecycle built without __init__, and every profile without the flags, behaves exactly as before.
    levern = None
    # The merged route (QWEN_FAST_LEVER_N=1 beside QWEN_PREFIX_REUSE=1 and sticky sessions, levern_route.merged_requested), its flags (levern_policy.merged_config)
    # and the prefills ADMISSION V2 parked to the host while a short one ran ({request id: its state}); None / empty on every other lifecycle.
    merged = False
    merged_cfg = None
    parked = None

    # request_id is assigned at five sites - construction, reset, the
    # EOS-at-first-token release, the prefill-to-decode handoff, and admission.
    # A property catches all of them and any added later. Patching the sites
    # individually is how one gets missed, which the EOS release already records:
    # run 35441524535 refused a second request with prefill_slot still naming the
    # first, because that site did not clear it.
    @property
    def request_id(self):
        return self._request_id

    @request_id.setter
    def request_id(self, value):
        previous = getattr(self, "_request_id", None)
        self._request_id = value
        prefill_gate().held = value
        if previous != value:
            try:
                from loguru import logger
                logger.info(
                    f"[PINDIAG] prefill gate: held={value!r} was={previous!r}")
            except BaseException:
                pass

    def __init__(self, worker, *, config, capture_factory, bridge_factory, eos_ids, cancelled,
                 packed_step=None, lanes=None):
        validate_fast_config(config)
        runner = worker.model_runner
        if (not worker.is_driver_worker or runner._pending_samples
                or getattr(worker, '_qwen_fast_lifecycle', None) is not None
                or not all(callable(value) for value in (capture_factory, bridge_factory, cancelled))):
            raise ValueError('Idle exclusive driver and explicit serving factories required')
        self.worker, self.runner = worker, runner
        self.capture_factory, self.bridge_factory = capture_factory, bridge_factory
        self.eos_ids, self.cancelled = tuple(eos_ids), cancelled
        self.packed_step = packed_step
        # request_id is the request in the PREFILL phase; decoding_id is the one
        # the hook serves. They were one field, which is why a second arrival
        # hit 'one complete fresh prefill' while the first was merely decoding.
        self.request_id = self.decoding_id = self.capture = self.hook = None
        # Every request the hook is serving, in arrival order. decoding_id stays as
        # the most recent, so existing single-user behaviour reads the same.
        self.decoding_ids = []
        self.prefill_pending = self.failed = self.closed = self.ignore_eos = False
        # A mid-prompt prefill chunk has run and produced no token. Distinct from
        # prefill_pending, which means the native seed IS due.
        self.chunk_in_flight = False
        # One entry per prefill chunk: True when the plugin's execute deferred its
        # sampler (returned None). docs/lever-n-plugin-contract-2026-09-19.md:55-72
        # establishes that the RUNNER zeroes next_token_ids for mid-prompt rows and
        # drops the placeholders when it builds the output, so an intermediate chunk
        # has nothing to sample - but whether the plugin then defers or returns that
        # output is not determinable from this repo. Both are accepted and recorded,
        # and a change of behaviour mid-prompt is refused, so the first rig run says
        # which it is instead of a guess deciding it here.
        self.chunk_deferred = []
        # QWEN_FAST_ANY_REQUEST (C2-any; default off), read once for the lifecycle's life.
        # Off, `quarantine` stays None and every path below is exactly what it was: a
        # refusal fails the engine, and only an EOS first token skips the bridge. On:
        # - D2: a host-side refusal of one request's own terms (the sampling contract at
        #   admission, or serving_request_factory.RequestRefused from the bridge: its sampling
        #   contract, budget or page table) ends that request as FINISHED_ABORTED through
        #   serving_request_quarantine, and the engine lives on. Step-shape refusals, a
        #   grammar_output, hook and round refusals and the factory's invariants stay fatal;
        # - D4: a first token that already exhausts max_tokens (or the context) is terminal
        #   like EOS - no bridge; the platform's warmup is max_tokens 1.
        # - one fresh prompt per prefill step on the scheduler the engine runs
        #   (serving_prefill_admission): any-length prompts can arrive together, and the
        #   TTScheduler the platform installs batches them into one step, which the step-shape
        #   check in _execute refuses (run 36211578069).
        self.any_request = any_request_enabled()
        # QWEN_FAST_STICKY_SESSIONS (phase 1; default off), read once for the lifecycle's life.
        # Off, nothing below reads the prefix registry and a hit is refused as it always was.
        self.sticky = sticky_sessions_enabled()
        self.max_model_len = getattr(getattr(config, 'model_config', None), 'max_model_len', None)
        self.quarantine = None
        # QWEN_FAST_LANE (serving_fast_lane.LaneRuntime; default off, None): one fast lane beside standard lanes. On, the
        # requests' lane marks are validated at admission (a bad mark is this request's own contract, quarantined like a bad
        # sampling parameter), the lane gate is installed on the scheduler class the engine runs (a step that schedules a request
        # the round did not plan is refused by name in the hook), and every hook is built with the runtime.
        self.lanes = lanes
        self.lane_gate = None
        # The qualified name of the scheduler class that admits one fresh prompt per step, or None.
        self.admission = None
        # The admission refusal of the request in the prefill phase, served at its sampler.
        self.refused = None
        if self.any_request:
            self.quarantine = self._install_quarantine(config)
            self.admission = self._install_admission(config)
        # Lever N (QWEN_FAST_LEVER_N=1 / QWEN_FAST_LEVERN_AUDIT=1; default off, None): the lifecycle announces each prefill step to the model route
        # (levern_route) and keeps the displacement and the release of a split prompt. The cap and the alternation live in the scheduler class's
        # install above: a chunked profile that never chunks is refused here, not served.
        self.levern = self._levern_module()
        if self.levern is not None and self.levern.mode()[0] and self.admission is None:
            raise ValueError('Lever N needs the one-fresh-prefill cap and its chunk cap installed on the scheduler the engine runs '
                             '(serving_prefill_admission.install); it could not be')
        self.merged = self.levern is not None and self.levern.mode()[0] and self.levern.merged_requested()
        if self.merged:
            import levern_policy

            self.merged_cfg = levern_policy.merged_config()
            self.parked = {}
        if self.lanes is not None:
            if self.quarantine is None or self.packed_step is None:
                raise ValueError('The fast lane needs the C2-any request quarantine and a packed step')
            self.lane_gate = self._install_lane_gate(config)
        self.original_execute, self.original_sample = worker.execute_model, worker.sample_tokens
        self.saved = []
        for name, value in (('_qwen_fast_lifecycle', self),
                ('execute_model', MethodType(self._execute, worker)),
                ('sample_tokens', MethodType(self._sample, worker)),
                ('take_draft_token_ids', MethodType(self._drafts, worker))):
            self.saved.append((name, name in vars(worker), vars(worker).get(name)))
            setattr(worker, name, value)

    @staticmethod
    def _levern_module():
        """levern_route when QWEN_FAST_LEVER_N or QWEN_FAST_LEVERN_AUDIT is set (strictly 0 or 1: anything else is refused), else None. The
        module is imported only then, so an image or a profile without the flags never loads it."""
        import os

        if os.environ.get('QWEN_FAST_LEVER_N', '0') == '0' and os.environ.get('QWEN_FAST_LEVERN_AUDIT', '0') == '0':
            return None
        import levern_route

        route, audit = levern_route.mode()
        return levern_route if route or audit else None

    def _levern_announce(self, request_id, start, end, total, source=None, park_owner=False):
        """Tell the model route which step the next execute_model call carries (levern_route.announce); a step it will not split is announced
        too, so the control arm's digests and the owner token know the request. Under the merged route every step also names where its starting
        state comes from (levern_route.SOURCES) and whether it takes the scratch from a suspended prefill that must be parked first."""
        self.levern.announce(self.runner.model.model[0], self.levern.Step(request_id, start, end, total, source, park_owner))

    def _levern_withdraw(self):
        if self.levern is not None:
            self.levern.withdraw(self.runner.model.model[0])

    def _check(self):
        if self.failed or self.closed:
            raise ValueError('Serving lifecycle is failed or closed')

    @staticmethod
    def _install_quarantine(config):
        """The scheduler-side consumer for D2 (serving_request_quarantine.install), or None when
        it cannot be installed - then a refusal stays fatal, exactly as without the flag,
        rather than strand a request the scheduler never aborts."""
        try:
            import serving_request_quarantine

            serving_request_quarantine.install(config)
            return serving_request_quarantine
        except Exception as failure:
            try:
                from loguru import logger
                logger.warning(
                    f"[PINDIAG] request quarantine NOT installed ({type(failure).__name__}: "
                    f"{str(failure)[:200]}); host-side refusals still fail the engine")
            except BaseException:
                pass
            return None

    @staticmethod
    def _install_admission(config):
        """One fresh prompt per prefill step (serving_prefill_admission.install) on the scheduler
        class this config names. Returns that class's qualified name, or None when it cannot be
        installed. In that case scheduling stays exactly as it was without the flag, and a step
        carrying two fresh prompts still fails the engine in _execute below."""
        try:
            import serving_prefill_admission

            return serving_prefill_admission.install(config)
        except Exception as failure:
            try:
                from loguru import logger
                logger.warning(
                    f"[PINDIAG] one fresh prefill per step NOT installed ({type(failure).__name__}: "
                    f"{str(failure)[:200]}); simultaneous arrivals can still share a prefill step, "
                    f"which fails the engine")
            except BaseException:
                pass
            return None

    @staticmethod
    def _install_lane_gate(config):
        """The lane gate (serving_fast_lane_scheduler.install) on the scheduler class this config names. Unlike the quarantine
        and the admission cap it is NOT optional: without it a solo round would schedule every standard user too and the
        round would be refused, so a class that cannot take it fails the attach instead of serving half the feature (memory:
        a mounted graft is not an executed graft)."""
        import serving_fast_lane_scheduler

        return serving_fast_lane_scheduler.install(config)

    def _quarantinable(self, failure):
        """Whether `failure` ends only this request: a host-side refusal with a live consumer."""
        if self.quarantine is None:
            return False
        from serving_request_factory import RequestRefused

        return isinstance(failure, RequestRefused)

    def _granted_resume(self, new, chunk):
        """Sticky sessions: where this new request's prefill resumes - R when it arrives at
        num_computed_tokens == R > 0 on a granted prefix hit, else 0 (a cold prompt, judged by
        the uncached clause in _execute exactly as without the flag).

        vLLM's hit reaches the worker only through the scheduler graft's trim, which cuts it to
        a 2048-token boundary Q with a GDN checkpoint, at or below the prompt minus 2048, and
        commits a grant for this request at start_pos == Q in this step
        (qwen_prefix_scheduler_patch). So R > 0 is admitted only when all of these hold, and
        anything else is a broken invariant that fails the engine (a ValueError here, as the
        uncached clause's refusal always was), never a silent cold prefill over cached blocks:
        - the registry holds this step's committed grant for the request, with Q == R;
        - R is a whole number of 2048-token chunks;
        - R <= P - 2048, so the draft window [P - 2048, P) is this prefill's own chunks;
        - the step carries the whole rest of the prompt, P - R tokens (prefix reuse never
          splits a prefill) - or, under the merged route (Lever N beside prefix reuse), it is a
          step of the plan from R (levern_policy.valid_step): the whole rest, or a boundary at or
          below S_last."""
        computed = getattr(new, 'num_computed_tokens', None)
        if type(computed) is not int or computed <= 0 or new.prompt_token_ids is None:
            return 0
        prompt = len(new.prompt_token_ids)
        grant = committed_grant(new.req_id)
        granted = None if grant is None else getattr(grant, 'q', None)
        problems = []
        if grant is None:
            problems.append('no committed prefix grant')
        elif type(granted) is not int or granted != computed:
            problems.append('the committed grant is at Q=%r' % (granted,))
        if computed % RESUME_CHUNK:
            problems.append('R is not a %d-token boundary' % RESUME_CHUNK)
        if computed > prompt - RESUME_CHUNK:
            problems.append('R is above the prompt minus %d (the draft window)' % RESUME_CHUNK)
        if self.merged:
            import levern_policy

            if not levern_policy.valid_step(prompt, computed, chunk):
                problems.append('the step carries %r of the %d tokens after R, which is not a step of the plan (the whole rest, or a '
                                '2048 boundary at or below S_last=%d)' % (chunk, prompt - computed, levern_policy.final_start(prompt)))
        elif chunk != prompt - computed:
            problems.append('the step carries %r of the %d tokens after R' % (chunk, prompt - computed))
        if problems:
            raise ValueError('Sticky admission refused: req=%r R=%d P=%d: %s'
                             % (new.req_id, computed, prompt, '; '.join(problems)))
        return computed

    def _terminal_at_seed(self, state):
        """D4: why vLLM stops this request at its first token anyway (so building a verifier
        for it would be wasted - and at max_tokens 1, fatal: the session is finished before
        the engine that requires an unfinished one is built), or None. vLLM's own stop check
        reads num_output_tokens >= max_tokens and num_tokens >= max_model_len."""
        emitted = len(state.output_token_ids)
        max_tokens = getattr(getattr(state, 'sampling_params', None), 'max_tokens', None)
        if type(max_tokens) is int and emitted >= max_tokens:
            return 'max_tokens %d reached at the first token' % max_tokens
        prompt = getattr(state, 'prompt_token_ids', None)
        if prompt is not None and type(self.max_model_len) is int and len(prompt) + emitted >= self.max_model_len:
            return 'context %d reached at the first token' % self.max_model_len
        return None

    def _end_at_seed(self, result):
        """The request is done with its first token: no bridge, no hook, the capture released
        and the prefill slot freed for the next arrival (the EOS branch's release)."""
        self.capture.close()
        self.capture = None
        self.request_id = None
        self.refused = None
        return result

    def _quarantine_request(self, result, reason):
        """D2: end the request in the prefill phase as FINISHED_ABORTED and keep the engine.
        Its seed is in `result`; the scheduler aborts it right after this step's output
        (serving_request_quarantine.abort_quarantined). Nothing of it reached the device beyond
        its prefill: a quarantined refusal is raised before the drafter, the slot adoption and
        the engine (serving_request_factory.RequestRefused)."""
        request_id = self.request_id
        self.quarantine.register(request_id, reason)
        try:
            from loguru import logger
            logger.warning(
                f"[PINDIAG] request quarantined: {request_id!r} ends as FINISHED_ABORTED, engine kept "
                f"({len(self.decoding_ids)} decoding): {str(reason)[:300]}")
        except BaseException:
            pass
        return self._end_at_seed(result)

    def _release_request(self):
        """Everything: the decode side and the prefill side. Only close() wants both
        at once - a finishing request releases the side it belongs to."""
        if self.prefill_pending:
            raise ValueError('Cannot release a pending native sampler')
        self._release_decoders()
        self._release_prefill()

    def _release_decoders(self):
        """The hook and the decoding set, once no request is left decoding on it.

        Leaves a prefill in flight alone. With Lever N the last decoder can finish
        between two chunks of the next user's prompt - at 131k and r=1 user 1 is
        expected to finish inside user 2's prefill - and tearing down that capture
        and gate with the hook sent user 2's remaining chunks to the stock runner
        untracked, freed the gate so user 3 was admitted, and the first step
        decoding user 2 beside the bridged user 3 was a mixed step. The prefill's
        final chunk now builds a fresh hook through _sample's `self.hook is None`
        branch, the same adoption every first user takes."""
        if self.hook is not None:
            self.hook.close()
            self.hook = None
        self.decoding_id = None
        self.decoding_ids = []

    def _release_prefill(self):
        """The capture and the prefill gate of the request in the prefill phase.

        Leaves the hook alone: a prompt aborted mid-prefill must not close the hook
        other users are still decoding on. chunk_in_flight is deliberately left as
        it was, which is what the combined release always did - the next admission
        resets it, and until then it only makes _sample defer to the plugin's own
        sampler rather than refuse."""
        if self.prefill_pending:
            raise ValueError('Cannot release a pending native sampler')
        if self.capture is not None:
            self.capture.close()
            self.capture = None
        if self.levern is not None and self.request_id is not None:
            # Lever N: a prompt finished or aborted between two of its steps leaves a scratch nobody continues (levern_route.release).
            self.levern.release(self.runner.model.model[0], self.request_id)
        self.request_id = None

    def _execute(self, worker, scheduled):
        self._check()
        try:
            if self.prefill_pending:
                raise ValueError('Prefill must be sampled before another execution')
            if self.quarantine is not None:
                # The step that quarantined a request ran the scheduler's update_from_output
                # after it, which aborts every registered request. One still registered and
                # scheduled here means that consumer is not the scheduler this engine runs:
                # fail loudly, before an unbridged request decodes on the stock path.
                stranded = self.quarantine.unconsumed(scheduled)
                if stranded:
                    raise ValueError('Quarantined requests %r are still scheduled: the scheduler never aborted '
                                     'them (request quarantine consumer on %s did not run)'
                                     % (stranded, self.quarantine.holder().installed))
            finished = set(scheduled.finished_req_ids)
            for request_id in [value for value in self.decoding_ids if value in finished]:
                # One finishing request must not tear down the hook the others are
                # still decoding on.
                self.decoding_ids.remove(request_id)
                if self.lanes is not None:
                    self.lanes.release(request_id)      # the fast lane frees the round its request is named finished
                if self.hook is not None and len(self.hook.bridges) > 1:
                    self.hook.detach(request_id)
                    if self.decoding_id == request_id:
                        self.decoding_id = self.decoding_ids[-1]
            # Two independent releases, not one. They were a single _release_request
            # under an `or`, so either side finishing tore down the other: the last
            # decoder finishing mid-way through someone else's chunked prefill
            # dropped that prefill's capture and gate, and a prompt aborted
            # mid-prefill closed the hook a live user was decoding on. Either left
            # a request decoding with no bridge beside one that has a bridge, which
            # the hook refuses.
            if self.decoding_id in finished and not self.decoding_ids:
                self._release_decoders()
            if self.request_id is not None and self.request_id in finished:
                self._release_prefill()
            for request_id in [value for value in (self.parked or ()) if value in finished]:
                self._release_parked(request_id)
            # ADMISSION V2: the step continues a prefill this lifecycle parked while a short one ran.
            if self._is_parked_continuation(scheduled):
                return self._resume_parked(scheduled)
            # A continuation of the prefill already in flight. It arrives as a CACHED
            # request, so without this it would either be handed to the hook (which
            # refuses a request it is not decoding) or fall into the fresh-prefill
            # contract below and be refused as 'one complete fresh prefill'.
            #
            # Tested BEFORE the hook branch, not after it. With a hook live (someone
            # else decoding) that branch delegates every step that is not new-only,
            # so a continuation placed below it was unreachable exactly when Lever N
            # needs it: v80 ran user 2's chunks 2..16 on the stock runner outside the
            # capture, the final chunk never set prefill_pending, user 2 was never
            # bridged, the gate stayed held through its whole decode, and every mixed
            # step decoded both users on the stock path. The device work of a chunk
            # is unchanged by this - it still reaches the stock runner, through the
            # hook's unknown-id pass-through - only the bookkeeping around it is.
            #
            # Zero-token steps are excluded so every shape other than a real chunk is
            # routed exactly as before (both branches below send those to the stock
            # path whether or not a hook is live).
            if (not self._legacy_continuation_order()
                    and self._is_continuation(scheduled)):
                return self._continue_prefill(scheduled)
            if self.hook is not None:
                # A prefill-only step for a NEW request, while another request
                # decodes. TTScheduler never mixes prefill and decode in one batch
                # (probe 35435453374: stock vllm gives new=['B'] cached=['A'], the
                # plugin gives new=['B'] cached=[]), so this is a clean prefill step
                # and belongs on the prefill path below. Delegating it to the hook
                # sends it to admit_scheduler_output, which refuses any new request.
                if not (scheduled.scheduled_new_reqs
                        and not scheduled.scheduled_cached_reqs.req_ids):
                    return self.original_execute(scheduled)
            elif scheduled.total_num_scheduled_tokens == 0:
                return self.original_execute(scheduled)
            # The pre-fix position, reachable only under the negative-control flag:
            # with no hook it behaves exactly as the test above, with one it is dead.
            if self._legacy_continuation_order() and self._is_continuation(scheduled):
                return self._continue_prefill(scheduled)
            # Nothing to admit. The clause below vets a request JOINING the fast
            # path, so a step carrying no new request was never an admission and
            # must not be judged as one. Run 35718626867 raised here on
            # prefill_slot=None new=[] cached=[one id] spec={...} - a plain decode,
            # arriving after _release_request tore down the hook because every
            # request the lifecycle TRACKS had finished while an untracked one was
            # still decoding.
            #
            # Delegating is what the two branches above already do for every other
            # shape the fast path does not own. Logged per request id rather than
            # silent: if this fires once every user is adopted, something else is
            # wrong and the marker is how that becomes visible.
            if not scheduled.scheduled_new_reqs:
                _qwen_seen = getattr(self, "_qwen_delegated_ids", None)
                if _qwen_seen is None:
                    _qwen_seen = self._qwen_delegated_ids = set()
                _qwen_ids = tuple(scheduled.scheduled_cached_reqs.req_ids)
                if _qwen_ids not in _qwen_seen:
                    _qwen_seen.add(_qwen_ids)
                    try:
                        from loguru import logger
                        logger.info(
                            f"[PINDIAG] lifecycle delegate: cached={list(_qwen_ids)} "
                            f"tracked={list(self.decoding_ids)} hook={self.hook is not None}")
                    except BaseException:
                        pass
                return self.original_execute(scheduled)
            # ADMISSION V2 (merged route, QWEN_FAST_LEVERN_PARK=host): a short prompt takes the scratch from the long prefill suspended on it. The long
            # one is parked (its capture, sampler bookkeeping and gate move aside, and the route takes its state to the host when this step runs); it
            # resumes through _resume_parked once the short one is bridged. Only a clean new-request step may do it.
            park_owner = self._park_current(scheduled)
            if (self.request_id is not None or len(scheduled.scheduled_new_reqs) != 1
                    or scheduled.scheduled_cached_reqs.req_ids
                    or scheduled.scheduled_spec_decode_tokens
                    or getattr(scheduled, 'has_structured_output_requests', False)
                    or getattr(scheduled, 'scheduled_encoder_inputs', {})):
                # Carrying the values, because run 35436193682 and 35441222051 both
                # died here and the clause had to be guessed from outside.
                raise ValueError('Fast serving requires one complete fresh prefill: '
                                 'prefill_slot=%r new=%r cached=%r spec=%r structured=%r encoder=%r'
                                 % (self.request_id,
                                    [getattr(value, 'req_id', None) for value in scheduled.scheduled_new_reqs],
                                    list(scheduled.scheduled_cached_reqs.req_ids),
                                    dict(scheduled.scheduled_spec_decode_tokens),
                                    getattr(scheduled, 'has_structured_output_requests', False),
                                    getattr(scheduled, 'scheduled_encoder_inputs', {})))
            new = scheduled.scheduled_new_reqs[0]
            # Split into a STEP shape and a REQUEST shape. Everything that protected the
            # request is unchanged - text only, uncached, one request in the step, and the
            # per-request and total token counts agreeing. The single clause relaxed is
            # that this step must carry the WHOLE prompt: with chunked prefill the first
            # step carries the first chunk, and the capture is still built for the full
            # prompt below, so the draft window stays anchored at the true position.
            chunk = scheduled.num_scheduled_tokens.get(new.req_id)
            # Sticky sessions: a granted prefix hit arrives at num_computed_tokens == R > 0
            # (_granted_resume, which refuses - fatally - a hit it cannot prove granted). Off,
            # resume is 0 and the clause below is exactly the uncached one.
            resume = self._granted_resume(new, chunk) if self.sticky else 0
            if (new.prompt_token_ids is None or new.num_computed_tokens != resume
                    or new.mm_features or new.prompt_embeds is not None or new.lora_request is not None
                    or set(scheduled.num_scheduled_tokens) != {new.req_id}
                    or not isinstance(chunk, int) or not 1 <= chunk <= len(new.prompt_token_ids) - resume
                    or scheduled.total_num_scheduled_tokens != chunk):
                raise ValueError('Text-only uncached prompt, whole or first chunk, required: '
                                 'computed=%r chunk=%r prompt=%r total=%r'
                                 % (getattr(new, 'num_computed_tokens', None), chunk,
                                    None if new.prompt_token_ids is None else len(new.prompt_token_ids),
                                    scheduled.total_num_scheduled_tokens))
            refused = None
            try:
                validate_request_sampling(new.sampling_params, prompt_tokens=len(new.prompt_token_ids),
                                          eos_ids=self.eos_ids)
                if self.lanes is not None:
                    import serving_fast_lane

                    serving_fast_lane.request_lane(new.sampling_params)
            except ValueError as refusal:
                # D2 (C2-any): this request's contract, not the engine's. It still prefills -
                # the runner has to account the step it was scheduled in - and is ended at its
                # first token instead of being bridged (_sample). A lane mark the contract does
                # not admit (serving_fast_lane.LaneRefused) is the same kind of refusal.
                if self.quarantine is None:
                    raise
                refused = 'sampling contract: %s' % refusal
            # A terminal first token only ends the request when EOS is honoured.
            self.ignore_eos = bool(getattr(new.sampling_params, 'ignore_eos', False))
            self.refused = refused
            self.request_id = new.req_id
            if resume:
                # The capture's ledger counts from R, and it must record the prefix-reuse route's
                # slot: the model prefills [R, P) through that route, which runs under this capture
                # only when the capture declares it records it.
                self.capture = self.capture_factory(len(new.prompt_token_ids), start=resume)
                if getattr(self.capture, 'records_prefix_route', False) is not True:
                    raise ValueError('Sticky admission of %r at %d: the prefill capture does not record the '
                                     'prefix-reuse route' % (new.req_id, resume))
                try:
                    from loguru import logger
                    logger.info(f"[PINDIAG] sticky admit req={new.req_id!r} Q={resume} "
                                f"P={len(new.prompt_token_ids)} tail={chunk}")
                except BaseException:
                    pass
            else:
                self.capture = self.capture_factory(len(new.prompt_token_ids))
            # Per-request, not per-process: without this the mid-prompt sampler
            # check below compares this prompt's first chunk against the PREVIOUS
            # request's last one.
            self.chunk_deferred = []
            self.chunk_in_flight = False
            # A step that carries the whole prompt keeps using capture(), whose contract
            # is the stricter one - a single segment must cover the prompt - so the
            # unchunked path behaves exactly as it did and needs no capture that knows
            # about suspension. segment() is only for a step that does NOT finish the
            # prompt, which cannot happen until chunked prefill is enabled.
            whole = resume + chunk == len(new.prompt_token_ids)
            scoped = False
            if self.levern is not None and not whole:
                # Lever N: the first step of a split prompt writes no decode slot (the route's last step does), so the resident engine stays
                # resident; the packed fixture write epoch still advances, as for every prefill chunk - unless the epoch scope is the route's
                # (_epoch_scoped), in which case what the route measured decides it after the step (_epoch_after).
                scoped = self._route_scope()
                if not scoped:
                    note_fixture_writer('prefill')
            else:
                # A native prefill rewrites GDN slot 0, so whichever engine's state it
                # held is gone; the next verify must restore its own carry.
                note_prefill()
            if self.levern is not None:
                self._levern_announce(new.req_id, resume, resume + chunk, len(new.prompt_token_ids),
                                      ('CHECKPOINT' if resume else 'COLD') if self.merged else None, park_owner)
            try:
                with (self.capture.capture() if whole else self.capture.segment()):
                    result = self.original_execute(scheduled)
            finally:
                self._levern_withdraw()
            if scoped:
                self._epoch_after('prefill')
            return self._after_prefill_chunk(result, whole)
        except BaseException:
            self.failed = True
            raise

    @staticmethod
    def _legacy_continuation_order():
        """QWEN_FAST_LEGACY_CONTINUATION_ORDER=1 restores the pre-fix routing, where
        the continuation test sat below the hook branch. A negative control only: it
        must reproduce v80's signature (chunks outside the capture, gate held through
        decode). Read per call so a test can set it; it reaches a served container
        only if the launcher forwards it with -e.

        Not token-exact with more than one user. Under this flag a continuation that
        arrives while a hook is live bypasses _continue_prefill, and with it
        _displace_after_continuation: the plugin hands a lone resumed prefill GDN slot
        0 whenever slot 0 is free (run v121), so a decoder still marked resident at row
        0 would skip its carry restore and decode from the other prompt's partial
        state, with no error. Use it as a single-user or signature-only control."""
        import os
        return os.environ.get('QWEN_FAST_LEGACY_CONTINUATION_ORDER') == '1'

    def _is_continuation(self, scheduled):
        """The step is the next chunk of the prefill this lifecycle holds: a capture
        open and incomplete, no new request, exactly the gate holder cached, and
        tokens actually scheduled. Needs chunked prefill to be true at all - without
        it no capture is ever left incomplete - so it is inert on unchunked arms."""
        return (self.capture is not None and self.request_id is not None
                and not self.capture.complete
                and not scheduled.scheduled_new_reqs
                and list(scheduled.scheduled_cached_reqs.req_ids) == [self.request_id]
                and bool(getattr(scheduled, 'total_num_scheduled_tokens', 1)))

    def _route_scope(self):
        """Whether the fixture epoch of a Lever N step is decided by what the route measured (QWEN_FAST_LEVERN_EPOCH_SCOPE=route on the merged
        route, with the disjoint-writer class engaged: verify_prestage.disjoint_engaged) instead of bumped before the step."""
        if not self.merged or getattr(self.merged_cfg, 'epoch_scope', 'global') != 'route':
            return False
        try:
            from verify_prestage import disjoint_engaged
        except ImportError:
            return False
        return disjoint_engaged()

    def _epoch_after(self, reason):
        """The epoch decision of a scoped step, after it ran: an intermediate step that wrote no decode slot, with the batched buffers bound exactly
        as before it (levern_route.IDENTITY_ATTR), is a disjoint writer and moves no epoch; any other step - the final one, one that wrote a slot,
        one the route did not vouch for - bumps the global epoch as before."""
        capture = self.capture
        model = self.runner.model.model[0]
        final_step = capture is None or capture.complete
        wrote = getattr(capture, 'segment_wrote_slot', None)
        identity = vars(model).get(self.levern.IDENTITY_ATTR)
        if final_step or wrote is not False or identity is not True:
            note_fixture_writer(reason)
            return
        from verify_prestage import note_disjoint

        note_disjoint(reason)

    def _is_parked_continuation(self, scheduled):
        """The step is the next chunk of a prefill this lifecycle parked (admission v2): no new request, exactly that one cached id, tokens
        scheduled, and no other prefill active."""
        if not self.parked or scheduled.scheduled_new_reqs or not getattr(scheduled, 'total_num_scheduled_tokens', 1):
            return False
        ids = list(scheduled.scheduled_cached_reqs.req_ids)
        return len(ids) == 1 and ids[0] in self.parked

    def _park_current(self, scheduled):
        """Admission v2: the step admits a new request while a prefill is suspended on the scratch - move that prefill's lifecycle state aside and
        return True so the new request's first step is announced as the one that takes the scratch from it (levern_route parks the state on the
        host). Returns False, changing nothing, in every other case: the flag is off, nothing is suspended, the step is not a clean single new
        request, or every park slot is taken (the step is then refused by the one-fresh-prefill check as it always was)."""
        if (not self.merged or self.merged_cfg.park != 'host' or self.request_id is None or self.capture is None or self.capture.complete
                or self.prefill_pending or len(scheduled.scheduled_new_reqs) != 1 or scheduled.scheduled_cached_reqs.req_ids
                or len(self.parked) >= self.merged_cfg.park_slots):
            return False
        suspended = self.request_id
        self.parked[suspended] = dict(capture=self.capture, ignore_eos=self.ignore_eos, refused=self.refused,
                                      chunk_deferred=self.chunk_deferred, chunk_in_flight=self.chunk_in_flight)
        self.capture = None
        self.request_id = None
        self.chunk_deferred = []
        self.chunk_in_flight = False
        try:
            from loguru import logger
            logger.info(f"[PINDIAG] prefill parked: req={suspended!r} at={self.parked[suspended]['capture'].cursor} "
                        f"parked={sorted(self.parked)} for req={scheduled.scheduled_new_reqs[0].req_id!r}")
        except BaseException:
            pass
        return True

    def _resume_parked(self, scheduled):
        """Admission v2: the scheduler resumed a parked prefill. Its lifecycle state comes back and the step runs as a continuation whose state the
        route restores from the host (source PARKED)."""
        if self.request_id is not None:
            raise ValueError('A parked prefill %r resumes while %r is still in its prefill phase: one prefill at a time owns the scratch'
                             % (list(scheduled.scheduled_cached_reqs.req_ids), self.request_id))
        request_id = list(scheduled.scheduled_cached_reqs.req_ids)[0]
        state = self.parked.pop(request_id)
        self.request_id = request_id
        self.capture = state['capture']
        self.ignore_eos, self.refused = state['ignore_eos'], state['refused']
        self.chunk_deferred, self.chunk_in_flight = state['chunk_deferred'], state['chunk_in_flight']
        return self._continue_prefill(scheduled, source='PARKED')

    def _release_parked(self, request_id):
        """A parked prefill that finished or was aborted: its capture and its parked state are nobody's any more."""
        state = self.parked.pop(request_id, None)
        if state is not None and state['capture'] is not None:
            state['capture'].close()
        if self.levern is not None:
            self.levern.release(self.runner.model.model[0], request_id)

    def _continue_prefill(self, scheduled, source=None):
        """One more chunk of the prefill already in flight.

        The capture stays open across the suspension (dflash_prefill_window.segment),
        so its cursor, chunks and children carry the no-gap ledger from the previous
        step. Nothing else about the request changes: request_id, ignore_eos and the
        sampling validation were all settled when the first chunk was admitted.
        """
        # One line per chunk, naming where it starts: the count of these per request
        # is the proof the lifecycle saw every chunk, which v80 could not show.
        try:
            from loguru import logger
            computed = list(getattr(scheduled.scheduled_cached_reqs, 'num_computed_tokens', None) or ())
            start = computed[0] if computed else None
            logger.info(
                f"[PINDIAG] lifecycle continuation: req={self.request_id!r} "
                f"start={start!r} "
                f"tokens={getattr(scheduled, 'total_num_scheduled_tokens', None)!r} "
                f"hook={self.hook is not None}")
        except BaseException:
            pass
        # Round-fence plan H1a: every Lever N chunk, whichever GDN slot it writes, bumps the
        # packed fixture write epoch (a pre-staged verify then restages in full).
        # Under the merged route's epoch scope (_route_scope) the bump is decided after the step from what the route measured instead (_epoch_after).
        levern = getattr(self, 'levern', None)
        scoped = levern is not None and self._route_scope()
        if not scoped:
            note_fixture_writer('prefill-chunk')
        if levern is not None:
            cached = scheduled.scheduled_cached_reqs
            begun = int(list(cached.num_computed_tokens)[0])
            self._levern_announce(self.request_id, begun, begun + int(scheduled.num_scheduled_tokens[self.request_id]),
                                  self.capture.position, (source or 'SCRATCH') if self.merged else None)
        try:
            with self.capture.segment():
                result = self.original_execute(scheduled)
        finally:
            if levern is not None:
                self._levern_withdraw()
        if scoped:
            self._epoch_after('prefill-chunk')
        self._displace_after_continuation()
        return self._after_prefill_chunk(result, False)

    def _displace_after_continuation(self):
        """A continuation chunk that wrote GDN slot 0 overwrote the resident decoder.

        Admission calls note_prefill before every first chunk; continuations did not,
        which was safe only while a resumed prompt never wrote slot 0. It can: the
        plugin re-allocates a prefill's slot on every prompt step and hands a lone
        prefill slot 0 whenever slot 0 is free (run v121: user 3's chunks went to [2],
        then [0] after user 1 finished). Every chunk writes a snapshot of the in-flight
        prompt into its slot, and slot 0 is the fast path's working row, so a decoder
        still marked resident would skip its carry restore (verifier_engine
        restore_carry) and decode from the other prompt's partial recurrence - wrong
        tokens, no error.

        Keyed on the slot this chunk actually wrote, so a chunk written anywhere else
        leaves residency exactly as before. Anything short of a positive nonzero slot
        (the single-sequence path, which writes the live state, or a capture that does
        not report one) displaces: a spurious restore costs one carry copy, a missed
        one costs the stream.
        """
        if getattr(self.capture, 'segment_wrote_slot', None) is False:
            # Lever N: an intermediate step of a split prompt wrote no decode slot (levern_route), so no resident engine's state was overwritten.
            return
        slot = getattr(self.capture, 'segment_slot', None)
        if type(slot) is int and slot > 0:
            return
        note_prefill()
        try:
            from loguru import logger
            logger.info(
                f"[PINDIAG] prefill chunk wrote working GDN slot {slot!r}: resident "
                f"engine displaced (req={self.request_id!r})")
        except BaseException:
            pass

    def _after_prefill_chunk(self, result, whole):
        """What follows a prefill chunk, whether it finished the prompt or not.

        `whole` says the step carried the entire prompt, which is the only shape that
        exists before chunked prefill is enabled. It is passed rather than derived so
        this never consults capture.complete on the unchunked path - keeping that path
        independent of whether the capture knows about suspension at all.
        """
        deferred = result is None
        if self.chunk_deferred and deferred != self.chunk_deferred[-1]:
            raise ValueError('Prefill chunk changed sampler behaviour mid-prompt: '
                             'deferred=%r after %r' % (deferred, self.chunk_deferred))
        self.chunk_deferred.append(deferred)
        if not whole and not self.capture.complete:
            # Mid-prompt. The runner has already zeroed next_token_ids for these rows
            # and drops the placeholders when it builds the output, so there is no
            # token here - the deferred sampler belongs to the one the FINAL chunk
            # produces, and prefill_pending stays clear until then.
            #
            # That does NOT mean _sample is skipped. vLLM calls sample_tokens on
            # every step (v1/engine/core.py:499), so it is entered here too, and run
            # 35687608717 died because the guard there saw prefill_pending clear and
            # refused. This flag is the third state the guard was missing.
            self.chunk_in_flight = True
            return result
        if not deferred:
            raise ValueError('Expected native prefill deferred sampler on the final chunk')
        self.chunk_in_flight = False
        self.prefill_pending = True
        return None

    def _sample(self, worker, grammar_output):
        self._check()
        try:
            if self.chunk_in_flight:
                # A mid-prompt chunk produced no token, so there is no native seed to
                # commit. The plugin's own sampler drops the zeroed placeholders the
                # runner wrote. Checked BEFORE the guard below, because that guard is
                # about misuse of native sampling and this is not native sampling at
                # all - including when grammar_output is set, which this path leaves
                # to the original sampler rather than refusing.
                return self.original_sample(grammar_output)
            if not self.prefill_pending or grammar_output is not None:
                raise ValueError('Only an unstructured pending prefill may use native sampling')
            result = self.original_sample(grammar_output)
            self.prefill_pending = False
            if (result is None or result.req_ids != [self.request_id]
                    or len(result.sampled_token_ids) != 1 or len(result.sampled_token_ids[0]) != 1):
                raise ValueError('Exactly one committed prefill seed required')
            state = self.runner.requests[self.request_id]
            if state.output_token_ids != result.sampled_token_ids[0]:
                raise ValueError('Native prefill output and runner state disagree')
            if state.output_token_ids[0] in self.eos_ids and not self.ignore_eos:
                # Finished at its first token: no bridge, no hook. The prefill slot
                # MUST be freed here - leaving it held is invisible with one user,
                # because that user is done, and fatal with two: run 35441524535
                # refused the second request with prefill_slot still naming the
                # first. A synthetic prompt that repeats one phrase invites exactly
                # this, so it is the common case on the bench rather than a corner.
                self.capture.close()
                self.capture = None
                self.request_id = None
                return result
            if self.any_request:
                # D4: vLLM stops it at this token (max_tokens 1 - the platform's warmup - or the
                # context): the EOS branch's release, no bridge. Checked before a refusal, since
                # a request that is finished anyway needs no abort.
                terminal = self._terminal_at_seed(state)
                if terminal is not None:
                    try:
                        from loguru import logger
                        logger.info(f"[PINDIAG] request {self.request_id!r} ends at its first token: {terminal}")
                    except BaseException:
                        pass
                    return self._end_at_seed(result)
                if self.refused is not None:
                    return self._quarantine_request(result, self.refused)
            try:
                bridge = self.bridge_factory(state, self.capture)
            except BaseException as failure:
                if not self._quarantinable(failure):
                    raise
                return self._quarantine_request(result, 'bridge refused: %s' % failure)
            try:
                if self.hook is None:
                    self.hook = FastWorkerHook(self.worker, bridge, cancelled=self.cancelled,
                                               packed_step=self.packed_step,
                                               **({'lanes': self.lanes} if self.lanes is not None else {}))
                else:
                    # A hook used to BE the binding of one request to the worker,
                    # which is why a second arrival could not decode. It now holds a
                    # registry, because probe 35436807668 showed the scheduler
                    # putting every live decode in one step: the worker serves them
                    # together or not at all.
                    self.hook.attach(bridge)
            except BaseException:
                bridge.close()
                raise
            # The request leaves the prefill phase and joins the decoding set, so the
            # prefill slot is free for the next arrival.
            self.decoding_ids.append(self.request_id)
            self.decoding_id = self.request_id
            self.request_id = None
            self.capture = None
            return result
        except BaseException:
            self.failed = True
            raise

    def _drafts(self, worker):
        self._check()
        return None

    def close(self):
        if self.closed:
            return
        for request_id in list(self.parked or ()):
            self._release_parked(request_id)
        self._release_request()
        if self.lanes is not None:
            self.lanes.publish(None)
        for name, existed, value in reversed(self.saved):
            if existed:
                setattr(self.worker, name, value)
            else:
                delattr(self.worker, name)
        self.closed = True
