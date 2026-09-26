"""Opt-in synchronous prefill-to-fast-decode worker lifecycle."""

from types import MethodType

from serving_fast_policy import any_request_enabled, validate_fast_config, validate_request_sampling
from serving_worker_hook import FastWorkerHook, note_fixture_writer

# The scheduler cannot see the lifecycle: one is the mounted plugin, the other the
# baked evidence tree. They do share the EngineCore process, so a module parked
# under a fixed sys.modules key reaches both without either importing the other by
# path. Readers treat absent-or-unset as "no prefill held", which is the behaviour
# before this existed.
PREFILL_GATE_KEY = '_qwen_prefill_gate'


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
    max_model_len = None

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
                 packed_step=None):
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
        self.max_model_len = getattr(getattr(config, 'model_config', None), 'max_model_len', None)
        self.quarantine = None
        # The qualified name of the scheduler class that admits one fresh prompt per step, or None.
        self.admission = None
        # The admission refusal of the request in the prefill phase, served at its sampler.
        self.refused = None
        if self.any_request:
            self.quarantine = self._install_quarantine(config)
            self.admission = self._install_admission(config)
        self.original_execute, self.original_sample = worker.execute_model, worker.sample_tokens
        self.saved = []
        for name, value in (('_qwen_fast_lifecycle', self),
                ('execute_model', MethodType(self._execute, worker)),
                ('sample_tokens', MethodType(self._sample, worker)),
                ('take_draft_token_ids', MethodType(self._drafts, worker))):
            self.saved.append((name, name in vars(worker), vars(worker).get(name)))
            setattr(worker, name, value)

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

    def _quarantinable(self, failure):
        """Whether `failure` ends only this request: a host-side refusal with a live consumer."""
        if self.quarantine is None:
            return False
        from serving_request_factory import RequestRefused

        return isinstance(failure, RequestRefused)

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
            if (new.prompt_token_ids is None or new.num_computed_tokens != 0
                    or new.mm_features or new.prompt_embeds is not None or new.lora_request is not None
                    or set(scheduled.num_scheduled_tokens) != {new.req_id}
                    or not isinstance(chunk, int) or not 1 <= chunk <= len(new.prompt_token_ids)
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
            except ValueError as refusal:
                # D2 (C2-any): this request's contract, not the engine's. It still prefills -
                # the runner has to account the step it was scheduled in - and is ended at its
                # first token instead of being bridged (_sample).
                if self.quarantine is None:
                    raise
                refused = 'sampling contract: %s' % refusal
            # A terminal first token only ends the request when EOS is honoured.
            self.ignore_eos = bool(getattr(new.sampling_params, 'ignore_eos', False))
            self.refused = refused
            self.request_id = new.req_id
            self.capture = self.capture_factory(len(new.prompt_token_ids))
            # Per-request, not per-process: without this the mid-prompt sampler
            # check below compares this prompt's first chunk against the PREVIOUS
            # request's last one.
            self.chunk_deferred = []
            self.chunk_in_flight = False
            # A native prefill rewrites GDN slot 0, so whichever engine's state it
            # held is gone; the next verify must restore its own carry.
            note_prefill()
            # A step that carries the whole prompt keeps using capture(), whose contract
            # is the stricter one - a single segment must cover the prompt - so the
            # unchunked path behaves exactly as it did and needs no capture that knows
            # about suspension. segment() is only for a step that does NOT finish the
            # prompt, which cannot happen until chunked prefill is enabled.
            whole = chunk == len(new.prompt_token_ids)
            with (self.capture.capture() if whole else self.capture.segment()):
                result = self.original_execute(scheduled)
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

    def _continue_prefill(self, scheduled):
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
        note_fixture_writer('prefill-chunk')
        with self.capture.segment():
            result = self.original_execute(scheduled)
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
                                               packed_step=self.packed_step)
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
        self._release_request()
        for name, existed, value in reversed(self.saved):
            if existed:
                setattr(self.worker, name, value)
            else:
                delattr(self.worker, name)
        self.closed = True
