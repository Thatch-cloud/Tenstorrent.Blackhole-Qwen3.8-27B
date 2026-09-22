"""Opt-in synchronous prefill-to-fast-decode worker lifecycle."""

from types import MethodType

from serving_fast_policy import validate_fast_config, validate_request_sampling
from serving_worker_hook import FastWorkerHook

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
    # verifier_engine is only importable where the target model is; elsewhere a
    # prefill has no resident engine to displace.
    try:
        from verifier_engine import note_prefill as displace
    except ImportError:
        return
    displace()


class FastServingLifecycle:
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

    def _release_request(self):
        if self.prefill_pending:
            raise ValueError('Cannot release a pending native sampler')
        if self.hook is not None:
            self.hook.close()
            self.hook = None
        if self.capture is not None:
            self.capture.close()
            self.capture = None
        self.request_id = self.decoding_id = None
        self.decoding_ids = []

    def _execute(self, worker, scheduled):
        self._check()
        try:
            if self.prefill_pending:
                raise ValueError('Prefill must be sampled before another execution')
            finished = set(scheduled.finished_req_ids)
            for request_id in [value for value in self.decoding_ids if value in finished]:
                # One finishing request must not tear down the hook the others are
                # still decoding on.
                self.decoding_ids.remove(request_id)
                if self.hook is not None and len(self.hook.bridges) > 1:
                    self.hook.detach(request_id)
                    if self.decoding_id == request_id:
                        self.decoding_id = self.decoding_ids[-1]
            if (self.request_id in finished
                    or (self.decoding_id in finished and not self.decoding_ids)):
                self._release_request()
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
            # A continuation of the prefill already in flight. It arrives as a CACHED
            # request, so without this it would either be handed to the hook (which
            # refuses a request it is not decoding) or fall into the fresh-prefill
            # contract below and be refused as 'one complete fresh prefill'.
            if (self.capture is not None and self.request_id is not None
                    and not self.capture.complete
                    and not scheduled.scheduled_new_reqs
                    and list(scheduled.scheduled_cached_reqs.req_ids) == [self.request_id]):
                return self._continue_prefill(scheduled)
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
            validate_request_sampling(new.sampling_params, prompt_tokens=len(new.prompt_token_ids), eos_ids=self.eos_ids)
            # A terminal first token only ends the request when EOS is honoured.
            self.ignore_eos = bool(getattr(new.sampling_params, 'ignore_eos', False))
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

    def _continue_prefill(self, scheduled):
        """One more chunk of the prefill already in flight.

        The capture stays open across the suspension (dflash_prefill_window.segment),
        so its cursor, chunks and children carry the no-gap ledger from the previous
        step. Nothing else about the request changes: request_id, ignore_eos and the
        sampling validation were all settled when the first chunk was admitted.
        """
        with self.capture.segment():
            result = self.original_execute(scheduled)
        return self._after_prefill_chunk(result, False)

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
            bridge = self.bridge_factory(state, self.capture)
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
