"""Opt-in synchronous prefill-to-fast-decode worker lifecycle."""

from types import MethodType

from serving_fast_policy import validate_fast_config, validate_request_sampling
from serving_worker_hook import FastWorkerHook


class FastServingLifecycle:
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
            if (new.prompt_token_ids is None or new.num_computed_tokens != 0
                    or new.mm_features or new.prompt_embeds is not None or new.lora_request is not None
                    or scheduled.num_scheduled_tokens != {new.req_id: len(new.prompt_token_ids)}
                    or scheduled.total_num_scheduled_tokens != len(new.prompt_token_ids)):
                raise ValueError('Text-only uncached complete prompt required')
            validate_request_sampling(new.sampling_params, prompt_tokens=len(new.prompt_token_ids), eos_ids=self.eos_ids)
            # A terminal first token only ends the request when EOS is honoured.
            self.ignore_eos = bool(getattr(new.sampling_params, 'ignore_eos', False))
            self.request_id = new.req_id
            self.capture = self.capture_factory(len(new.prompt_token_ids))
            with self.capture.capture():
                result = self.original_execute(scheduled)
            if result is not None:
                raise ValueError('Expected native prefill deferred sampler')
            self.prefill_pending = True
            return None
        except BaseException:
            self.failed = True
            raise

    def _sample(self, worker, grammar_output):
        self._check()
        try:
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
