"""Opt-in synchronous prefill-to-fast-decode worker lifecycle."""

from types import MethodType

from serving_fast_policy import validate_fast_config, validate_request_sampling
from serving_worker_hook import FastWorkerHook


class FastServingLifecycle:
    def __init__(self, worker, *, config, capture_factory, bridge_factory, eos_ids, cancelled):
        validate_fast_config(config)
        runner = worker.model_runner
        if (not worker.is_driver_worker or runner._pending_samples
                or getattr(worker, '_qwen_fast_lifecycle', None) is not None
                or not all(callable(value) for value in (capture_factory, bridge_factory, cancelled))):
            raise ValueError('Idle exclusive driver and explicit serving factories required')
        self.worker, self.runner = worker, runner
        self.capture_factory, self.bridge_factory = capture_factory, bridge_factory
        self.eos_ids, self.cancelled = tuple(eos_ids), cancelled
        self.request_id = self.capture = self.hook = None
        self.prefill_pending = self.failed = self.closed = False
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
        self.request_id = None

    def _execute(self, worker, scheduled):
        self._check()
        try:
            if self.prefill_pending:
                raise ValueError('Prefill must be sampled before another execution')
            if self.request_id in scheduled.finished_req_ids:
                self._release_request()
            if self.hook is not None:
                return self.original_execute(scheduled)
            if scheduled.total_num_scheduled_tokens == 0:
                return self.original_execute(scheduled)
            if (self.request_id is not None or len(scheduled.scheduled_new_reqs) != 1
                    or scheduled.scheduled_cached_reqs.req_ids
                    or scheduled.scheduled_spec_decode_tokens
                    or getattr(scheduled, 'has_structured_output_requests', False)
                    or getattr(scheduled, 'scheduled_encoder_inputs', {})):
                raise ValueError('Fast serving requires one complete fresh prefill')
            new = scheduled.scheduled_new_reqs[0]
            if (new.prompt_token_ids is None or new.num_computed_tokens != 0
                    or new.mm_features or new.prompt_embeds is not None or new.lora_request is not None
                    or scheduled.num_scheduled_tokens != {new.req_id: len(new.prompt_token_ids)}
                    or scheduled.total_num_scheduled_tokens != len(new.prompt_token_ids)):
                raise ValueError('Text-only uncached complete prompt required')
            validate_request_sampling(new.sampling_params, prompt_tokens=len(new.prompt_token_ids), eos_ids=self.eos_ids)
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
            if state.output_token_ids[0] in self.eos_ids:
                self.capture.close()
                self.capture = None
                return result
            bridge = self.bridge_factory(state, self.capture)
            try:
                self.hook = FastWorkerHook(self.worker, bridge, cancelled=self.cancelled)
            except BaseException:
                bridge.close()
                raise
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
