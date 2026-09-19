"""Request-scoped worker routing for an already prepared fast verifier bridge."""

from types import MethodType


class FastWorkerHook:
    def __init__(self, worker, bridge, *, cancelled):
        runner = worker.model_runner
        if (not worker.is_driver_worker or runner is not bridge.runner
                or runner.non_dp_async_scheduling or runner.tt_data_parallel_size != 1
                or runner._pending_samples or not callable(cancelled)):
            raise ValueError('Idle synchronous driver worker owning the prepared bridge required')
        if getattr(runner, '_qwen_fast_hook', None) is not None:
            raise ValueError('A fast request already owns this runner')
        self.worker, self.runner, self.bridge = worker, runner, bridge
        self.cancelled = cancelled
        self.closed = False
        self.saved = []
        # Kept so a step that is not this request's decode can still reach the
        # runner. Captured before _replace, or it would capture the replacement.
        self.original_execute = runner.execute_model
        self._replace(runner, '_qwen_fast_hook', self)
        self._replace(runner, 'execute_model', MethodType(self._execute, runner))
        self._replace(runner, 'sample_tokens', MethodType(self._sample, runner))
        self._replace(worker, 'take_draft_token_ids', MethodType(self._drafts, worker))

    def _replace(self, owner, name, value):
        self.saved.append((owner, name, name in vars(owner), vars(owner).get(name)))
        setattr(owner, name, value)

    def _execute(self, runner, scheduled):
        if self.closed or runner is not self.runner or runner._pending_samples:
            raise ValueError('Fast worker ownership or sampler queue changed')
        # A step carrying new requests is a PREFILL step for someone else. This hook
        # owns one request's decode, and TTScheduler never mixes prefill and decode
        # in one batch (probe 35435453374), so such a step contains none of this
        # request's work and must reach the runner rather than the decode contract.
        # Without this, admit_scheduler_output refuses it for carrying a new request
        # and a second arrival cannot prefill while the first decodes.
        if getattr(scheduled, 'scheduled_new_reqs', None):
            return self.original_execute(scheduled)
        return self.bridge.execute_decode(scheduled, cancelled=self.cancelled)

    def _sample(self, runner, grammar_output):
        raise RuntimeError('Fast execution returns committed output directly; deferred sampling is forbidden')

    def _drafts(self, worker):
        if self.closed or worker is not self.worker:
            raise ValueError('Live fast worker owner required')
        return self.bridge.drafts()

    def close(self):
        if self.closed:
            return
        self.bridge.close()
        for owner, name, existed, value in reversed(self.saved):
            if existed:
                setattr(owner, name, value)
            else:
                delattr(owner, name)
        self.closed = True
