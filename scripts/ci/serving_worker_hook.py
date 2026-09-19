"""Request-scoped worker routing for an already prepared fast verifier bridge."""

from types import MethodType


class FastWorkerHook:
    def __init__(self, worker, bridge, *, cancelled, packed_step=None):
        runner = worker.model_runner
        if (not worker.is_driver_worker or runner is not bridge.runner
                or runner.non_dp_async_scheduling or runner.tt_data_parallel_size != 1
                or runner._pending_samples or not callable(cancelled)):
            raise ValueError('Idle synchronous driver worker owning the prepared bridge required')
        if getattr(runner, '_qwen_fast_hook', None) is not None:
            raise ValueError('A fast request already owns this runner')
        self.worker, self.runner, self.bridge = worker, runner, bridge
        # One hook, several requests. The hook used to BE the binding of one request
        # to the worker, which is why a second arrival could not decode; probe
        # 35436807668 showed TTScheduler scheduling every live decode in one step,
        # so the worker has to serve them together or not at all.
        self.bridges = {bridge.request.session.request_id: bridge}
        self.packed_step = packed_step
        self.cancelled = cancelled
        self.closed = False
        self.saved = []
        # Kept so a step that is not this request's decode can still reach the
        # runner. Captured before _replace, or it would capture the replacement.
        self.original_execute = runner.execute_model
        # Same reason: a request that is not this hook's decode still needs the
        # native deferred sampler for its prefill.
        self.original_sample = runner.sample_tokens
        self._replace(runner, '_qwen_fast_hook', self)
        self._replace(runner, 'execute_model', MethodType(self._execute, runner))
        self._replace(runner, 'sample_tokens', MethodType(self._sample, runner))
        self._replace(worker, 'take_draft_token_ids', MethodType(self._drafts, worker))

    def _replace(self, owner, name, value):
        self.saved.append((owner, name, name in vars(owner), vars(owner).get(name)))
        setattr(owner, name, value)

    def attach(self, bridge):
        """Admit another request to this worker's packed block."""
        if self.closed or bridge.runner is not self.runner or self.packed_step is None:
            raise ValueError('An open hook on the same runner with a packed step is required')
        request_id = bridge.request.session.request_id
        if request_id in self.bridges:
            raise ValueError('That request already decodes on this worker')
        self.bridges[request_id] = bridge
        return self

    def detach(self, request_id):
        bridge = self.bridges.pop(request_id, None)
        if bridge is None:
            raise ValueError('That request does not decode on this worker')
        bridge.close()
        return self.bridges

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
        # A step that schedules no tokens is a bookkeeping step - a request
        # finishing, for instance - not this hook's decode. With one bridge the
        # lifecycle released the hook before such a step could arrive; holding
        # several, it does not, so the pass-through has to be here.
        if not getattr(scheduled, 'total_num_scheduled_tokens', 1):
            return self.original_execute(scheduled)
        if len(self.bridges) == 1 and self.packed_step is None:
            return self.bridge.execute_decode(scheduled, cancelled=self.cancelled)
        from serving_packed_bridge import execute_packed_decode

        return execute_packed_decode(self.bridges, scheduled, cancelled=self.cancelled,
                                     packed_step=self.packed_step)

    def _sample(self, runner, grammar_output):
        # The hook's own decode returns committed output directly and never defers,
        # so any sampler call arriving here belongs to ANOTHER request's prefill -
        # which is exactly what a second user needs before it can join the block.
        # The gate on that stays in the lifecycle, which is the only party that
        # knows whether a prefill is actually pending.
        if self.closed or runner is not self.runner:
            raise ValueError('Fast worker ownership changed')
        return self.original_sample(grammar_output)

    def _drafts(self, worker):
        if self.closed or worker is not self.worker:
            raise ValueError('Live fast worker owner required')
        if len(self.bridges) == 1 and self.packed_step is None:
            return self.bridge.drafts()
        from serving_vllm_packed import packed_draft_token_ids

        return packed_draft_token_ids([bridge.request for bridge in self.bridges.values()])

    def close(self):
        if self.closed:
            return
        for bridge in list(self.bridges.values()):
            bridge.close()
        self.bridges.clear()
        for owner, name, existed, value in reversed(self.saved):
            if existed:
                setattr(owner, name, value)
            else:
                delattr(owner, name)
        self.closed = True
