"""Request-scoped worker routing for an already prepared fast verifier bridge."""

from functools import partial
import os
from types import MethodType


PHASE_LOG = os.environ.get('QWEN_FAST_PHASE_LOG') == '1'


def phase(name, request_id, call):
    """Run `call` between a begin and an end line so a hang names its phase."""
    if not PHASE_LOG:
        return call()
    import time
    from loguru import logger

    logger.info('[PHASE] {} {} begin', name, request_id)
    started = time.perf_counter()
    result = call()
    logger.info('[PHASE] {} {} end {:.1f} ms', name, request_id, (time.perf_counter() - started) * 1000)
    return result


def real_remaining_budget(bridge):
    """The request's REAL, externally-imposed remaining token budget - the one vLLM's
    own scheduler enforces - or `None` when it cannot be read.

    `GreedySession.max_new_tokens` is a fixed 256 (the full device capture ceiling),
    never the request's own `max_tokens`: `serving_request_factory.from_prefill` never
    threads `sampling_params.max_tokens` into the session, because stopping a request
    at its OWN shorter budget is left entirely to vLLM's scheduler, external to this
    engine - `session.finished` only ever goes True from an EOS token or from actually
    reaching all 256 slots. So a request can be a round away from vLLM excluding it
    from the NEXT schedule (`len(state.output_token_ids) >= sampling_params.max_tokens`,
    checked by the scheduler, not by us) while `session.finished` still reads False:
    run 35567165791 hit exactly this, one round after run 35564623068's fix landed -
    the finishing request's OWN draft still ran a full ~205 ms proposal in the very
    tick that turned out to draft its LAST round, because nothing here knew its real
    budget was already exhausted.

    `bridge.state` is `runner.requests[request_id]` (`FastRunnerBridge.__init__`) -
    the same object `apply_committed_output` (serving_vllm_state.py) extends every
    committed round, so its `output_token_ids` is exactly what the scheduler's own
    stop check reads, kept in sync one round ahead of anything `session` can see.
    """
    state = getattr(bridge, 'state', None)
    max_tokens = getattr(getattr(state, 'sampling_params', None), 'max_tokens', None)
    output_token_ids = getattr(state, 'output_token_ids', None)
    if type(max_tokens) is not int or output_token_ids is None:
        return None
    return max_tokens - len(output_token_ids)


def discard_stale_ticket(request, packed_rows):
    """A pending ticket the coming round's shared width would replace.

    A request can go several ticks between drafting a ticket and being stepped on it -
    a tick that turns out to be some other request's prefill, or bookkeeping for a
    partner finishing (`FastWorkerHook._execute`) - so its pending ticket can still be
    the width every OTHER live request drafted THEN, not the width `proposal_rows`
    decides for every live request NOW. Left alone, that stale ticket rides into a
    round every other entry drafts at `packed_rows`: `serving_packed_step.ineligible`
    refuses the round for the width mismatch, and beside the four-user block the
    per-request engines capture only the sequential widths
    (packed_shapes.sequential_capture_rows), so the fresh block-width tickets have no
    capture anywhere to fall back to - `packed_device_step`'s hard ValueError (run
    35535533720). Discarding the stale ticket here, before drafting, keeps every live
    request's pending ticket at one width per step, so that round is never formed.

    `packed_rows` is `None` exactly when `proposal_rows` decided this round is not the
    block's - too few live requests for a full group (a partner just finished, run
    35564623068) or a survivor's remaining budget narrower than a block round - and a
    ticket already pending at some OTHER width (typically the block's) is exactly as
    stale here as a width mismatch against a real `packed_rows` number: `len(ticket.
    tokens) == None` is never true, so a pending ticket is always cleared when the
    round has no shared width to match it against, and `drafts()` then redrafts fresh
    at the engine's own native width - the one width `serving_packed_step.unservable`
    always finds a capture for.

    Never verified - drafting only runs the draft device and caches a proposal, never
    the target verifier or the block - so dropping it is a pure host-side reset back to
    `idle`; the request's own `drafts()` then redrafts fresh at `packed_rows`, exactly
    as it would with no pending ticket at all.
    """
    session = request.session
    ticket = session.pending
    if ticket is None or session.phase != 'pending' or len(ticket.tokens) == packed_rows:
        return
    discard = getattr(request.runtime, 'discard_proposal', None)
    if callable(discard):
        discard()
    session.pending, session.phase = None, 'idle'


def pipelined_device(bridge):
    """The DFlashDevice a bridge's proposal would run on this round, when its
    runtime exposes one (dflash2's DFlashRequestRuntime.drafter) and the bridge
    is actually eligible to draft - the same eligibility FastRunnerBridge.drafts()
    itself checks (bridge.failed, request.closed/cancelled, session.finished,
    session.pending) - so prepare_pipelined_drafts() never prewarms a bridge whose
    drafts() call would skip or refuse drafting anyway. None for anything else:
    a bridge with no draft need this round, a dspark request (DSparkRequestRuntime
    subclasses DFlashRequestRuntime and has a `.drafter`, but every DSparkDevice
    class - dspark_device.py, dspark_prepared_proposal.py, dspark_t32_device.py -
    has no `prepare_device` method, unlike this file only adds to DFlashDevice),
    or one with no captured trace at all (DFlashDevice.prepare_device's own
    `proposal_capture is None` no-op, QWEN_FAST_EAGER_PROPOSAL) - the caller just
    leaves it to run its normal blocking drafts() in phase B."""
    request = bridge.request
    if getattr(bridge, 'failed', False) or getattr(request, 'closed', False) or getattr(request, 'cancelled', False):
        return None
    session = request.session
    if getattr(session, 'finished', False) or session.pending is not None:
        return None
    device = getattr(getattr(request, 'runtime', None), 'drafter', None)
    return device if device is not None and callable(getattr(device, 'prepare_device', None)) else None


def prepare_pipelined_drafts(bridges):
    """QWEN_FAST_PIPELINED_PROPOSALS phase A: enqueue every eligible bridge's
    proposal device work (DFlashDevice.prepare_device -> dflash_proposal_trace.
    PreparedDFlashProposal.prepare_device) without waiting on any of it, then
    fence the shared mesh ONCE instead of once per bridge, before phase B (the
    unmodified per-bridge drafts() loop in _drafts, below) reads any of it back -
    the execute_trace(blocking=False) x N + one synchronize_device pattern
    mlp-sweep.py already uses for N independent replays on one mesh. Every
    device's own .mesh is the SAME physical mesh (DFlashDevice.mesh is model.
    mesh_device, shared by every concurrent request), so ANY one prepared
    device's .operations/.mesh fences all of them; phase B needs no width or
    count from here; DFlashDevice.propose() finds each prepared bridge's matching
    seed by itself when the harness eventually calls it and finishes the prewarm
    in place of redoing the work, in the SAME original bridge order phase B
    already iterates in.

    A device that raises while being prepared fails the round exactly as its own
    drafts() call would have (the exception is never swallowed) - but only after
    every OTHER already-prepared device's enqueued work is fenced and released,
    so a mid-loop failure never leaves the shared mesh with in-flight work whose
    transients this function is about to free out from under it."""
    prepared, fence = [], None
    try:
        for bridge in bridges:
            device = pipelined_device(bridge)
            if device is None:
                continue
            if device.prepare_device(bridge.request.session.seed):
                prepared.append(device)
                if fence is None:
                    fence = (device.operations, device.mesh)
    except BaseException:
        if fence is not None:
            fence[0].synchronize_device(fence[1])
        for device in prepared:
            device.proposal_capture.discard_pending()
        raise
    if fence is not None:
        fence[0].synchronize_device(fence[1])
    return prepared


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
        if PHASE_LOG:
            # One line per scheduled step: a stream of zero-token steps is a
            # scheduler livelock, silence after a line is the device.
            from loguru import logger
            cached = getattr(getattr(scheduled, 'scheduled_cached_reqs', None), 'req_ids', ()) or ()
            # finished and preempted too: run 35484349353 was refused on one of the
            # step-level conditions and this line, printed just before, named neither.
            logger.info('[PHASE] execute total={} new={} cached={} spec={} finished={} preempted={}',
                        getattr(scheduled, 'total_num_scheduled_tokens', None),
                        len(getattr(scheduled, 'scheduled_new_reqs', None) or ()), len(list(cached)),
                        len(getattr(scheduled, 'scheduled_spec_decode_tokens', None) or {}),
                        sorted(getattr(scheduled, 'finished_req_ids', None) or (), key=str),
                        sorted(getattr(scheduled, 'preempted_req_ids', None) or (), key=str))
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
        # Each bridge's own drafts(), not a direct read of the tickets: drafts() is
        # where a request PREPARES its ticket when none is pending. Reading the
        # tickets directly skipped that and every packed step was refused with
        # 'Live prepared request ticket required' (run 35475786321).
        from vllm.v1.outputs import DraftTokenIds

        # The ticket width of the coming round, decided here, where every live request is
        # known: the packed step's rows per user when the block will serve the round as one
        # pass, else None and each engine proposes at its own captured width. Beside the
        # 64-row block the engines capture only the sequential widths (1, 2, 4), so a
        # block-served round must be drafted at the block's width and a sequential one
        # (survivors, a user with fewer than sixteen tokens left) at the engines'. A step
        # without the policy (the sequential step) leaves every proposal as before.
        policy = getattr(self.packed_step, 'proposal_rows', None)
        have_policy = callable(policy)
        bridges = list(self.bridges.values())
        packed_rows = policy([bridge.request for bridge in bridges]) if have_policy else None
        if packed_rows is not None:
            # proposal_rows only sees session.finished (EOS or the full 256-slot
            # ceiling), never a request's own shorter max_tokens - that budget is
            # vLLM's own, enforced by excluding the request from the NEXT schedule,
            # not by anything here. A live request already at or past ITS real
            # budget is about to be one of those exclusions, and drafting the round
            # at the block's width anyway repeats run 35567165791: the survivors'
            # fresh block-width tickets ride into a round the scheduler admits one
            # entry short, with no capture anywhere to serve them standalone
            # (real_remaining_budget's docstring). Caught here, before drafting,
            # the round degrades to native widths THIS tick instead of next.
            for bridge in bridges:
                if getattr(bridge.request.session, 'finished', False):
                    continue
                remaining = real_remaining_budget(bridge)
                if remaining is not None and remaining < packed_rows:
                    packed_rows = None
                    break
        # Before drafting, not after: a bridge whose pending ticket is already
        # this round's width is untouched (the two-user block and the sequential
        # default never trim their engines' captures, so this never fires for
        # them - packed_shapes.sequential_capture_rows), and one that is not gets
        # a clean redraft at packed_rows instead of riding into a mixed round.
        #
        # This runs whenever a packed policy is CONFIGURED, even when the policy
        # answers None for this particular round - fewer live requests than the
        # block's users (a partner just finished) or a survivor's remaining budget
        # narrower than a block round both answer None here exactly as 'no block
        # round today' does. A ticket already pending at the block's width does not
        # stop being stale just because this round has nowhere shared to put it:
        # left alone, it rides into a round with fewer (or oddly shaped) entries
        # than the block's users, which `serving_packed_step.ineligible` refuses on
        # ENTRY COUNT alone, and whose block-width tickets the survivors' own
        # trimmed engines never captured (`packed_shapes.sequential_capture_rows`) -
        # so `unservable` finds no fallback and `packed_device_step` calls
        # `refuse_round`, failing every survivor's session instead of the one
        # partner who actually finished (run 35564623068). Discarding it here lets
        # `drafts()` redraft fresh at each engine's own native width instead, which
        # the sequential step can always fall back to. Only a packed_step with no
        # `proposal_rows` at all (the plain sequential default) skips this - there
        # is no shared width concept to go stale against.
        if have_policy:
            for bridge in bridges:
                discard_stale_ticket(bridge.request, packed_rows)
        if os.environ.get('QWEN_FAST_PIPELINED_PROPOSALS') == '1':
            # Phase A only - prewarms whichever bridges are eligible and fences
            # them once. Phase B is the loop below, completely unchanged: every
            # bridge still calls its own drafts() in this same original order,
            # and a bridge this left unprepared just runs it exactly as today.
            # Must run after discard_stale_ticket above: a bridge a stale ticket
            # left with session.pending set would otherwise look ineligible here
            # (pipelined_device's own session.pending check) a moment before that
            # same pending gets cleared for phase B, losing the prewarm for exactly
            # the case the discard exists to redraft.
            #
            # Wrapped in the same [PHASE] begin/end lines phase B's own drafts()
            # calls already get (QWEN_FAST_PHASE_LOG=1): phase A was previously
            # invisible to that log, so a slow round's phase-B total (the sum of
            # its 'propose' lines) could look like the whole round when phase A -
            # every eligible bridge's prewarm plus the one shared fence - was
            # actually where a chunk of the time went.
            ids = ','.join(str(bridge.request.session.request_id)[:48] for bridge in bridges)
            # QWEN_FAST_PACKED_PROPOSAL (dflash_packed_proposal.packed_proposal_enabled,
            # dflash_packed_proposal_coordinator.PackedProposalCoordinator): pairs eligible
            # bridges by fixed pool slot and runs one traced two-user pass per full pair
            # instead of two single-user passes, then falls back to prepare_pipelined_drafts'
            # own per-bridge loop for anything left unpaired. Off by default; with it unset
            # this is exactly the unmodified prepare_pipelined_drafts(bridges) call below, and
            # self never gains a _packed_coordinator attribute at all.
            from dflash_packed_proposal import packed_proposal_enabled

            if packed_proposal_enabled():
                coordinator = getattr(self, '_packed_coordinator', None)
                if coordinator is None:
                    from dflash_packed_proposal_coordinator import PackedProposalCoordinator

                    coordinator = self._packed_coordinator = PackedProposalCoordinator()
                phase('prepare_proposals', ids, lambda: coordinator.prepare(bridges))
            else:
                phase('prepare_proposals', ids, lambda: prepare_pipelined_drafts(bridges))
        request_ids, tokens = [], []
        for bridge in bridges:
            # Phase lines around each proposal: run 35482551725 stalled with both
            # requests still running and neither device past its FIRST proposal, so
            # the next run has to say whether it is a proposal or a step that never
            # returns, and whose.
            drafts = phase('propose', bridge.request.session.request_id,
                           bridge.drafts if packed_rows is None else partial(bridge.drafts, packed_rows=packed_rows))
            if drafts is None:
                continue
            request_ids.extend(drafts.req_ids)
            tokens.extend(drafts.draft_token_ids)
        if not request_ids:
            return None
        return DraftTokenIds(req_ids=request_ids, draft_token_ids=tokens)

    def close(self):
        if self.closed:
            return
        # Only present at all if QWEN_FAST_PACKED_PROPOSAL was ever taken this hook's
        # life (serving_worker_hook._drafts) - a hook that never did carries no such
        # attribute, and getattr below is then exactly the no-op it always was.
        coordinator = getattr(self, '_packed_coordinator', None)
        if coordinator is not None:
            coordinator.close()
        for bridge in list(self.bridges.values()):
            bridge.close()
        self.bridges.clear()
        for owner, name, existed, value in reversed(self.saved):
            if existed:
                setattr(owner, name, value)
            else:
                delattr(owner, name)
        self.closed = True
