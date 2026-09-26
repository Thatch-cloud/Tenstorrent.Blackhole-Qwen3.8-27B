"""D2 request quarantine: end ONE request as FINISHED_ABORTED instead of failing the engine.

Before this, every refusal inside the fast path's lifecycle set `failed` and re-raised
(serving_lifecycle.py, the `except` at the end of _execute and of _sample), so the engine
died and every live user with it - smoke v3 died on the platform's 54-token warmup that way.
Under QWEN_FAST_ANY_REQUEST (serving_fast_policy.any_request_enabled) a host-side refusal of
ONE request's own terms - raised before that request touched any device state - is quarantined
instead. That is exactly two sources: the sampling contract at admission
(serving_fast_policy.validate_request_sampling in the lifecycle), and
serving_request_factory.RequestRefused from the bridge, which covers only the request's sampling
contract, budget and page table. Everything else still fails the engine, deliberately:
- the step-shape refusals in serving_lifecycle ('one complete fresh prefill', 'Text-only uncached
  prompt') - the scheduler handed the worker a step it cannot run;
- a grammar_output reaching _sample;
- every hook and packed-step round refusal;
- the factory's engine and runner invariants (frontier, helpers, a seed outside the vocabulary,
  an EOS seed without ignore_eos, the KV group count): evidence of a sampler, device or lifecycle
  fault, on which the other live users must not keep decoding.
In practice the edge contract (serving_c2_contract.enforce_request) already refuses or coerces
every sampling-contract violation before the engine sees it, so this path is rarely reached; it
exists so that one that slips through ends one request, not every live user's.

A quarantined request:

1. the worker (serving_lifecycle) has already run the request's prefill and sampled its first
   token; it builds no bridge, closes the capture, frees the prefill gate, returns the step's
   output as usual and registers the request here;
2. in the SAME engine step the scheduler's update_from_output runs next (vLLM's EngineCore:
   execute_model, sample_tokens, then update_from_output), and the wrapper installed below
   finishes every registered request with RequestStatus.FINISHED_ABORTED and marks its
   EngineCoreOutput finished with FinishReason.ABORT - the client gets the first token and
   finish_reason "abort", and the scheduler frees its KV blocks;
3. the next step names it in finished_req_ids, which the runner and the lifecycle already
   treat as any finished request.

The two sides meet through a holder parked under a fixed sys.modules key, the pattern
serving_lifecycle.prefill_gate uses: the worker and the scheduler share the EngineCore
process (uniproc executor), and neither imports the other.

WHICH scheduler is wrapped matters more than how (memory: graft-mounted-is-not-graft-executed).
v235's EngineCore logged "Using custom scheduler class vllm_tt_plugin.scheduler.TTScheduler"
although the API server had installed serving_one_in_flight.OneInFlightScheduler, so a
consumer placed in that subclass would never run. The wrapper goes on the class the worker's
own config names (scheduler_class), and two markers say whether it ran: one line from inside
the wrapper the first time it is called (the positive control) and one per aborted request.
On the TT platform the first reads exactly
    [PINDIAG] request quarantine consumer live in TTScheduler
on the engine's first step (the platform's warmup): a c2 gate or smoke must require it, because
a quarantine installed on a class the engine never builds is silent until a refusal strands a
request. test_serving_request_quarantine.InstalledVllmQuarantineTests drives the consumer through
TTScheduler(AsyncScheduler) on the installed vLLM, which only qwen-fast-vllm-cpu.yml has.
If a registered request is still pending when the next step executes, the wrapper did not run
in this engine, and serving_lifecycle fails the engine loudly (unconsumed) rather than let an
unbridged request decode.

S2 W5b adds a second worker-side source: serving_packed_step.refuse_round, the residual path of
a packed round no engine captures even narrowed, registers every request of that round here. Those
are RUNNING decodes, not prefills, and the step returns ZERO new tokens for them, so vLLM builds no
EngineCoreOutput of its own: abort_quarantined adds the finishing one (new_token_ids=[],
finish_reason ABORT), and finish_requests frees the request's blocks whatever its inflated frontier.

Default off: nothing here is imported unless QWEN_FAST_ANY_REQUEST=1 (serving_packed_step imports it
only to ask consumer_installed, which reads the holder without creating it).
"""

import importlib
import sys
import types

HOLDER_KEY = '_qwen_request_quarantine'
WRAPPED = '_qwen_request_quarantine'


def _log(message, *values):
    try:
        from loguru import logger
    except ImportError:
        print(message.format(*values), flush=True)
        return
    logger.info(message, *values)


def holder():
    """The shared holder, created on first use by whichever side runs first.

    pending: request id -> reason, registered by the worker, consumed by the scheduler.
    installed: the qualified name of the scheduler class wrapped, or None.
    live: True once the wrapped update_from_output has run in this process."""
    value = sys.modules.get(HOLDER_KEY)
    if value is None:
        value = types.ModuleType(HOLDER_KEY)
        value.pending = {}
        value.installed = None
        value.live = False
        sys.modules[HOLDER_KEY] = value
    return value


def consumer_installed():
    """Whether install() wrapped a scheduler class in this process, read without creating the
    holder. serving_packed_step.abort_refused (S2 W5b) asks before it ends a refused packed
    round's requests here; register() below refuses without one anyway."""
    value = sys.modules.get(HOLDER_KEY)
    return value is not None and getattr(value, 'installed', None) is not None


def register(request_id, reason):
    """The worker ended `request_id` host-side; the scheduler aborts it in this step."""
    if not isinstance(request_id, str) or not request_id:
        raise ValueError('A request id is required to quarantine a request')
    shared = holder()
    if shared.installed is None:
        raise ValueError('No scheduler consumes the request quarantine; refusing to strand %r' % request_id)
    shared.pending[request_id] = str(reason)


def unconsumed(scheduled):
    """Registered requests the scheduler has not aborted yet, that this step still schedules.

    Called at the top of the next execute_model: by then update_from_output of the step that
    registered them has run. One still pending and scheduled means the wrapper never ran here."""
    pending = holder().pending
    if not pending:
        return []
    finished = set(getattr(scheduled, 'finished_req_ids', None) or ())
    for request_id in [request_id for request_id in pending if request_id in finished]:
        # Finished some other way before the scheduler saw the registration (a client abort
        # between the two): nothing left to abort.
        del pending[request_id]
    cached = list(getattr(getattr(scheduled, 'scheduled_cached_reqs', None), 'req_ids', None) or ())
    counts = getattr(scheduled, 'num_scheduled_tokens', None)
    named = set(cached) | (set(counts) if isinstance(counts, dict) else set())
    return sorted(request_id for request_id in pending if request_id in named)


def scheduler_class(config, importer=importlib.import_module):
    """The scheduler class the EngineCore builds from this config.

    `scheduler_cls` as a class or a dotted path (the TT platform names
    'vllm_tt_plugin.scheduler.TTScheduler'); unset, whatever the config's own
    get_scheduler_cls answers, else vLLM's Scheduler."""
    scheduler_config = config.scheduler_config
    named = getattr(scheduler_config, 'scheduler_cls', None)
    if isinstance(named, type):
        return named
    if isinstance(named, str) and named:
        module, _, attribute = named.rpartition('.')
        if not module:
            raise ValueError('Dotted scheduler class path required, got %r' % named)
        return getattr(importer(module), attribute)
    getter = getattr(scheduler_config, 'get_scheduler_cls', None)
    if callable(getter):
        return getter()
    return getattr(importer('vllm.v1.core.sched.scheduler'), 'Scheduler')


def vllm_types():
    """(EngineCoreOutput, EngineCoreOutputs, FinishReason, RequestStatus) from the installed vLLM."""
    from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
    from vllm.v1.request import RequestStatus

    return EngineCoreOutput, EngineCoreOutputs, FinishReason, RequestStatus


def abort_quarantined(scheduler, outputs, *, kinds=None, log=None):
    """Finish every registered request as FINISHED_ABORTED and mark its output finished.

    `outputs` is what the scheduler's update_from_output returned: EngineCoreOutputs per client
    index. A request that already finished in that same update (its first token was a stop, or
    it hit max_tokens) is simply dropped from the registry."""
    shared = holder()
    if not shared.pending:
        return outputs
    log = _log if log is None else log
    output_type, outputs_type, finish_reason, request_status = vllm_types() if kinds is None else kinds
    pending = list(shared.pending.items())
    shared.pending.clear()
    for request_id, reason in pending:
        request = scheduler.requests.get(request_id)
        if request is None:
            log('[PINDIAG] request quarantine: {} had already finished; nothing to abort ({})', request_id, reason)
            continue
        client = getattr(request, 'client_index', 0)
        scheduler.finish_requests(request_id, request_status.FINISHED_ABORTED)
        if isinstance(outputs, dict):
            batch = outputs.get(client)
            if batch is None:
                batch = outputs[client] = outputs_type(outputs=[])
        else:
            batch = outputs
        for output in batch.outputs:
            if output.request_id == request_id:
                output.finish_reason = finish_reason.ABORT
                break
        else:
            batch.outputs.append(output_type(request_id=request_id, new_token_ids=[],
                                             finish_reason=finish_reason.ABORT))
        log('[PINDIAG] request quarantine: {} ended as FINISHED_ABORTED, engine kept: {}', request_id, reason)
    return outputs


def install(config, *, importer=importlib.import_module, log=None):
    """Wrap the configured scheduler class's update_from_output; returns its qualified name.

    Idempotent. Raises if the class cannot be resolved: the caller then keeps refusals fatal."""
    log = _log if log is None else log
    cls = scheduler_class(config, importer)
    name = '%s.%s' % (cls.__module__, cls.__qualname__)
    shared = holder()
    original = cls.update_from_output
    if getattr(original, WRAPPED, False):
        shared.installed = name
        return name

    def update_from_output(self, scheduler_output, model_runner_output):
        outputs = original(self, scheduler_output, model_runner_output)
        if not shared.live:
            shared.live = True
            log('[PINDIAG] request quarantine consumer live in {}', type(self).__name__)
        return abort_quarantined(self, outputs, log=log)

    setattr(update_from_output, WRAPPED, True)
    update_from_output.__wrapped__ = original
    cls.update_from_output = update_from_output
    shared.installed = name
    log('[PINDIAG] request quarantine installed on {}', name)
    return name
