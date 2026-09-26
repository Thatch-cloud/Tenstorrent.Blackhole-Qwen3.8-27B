"""Build a request from completed serving prefill, without reference generation."""

from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
from types import SimpleNamespace

from serving_fast_request import FastRequest
from serving_fast_policy import OUTPUT_BUDGET, any_request_enabled, validate_request_sampling
from serving_page_binding import validate_initial_capture_pages


# The widest capture below which an engine's verify never replays attention: model_batch
# replays only blocks of eight rows or more (ModelBatch.attention_replay = replay and rows >= 8)
# and attention_request_plan captures every narrower width exactly once, at the request's own
# position, replay or not. Beside the four-user block the engines capture 1, 2 and 4 rows
# (packed_shapes.M3_SEQUENTIAL_CAPTURE_ROWS), so none of their device programs depends on the
# replay switch.
REPLAY_MIN_ROWS = 8


class RequestRefused(ValueError):
    """A host-side refusal of ONE request's own terms, raised before it has touched any device
    state: its sampling contract (validate_request_sampling), its budget (request_budget) and its
    page table (validate_initial_capture_pages) at the top of from_prefill. Under
    QWEN_FAST_ANY_REQUEST the lifecycle ends that request as FINISHED_ABORTED and keeps the engine
    (serving_lifecycle); without it nothing raises this.

    S2 W6b adds one host-side refusal that is not of the request's own terms but is still raised before
    it touches device state: under QWEN_FAST_EXTENT_REPLAY=1 the DRAM backstop (dram_backstop) refuses a
    request whose engine build would not fit the DRAM its prefill left. The scheduler's hold
    (serving_prefill_admission) normally keeps such a prompt waiting before its prefill; this is the
    backstop when the hold was lifted or did not run.

    Deliberately NOT one: the frontier, the one-emitted-seed and GDN-helper count, a seed outside
    the vocabulary, an EOS seed that reached the factory without ignore_eos, and the KV group count.
    Those are engine and runner invariants - a bad sampled seed is evidence of a sampler or device
    fault, an EOS seed here is a lifecycle bug - so they stay plain, engine-fatal ValueErrors: the
    other live users must not keep decoding on suspect state."""


@contextmanager
def host_refusals(enabled):
    """Re-raise a ValueError from the enclosed host-side checks as RequestRefused when
    `enabled`, with the same message; disabled, the original exception propagates untouched."""
    try:
        yield
    except RequestRefused:
        raise
    except ValueError as refusal:
        if not enabled:
            raise
        raise RequestRefused(str(refusal)) from refusal


def request_budget(parameters, *, prompt_tokens, capacity, ceiling=None):
    """One request's output budget under QWEN_FAST_ANY_REQUEST: its own max_tokens, within the
    server ceiling (OUTPUT_BUDGET) and the room its page table leaves after the prompt.

    The session counts the prefill seed as its first emitted token, exactly as vLLM counts it
    against max_tokens, so a session built with this budget finishes on the round vLLM stops
    scheduling the request; real_remaining_budget (serving_worker_hook) stays the backstop."""
    ceiling = OUTPUT_BUDGET if ceiling is None else ceiling
    max_tokens = getattr(parameters, 'max_tokens', None)
    if any(type(value) is not int for value in (max_tokens, prompt_tokens, capacity, ceiling)):
        raise ValueError('Integer max_tokens, prompt length, page capacity and ceiling required for a request budget')
    if max_tokens < 1 or ceiling < 1 or not 1 <= prompt_tokens < capacity:
        raise ValueError('A request budget needs max_tokens >= 1 and a prompt inside its page capacity; '
                         'max_tokens=%d prompt=%d capacity=%d' % (max_tokens, prompt_tokens, capacity))
    return min(max_tokens, ceiling, capacity - prompt_tokens)


def sequential_captures(capture_rows):
    """Whether an engine capped at `capture_rows` captures only widths that never replay."""
    return capture_rows is not None and capture_rows < REPLAY_MIN_ROWS


# S2 (s2-design.md): the per-user extent replay beside the 64-row block, profiles c2-packed and c2-packed-gate
# only. Its memory items (W6) key on this flag, never on "a block is built": exact and c2-gate build a block
# too and keep every byte they had [A5].
EXTENT_REPLAY_FLAG = 'QWEN_FAST_EXTENT_REPLAY'
# W6a: the one proposal bucket every engine captures under the flag.
SINGLE_PROPOSAL_CONTEXT = 2048
DRAM_REGISTERED = '[PINDIAG] dram admission hold registered: '
DRAM_BACKSTOP_REFUSED = '[PINDIAG] dram backstop refused request '


def extent_replay_enabled(environ=None):
    """QWEN_FAST_EXTENT_REPLAY: '1' on, unset or '0' off, anything else refused (ValueError)."""
    value = (os.environ if environ is None else environ).get(EXTENT_REPLAY_FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1, got %r' % (EXTENT_REPLAY_FLAG, value))
    return value == '1'


def single_bucket_contexts(position, max_new_tokens):
    """S2 W6a: dflash_proposal_inputs.proposal_contexts' bounds and refusals, and always the one 2048 bucket.

    The ladder builds one bucket per rung from min(position, 2048) to min(2048, position + budget), all at
    once: four buckets, 1.35 GB per engine instead of 0.80 GB, for a prompt under 256 tokens with a long
    answer (MEM goal-c2-serving: run 36218104858). Beside the block's 3.84 GB four such engines do not fit
    (s2-design.md section 3.1). A 2048 bucket serves every history below it: proposal_inputs masks the
    rows past history_rows with -inf and admits history_rows <= context_rows, and
    PreparedDFlashProposal picks the first bucket at least history_rows wide (dflash_proposal_trace.py)."""
    from dflash_proposal_inputs import proposal_contexts

    proposal_contexts(position, max_new_tokens)
    return (SINGLE_PROPOSAL_CONTEXT,)


@contextmanager
def single_proposal_bucket():
    """S2 W6a: build a request's proposal capture with the single bucket, by a scoped patch of the name
    dflash_proposal_trace resolves at call time - its bytes stay unchanged (pinned-adjacent, s2-design.md Q14),
    the repo's pattern for such overrides (matched_target_geometry.geometry_scope)."""
    from unittest.mock import patch
    import dflash_proposal_trace

    with patch.object(dflash_proposal_trace, 'proposal_contexts', single_bucket_contexts):
        yield


def register_dram_admission(pool, *, log=None):
    """S2 W6b: park the scheduler-side DRAM admission hold's predicate (serving_prefill_admission) for this
    attach: it reads the smallest largest-free block over the chips through `pool`, against the one set of
    defaults there and the coordinator's DRAM reserve (QWEN_FAST_PACKED_PROPOSAL_DRAM_RESERVE_MB). Returns
    the callable that removes it; serving_runtime registers it in the attach's scope, so it goes before
    the pool closes."""
    import serving_prefill_admission as admission
    from dflash_packed_proposal_coordinator import dram_reserve_bytes

    reserve = dram_reserve_bytes()
    unregister = admission.register_dram_predicate(admission.dram_predicate(pool, reserve))
    largest, reason = admission.largest_free(pool)
    (_log if log is None else log)(
        DRAM_REGISTERED + 'need = engine {} + build margin {} + prefill {} at >= {} prompt tokens + reserve {} bytes '
        'per chip; largest_free now {}', admission.ENGINE_BUILD_BYTES, admission.ENGINE_BUILD_MARGIN_BYTES,
        admission.PREFILL_TRANSIENT_BYTES, admission.PREFILL_TRANSIENT_FROM, reserve,
        largest if largest is not None else 'unavailable (%s)' % reason)
    return unregister


def dram_backstop(pool, *, request_id, reserve=None, log=None):
    """S2 W6b's post-prefill backstop: RequestRefused (quarantined under QWEN_FAST_ANY_REQUEST, so the
    request ends FINISHED_ABORTED and the engine lives) when the smallest largest-free DRAM block over the
    chips is below the engine build's peak plus the reserve. Returns that reading, or None when the pool
    cannot be read: then it is a diagnostic, as the coordinator's headroom is (the attach refuses such a pool
    under the flag, W7)."""
    import serving_prefill_admission as admission

    if reserve is None:
        from dflash_packed_proposal_coordinator import dram_reserve_bytes

        reserve = dram_reserve_bytes()
    need = admission.backstop_need(reserve)
    largest, reason = admission.largest_free(pool)
    log = _log if log is None else log
    if largest is None:
        log('[PINDIAG] dram backstop unavailable for request {}: {} (not refused)', request_id, reason)
        return None
    if largest < need:
        log(DRAM_BACKSTOP_REFUSED + '{}: largest_free={} need={} bytes per chip', request_id, largest, need)
        raise RequestRefused('DRAM backstop: the largest free DRAM block (%d bytes on the smallest chip) is below '
                             'the engine build peak plus the reserve (%d bytes)' % (largest, need))
    return largest


_ATTACH_QUALIFICATION = {}


def attach_source_check(directory=None, *, qualify=None, log=None):
    """The source qualification the per-request T16 gate used to carry, run once per process.

    An engine built with target_attention_t16=True runs, at every admission,
    target_t16_attention_gate.qualify(<verifier_engine's directory>) (verifier_engine.py:167-169).
    Staged, that is frozen_combined_runtime.qualify_target -> qualify: every component source the
    frozen evidence reports pin (draft-numerical.json and target-replay.json 'sources'), hashed
    against the tree the process runs - the check that refused image v42. Under
    QWEN_FAST_ANY_REQUEST the engines are built without that gate, so this calls the SAME
    function on the SAME directory, once, at attach: a mismatch raises and the attach fails,
    before any request. Cached per directory, so a second attach in one process does not rehash.

    What the marker can say is what `qualify` returns, and that depends on the tree. Staged at
    the served geometry (the frozen runtime's request context, 131072), the gate's qualify IS
    frozen_combined_runtime.qualify_target, which hashes every pinned source inside qualify and
    returns only dict(<qualify's 'target' evidence>, report_sha256=<the target-replay.json pin>)
    - no source list. So the marker reports how many sources were hashed only when the evidence
    carries runtime_component_sources, and otherwise says the count was not returned; its report
    is whichever report_sha256 the evidence names (target-replay.json's, under qualify_target).

    `qualify` and `log` are injectable for tests; by default the image's own (staged)
    target_t16_attention_gate.qualify and this module's _log."""
    if directory is None:
        import verifier_engine

        directory = Path(verifier_engine.__file__).parent
    directory = Path(directory)
    key = str(directory)
    if key in _ATTACH_QUALIFICATION:
        return _ATTACH_QUALIFICATION[key]
    if qualify is None:
        from target_t16_attention_gate import qualify
    evidence = qualify(directory)
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError('The attach-time source qualification must return its evidence, got %r' % (evidence,))
    _ATTACH_QUALIFICATION[key] = evidence
    sources = evidence.get('runtime_component_sources')
    (_log if log is None else log)(
        '[PINDIAG] attach source check: {} qualified once per process for engines without the per-request '
        'T16 gate ({}; evidence keys {}; report_sha256 {})', key,
        '%d pinned component sources hashed' % len(sources) if isinstance(sources, dict)
        else 'sources hashed inside qualify, count not returned',
        ','.join(sorted(str(name) for name in evidence)), evidence.get('report_sha256', 'absent'))
    return evidence


def device_components():
    from dflash_device import DFlashDevice
    from dflash_proposal_trace import PreparedDFlashProposal
    from dflash_request_runtime import DFlashRequestRuntime
    from greedy_session import GreedySession
    from verifier_engine import VerifierEngine
    from models.tt_transformers.tt.ccl import TT_CCL

    return SimpleNamespace(device=DFlashDevice, proposal=PreparedDFlashProposal,
        runtime=DFlashRequestRuntime, session=GreedySession, engine=VerifierEngine, collectives=TT_CCL)


def _log(message, *values):
    try:
        from loguru import logger
    except ImportError:
        print(message.format(*values), flush=True)
        return
    logger.info(message, *values)


def _proposal_ladder(position, budget):
    """The proposal buckets a request's drafter builds at this position and budget, for the log
    only: 'unavailable' rather than fail a request over a diagnostic. Under QWEN_FAST_EXTENT_REPLAY=1
    the one 2048 bucket (S2 W6a, single_bucket_contexts)."""
    try:
        if extent_replay_enabled():
            return single_bucket_contexts(position, budget)
        from dflash_proposal_inputs import proposal_contexts

        return proposal_contexts(position, budget)
    except Exception as failure:
        return 'unavailable (%s)' % type(failure).__name__


def adopt_prefill_slot(helpers, capture, request_id):
    """Bring the prefill's GDN state to slot 0 before anything reads it.

    The plugin's batched prefill writes the admitted user's recurrent and conv state
    into its decode slot - empty_slots, [1] for the second concurrent user - and leaves
    the live rows alone, while the engine's initial snapshot, its carry and every
    per-step save and restore read native index 0 (gdn_snapshot.ActiveSnapshot). Runs
    35492676194 and 35493208438: the second user's admission snapshot was the FIRST
    user's post-prefill state and its own sat unread in slot 1. Copy it over once,
    here, so the initial save the engine makes next reads this request's own state;
    decode, restore and commit stay at slot 0. Slot 0 and a single-sequence prefill
    (prefill_slot None) have nothing to adopt, so a single user runs exactly as before.
    """
    if not hasattr(capture, 'prefill_slot'):
        raise ValueError('Prefill capture must record the native slot the batched prefill wrote')
    slot = capture.prefill_slot
    if slot is None or slot == 0:
        _log('[PINDIAG] prefill slot {}, nothing to adopt for request {}', slot, request_id)
        return
    # Each helper verifies its conv-state slices against a host readback before
    # writing (rec_state by metadata only) and reports the chips it verified on; one
    # count for every layer, or the copy is not the proof the gate run needs.
    chips = {helper.adopt_slot(slot, layer=layer) for layer, helper in enumerate(helpers)}
    if len(chips) != 1 or not isinstance(next(iter(chips)), int) or next(iter(chips)) < 1:
        raise ValueError('Every GDN layer must verify its adopted slot on the same chips; got %r' % (sorted(chips, key=repr),))
    (verified,) = chips
    _log('[PINDIAG] adopted GDN slot {} into slot 0: {} layers, conv slices verified on {} for request {}',
         slot, len(helpers), 'both chips' if verified == 2 else '%d chips' % verified, request_id)


def from_prefill(operations, model, sampler, pages, helpers, *, state, capture, fixtures, eos_ids,
                 collectives=None, buffer_pool=None, shared_weights=None, capture_rows=None):
    from dflash_request_runtime import TARGET_TAPS

    # QWEN_FAST_ANY_REQUEST (C2-any, serving_fast_policy.any_request_enabled; default off).
    # Off, every value below is what it always was: the server's OUTPUT_BUDGET for every
    # request, the snapshot EOS for every session, replay attention and the T16 gate for every
    # engine, and plain ValueErrors from the host checks.
    any_request = any_request_enabled()
    # S2 W6 (QWEN_FAST_EXTENT_REPLAY, default off): one 2048 proposal bucket (W6a), the DRAM backstop (W6b) and
    # the engine build's ledger point (W6d). Off, none of the three runs.
    extent_memory = extent_replay_enabled()
    # Phase 0 (below) builds the engine without the per-request T16 gate, whose source
    # qualification then runs only at attach (attach_source_check, from serving_runtime). An
    # image whose serving_runtime predates that call would serve without any source check at
    # all, so Phase 0 is refused - for the engine, not the request: it is configuration - until
    # the check has run in this process. First of all, so that on such an image this is the
    # error the first request meets, not a request check that only looks like the fault.
    sequential = any_request and sequential_captures(capture_rows)
    if sequential and not _ATTACH_QUALIFICATION:
        raise ValueError('QWEN_FAST_ANY_REQUEST=1 builds engines without the per-request T16 source check, but '
                         'the attach-time check never ran in this process (serving_runtime.attach_combined_runtime '
                         'must call serving_request_factory.attach_source_check)')
    # Everything up to the slot adoption is host-side: nothing below has touched the device
    # yet. Under C2-any a refusal of the request's own terms - its sampling contract, its
    # budget, its page table - is this request's alone (host_refusals -> RequestRefused), which
    # the lifecycle ends as FINISHED_ABORTED instead of failing the engine. The invariants in
    # between stay plain ValueErrors either way (RequestRefused says why). The checks run in
    # the order they always did.
    prompt = tuple(state.prompt_token_ids)
    with host_refusals(any_request):
        validate_request_sampling(state.sampling_params, prompt_tokens=len(prompt), eos_ids=eos_ids)
    # The frontier. Under whole-prompt prefill this is zero: the request arrives fresh
    # and is prefilled inside the step, so anything else means it had been advanced
    # already. Under CHUNKED prefill the seed is emitted on the final chunk, by which
    # point the earlier chunks are counted - run 35696842354 reached here with 30720 of
    # 32768 after sixteen chunks, and was refused for it.
    #
    # What the clause protects is that the request has not DECODED: the seed being
    # adopted must be this prefill's own first token, not a continuation of a stream
    # already in flight. That is frontier < len(prompt), true of both shapes, and still
    # false for an advanced request. On the unchunked path nothing has run, so zero
    # stays the only value it can take there.
    frontier = state.num_computed_tokens
    if (len(state.output_token_ids) != 1
            or type(frontier) is not int or not 0 <= frontier < len(prompt)
            or not isinstance(state.req_id, str) or not state.req_id
            or len(helpers) != 48):
        raise ValueError('Fresh native prefill with the frontier inside the prompt, one emitted seed '
                         'and all GDN helpers required; frontier=%r prompt=%d emitted=%d helpers=%d'
                         % (frontier, len(prompt), len(state.output_token_ids), len(helpers)))
    seed = state.output_token_ids[0]
    if type(seed) is not int or not 0 <= seed < model.args.vocab_size:
        raise ValueError('Valid target-selected prefill seed required')
    # Only when EOS is honoured. Under ignore_eos the request keeps decoding, so it
    # needs a verifier exactly like any other - run 35442208627 stopped here after
    # the lifecycle had already been corrected for the same assumption one layer up.
    ignore_eos = getattr(state.sampling_params, 'ignore_eos', False)
    if seed in eos_ids and not ignore_eos:
        raise ValueError('Terminal prefill must finish without allocating a verifier')
    if len(state.block_ids) != 1:
        raise ValueError('One scheduler-owned KV group required')
    with host_refusals(any_request):
        # The request's own budget under C2-any (request_budget: its max_tokens within the
        # ceiling and the page room after the prompt), so the session, the drafter and its
        # proposal trace stop where vLLM stops the request; the server ceiling otherwise.
        budget = (request_budget(state.sampling_params, prompt_tokens=len(prompt), capacity=int(pages.shape[1]) * 64)
                  if any_request else OUTPUT_BUDGET)
        validate_initial_capture_pages(pages, state.block_ids[0], position=len(prompt), output_budget=budget)
    if extent_memory:
        # S2 W6b: the post-prefill DRAM backstop, host-side and before any device state, like the refusals
        # above (RequestRefused). The scheduler's hold keeps such a prompt waiting before its prefill.
        dram_backstop(buffer_pool, request_id=state.req_id)
    # After the host-side refusals, so a rejected request touches no device state, and
    # before the drafter, the engine and every other reader of slot 0.
    adopt_prefill_slot(helpers, capture, state.req_id)
    components = device_components()
    _, layers, projection, selector = fixtures
    capture_released = False

    def release_capture():
        nonlocal capture_released
        if not capture_released:
            capture.close()
            capture_released = True

    with ExitStack() as owned:
        owned.callback(release_capture)
        # T1 diagnostic. dflash_device's pin raises one message for fourteen or-ed
        # terms and names none of them, and run 35433428038 hit it on the FIRST
        # request once two users were admitted. Report what is about to be passed,
        # so which term fires is read rather than guessed.
        _qwen_outputs = capture.outputs()
        try:
            from loguru import logger as _qwen_logger
            _qwen_logger.info(
                "[PINDIAG] position={} feature_start={} features={} block_ids={}",
                len(prompt), max(0, len(prompt) - 2048),
                [tuple(getattr(v, 'shape', ())) for v in _qwen_outputs],
                getattr(state, 'block_ids', None))
        except BaseException:
            pass
        # ONE collectives object for every request. Each used to build its own
        # TT_CCL, and two of them cycle semaphore handles over the same mesh: run
        # 35477522469 had the draft head refuse non-finite candidates and run
        # 35478872085, with the trace disabled, had the two chips' replicated
        # selector features disagree. Both are what interleaved collectives look
        # like. The sequential step runs users one at a time, so a shared object is
        # used exactly as serially as it is with a single request.
        # QWEN_FAST_SHARED_CCL=0 gives each request its own TT_CCL for the drafter.
        # Run 35478872085 ran that way with unprotected buffers and diverged but never
        # hung; every hang (35481903377, 35482551725) has been on the shared object
        # with the buffers protected. A trace replay does not advance the host's
        # semaphore counter, so the next EAGER gather may reuse a handle the other
        # request's trace just used - the one hazard the collectives audit could
        # construct was a hang. This is the A/B for it.
        if extent_memory:
            # S2 W6d: the allocator just before the engine build (drafter, verifier captures, proposal bucket),
            # against its estimated peak; a no-op unless QWEN_FAST_MEMORY_LEDGER=1.
            import memory_ledger
            import serving_prefill_admission

            memory_ledger.before('engine', estimate=serving_prefill_admission.engine_build_peak(),
                                 point='req=%s' % memory_ledger.short_id(state.req_id), request=str(state.req_id))
        shared_ccl = os.environ.get('QWEN_FAST_SHARED_CCL', '1') == '1'
        device = components.device(operations, model,
            collectives if collectives is not None and shared_ccl else components.collectives(model.mesh_device),
            layers, projection, selector, _qwen_outputs, position=len(prompt),
            block_rows=16, proposal_capture=True, max_new_tokens=budget,
            fused_convolution=True, feature_start=max(0, len(prompt) - 2048),
            cache_history=True, cache_projection_capture=False, live_query_qk=False,
            native_proposal_attention=True, defer_proposal_capture=True, buffer_pool=buffer_pool,
            shared_weights=shared_weights)
        owned.callback(device.close)
        release_capture()
        runtime = components.runtime(device, position=len(prompt))
        # D4 (C2-any only): under ignore_eos vLLM keeps scheduling the request past an EOS, so
        # the session must not finish there either - with the snapshot EOS it did, and the
        # request then had nothing to decode on (B4 5b). eos_ids=() leaves only the budget.
        session = components.session(state.req_id, prompt, seed, vocab_size=model.args.vocab_size,
            max_new_tokens=budget, eos_ids=() if any_request and ignore_eos else eos_ids,
            neural={'dflash2': runtime}, verifier_rows=16, lookup_enabled=False)

        def prepare_proposal(engine):
            if device.proposal_capture is not None:
                raise ValueError('Proposal trace already captured before verifier allocation')
            if os.environ.get('QWEN_FAST_EAGER_PROPOSAL') == '1':
                # Each request captures its own device trace for the proposal pass.
                # Two requests mean two traces on one mesh, and run 35477522469 had
                # the draft vocabulary head refuse its own candidates after two
                # committed blocks - which is what a clobbered trace would look
                # like. DFlashDevice.propose already runs eagerly when this is None,
                # so leaving it unset isolates the trace as a variable.
                return
            if extent_memory:
                # S2 W6a: one 2048 bucket whatever the prompt, 0.80 GB per engine instead of up to 1.35 GB
                # (single_bucket_contexts). Solo on the same profile builds the same bucket.
                with single_proposal_bucket():
                    device.proposal_capture = components.proposal(device, max_new_tokens=budget)
                return
            device.proposal_capture = components.proposal(device, max_new_tokens=budget)

        # The T16 gate compares these four for equality and run 35474038724 passed
        # position=32768 from len(prompt) yet still failed, so report what the gate
        # will actually read rather than guessing which term differs.
        try:
            from loguru import logger as _qwen_gate_logger
            _qwen_gate_logger.info(
                '[PINDIAG] gate rows={} position={}/{} remaining={}/{} maxnew={}/{} emitted={}',
                16, session.position, type(session.position).__name__,
                session.max_new_tokens - len(session.emitted),
                type(session.max_new_tokens - len(session.emitted)).__name__,
                session.max_new_tokens, type(session.max_new_tokens).__name__,
                len(session.emitted))
        except BaseException:
            pass
        # The verifier's storage comes from the same pool slot the device borrowed:
        # initial and carried GDN state, per-width checkpoints, feature taps and the
        # fixtures' inputs were allocated at attach, before any request's trace, so an
        # earlier request's replays cannot overwrite them (serving_buffer_pool.py).
        # Only when the slot carries it, so a pool without verifier storage leaves the
        # engine allocating as before.
        verifier_storage = getattr(getattr(device, 'pool_slot', None), 'verifier', None)
        # Phase 0 (C2-any only, plan S1 item 1). Beside the four-user block the runtime caps
        # the captures below eight rows, which never replay (REPLAY_MIN_ROWS) and are captured
        # once each at the request's own position either way, so replay attention changes none
        # of this engine's device programs - it only keys the buckets by a mask-family plan and
        # routes the engine through the T16 gate, which admits nothing but the frozen context
        # with <= 256 left (frozen_combined_runtime.validate_target_option; smoke v3 died on
        # a 54-token warmup there). Both off here; the gate's source check runs once at attach
        # instead (attach_source_check). The pool lends buckets by width alone
        # (serving_buffer_pool.VerifierSlot.take) and its replay tables exist only for rows
        # >= 8, so the engine borrows exactly what it did. Anywhere else, unchanged.
        # UNVERIFIED on hardware: that equivalence is host-side (test_serving_any_request stops at
        # before_capture). No run has replayed a 1/2/4-row per-request capture more than 256
        # positions past where it was captured - every served budget was <= 256 - and under c2
        # that span reaches the budget (up to 16383) through the serial singleton reader
        # (QWEN_SKIP_UNUSED_SINGLETON_POSITIONS=1, the SDPA flags). The device programs are
        # exact's; exact never ran them over long spans. G4 (full 2-4k answers against a solo
        # reference on the same image) is what qualifies it.
        # The proposal ladder is reported with it (R3, G5): PreparedDFlashProposal allocates and
        # captures one bucket per rung from min(position, 2048) to min(2048, position + budget) at
        # build, all at once - one rung from position 1025 on, four (256..2048) for a prompt under
        # 256 tokens once position + budget passes 1024 (a 60-token prompt at any budget over 964)
        # - so the short-prompt worst case is read, not inferred.
        if sequential:
            _log('[PINDIAG] any-request engine for {}: captures <= {} rows, replay attention and the T16 gate '
                 'off, budget {} of max_tokens {} at position {}{}, proposal ladder {}', state.req_id, capture_rows,
                 budget, state.sampling_params.max_tokens, len(prompt), ', ignore_eos' if ignore_eos else '',
                 _proposal_ladder(len(prompt), budget))
        # capture_rows: the serving runtime's cap on this engine's captures beside a packed
        # block (packed_shapes.sequential_capture_rows); only when it caps, so a runtime
        # without one calls the engine exactly as before.
        engine = components.engine(model, session, pages, helpers, sampler=sampler,
            norm_batch=True, attention_replay=not sequential, replay_group_rows=4, max_verify_rows=16,
            native_sampling_rows=True, retain_feature_taps=TARGET_TAPS,
            commit_only_gdn=True, target_attention_t16=not sequential, before_capture=prepare_proposal,
            **(dict(storage=verifier_storage) if verifier_storage is not None else {}),
            **(dict(capture_rows=capture_rows) if capture_rows is not None else {}))
        owned.callback(engine.close)
        # QWEN_FAST_PUBLISH_PREWARM (M3NATIVE_PUBLISH_PREWARM; default off): the drafter's
        # publication prepared and discarded once per process per captured (rows, prefix),
        # so its eager programs exist before the first sequential commit needs them
        # (publish_prewarm.py). Unset, nothing is imported or called.
        if os.environ.get('QWEN_FAST_PUBLISH_PREWARM') == '1':
            import publish_prewarm

            publish_prewarm.warm(device, engine)
        request = FastRequest(session, engine, runtime, release_drafter=device.close,
            collect_timings=os.environ.get('QWEN_FAST_PHASE_TIMING') == '1')
        owned.pop_all()
    return request
