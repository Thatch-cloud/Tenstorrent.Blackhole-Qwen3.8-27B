"""Build a request from completed serving prefill, without reference generation."""

from contextlib import ExitStack
import os
from types import SimpleNamespace

from serving_fast_request import FastRequest
from serving_fast_policy import validate_request_sampling
from serving_page_binding import validate_initial_capture_pages


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
                 collectives=None, buffer_pool=None, shared_weights=None):
    from dflash_request_runtime import TARGET_TAPS

    prompt = tuple(state.prompt_token_ids)
    validate_request_sampling(state.sampling_params, prompt_tokens=len(prompt), eos_ids=eos_ids)
    if (len(state.output_token_ids) != 1 or state.num_computed_tokens != 0
            or not isinstance(state.req_id, str) or not state.req_id
            or len(helpers) != 48):
        raise ValueError('Fresh native prefill with pre-step frontier zero, one emitted seed and all GDN helpers required')
    seed = state.output_token_ids[0]
    if type(seed) is not int or not 0 <= seed < model.args.vocab_size:
        raise ValueError('Valid target-selected prefill seed required')
    # Only when EOS is honoured. Under ignore_eos the request keeps decoding, so it
    # needs a verifier exactly like any other - run 35442208627 stopped here after
    # the lifecycle had already been corrected for the same assumption one layer up.
    if seed in eos_ids and not getattr(state.sampling_params, 'ignore_eos', False):
        raise ValueError('Terminal prefill must finish without allocating a verifier')
    if len(state.block_ids) != 1:
        raise ValueError('One scheduler-owned KV group required')
    validate_initial_capture_pages(pages, state.block_ids[0], position=len(prompt), output_budget=256)
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
                len(prompt), len(prompt) - 2048,
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
        shared_ccl = os.environ.get('QWEN_FAST_SHARED_CCL', '1') == '1'
        device = components.device(operations, model,
            collectives if collectives is not None and shared_ccl else components.collectives(model.mesh_device),
            layers, projection, selector, _qwen_outputs, position=len(prompt),
            block_rows=16, proposal_capture=True, max_new_tokens=256,
            fused_convolution=True, feature_start=len(prompt) - 2048,
            cache_history=True, cache_projection_capture=False, live_query_qk=False,
            native_proposal_attention=True, defer_proposal_capture=True, buffer_pool=buffer_pool,
            shared_weights=shared_weights)
        owned.callback(device.close)
        release_capture()
        runtime = components.runtime(device, position=len(prompt))
        session = components.session(state.req_id, prompt, seed, vocab_size=model.args.vocab_size,
            max_new_tokens=256, eos_ids=eos_ids, neural={'dflash2': runtime},
            verifier_rows=16, lookup_enabled=False)

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
            device.proposal_capture = components.proposal(device, max_new_tokens=256)

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
        engine = components.engine(model, session, pages, helpers, sampler=sampler,
            norm_batch=True, attention_replay=True, replay_group_rows=4, max_verify_rows=16,
            native_sampling_rows=True, retain_feature_taps=TARGET_TAPS,
            commit_only_gdn=True, target_attention_t16=True, before_capture=prepare_proposal,
            **(dict(storage=verifier_storage) if verifier_storage is not None else {}))
        owned.callback(engine.close)
        request = FastRequest(session, engine, runtime, release_drafter=device.close,
            collect_timings=os.environ.get('QWEN_FAST_PHASE_TIMING') == '1')
        owned.pop_all()
    return request
