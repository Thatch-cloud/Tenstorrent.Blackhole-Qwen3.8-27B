"""Build a request from completed serving prefill, without reference generation."""

from contextlib import ExitStack
from types import SimpleNamespace

from serving_fast_request import FastRequest
from serving_fast_policy import validate_request_sampling


def device_components():
    from dflash_device import DFlashDevice
    from dflash_proposal_trace import PreparedDFlashProposal
    from dflash_request_runtime import DFlashRequestRuntime
    from greedy_session import GreedySession
    from verifier_engine import VerifierEngine
    from models.tt_transformers.tt.ccl import TT_CCL

    return SimpleNamespace(device=DFlashDevice, proposal=PreparedDFlashProposal,
        runtime=DFlashRequestRuntime, session=GreedySession, engine=VerifierEngine, collectives=TT_CCL)


def from_prefill(operations, model, sampler, pages, helpers, *, state, capture, fixtures, eos_ids):
    from dflash_request_runtime import TARGET_TAPS

    prompt = tuple(state.prompt_token_ids)
    validate_request_sampling(state.sampling_params, prompt_tokens=len(prompt))
    if (len(state.output_token_ids) != 1 or state.num_computed_tokens != len(prompt)
            or not isinstance(state.req_id, str) or not state.req_id
            or len(helpers) != 48):
        raise ValueError('Completed native prefill with exactly one emitted seed and all GDN helpers required')
    seed = state.output_token_ids[0]
    if type(seed) is not int or not 0 <= seed < model.args.vocab_size:
        raise ValueError('Valid target-selected prefill seed required')
    if seed in eos_ids:
        raise ValueError('Terminal prefill must finish without allocating a verifier')
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
        device = components.device(operations, model, components.collectives(model.mesh_device),
            layers, projection, selector, capture.outputs(), position=len(prompt),
            block_rows=16, proposal_capture=True, max_new_tokens=256,
            fused_convolution=True, feature_start=len(prompt) - 2048,
            cache_history=True, cache_projection_capture=False, live_query_qk=False,
            native_proposal_attention=True, defer_proposal_capture=True)
        owned.callback(device.close)
        release_capture()
        runtime = components.runtime(device, position=len(prompt))
        session = components.session(state.req_id, prompt, seed, vocab_size=model.args.vocab_size,
            max_new_tokens=256, eos_ids=eos_ids, neural={'dflash2': runtime},
            verifier_rows=16, lookup_enabled=False)

        def prepare_proposal(engine):
            if device.proposal_capture is not None:
                raise ValueError('Proposal trace already captured before verifier allocation')
            device.proposal_capture = components.proposal(device, max_new_tokens=256)

        engine = components.engine(model, session, pages, helpers, sampler=sampler,
            norm_batch=True, attention_replay=True, replay_group_rows=4, max_verify_rows=16,
            native_sampling_rows=True, retain_feature_taps=TARGET_TAPS,
            commit_only_gdn=True, target_attention_t16=True, before_capture=prepare_proposal)
        owned.callback(engine.close)
        request = FastRequest(session, engine, runtime, release_drafter=device.close)
        owned.pop_all()
    return request
