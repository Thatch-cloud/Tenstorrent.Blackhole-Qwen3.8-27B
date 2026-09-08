"""Actual lookup drafting pilot, not a representative coding-quality benchmark."""

import hashlib
import json
from pathlib import Path
import time

from greedy_session import GreedySession
from verifier_engine import VerifierEngine, validate_replay_options


class RequestMismatch(AssertionError):
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__(f"Actual drafted generation differs from native at token {evidence['token_index']}")


def check_committed_prefix(actual, expected, block):
    for index, token in enumerate(actual):
        if index >= len(expected) or token != expected[index]:
            raise RequestMismatch(dict(kind='committed-token-mismatch', token_index=index,
                actual_token=token, expected_token=expected[index] if index < len(expected) else None,
                actual_prefix=list(actual), expected_prefix=expected[:len(actual)], block=block))


def terminal_ids(weights, vocab_size):
    config = json.loads((Path(weights) / 'generation_config.json').read_text())
    tokens = config.get('eos_token_id')
    tokens = (tokens,) if type(tokens) is int else tuple(tokens) if isinstance(tokens, list) else ()
    if not tokens or any(type(token) is not int or not 0 <= token < vocab_size for token in tokens):
        raise ValueError('Frozen generation config must declare valid terminal IDs')
    return tokens


def measure_request(model, sampler, prompt, pages, helpers, *, prefill, decode, live_digest,
                    kv_digest, inactive_digest, eos_ids=(), max_new_tokens=129, norm_batch=False,
                    attention_replay=False, family_routing=False, attention_mask_once=False, replay_group_rows=4,
                    lookup_max_rows=32, engine_factory=None, neural=None, selected_drafter=None, lookup_enabled=True,
                    mtp_runtime=None, mtp_factory=None, progress=None, native_sampling_rows=False, short_context=False,
                    attention_audit=False, feature_factory=None, commit_only_gdn=False, audit_commit_only_gdn=False):
    if (type(commit_only_gdn) is not bool or type(audit_commit_only_gdn) is not bool
            or (audit_commit_only_gdn and not commit_only_gdn)):
        raise ValueError('Explicit commit-only GDN selection required before auditing deferred state')
    if type(attention_audit) is not bool or (attention_audit and not (short_context and attention_replay)):
        raise ValueError('Attention diagnostics require explicit short-context parallel attention')
    if type(native_sampling_rows) is not bool or (native_sampling_rows and sampler is None):
        raise ValueError('Native-row experiment requires explicit device sampling')
    if type(short_context) is not bool or (short_context and
            (not family_routing or not norm_batch or not native_sampling_rows or lookup_max_rows != 8 or replay_group_rows != 4)):
        raise ValueError('Short-context requests require bounded family routing, native-row sampling and norm-batched T8')
    if progress is not None and not callable(progress):
        raise ValueError('Request progress must be callable')
    if mtp_factory is not None and (not callable(mtp_factory) or mtp_runtime is not None):
        raise ValueError('Choose a prepared MTP runtime or one post-prefill factory')
    if feature_factory is not None:
        if (not callable(feature_factory) or mtp_factory is not None or mtp_runtime is not None
                or neural or selected_drafter is not None or engine_factory is not None):
            raise ValueError('One post-prefill feature-drafter factory must own request routing')
        neural, selected_drafter, lookup_enabled = {'dflash2': feature_factory}, 'dflash2', False
    if mtp_runtime is not None or mtp_factory is not None:
        if (neural or selected_drafter is not None or engine_factory is not None
                or (mtp_runtime is not None and not all(
                    callable(getattr(mtp_runtime, name, None)) for name in ('bind', 'publish', '__call__')))):
            raise ValueError('MTP request bridge owns neural routing and requires the hidden-retaining verifier')
        neural, selected_drafter, lookup_enabled = {'mtp': mtp_runtime or mtp_factory}, 'mtp', False
    neural = dict(neural or {})
    if (type(lookup_enabled) is not bool or (not lookup_enabled and not neural)
            or any(not isinstance(name, str) or not name or name in ('lookup', 'target')
            or not callable(adapter) for name, adapter in neural.items())
            or (selected_drafter is not None and (not isinstance(selected_drafter, str) or selected_drafter not in neural))
            or (neural and selected_drafter is None)):
        raise ValueError('Explicit registered neural adapter selection required')
    if type(lookup_max_rows) is not int or lookup_max_rows not in (1, 2, 4, 8, 16, 32):
        raise ValueError('Explicit supported lookup width cap required')
    if lookup_max_rows != 32 and (family_routing or attention_replay) and not short_context:
        raise ValueError('Lookup width-cap experiment requires native attention')
    if type(norm_batch) is not bool:
        raise ValueError('Explicit boolean norm-batch selection required')
    if type(attention_replay) is not bool or type(family_routing) is not bool:
        raise ValueError('Explicit boolean attention and routing selection required')
    if attention_replay and (not norm_batch or not family_routing):
        raise ValueError('Replay attention requires norm batching and bounded family routing')
    validate_replay_options(attention_replay, attention_mask_once, replay_group_rows)
    started = time.perf_counter()
    gold = [prefill(prompt)]
    native_prefill_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    while len(gold) < max_new_tokens and gold[-1] not in eos_ids:
        logits = decode(gold[-1], len(prompt) + len(gold) - 1, True)
        gold.append(int(logits.reshape(-1, model.args.vocab_size)[0].float().argmax()))
    native_decode_ms = (time.perf_counter() - started) * 1000
    gold_state = live_digest()
    gold_kv = kv_digest(len(prompt) + len(gold) - 1)
    started = time.perf_counter()
    seed = prefill(prompt)
    prefill_ms = (time.perf_counter() - started) * 1000
    if seed != gold[0]:
        raise AssertionError(f'Fresh request prefill changed the native seed: native={gold[0]}, candidate={seed}')
    mtp_setup_ms = 0.0
    feature_setup_ms = 0.0
    feature_runtime = None
    if mtp_factory is not None:
        started = time.perf_counter()
        mtp_runtime = mtp_factory()
        mtp_setup_ms = (time.perf_counter() - started) * 1000
        if not all(callable(getattr(mtp_runtime, name, None)) for name in ('bind', 'publish', '__call__')):
            raise ValueError('MTP factory must return a prepared request runtime')
        neural = {'mtp': mtp_runtime}
    if feature_factory is not None:
        from dflash_request_runtime import TARGET_TAPS

        started = time.perf_counter()
        feature_runtime = feature_factory()
        feature_setup_ms = (time.perf_counter() - started) * 1000
        if (not all(callable(getattr(feature_runtime, name, None)) for name in ('bind', 'publish', '__call__'))
                or tuple(feature_runtime.tap_ids) != TARGET_TAPS):
            raise ValueError('DFlash2 factory must return a complete target-feature request bridge')
        neural = {'dflash2': feature_runtime}
    runtime = mtp_runtime if mtp_runtime is not None else feature_runtime
    inactive_before = inactive_digest()
    session = GreedySession('lookup-pilot', prompt, seed, vocab_size=model.args.vocab_size,
        max_new_tokens=max_new_tokens, eos_ids=eos_ids, verifier_rows=32, neural=neural, lookup_enabled=lookup_enabled)
    engine = None
    blocks = []
    setup_ms = decode_ms = 0.0
    capture_count = 0
    gdn_verify_checks = []
    try:
        if not session.finished:
            plan = None
            if family_routing:
                from attention_request_plan import capture_plan
                plan = capture_plan(session.position, pages.shape[1] * 64, session.verifier_rows,
                    session.max_new_tokens - len(session.emitted), max_verify_rows=lookup_max_rows, short_context=short_context)
            factory = VerifierEngine if engine_factory is None else engine_factory
            engine = factory(model, session, pages, helpers, sampler=sampler, norm_batch=norm_batch,
                attention_replay=attention_replay, attention_mask_once=attention_mask_once,
                replay_group_rows=replay_group_rows,
                **(dict(native_sampling_rows=True) if native_sampling_rows else {}),
                **(dict(short_context=True) if short_context else {}),
                **(dict(attention_audit=True) if attention_audit else {}),
                **(dict(retain_mtp_hidden=True) if mtp_runtime is not None else {}),
                **(dict(retain_feature_taps=feature_runtime.tap_ids) if feature_runtime is not None else {}),
                **(dict(commit_only_gdn=True) if commit_only_gdn else {}),
                **(dict(max_verify_rows=lookup_max_rows) if lookup_max_rows != 32 else {}))
            if runtime is not None:
                runtime.bind(session, engine)
            publish = engine.publish if runtime is None else runtime.publish
            capture_count = len(engine.buckets)
            setup_ms = engine.setup_ms
            started = time.perf_counter()
            while not session.finished:
                block_started = time.perf_counter()
                maximum = plan.max_rows(session.position, session.max_new_tokens - len(session.emitted)) if plan else 32
                maximum = min(maximum, lookup_max_rows)
                if attention_replay and maximum != engine.proposal_rows():
                    raise AssertionError('Engine and matched request disagree on safe proposal width')
                ticket = session.propose(session.request_id, max_rows=maximum, selected=selected_drafter)
                drafted = time.perf_counter()
                before_verify = live_digest() if audit_commit_only_gdn and len(ticket.tokens) > 1 else None
                predictions, components = engine.verify(ticket)
                if before_verify is not None:
                    if live_digest() != before_verify:
                        raise AssertionError('Commit-only verification modified native GDN state before the decision')
                    gdn_verify_checks.append(dict(position=ticket.position, rows=len(ticket.tokens), unchanged=True))
                verified = time.perf_counter()
                decision = session.commit(session.request_id, ticket, predictions, publish)
                finished = time.perf_counter()
                blocks.append(dict(rows=len(ticket.tokens), source=ticket.source, accepted=decision.accepted,
                    match_length=ticket.match_length, position=ticket.position, input_tokens=list(ticket.tokens),
                    committed=len(decision.emitted), draft_ms=(drafted - block_started) * 1000,
                    select_commit_ms=(finished - verified) * 1000,
                    cycle_ms=(finished - block_started) * 1000, **components))
                check_committed_prefix(session.emitted, gold, dict(blocks[-1],
                    predictions=list(predictions), emitted=list(decision.emitted)))
                if progress is not None:
                    progress(blocks[-1])
            decode_ms = (time.perf_counter() - started) * 1000
            engine.close()
            engine = None
        if session.emitted != gold:
            mismatch = next((index for index, pair in enumerate(zip(session.emitted, gold)) if pair[0] != pair[1]),
                            min(len(session.emitted), len(gold)))
            raise AssertionError(f'Actual drafted generation differs from native at token {mismatch}')
        if session.committed_decode_tokens != len(gold) - 1 or sum(block['committed'] for block in blocks) != len(gold) - 1:
            raise AssertionError('Committed token accounting must exclude the prefill seed')
        if live_digest() != gold_state or kv_digest(session.position) != gold_kv or inactive_digest() != inactive_before:
            raise AssertionError('Actual request final active GDN, valid KV or inactive slots differ')
        return dict(length=len(prompt), kind='Synthetic repeated-code lookup pilot; not a coding-quality benchmark',
            exact=True, state_exact=True, inactive_exact=True, blocks=blocks, norm_batch=norm_batch,
            commit_only_gdn=commit_only_gdn, gdn_verify_checks=gdn_verify_checks,
            attention_replay=attention_replay, family_routing=family_routing, capture_count=capture_count,
            attention_mask_once=attention_mask_once, replay_group_rows=replay_group_rows,
            lookup_max_rows=lookup_max_rows, native_sampling_rows=native_sampling_rows, short_context=short_context,
            attention_audit=attention_audit, instrumented_timing=attention_audit or audit_commit_only_gdn,
            selected_drafter=selected_drafter,
            drafting_policy='lookup-first' if lookup_enabled else 'neural-with-target-fallback',
            prompt_tokens=list(prompt), emitted=gold, max_new_tokens=max_new_tokens, eos_ids=list(eos_ids),
            vocab_size=model.args.vocab_size,
            prompt_sha256=hashlib.sha256(json.dumps(list(prompt)).encode()).hexdigest(),
            output_sha256=hashlib.sha256(json.dumps(gold).encode()).hexdigest(),
            committed_decode_tokens=session.committed_decode_tokens, proposed=session.committed_block_proposals,
            accepted=session.accepted_proposals, prefill_ms=prefill_ms, engine_setup_ms=setup_ms,
            decode_ms=decode_ms, mtp_setup_ms=mtp_setup_ms, feature_setup_ms=feature_setup_ms,
            native_prefill_ms=native_prefill_ms, native_decode_ms=native_decode_ms,
            committed_tokens_per_second=1000 * session.committed_decode_tokens / decode_ms
                if decode_ms and not (attention_audit or audit_commit_only_gdn) else None,
            post_seed_including_setup_ms=mtp_setup_ms + feature_setup_ms + setup_ms + decode_ms,
            prefill_setup_decode_ms=prefill_ms + mtp_setup_ms + feature_setup_ms + setup_ms + decode_ms,
            setup_amortized=False, cross_request_trace_reuse=False)
    finally:
        if engine is not None:
            if engine.phase == 'verified' and session.phase == 'pending':
                session.abort(session.request_id, session.pending,
                              engine.publish if runtime is None else runtime.publish)
            engine.close()
        session.close(session.request_id)
