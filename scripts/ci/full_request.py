"""Actual lookup drafting pilot, not a representative coding-quality benchmark."""

import hashlib
import json
from pathlib import Path
import time

from greedy_session import GreedySession
from verifier_engine import VerifierEngine, validate_replay_options


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
                    lookup_max_rows=32, engine_factory=None, neural=None, selected_drafter=None, lookup_enabled=True):
    neural = dict(neural or {})
    if (type(lookup_enabled) is not bool or (not lookup_enabled and not neural)
            or any(not isinstance(name, str) or not name or name in ('lookup', 'target')
            or not callable(adapter) for name, adapter in neural.items())
            or (selected_drafter is not None and (not isinstance(selected_drafter, str) or selected_drafter not in neural))
            or (neural and selected_drafter is None)):
        raise ValueError('Explicit registered neural adapter selection required')
    if type(lookup_max_rows) is not int or lookup_max_rows not in (1, 2, 4, 8, 16, 32):
        raise ValueError('Explicit supported lookup width cap required')
    if lookup_max_rows != 32 and (family_routing or attention_replay):
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
        raise AssertionError('Fresh request prefill changed the native seed')
    inactive_before = inactive_digest()
    session = GreedySession('lookup-pilot', prompt, seed, vocab_size=model.args.vocab_size,
        max_new_tokens=max_new_tokens, eos_ids=eos_ids, verifier_rows=32, neural=neural, lookup_enabled=lookup_enabled)
    engine = None
    blocks = []
    setup_ms = decode_ms = 0.0
    capture_count = 0
    try:
        if not session.finished:
            plan = None
            if family_routing:
                from attention_request_plan import capture_plan
                plan = capture_plan(session.position, pages.shape[1] * 64, session.verifier_rows,
                    session.max_new_tokens - len(session.emitted))
            factory = VerifierEngine if engine_factory is None else engine_factory
            engine = factory(model, session, pages, helpers, sampler=sampler, norm_batch=norm_batch,
                attention_replay=attention_replay, attention_mask_once=attention_mask_once,
                replay_group_rows=replay_group_rows,
                **(dict(max_verify_rows=lookup_max_rows) if lookup_max_rows != 32 else {}))
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
                predictions, components = engine.verify(ticket)
                verified = time.perf_counter()
                decision = session.commit(session.request_id, ticket, predictions, engine.publish)
                finished = time.perf_counter()
                blocks.append(dict(rows=len(ticket.tokens), source=ticket.source, accepted=decision.accepted,
                    match_length=ticket.match_length, position=ticket.position, input_tokens=list(ticket.tokens),
                    committed=len(decision.emitted), draft_ms=(drafted - block_started) * 1000,
                    select_commit_ms=(finished - verified) * 1000,
                    cycle_ms=(finished - block_started) * 1000, **components))
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
            attention_replay=attention_replay, family_routing=family_routing, capture_count=capture_count,
            attention_mask_once=attention_mask_once, replay_group_rows=replay_group_rows,
            lookup_max_rows=lookup_max_rows,
            selected_drafter=selected_drafter,
            drafting_policy='lookup-first' if lookup_enabled else 'neural-with-target-fallback',
            prompt_tokens=list(prompt), emitted=gold, max_new_tokens=max_new_tokens, eos_ids=list(eos_ids),
            vocab_size=model.args.vocab_size,
            prompt_sha256=hashlib.sha256(json.dumps(list(prompt)).encode()).hexdigest(),
            output_sha256=hashlib.sha256(json.dumps(gold).encode()).hexdigest(),
            committed_decode_tokens=session.committed_decode_tokens, proposed=session.committed_block_proposals,
            accepted=session.accepted_proposals, prefill_ms=prefill_ms, engine_setup_ms=setup_ms,
            decode_ms=decode_ms, native_prefill_ms=native_prefill_ms, native_decode_ms=native_decode_ms,
            committed_tokens_per_second=1000 * session.committed_decode_tokens / decode_ms if decode_ms else None,
            post_seed_including_setup_ms=setup_ms + decode_ms,
            prefill_setup_decode_ms=prefill_ms + setup_ms + decode_ms,
            setup_amortized=False, cross_request_trace_reuse=False)
    finally:
        if engine is not None:
            if engine.phase == 'verified' and session.phase == 'pending':
                session.abort(session.request_id, session.pending, engine.publish)
            engine.close()
        session.close(session.request_id)
