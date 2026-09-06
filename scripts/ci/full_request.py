"""Actual lookup drafting pilot, not a representative coding-quality benchmark."""

import hashlib
import json
from pathlib import Path
import time

from greedy_session import GreedySession
from verifier_engine import VerifierEngine


def terminal_ids(weights, vocab_size):
    config = json.loads((Path(weights) / 'generation_config.json').read_text())
    tokens = config.get('eos_token_id')
    tokens = (tokens,) if type(tokens) is int else tuple(tokens) if isinstance(tokens, list) else ()
    if not tokens or any(type(token) is not int or not 0 <= token < vocab_size for token in tokens):
        raise ValueError('Frozen generation config must declare valid terminal IDs')
    return tokens


def measure_request(model, sampler, prompt, pages, helpers, *, prefill, decode, live_digest,
                    kv_digest, inactive_digest, eos_ids=(), max_new_tokens=129, norm_batch=False):
    if type(norm_batch) is not bool:
        raise ValueError('Explicit boolean norm-batch selection required')
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
        max_new_tokens=max_new_tokens, eos_ids=eos_ids, verifier_rows=32)
    engine = None
    blocks = []
    setup_ms = decode_ms = 0.0
    try:
        if not session.finished:
            engine = VerifierEngine(model, session, pages, helpers, sampler=sampler, norm_batch=norm_batch)
            setup_ms = engine.setup_ms
            started = time.perf_counter()
            while not session.finished:
                block_started = time.perf_counter()
                ticket = session.propose(session.request_id, max_rows=32)
                drafted = time.perf_counter()
                predictions, components = engine.verify(ticket)
                verified = time.perf_counter()
                decision = session.commit(session.request_id, ticket, predictions, engine.publish)
                finished = time.perf_counter()
                blocks.append(dict(rows=len(ticket.tokens), source=ticket.source, accepted=decision.accepted,
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
            emitted=gold, output_sha256=hashlib.sha256(json.dumps(gold).encode()).hexdigest(),
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
