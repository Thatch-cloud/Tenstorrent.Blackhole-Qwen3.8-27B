"""Offline native-token tape scoring, not device verification or throughput.

Native predictions are valid only until the first draft mismatch. Later tape
entries and padding are deliberately unused by prefix selection; they are not
target predictions for the rejected draft branch. Drafts see committed history
only. No offline result certifies cache rollback or accelerator correctness.
"""

import argparse
import hashlib
import json
from pathlib import Path

from greedy_session import GreedySession


def score_lookup(prompt, emitted, *, vocab_size, max_new_tokens, eos_ids=(), max_rows=32,
                 minimum_match=1, prefer_full_suffix=False):
    prompt, emitted, eos_ids = list(prompt), list(emitted), tuple(eos_ids)
    if type(minimum_match) is not int or minimum_match < 1:
        raise ValueError('Positive minimum match required')
    if type(max_rows) is not int or max_rows not in (1, 2, 4, 8, 16, 32):
        raise ValueError('Supported verifier bucket required')
    if type(vocab_size) is not int or vocab_size < 1:
        raise ValueError('Positive vocabulary size required')
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError('Positive generation budget required')
    if not emitted or len(emitted) > max_new_tokens:
        raise ValueError('Nonempty native output within budget required')
    if any(type(token) is not int or not 0 <= token < vocab_size for token in (*prompt, *emitted, *eos_ids)):
        raise ValueError('Tokens must belong to the target vocabulary')
    if any(token in eos_ids for token in emitted[:-1]):
        raise ValueError('Native output must stop at its first terminal token')
    if len(emitted) != max_new_tokens and emitted[-1] not in eos_ids:
        raise ValueError('Native tape is incomplete without budget exhaustion or EOS')
    session = GreedySession('offline-lookup', prompt, emitted[0], vocab_size=vocab_size,
        max_new_tokens=max_new_tokens, eos_ids=eos_ids, verifier_rows=32,
        prefer_full_suffix=prefer_full_suffix)
    blocks = []
    try:
        while not session.finished:
            cursor = len(session.emitted)
            remaining = max_new_tokens - cursor
            limit = min(max_rows, remaining)
            match_length = 0
            if limit > 1:
                unused_tokens, match_length = session.drafter.lookup.propose_with_match(session.request_id, limit - 1)
            selected_rows = max_rows if match_length >= minimum_match else 1
            ticket = session.propose(session.request_id, max_rows=selected_rows)
            predictions = emitted[cursor:cursor + len(ticket.tokens)]
            predictions += [0] * (len(ticket.tokens) - len(predictions))
            decision = session.commit(session.request_id, ticket, predictions, lambda prefix: None)
            if session.emitted != emitted[:len(session.emitted)]:
                raise AssertionError('Offline scoring consumed beyond the native tape')
            blocks.append(dict(rows=len(ticket.tokens), match_length=match_length,
                source=ticket.source, accepted=decision.accepted, committed=len(decision.emitted),
                position=ticket.position))
        if session.emitted != emitted:
            raise AssertionError('Offline scoring did not consume the complete native tape')
        return dict(scope='Offline acceptance only; no device, timing or coding-quality certification',
            max_rows=max_rows, minimum_match=minimum_match, prefer_full_suffix=prefer_full_suffix,
            accepted=session.accepted_proposals, proposed=session.committed_block_proposals,
            committed=session.committed_decode_tokens, blocks=blocks,
            mean_committed_per_block=session.committed_decode_tokens / len(blocks) if blocks else None)
    finally:
        session.close(session.request_id)


def score_report(report):
    if report.get('passed') is not True or not report.get('request_checks'):
        raise ValueError('A passed native-checked request report is required')
    results = []
    for request in report['request_checks']:
        if any(request.get(key) is not True for key in ('exact', 'state_exact', 'inactive_exact')):
            raise ValueError('Request correctness checks must pass')
        for tokens, digest in (('prompt_tokens', 'prompt_sha256'), ('emitted', 'output_sha256')):
            expected = hashlib.sha256(json.dumps(request[tokens]).encode()).hexdigest()
            if request[digest] != expected:
                raise ValueError('Recorded token hash mismatch')
        arms = []
        for maximum in (1, 2, 4, 8, 16, 32):
            for minimum in (1, 2, 4, 8):
                for prefer_full in (False, True):
                    arms.append(score_lookup(request['prompt_tokens'], request['emitted'],
                        vocab_size=request['vocab_size'], max_new_tokens=request['max_new_tokens'],
                        eos_ids=request['eos_ids'], max_rows=maximum, minimum_match=minimum,
                        prefer_full_suffix=prefer_full))
        results.append(dict(prompt_sha256=request['prompt_sha256'], output_sha256=request['output_sha256'], arms=arms))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    options = parser.parse_args()
    print(json.dumps(score_report(json.loads(options.report.read_text())), indent=2))


if __name__ == '__main__':
    main()
