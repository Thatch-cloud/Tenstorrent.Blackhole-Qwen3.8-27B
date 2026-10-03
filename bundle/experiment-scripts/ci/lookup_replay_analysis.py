"""Counterfactual routing against a recorded greedy oracle; no latency measurement."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'speculative-decoding' / 'harness'))
from lookup_draft import LookupDraft


def replay(request, *, min_match=1, max_rows=32):
    if type(min_match) is not int or not 1 <= min_match <= 128 or type(max_rows) is not int or max_rows not in (1, 2, 4, 8, 16, 32):
        raise ValueError('Bounded lookup policy required')
    if request.get('family_routing') is not False or not all(request.get(key) is True for key in ('exact', 'state_exact', 'inactive_exact')):
        raise ValueError('Correctness-gated native-attention request required')
    gold = request['emitted']
    if not gold or request['committed_decode_tokens'] != len(gold) - 1:
        raise ValueError('Complete post-seed oracle accounting required')
    history = LookupDraft('offline', request['prompt_tokens'], max_proposals=31)
    history.commit('offline', gold[:1])
    cursor, blocks = 1, []
    while cursor < len(gold):
        maximum = min(max_rows, len(gold) - cursor)
        proposal, match = history.propose_with_match('offline', maximum - 1) if maximum > 1 else ([], 0)
        if match < min_match:
            proposal = []
        rows = max(width for width in (1, 2, 4, 8, 16, 32) if width <= min(maximum, len(proposal) + 1))
        proposal = proposal[:rows - 1]
        accepted = 0
        for token, expected in zip(proposal, gold[cursor:]):
            if token != expected:
                break
            accepted += 1
        count = accepted + 1
        blocks.append(dict(rows=rows, source='lookup' if rows > 1 else 'target',
            match_length=match if rows > 1 else 0, input_tokens=[gold[cursor - 1], *proposal],
            position=len(request['prompt_tokens']) + cursor - 1, accepted=accepted, committed=count))
        history.commit('offline', gold[cursor:cursor + count])
        cursor += count
    history.close('offline')
    return blocks


def analyze(request):
    reproduced = replay(request)
    if len(reproduced) != len(request['blocks']) or any(
            any(block[key] != actual[key] for key in block)
            for block, actual in zip(reproduced, request['blocks'], strict=True)):
        raise ValueError('Unrestricted replay must first reproduce every recorded routing decision')
    policies = []
    for min_match, max_rows in ((1, 32), (1, 8), (2, 8), (4, 8), (8, 8), (1, 1)):
        blocks = replay(request, min_match=min_match, max_rows=max_rows)
        policies.append(dict(min_match=min_match, max_rows=max_rows, calls=len(blocks),
            proposed=sum(block['rows'] - 1 for block in blocks), accepted=sum(block['accepted'] for block in blocks),
            width_histogram=dict(sorted(Counter(block['rows'] for block in blocks).items()))))
    return dict(scope=__doc__, recorded_routing_exact=True, policies=policies)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    options = parser.parse_args()
    report = json.loads(options.report.read_text())
    print(json.dumps(analyze(report['request_checks'][0]), indent=2))
