"""Offline attribution of committed request latency, not kernel timing."""

import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(request):
    blocks = request['blocks']
    committed = request['committed_decode_tokens']
    if sum(block['committed'] for block in blocks) != committed:
        raise ValueError('Block accounting differs from committed request tokens')
    if not all(request.get(key) is True for key in ('exact', 'state_exact', 'inactive_exact')):
        raise ValueError('Only correctness-gated requests may be summarized')
    widths = Counter(block['rows'] for block in blocks)
    components = {key: sum(block[key] for block in blocks) for key in
                  ('draft_ms', 'input_ms', 'verify_readback_ms', 'select_commit_ms')}
    decode_ms = request['decode_ms']
    return dict(
        context=request['length'], replay_group_rows=request['replay_group_rows'],
        committed_tokens=committed, verification_calls=len(blocks),
        width_histogram=dict(sorted(widths.items())),
        committed_per_verification=committed / len(blocks) if blocks else None,
        committed_tokens_per_second=1000 * committed / decode_ms if decode_ms else None,
        decode_ms=decode_ms, component_ms=components,
        unassigned_decode_ms=decode_ms - sum(components.values()),
        width8_verify_readback_ms=sum(block['verify_readback_ms'] for block in blocks if block['rows'] == 8),
        target_200_decode_budget_ms=5 * committed,
        scope='Host wall time; verification includes readback, not isolated device kernels')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    options = parser.parse_args()
    report = json.loads(options.report.read_text())
    requests = report['request_checks']
    if not requests:
        raise ValueError('No completed request measurements')
    print(json.dumps([summarize(request) for request in requests], indent=2))


if __name__ == '__main__':
    main()
