"""Summarize complete-request timing without counting audited or nested timers twice."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def summarize(report, target_tg=200.0):
    if not math.isfinite(target_tg) or target_tg <= 0:
        raise ValueError('Positive finite target throughput required')
    if report.get('passed') is not True or report.get('streams') != 1:
        raise ValueError('Passed single-stream request evidence required')
    grouped = {}
    for request in report['request_checks']:
        if any(request.get(name) is not True for name in ('exact', 'state_exact', 'inactive_exact')):
            raise ValueError('All requests must retain exactness checks')
        if request.get('instrumented_timing') is True:
            continue
        if request.get('instrumented_timing') is not False:
            raise ValueError('Explicit timing classification required')
        grouped.setdefault(request['arm'], []).append(request)
    arms = {}
    for arm, requests in grouped.items():
        blocks = [block for request in requests for block in request['blocks']]
        if not blocks:
            raise ValueError('Timed blocks required')
        fields = ('draft_ms', 'verify_readback_ms', 'select_commit_ms', 'cycle_ms', 'committed')
        for block in blocks:
            if any(type(block.get(name)) not in (int, float) or not math.isfinite(block[name])
                    or block[name] < 0 for name in fields) or block['cycle_ms'] <= 0:
                raise ValueError('Finite nonnegative block measurements required')
        means = {name: sum(block[name] for block in blocks) / len(blocks) for name in fields}
        arms[arm] = dict(requests=len(requests), blocks=len(blocks), mean=means,
            target_cycle_ms=1000 * means['committed'] / target_tg,
            block_only_tg=1000 * means['committed'] / means['cycle_ms'],
            request_tg=report['request_comparison']['arms'][arm]['committed_tg'])
    if not arms:
        raise ValueError('No uninstrumented requests')
    return dict(context=report['ctx_tokens'], streams=1, target_tg=target_tg, arms=arms,
        scope='Observed cycle budget, not a speedup prediction; verifier timer includes blocking replay',
        held_out_coding_quality=False, serving_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    options = parser.parse_args()
    raw = options.report.read_bytes()
    result = summarize(json.loads(raw))
    result['report_sha256'] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
