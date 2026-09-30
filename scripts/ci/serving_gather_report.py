"""Fail-closed combined HTTP comparison; exclude both compilation warmups."""

import argparse
import hashlib
import json
import math
from pathlib import Path


ARMS = ('control', 'grouped', 'control', 'grouped', 'grouped', 'control')
REPORT_SHA256 = 'e645f7de77e3a31b086c388ef120ebd6522d51087c66084945c0960b2faff91b'


def records(log):
    result = []
    for line in log.splitlines():
        start = line.find('{')
        if start < 0:
            continue
        try:
            record, _ = json.JSONDecoder().raw_decode(line[start:])
        except ValueError:
            continue
        if isinstance(record, dict) and record.get('stage') in (
                'fast_serving_phases', 'gather_comparison_request'):
            result.append(record)
    return result


def summarize(canary, http, events, kernel, *, candidate_arm='grouped', report_sha256=REPORT_SHA256):
    if candidate_arm not in ('grouped', 'gate_exp'):
        raise ValueError('Known recurrence comparison required')
    arms = tuple(candidate_arm if arm == 'grouped' else arm for arm in ARMS)
    if (canary.get('passed') is not True or canary.get('forced_shutdown')
            or canary.get('server_exit_code') not in (0, -15)
            or canary.get('shutdown') != dict(worker_closed=True, devices_closed=True, engine_forced=False)
            or http.get('passed') is not True or len(http.get('requests', [])) != 6
            or len(events) != 12):
        raise ValueError('Six complete exact requests and clean hardware shutdown required')
    requests = []
    expected_tokens = http['requests'][0]['output_tokens']
    for ordinal, arm in enumerate(arms):
        phase, scope = events[ordinal * 2:ordinal * 2 + 2]
        response = http['requests'][ordinal]
        if (phase.get('stage') != 'fast_serving_phases' or phase.get('finished') is not True
                or phase.get('cancelled') is not False or phase.get('added_device_fences') is not False
                or scope.get('stage') != 'gather_comparison_request' or scope.get('ordinal') != ordinal
                or scope.get('arm') != arm or scope.get('warmup') is not (ordinal < 2)
                or response.get('ordinal') != ordinal or response.get('exact') is not True
                or response.get('context') != 4096 or response.get('output_tokens') != expected_tokens
                or len(expected_tokens) != 122):
            raise ValueError('Ordered identical-context exact-token warmups and ABBA required')
        audit = scope.get('audit')
        if arm == candidate_arm:
            if (not audit or audit.get('restored') is not True
                    or audit.get('report_sha256') != report_sha256 or not audit.get('kernels')
                    or (candidate_arm == 'gate_exp' and len(audit['kernels']) != 96)
                    or any(item != kernel for item in audit['kernels'])):
                raise ValueError('Every candidate kernel must match admitted simulation and restore')
        elif audit is not None:
            raise ValueError('Control must not use candidate scope')
        blocks = phase.get('blocks', [])
        position = 4096
        totals = dict.fromkeys(('draft_ms', 'verify_host_ms', 'commit_host_ms',
            'outside_phases_ms', 'cycle_ms', 'blocking_trace_host_ms'), 0.0)
        signature = []
        for block in blocks:
            committed = block['committed']
            if (type(committed) is not int or not 0 < committed <= block['rows'] <= 16
                    or block['position'] != position):
                raise ValueError('Contiguous committed-token accounting required')
            position += committed
            signature.append((block['rows'], committed))
            values = {key: block[key] for key in totals if key != 'blocking_trace_host_ms'}
            values['blocking_trace_host_ms'] = block['verifier']['blocking_trace_host_ms']
            if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                    for value in values.values()):
                raise ValueError('Finite nonnegative timings required')
            if (abs(sum(values[key] for key in ('draft_ms', 'verify_host_ms', 'commit_host_ms',
                    'outside_phases_ms')) - values['cycle_ms']) > 0.01
                    or values['blocking_trace_host_ms'] > values['verify_host_ms'] + 0.01):
                raise ValueError('Phase accounting or nested trace timing invalid')
            for key, value in values.items():
                totals[key] += value
        if position != 4096 + len(expected_tokens) - 1 or totals['cycle_ms'] <= 0:
            raise ValueError('All decode tokens after the prefill seed must be measured')
        requests.append(dict(ordinal=ordinal, arm=arm, warmup=ordinal < 2, blocks=len(blocks),
            signature=signature, totals=totals, decode_tokens=position - 4096,
            cycle_tokens_per_second=(position - 4096) * 1000 / totals['cycle_ms'],
            http_delivery_tokens_per_second=response['stream_delivery_tokens_per_second']))
    if any(item['signature'] != requests[2]['signature'] for item in requests[2:]):
        raise ValueError('Measured comparison changed acceptance or verifier buckets')
    aggregates = {}
    for arm in ('control', candidate_arm):
        selected = [item for item in requests[2:] if item['arm'] == arm]
        aggregates[arm] = dict(cycle_tokens_per_second=sum(item['decode_tokens'] for item in selected)
            * 1000 / sum(item['totals']['cycle_ms'] for item in selected),
            mean_verify_ms=sum(item['totals']['verify_host_ms'] for item in selected)
            / sum(item['blocks'] for item in selected))
    pairs = []
    for control, candidate in ((requests[2], requests[3]), (requests[5], requests[4])):
        pairs.append(dict(control_ordinal=control['ordinal'], candidate_ordinal=candidate['ordinal'],
            cycle_speedup=candidate['cycle_tokens_per_second'] / control['cycle_tokens_per_second'],
            verify_ms_change=(candidate['totals']['verify_host_ms'] - control['totals']['verify_host_ms'])
            / control['blocks']))
    return dict(passed=True, requests=requests, measured=aggregates, pairs=pairs,
        repeatable_two_percent_screen=all(pair['cycle_speedup'] > 1.02 for pair in pairs),
        cycle_speedup=aggregates[candidate_arm]['cycle_tokens_per_second'] / aggregates['control']['cycle_tokens_per_second'],
        performance_qualified=False, serving_qualified=False,
        scope='One fixture, warm ABBA HTTP screen; not held-out quality, state parity or production acceptance')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--candidate', choices=('grouped', 'gate_exp'), default='grouped')
    arguments = parser.parse_args()
    expected_sha = REPORT_SHA256
    if arguments.candidate == 'gate_exp':
        from gdn_gate_exp_gate import REPORT_SHA256 as expected_sha
        from gdn_gate_exp_report import inspect

        inspect(arguments.evidence, Path(__file__).parent)
    raw = (arguments.evidence / 'gdn-shared-recurrence.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('Pinned simulator evidence required')
    root = arguments.results
    report = summarize(json.loads((root / 'canary.json').read_text()),
        json.loads((root / 'http-reference.json').read_text()), records((root / 'server.log').read_text(encoding='utf-8')),
        json.loads(raw)['generated_kernels'][0], candidate_arm=arguments.candidate, report_sha256=expected_sha)
    report['candidate'] = arguments.candidate
    (root / 'gather-comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(passed=report['passed'], measured=report['measured'], pairs=report['pairs'],
        repeatable_two_percent_screen=report['repeatable_two_percent_screen'],
        cycle_speedup=report['cycle_speedup'], performance_qualified=False)))


if __name__ == '__main__':
    main()
