"""Complete combined-request diagnostic attribution; sampled cycles, not TG."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from mlp_compute_clock import ZONES
from mlp_read_order_report import validate_audit


def validate(report):
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('ctx_tokens') != 4096 or report.get('streams') != 1
            or report.get('fresh_context_audit') is not True
            or report.get('pp') is not None or report.get('committed_tg') is not None):
        raise ValueError('Clean complete 4K single-stream diagnostic required')
    for before, after in (('sources', 'sources_after'), ('native_sources', 'native_sources_after'),
            ('mlp_compute_clock_sources', 'mlp_compute_clock_sources_after')):
        if not report.get(before) or report.get(before) != report.get(after):
            raise ValueError('Unchanged runtime and diagnostic source fingerprints required')
    requests = report.get('request_checks', [])
    if len(requests) != 1:
        raise ValueError('Exactly one combined native-reference request required')
    request = requests[0]
    validate_audit(request)
    if (request.get('length') != 4096 or request.get('arm') != 'publication'
            or any(request.get(name) is not True for name in ('exact', 'state_exact', 'inactive_exact', 'instrumented_timing'))
            or request.get('committed_tokens_per_second') is not None):
        raise ValueError('Exact native tokens, target state and inactive slots required')
    fusion = request.get('fused_t16_mlp', {})
    if (fusion.get('rows') != 16 or fusion.get('restored') is not True
            or fusion.get('native_bindings_unchanged') is not True or len(fusion.get('hits', [])) != 64
            or any(type(count) is not int or count <= 0 for count in fusion['hits'])
            or request.get('gdn_norm_prefetch', {}).get('enabled') is not True
            or request.get('incremental_history', {}).get('enabled') is not True):
        raise ValueError('All fused layers and winning norm/publication paths required')
    summary = report.get('mlp_compute_clock_combined', {})
    if (summary != request.get('mlp_compute_clock_combined') or summary.get('diagnostic_only') is not True
            or summary.get('layers') != 64 or summary.get('sampled_verifier_replays') != 2
            or summary.get('compute_samples_releasable') is not True or summary.get('committed_tg') is not None):
        raise ValueError('Complete closed per-layer capture summary required')
    positions = [block['position'] for block in request.get('blocks', []) if block.get('rows') == 16][:2]
    records = summary.get('records', [])
    if len(positions) != 2 or len(records) != 2 or [record.get('position') for record in records] != positions:
        raise ValueError('Samples must match the first two actual T16 verifier blocks')
    expected = [(chip, processor, 0, name, block, subblock)
        for chip in range(2) for processor in range(3) for name, predicate, block, subblock in ZONES]
    groups = defaultdict(list)
    for replay, record in enumerate(records):
        layers = record.get('layers', [])
        if record.get('rows') != 16 or [layer.get('layer') for layer in layers] != list(range(64)):
            raise ValueError('All 64 ordered target layers required for each replay')
        for layer in layers:
            capture = layer.get('capture', {})
            samples = capture.get('samples', [])
            if (capture.get('label') != f'verify-{replay}-layer-{layer["layer"]}'
                    or capture.get('poisoned_before_execution') is not True
                    or [tuple(sample.get(field) for field in ('chip', 'processor', 'worker', 'zone', 'block', 'subblock'))
                        for sample in samples] != expected):
                raise ValueError('Fresh complete two-chip sample identities required')
            previous = {}
            for sample in samples:
                start, end, duration = (sample.get(name) for name in ('start_cycle', 'end_cycle', 'duration_cycles'))
                identity = sample['chip'], sample['processor']
                if (any(type(value) is not int for value in (start, end, duration))
                        or not 0 <= start <= end < 2**64 or not 0 <= duration <= 100_000_000
                        or duration != end - start or start < previous.get(identity, 0)):
                    raise ValueError('Bounded ordered internally consistent clock samples required')
                previous[identity] = end
                groups[sample['chip'], sample['processor'], sample['zone']].append(duration)
    return dict(passed=True, context=4096, layers=64, verifier_replays=2, sample_count=4608,
        diagnostic_only=True, active_utilization_measured=False, committed_tg=None,
        scope='Selected compute processor intervals; not active utilization or additive critical-path attribution',
        groups=[dict(chip=chip, processor=processor, zone=zone, samples=len(values), minimum_cycles=min(values),
            median_cycles=statistics.median(values), maximum_cycles=max(values))
            for (chip, processor, zone), values in sorted(groups.items())])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    print(json.dumps(dict(validate(json.loads(raw)), report_sha256=hashlib.sha256(raw).hexdigest()), indent=2))
