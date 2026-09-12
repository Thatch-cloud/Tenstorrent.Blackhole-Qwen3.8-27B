"""Whole-request matched target-collective link ABBA, not a sampler-only or kernel-only bandwidth result."""

import argparse
import hashlib
import json
from pathlib import Path

from full_dflash_request import summarize_dflash_abba_requests
from model_link_policy import AXES, target_links
from sampling_link_policy import SOURCES as FABRIC_SOURCES


SOURCES = ('model_link_policy.py', 'target_link_request.py', 'full_dflash_request.py', 'full_request.py',
    'full-prefix.py', 'sampling_link_policy.py', 'dflash_benchmark_report.py')


def source_hashes(root):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES}


def measure_target_links(measure, operations, model, *args, candidate, **options):
    if (type(candidate) is not bool or not callable(measure)
            or any(not callable(options.get(name)) for name in ('live_digest', 'kv_digest', 'inactive_digest'))):
        raise ValueError('Explicit target-link experiment arm and native request measurement required')
    with target_links(model, 4 if candidate else 2) as audit:
        result = measure(operations, model, *args, **options)
        result['final_target_digests'] = dict(active_gdn=options['live_digest'](),
            valid_kv=options['kv_digest'](len(result['prompt_tokens']) + result['committed_decode_tokens']),
            inactive=options['inactive_digest']())
    result.update(target_four_links=candidate, target_link_audit=audit)
    return result


def summarize(requests):
    if not isinstance(requests, list) or len(requests) != 6:
        raise ValueError('Two audit requests followed by complete ABBA required')
    for entry in requests:
        candidate, audit = entry.get('target_four_links'), entry.get('target_link_audit')
        if (type(candidate) is not bool or not isinstance(audit, dict) or audit.get('restored') is not True
                or type(audit.get('owners_validated')) is not int or audit['owners_validated'] != 193
                or type(audit.get('requested_links')) is not int or audit['requested_links'] != (4 if candidate else 2)
                or any(entry.get(name) is not True for name in ('commit_only_gdn', 'fused_convolution', 'cache_history'))
                or any(entry.get(name, False) is not False for name in ('cache_projection_capture', 'live_query_qk'))
                or len(entry.get('prompt_tokens', [])) != 4096 or entry.get('sampler_num_links') != 4
                or entry.get('fabric_sources') != FABRIC_SOURCES):
            raise ValueError('Isolated cached 4K target-link experiment with unchanged four-link sampler required')
        for field in ('original_requests', 'effective_requests', 'calls'):
            values = audit.get(field)
            if (not isinstance(values, dict) or set(values) != {name for name, unused in AXES}
                    or any(type(value) is not int or value < 0 for value in values.values())):
                raise ValueError('Complete integer axis policy and engagement audit required')
        if (audit['effective_requests'] != {name: audit['requested_links'] for name, unused in AXES}
                or any(value not in (1, 2, 4) for value in audit['original_requests'].values())
                or audit['calls']['axis0'] < 64):
            raise ValueError('Every target MLP must actually request its matched link policy')
        digests = entry.get('final_target_digests')
        if (not isinstance(digests, dict) or set(digests) != {'active_gdn', 'valid_kv', 'inactive'}
                or any(not isinstance(values, list) or not values or any(not isinstance(value, str) or len(value) != 64
                    or any(character not in '0123456789abcdef' for character in value) for value in values)
                    for values in digests.values())):
            raise ValueError('Full final-state digest vectors required for cross-arm comparison')
        if digests != requests[0].get('final_target_digests'):
            raise ValueError('Changing target link counts changed final GDN, valid KV or inactive state')
    if any(entry['target_link_audit']['original_requests'] != requests[0]['target_link_audit']['original_requests']
            for entry in requests):
        raise ValueError('Matched requests must start from the same original target link policy')
    result = summarize_dflash_abba_requests(requests, arm_key='target_four_links')
    result['control']['target_collective_links'] = 2
    result['candidate']['target_collective_links'] = 4
    result['collective_scope'] = 'Target model two versus four links; sampler four; drafter unchanged'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware-result', type=Path, required=True)
    options = parser.parse_args()
    report = json.loads(options.hardware_result.read_text())
    if (report.get('passed') is not True or report.get('target_link_request_sources') != source_hashes(Path(__file__).parent)
            or report.get('request_summary') != summarize(report.get('request_checks', []))):
        raise ValueError('Current source-bound complete target-link ABBA result required')
    print(json.dumps(report['request_summary']))


if __name__ == '__main__':
    main()
