"""Complete T16 DFlash composed/native comparison; no automatic promotion."""

import math

from drafter_comparison_report import validate_target_components

SCHEDULE = ((False, True), (True, True), (False, False), (True, False), (True, False), (False, False))
BLOCK_FIELDS = ('rows', 'source', 'accepted', 'match_length', 'position', 'input_tokens', 'committed')


def latency_budget(entries):
    blocks = [block for entry in entries for block in entry['blocks']]
    tokens = sum(entry['committed_decode_tokens'] for entry in entries)
    verify_ms = sum(block['verify_readback_ms'] for block in blocks)
    decode_ms = sum(entry['decode_ms'] for entry in entries)
    if not blocks or tokens <= 0 or verify_ms <= 0 or decode_ms <= 0:
        raise ValueError('Positive measured committed-token and verification totals required')
    return dict(target_tg=200, committed_tokens=tokens, blocks=len(blocks),
        committed_tokens_per_block=tokens / len(blocks),
        allowed_total_decode_ms=tokens * 5.0,
        allowed_mean_cycle_ms=tokens * 5.0 / len(blocks),
        measured_decode_ms=decode_ms,
        required_decode_reduction_fraction=max(0., 1 - tokens * 5.0 / decode_ms),
        verification_only_upper_bound_tg=1000 * tokens / verify_ms,
        verifier_alone_exceeds_budget=verify_ms > tokens * 5.0,
        scope='Conditional bound at measured acceptance and verification cost; not a predicted speed result')


def acceptance(entry, *, max_rows=16):
    if type(max_rows) is not int or max_rows not in (16, 32):
        raise ValueError('Explicit T16 or T32 proposal accounting required')
    blocks = entry.get('blocks')
    emitted, prompt = entry.get('emitted'), entry.get('prompt_tokens')
    if not isinstance(blocks, list) or not blocks or not isinstance(emitted, list) or not isinstance(prompt, list):
        raise ValueError('Complete proposal and committed-token tapes required')
    offset = proposed = accepted = 0
    for block in blocks:
        if (not isinstance(block, dict) or any(type(block.get(name)) is not int
                for name in ('rows', 'accepted', 'committed', 'position'))
                or block['source'] != 'dflash2' or block['rows'] not in (1, 2, 4, 8, 16, 32)
                or block['rows'] > max_rows
                or not isinstance(block.get('input_tokens'), list) or len(block['input_tokens']) != block['rows']
                or any(type(token) is not int or not 0 <= token < entry['vocab_size'] for token in block['input_tokens'])
                or not 0 <= block['accepted'] < block['rows'] or not 1 <= block['committed'] <= block['accepted'] + 1
                or block['position'] != len(prompt) + offset or offset + block['committed'] >= len(emitted)
                or block['input_tokens'][0] != emitted[offset]):
            raise ValueError('Bounded proposals and contiguous committed frontiers required')
        matched = min(block['accepted'], block['committed'])
        if block['input_tokens'][1:1 + matched] != emitted[offset + 1:offset + 1 + matched]:
            raise ValueError('Accepted proposal prefix differs from the exact committed token tape')
        for name in ('draft_ms', 'input_ms', 'verify_readback_ms', 'select_commit_ms', 'cycle_ms'):
            if type(block.get(name)) not in (int, float) or not math.isfinite(block[name]) or block[name] < 0:
                raise ValueError('All per-block costs must be retained and finite')
        offset += block['committed']
        proposed += block['rows'] - 1
        accepted += block['accepted']
    if (offset != entry['committed_decode_tokens'] or offset != len(emitted) - 1
            or type(entry.get('proposed')) is not int or type(entry.get('accepted')) is not int
            or entry['proposed'] != proposed or entry['accepted'] != accepted
            or entry['dflash']['proposal_calls'] != len(blocks)):
        raise ValueError('Proposal, acceptance, block and committed-token counters must reconcile')
    return dict(blocks=len(blocks), proposed=proposed, accepted=accepted, rejected=proposed - accepted,
        zero_acceptance_blocks=sum(block['rows'] > 1 and block['accepted'] == 0 for block in blocks),
        mixed_acceptance_blocks=sum(0 < block['accepted'] < block['rows'] - 1 for block in blocks),
        fully_accepted_blocks=sum(block['rows'] > 1 and block['accepted'] == block['rows'] - 1 for block in blocks))


def summarize(requests, *, weight_transport=False, bulk_pipeline=False):
    from full_dflash_request import summarize_dflash_requests

    if type(bulk_pipeline) is not bool or (bulk_pipeline and weight_transport):
        raise ValueError('One explicit weight-transport comparison policy required')
    transport = weight_transport or bulk_pipeline
    if (not isinstance(requests, list) or len(requests) != 6
            or [entry.get('native_proposal_attention') for entry in requests] !=
                ([True] * 6 if transport else [False, True, False, True, True, False])
            or [entry.get('instrumented_timing') for entry in requests] != [True, True, False, False, False, False]):
        raise ValueError('Two policy audits followed by a complete native-proposal ABBA required')
    if weight_transport and [('block_stream' in entry) for entry in requests] != [False, True, False, True, True, False]:
        raise ValueError('Two transport audits followed by unchanged/native-stream ABBA required')
    if bulk_pipeline and (any('block_stream' not in entry for entry in requests)
            or [entry['block_stream'].get('bulk_pipeline', False) for entry in requests]
                != [False, True, False, True, True, False]):
        raise ValueError('Serial/pipeline audits and ABBA on the same bulk streams required')
    if any(len(entry.get('prompt_tokens', [])) != 4096 or entry.get('max_new_tokens') != 256 for entry in requests):
        raise ValueError('Matched CTX4096 and 256-output T16 comparison required')
    reference = requests[0]
    identity = ('prompt_tokens', 'emitted', 'max_new_tokens', 'eos_ids', 'vocab_size', 'committed_decode_tokens',
        'fabric_sources', 'sources')
    draft_identity = ('checkpoints', 'block_rows', 'proposal_capture', 'proposal_contexts',
        'committed_feature_rows', 'target_taps', 'policy')
    for entry in requests:
        if (type(entry.get('native_proposal_attention')) is not bool
                or any(entry.get(key) != reference.get(key) for key in identity)
                or not entry.get('sources') or not entry.get('fabric_sources')
                or any(entry.get(key) is not True for key in ('commit_only_gdn', 'fused_convolution', 'cache_history'))
                or any(entry.get(key, False) is not False for key in
                    ('cache_projection_capture', 'live_query_qk', 'target_four_links'))
                or any(key not in entry['dflash'] or entry['dflash'][key] != reference['dflash'].get(key)
                    for key in draft_identity)
                or entry['dflash']['block_rows'] != 16 or entry['dflash']['proposal_capture'] is not True
                or entry['dflash'].get('native_proposal_attention', False) is not entry['native_proposal_attention']):
            raise ValueError('Only proposal arithmetic may change; target, outputs, source and cache policy must match')
        validate_target_components(entry, block_stream=transport and 'block_stream' in entry)
        acceptance(entry)
    if transport:
        trajectories = [[[block[key] for key in BLOCK_FIELDS] for block in entry['blocks']] for entry in requests]
        if any(trajectory != trajectories[0] for trajectory in trajectories[1:]):
            raise ValueError('Weight transport must not change proposals or acceptance')
    output = {}
    for name, indices in (('control', (0, 2, 5)), ('candidate', (1, 3, 4))):
        group = [requests[index] for index in indices]
        expected = [[block[key] for key in BLOCK_FIELDS] for block in group[0]['blocks']]
        if any([[block[key] for key in BLOCK_FIELDS] for block in entry['blocks']] != expected for entry in group[1:]):
            raise ValueError('Each policy must reproduce its own audited proposal and acceptance trajectory')
        summary = summarize_dflash_requests(group)
        counts = [acceptance(entry) for entry in group[1:]]
        totals = {key: sum(count[key] for count in counts) for key in counts[0]}
        totals.update(acceptance_fraction=totals['accepted'] / totals['proposed'] if totals['proposed'] else 0.,
            committed_tokens_per_block=summary['committed_tokens'] / totals['blocks'])
        summary['acceptance'] = totals
        summary['per_request_acceptance'] = counts
        summary['mean_block_costs_ms'] = {key: sum(block[key] for entry in group[1:] for block in entry['blocks'])
            / totals['blocks'] for key in ('draft_ms', 'input_ms', 'verify_readback_ms', 'select_commit_ms', 'cycle_ms')}
        summary['latency_budget'] = latency_budget(group[1:])
        output[name] = summary
    return dict(**output, measured_order=['control', 'candidate', 'candidate', 'control'],
        committed_tokens_per_second=output['candidate']['committed_tokens_per_second'],
        candidate_over_control=output['candidate']['committed_tokens_per_second'] / output['control']['committed_tokens_per_second'],
        target_reached=output['candidate']['target_reached'], proposal_trajectories_may_differ=not transport,
        performance_promoted=False, serving_qualified=False, held_out_coding_quality=False,
        scope=('Only target MLP weight transport changes; exact proposals and target tokens/state; not serving certification'
            if transport else 'Different draft arithmetic with exact native target tokens/state; not held-out quality or serving certification'))


def qualify_report(report):
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True or report.get('error')
            or not report.get('drafter_comparison_sources')
            or report['drafter_comparison_sources'] != report.get('drafter_comparison_sources_after')):
        raise ValueError('Clean complete combined report with unchanged sources required')
    result = summarize(report.get('request_checks', []))
    if result != report.get('dflash_native_comparison'):
        raise ValueError('Recorded comparison does not match complete request evidence')
    return result


if __name__ == '__main__':
    import hashlib
    import json
    from pathlib import Path
    import sys

    data = Path(sys.argv[1]).read_bytes()
    print(json.dumps(dict(qualify_report(json.loads(data)), raw_sha256=hashlib.sha256(data).hexdigest()), indent=2))
