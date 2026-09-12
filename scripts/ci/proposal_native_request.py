"""Different draft arithmetic, unchanged exact target: isolated complete-request ABBA."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from proposal_native_attention import POLICY
from proposal_native_attention_gate import SOURCES, ORIGINAL, PACKER, hashes, native_hashes, qualify


INTEGRATION = ('proposal_native_request.py', 'draft_attention_branch.py', 'dflash_device.py',
    'dflash_proposal_trace.py', 'dflash_proposal_inputs.py', 'dflash_request_runtime.py',
    'full_dflash_request.py', 'full_request.py', 'verifier_engine.py', 'model_batch.py',
    'gdn_device_loop_state.py', 'prepared_target_features.py', 'draft_kv_history.py',
    'full-prefix.py', 'run-baseline.sh', 'baseline-suite.sh', 'dflash_benchmark_report.py')
BLOCK_FIELDS = ('rows', 'source', 'accepted', 'match_length', 'position', 'input_tokens', 'committed')


def source_hashes():
    return hashes(Path(__file__).parent, tuple(dict.fromkeys((*SOURCES, *INTEGRATION))))


def input_evidence(native):
    if not isinstance(native, dict) or native.get(PACKER) != ORIGINAL[PACKER]:
        raise ValueError('Hardware must use the original packer, not the simulator exception')
    root = Path(__file__).parent
    reports = {}
    for context in (31, 2048):
        report_path = root / f'proposal-native-attention-simulator-{context}.json'
        status_path = root / f'proposal-native-attention-simulator-{context}.exit-status'
        if status_path.read_text().strip() != '0':
            raise ValueError('Both successful outer simulator exits are required')
        result = qualify(json.loads(report_path.read_text()), context, hashes(root, SOURCES), native)
        reports[str(context)] = dict(report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
            exit_sha256=hashlib.sha256(status_path.read_bytes()).hexdigest(), gate=result)
    return dict(policy=POLICY, reports=reports, native_sources=native)


def qualify_inputs(metal_root):
    return input_evidence(native_hashes(metal_root))


def acceptance(entry):
    blocks = entry.get('blocks')
    emitted, prompt = entry.get('emitted'), entry.get('prompt_tokens')
    if not isinstance(blocks, list) or not blocks or not isinstance(emitted, list) or not isinstance(prompt, list):
        raise ValueError('Complete proposal and committed-token tapes required')
    offset = proposed = accepted = 0
    for block in blocks:
        if (not isinstance(block, dict) or any(type(block.get(name)) is not int
                for name in ('rows', 'accepted', 'committed', 'position'))
                or block['source'] != 'dflash2' or block['rows'] not in (1, 2, 4, 8)
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


def summarize(requests):
    from full_dflash_request import summarize_dflash_requests

    if (not isinstance(requests, list) or len(requests) != 6
            or [entry.get('native_proposal_attention') for entry in requests] != [False, True, False, True, True, False]
            or [entry.get('instrumented_timing') for entry in requests] != [True, True, False, False, False, False]):
        raise ValueError('Two policy audits followed by a complete native-proposal ABBA required')
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
                or entry['dflash']['block_rows'] != 8 or entry['dflash']['proposal_capture'] is not True
                or entry['dflash'].get('native_proposal_attention', False) is not entry['native_proposal_attention']):
            raise ValueError('Only proposal arithmetic may change; target, outputs, source and cache policy must match')
        acceptance(entry)
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
        output[name] = summary
    return dict(**output, measured_order=['control', 'candidate', 'candidate', 'control'],
        committed_tokens_per_second=output['candidate']['committed_tokens_per_second'],
        candidate_over_control=output['candidate']['committed_tokens_per_second'] / output['control']['committed_tokens_per_second'],
        target_reached=output['candidate']['target_reached'], proposal_trajectories_may_differ=True,
        scope='Different draft arithmetic with exact native target tokens/state; not held-out quality or serving certification')


def qualify_request(report):
    from dflash_benchmark_report import report_rows

    if (report.get('passed') is not True or report.get('error') or report.get('closed_cleanly') is not True
            or report.get('context_lengths') != [4096] or report.get('eligible_for_serving_gate') is not False):
        raise ValueError('Clean complete 4K request comparison without serving promotion required')
    evidence = report.get('proposal_native_inputs')
    if not isinstance(evidence, dict) or evidence != input_evidence(evidence.get('native_sources')):
        raise ValueError('Both original-runtime simulator prerequisites required')
    expected = source_hashes()
    requests = report.get('request_checks', [])
    if any(any(entry.get('sources', {}).get(name) != digest for name, digest in expected.items()) for entry in requests):
        raise ValueError('Current native proposal and unchanged target integration source fingerprints required')
    summary = summarize(requests)
    if report.get('request_summary') != summary:
        raise ValueError('Saved proposal-policy summary must match every raw request and acceptance count')
    return dict(summary=summary, benchmark=report_rows(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--metal-root', type=Path)
    mode.add_argument('--hardware-result', type=Path)
    options = parser.parse_args()
    result = qualify_inputs(options.metal_root) if options.metal_root else qualify_request(
        json.loads(options.hardware_result.read_text()))
    print(json.dumps(dict(passed=True, result=result)))


if __name__ == '__main__':
    main()
