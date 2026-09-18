"""Matched complete T16/T32 requests; width-dependent proposals are not target output."""

from dflash_native_comparison_report import acceptance, latency_budget, BLOCK_FIELDS
from drafter_comparison_report import validate_target_components
from dflash_t32_combined_request import validate_request as validate_t32


WIDTHS = (16, 32, 16, 32, 32, 16)


def summarize(requests):
    from full_dflash_request import summarize_dflash_requests

    if (not isinstance(requests, list) or len(requests) != 6
            or tuple(entry.get('dflash', {}).get('block_rows') for entry in requests) != WIDTHS
            or [entry.get('instrumented_timing') for entry in requests] != [True, True, False, False, False, False]):
        raise ValueError('T16/T32 audits followed by complete width ABBA required')
    reference = requests[0]
    identity = ('prompt_tokens', 'emitted', 'max_new_tokens', 'eos_ids', 'vocab_size',
        'committed_decode_tokens', 'fabric_sources', 'sources')
    draft_identity = ('checkpoints', 'committed_feature_rows', 'target_taps', 'policy')
    for entry, rows in zip(requests, WIDTHS):
        if (len(entry.get('prompt_tokens', [])) != 4096 or entry.get('max_new_tokens') != 256
                or any(entry.get(key) != reference.get(key) for key in identity)
                or not entry.get('sources') or not entry.get('fabric_sources')
                or any(entry.get(key) is not True for key in
                    ('commit_only_gdn', 'fused_convolution', 'cache_history', 'native_proposal_attention'))
                or any(entry.get(key, False) is not False for key in
                    ('cache_projection_capture', 'live_query_qk', 'target_four_links'))
                or entry['dflash'].get('proposal_capture') is not True
                or entry['dflash'].get('native_proposal_attention') is not True
                or any(key not in entry['dflash'] or entry['dflash'][key] != reference['dflash'].get(key)
                    for key in draft_identity)):
            raise ValueError('Matched target tokens, prompt, precision policy, checkpoints and cache policy required')
        if rows == 32:
            validate_t32(entry)
        else:
            validate_target_components(entry, block_stream=True)
        acceptance(entry, max_rows=rows)
    output = {}
    for name, indices in (('control', (0, 2, 5)), ('candidate', (1, 3, 4))):
        group = [requests[index] for index in indices]
        tapes = [[[block[key] for key in BLOCK_FIELDS] for block in entry['blocks']] for entry in group]
        if any(tape != tapes[0] for tape in tapes[1:]):
            raise ValueError('Each width must reproduce its audited proposals and acceptance')
        summary = summarize_dflash_requests(group)
        measured = group[1:]
        counts = [acceptance(entry, max_rows=entry['dflash']['block_rows']) for entry in measured]
        totals = {key: sum(count[key] for count in counts) for key in counts[0]}
        totals.update(acceptance_fraction=totals['accepted'] / totals['proposed'] if totals['proposed'] else 0.,
            committed_tokens_per_block=summary['committed_tokens'] / totals['blocks'])
        summary.update(acceptance=totals, latency_budget=latency_budget(measured),
            mean_block_costs_ms={key: sum(block[key] for entry in measured for block in entry['blocks'])
                / totals['blocks'] for key in
                ('draft_ms', 'input_ms', 'verify_readback_ms', 'select_commit_ms', 'cycle_ms')})
        output[name] = summary
    return dict(**output, measured_order=[16, 32, 32, 16], ctx_tokens=4096, streams=1,
        candidate_over_control=output['candidate']['committed_tokens_per_second']
            / output['control']['committed_tokens_per_second'],
        target_reached=output['candidate']['target_reached'], proposal_trajectories_may_differ=True,
        performance_promoted=False, serving_qualified=False, held_out_coding_quality=False)


def qualify_report(report):
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True or report.get('error')
            or report.get('streams') != 1 or report.get('ctx_tokens') != 4096 or report.get('sampler_links') != 4
            or not report.get('drafter_comparison_sources')
            or report['drafter_comparison_sources'] != report.get('drafter_comparison_sources_after')):
        raise ValueError('Clean complete combined report with unchanged sources required')
    pool = report.get('block_stream_pool', {})
    if (pool.get('released') is not True or pool.get('native_bindings_unchanged') is not True
            or pool.get('allocated_layers') != 64 or pool.get('serving_defaults_changed') is not False
            or len(pool.get('admission', [])) != 2 or pool.get('setup_ms', 0) <= 0):
        raise ValueError('Complete admitted weight pool with successful release required')
    result = summarize(report.get('request_checks'))
    if result != report.get('dflash_t32_comparison'):
        raise ValueError('Recorded T32 comparison differs from complete request evidence')
    return result


if __name__ == '__main__':
    import hashlib
    import json
    from pathlib import Path
    import sys

    raw = Path(sys.argv[1]).read_bytes()
    print(json.dumps(dict(qualify_report(json.loads(raw)), raw_sha256=hashlib.sha256(raw).hexdigest()), indent=2))
