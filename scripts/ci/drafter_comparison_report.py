"""Compare complete requests without requiring different drafters to propose identical tokens."""

import math

SCHEDULE = (('dspark', True), ('dflash2', True), ('dspark', False),
            ('dflash2', False), ('dflash2', False), ('dspark', False))


def summarize(requests):
    from dspark_request_experiment import summarize as summarize_dspark
    from full_dflash_request import summarize_dflash_requests
    from cumulative_fusion_validation import validate_fusion_policy
    from shared_qk_norm_scatter_gate import REPORT_SHA256 as NORM_SHA256
    from gdn_direct_window_gate import REPORT_SHA256 as WINDOW_SHA256
    from mlp_down_grid_gate import REPORT_SHA256 as DOWN_SHA256

    if [(value.get('comparison_drafter'), value.get('instrumented_timing')) for value in requests] != list(SCHEDULE):
        raise ValueError('Fresh per-drafter audits and complete ABBA schedule required')
    first = requests[0]
    identity = ('prompt_tokens', 'emitted', 'max_new_tokens', 'eos_ids', 'vocab_size', 'committed_decode_tokens')
    signatures = {}
    for request in requests:
        drafter = request['comparison_drafter']
        if (request.get('selected_drafter') != drafter or len(request.get('prompt_tokens', [])) != 4096
                or request.get('max_new_tokens') != 256
                or any(request.get(key) != first.get(key) for key in identity)
                or any(request.get(key) is not True for key in ('exact', 'state_exact', 'inactive_exact', 'commit_only_gdn'))
                or request.get('sampler_num_links') != 4):
            raise ValueError('Same workload, target outputs, state, sampling and precision policy required')
        signature = [tuple(block[key] if key != 'input_tokens' else tuple(block[key]) for key in
            ('rows', 'source', 'accepted', 'position', 'input_tokens', 'committed')) for block in request['blocks']]
        if drafter in signatures and signature != signatures[drafter]:
            raise ValueError('Each drafter must reproduce its own audited proposals and acceptance')
        signatures[drafter] = signature
        validate_fusion_policy(request, 'register')
        norm, windows, down = (request.get(key, {}) for key in ('norm_reader', 'gdn_direct_window', 'mlp_down_grid'))
        shared = request.get('gdn_shared_qk', {})
        loads = shared.get('loads', [])
        if (shared.get('restored') is not True or shared.get('released') is not True
                or shared.get('admission', {}).get('report_sha256') != NORM_SHA256
                or len(loads) < 48 or len(loads) % 48
                or norm.get('policy') != 'scatter' or norm.get('report_sha256') != NORM_SHA256
                or norm.get('restored') is not True or norm.get('builds') != len(loads)
                or windows.get('direct') is not True or windows.get('restored') is not True
                or windows.get('report_sha256') != WINDOW_SHA256 or windows.get('hits') != len(loads)
                or down.get('wider_down') is not True or down.get('restored') is not True
                or down.get('report_sha256') != DOWN_SHA256
                or down.get('hits') != request['fused_t16_mlp']['hits']):
            raise ValueError('Both drafters must execute the same promoted target components')
        if any(type(request.get(key)) not in (int, float) or not math.isfinite(request[key]) or request[key] <= 0
               for key in ('decode_ms', 'prefill_ms')):
            raise ValueError('Finite positive complete request timings required')
    dspark = summarize_dspark([requests[index] for index in (0, 2, 5)])
    dflash = summarize_dflash_requests([requests[index] for index in (1, 3, 4)])
    return dict(dspark=dspark, dflash2=dflash,
        committed_tg_change_percent=100 * (dflash['committed_tokens_per_second'] / dspark['committed_tg'] - 1),
        target_rows=16, target_context=4096, streams=1,
        draft_history=dict(dspark='full history', dflash2='2048-token window'),
        performance_promoted=False, serving_qualified=False, held_out_coding_quality=False)


if __name__ == '__main__':
    import hashlib
    import json
    from pathlib import Path
    import sys

    path = Path(sys.argv[1])
    report = json.loads(path.read_text())
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or not report.get('drafter_comparison_sources')
            or report['drafter_comparison_sources'] != report.get('drafter_comparison_sources_after')):
        raise ValueError('Complete clean hardware report with unchanged sources required')
    result = summarize(report['request_checks'])
    if result != report.get('drafter_comparison'):
        raise ValueError('Independent combined drafter summary differs from recorded result')
    result['report_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps(result, indent=2))
