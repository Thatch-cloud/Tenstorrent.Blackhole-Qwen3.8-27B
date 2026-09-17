"""Reconcile frozen control replay evidence; never infer broad quality acceptance."""

import argparse
import hashlib
import json
import math
from pathlib import Path


ORIGINALS = {
    4096: ('4b3f90b001c8c91011666194e04f74b1157c70a6',
        '73151825723aee8fa1fadb2e64754f797dff151adc0c7439c460b1a30129eacb'),
    8192: ('8c102b20df22329106955b4006bf4d650bb94e40',
        'c4ab877ca9db65a5491b04afa5456ceedff5f4451154f96d16d39ebd4a0b804c'),
}


def metrics(report):
    results = {}
    requests = report['request_checks']
    if len(requests) != 6:
        raise ValueError('Six original combined requests required')
    for arm in ('control', 'publication'):
        selected = [request for request in requests if request['arm'] == arm]
        timed = [request for request in selected if request['instrumented_timing'] is False]
        audited = [request for request in selected if request['instrumented_timing'] is True]
        if len(timed) != 2 or len(audited) != 1:
            raise ValueError('One audited and two timed requests per arm required')
        for request in selected:
            if any(request.get(key) is not True for key in ('exact', 'state_exact', 'inactive_exact')):
                raise ValueError('Exact output, active and inactive state required')
            if (len(request['prompt_tokens']) != report['ctx_tokens']
                    or request['max_new_tokens'] != 256
                    or sum(block['committed'] for block in request['blocks']) != request['committed_decode_tokens']):
                raise ValueError('Complete original prompt, output budget and committed count required')
        for request in timed:
            if any(not math.isfinite(request[key]) or request[key] <= 0 for key in ('decode_ms', 'prefill_ms')):
                raise ValueError('Finite positive whole-request timing required')
        results[arm] = dict(
            pp=sum(len(request['prompt_tokens']) for request in timed) * 1000
                / sum(request['prefill_ms'] for request in timed),
            ctx=report['ctx_tokens'],
            committed_tg=sum(request['committed_decode_tokens'] for request in timed) * 1000
                / sum(request['decode_ms'] for request in timed),
            committed_tokens=sum(request['committed_decode_tokens'] for request in timed))
    if not math.isclose(results['publication']['committed_tg'], report['committed_tg'], rel_tol=1e-9):
        raise ValueError('Reported TG differs from complete-loop timing')
    return results


def compare(original, replay):
    context = original['ctx_tokens']
    revision = ORIGINALS[context][0]
    identity = ('ctx_tokens', 'drafter_history_rows', 'proposal_rows', 'streams', 'source_revision',
        'target_index_sha256', 'target_config_sha256', 'parameter_sha256', 'target_cache_formats',
        'sampler_links', 'sources', 'native_sources', 'comparison_axis')
    if original['source_revision'] != revision or any(original[key] != replay[key] for key in identity):
        raise ValueError('Replay differs from the frozen complete-runtime identity')
    for report in (original, replay):
        if (any(report.get(key) is not True for key in
                ('passed', 'closed_cleanly', 'checkpoint_closed')) or report.get('stage') != 'complete'
                or report['sources'] != report['sources_after']
                or report['native_sources'] != report['native_sources_after']
                or len(report['device_parameter_checks']) != 120
                or any(check.get('exact') is not True for check in report['device_parameter_checks'])):
            raise ValueError('Complete source-stable exact-weight runtime required')
    before, after = metrics(original), metrics(replay)
    fields = ('arm', 'instrumented_timing', 'prompt_sha256', 'output_sha256', 'emitted',
        'committed_decode_tokens', 'proposed', 'accepted', 'native_attention_kernel',
        'commit_only_gdn', 'target_attention_t16', 'norm_batch', 'capture_count')
    for expected, actual in zip(original['request_checks'], replay['request_checks'], strict=True):
        if any(expected[key] != actual[key] for key in fields):
            raise ValueError('Request identity, acceptance or enabled runtime path changed')
        block_fields = ('rows', 'source', 'position', 'input_tokens', 'accepted', 'committed')
        if len(expected['blocks']) != len(actual['blocks']) or any(
                any(first[key] != second[key] for key in block_fields)
                for first, second in zip(expected['blocks'], actual['blocks'], strict=True)):
            raise ValueError('Proposal sequence or commit decisions changed')
    return dict(runtime_identity_and_outputs_match=True, original=before, replay=after,
        candidate_tg_change_percent=(after['publication']['committed_tg']
            / before['publication']['committed_tg'] - 1) * 100,
        held_out_coding_quality=False, serving_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original', type=Path)
    parser.add_argument('replay', type=Path)
    args = parser.parse_args()
    original = json.loads(args.original.read_bytes())
    if hashlib.sha256(args.original.read_bytes()).hexdigest() != ORIGINALS[original['ctx_tokens']][1]:
        raise ValueError('Exact original winning report required')
    result = compare(original, json.loads(args.replay.read_bytes()))
    result['replay_sha256'] = hashlib.sha256(args.replay.read_bytes()).hexdigest()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
