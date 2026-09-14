"""Repeated full-budget SFPU requests after bounded per-block correctness audits."""

from contextlib import contextmanager
import hashlib
import inspect
import json
import math
from pathlib import Path
from unittest.mock import patch

from dspark_request_variants import proposal_signature
from dspark_sfpu_request_screen import summarize_screen
from dspark_score_sfpu_request_gate import qualify as qualify_numerical


SCREEN_RUN = 34819480314
SCREEN_SHA256 = '800f211f5258a4bccd1dbacd4641f3f67d025709e68c951d667f475f1878ecee'


def qualify(directory, report_directory=None):
    directory = Path(directory)
    reports = directory if report_directory is None else Path(report_directory)
    qualify_numerical(directory, reports / 'dspark-score-sfpu-hardware.json')
    payload = (reports / 'dspark-sfpu-request-screen.json').read_bytes()
    if hashlib.sha256(payload).hexdigest() != SCREEN_SHA256:
        raise ValueError('Exact completed combined-request audit required')
    report = json.loads(payload)
    if any(report.get(name) is not True for name in ('passed', 'closed_cleanly', 'correctness_screen_passed')):
        raise ValueError('Completed clean request audit required')
    if report['sources'] != report['sources_after'] or report['native_sources'] != report['native_sources_after']:
        raise ValueError('Unchanged audited request sources required')
    summarize_screen(report['request_checks'])
    for name in ('full_dspark_request.py', 'dspark_request_experiment.py', 'dspark_prepared_proposal.py',
            'dspark_64k_scope.py', 'dspark_score_sfpu.py', 'dspark_score_sfpu_hardware.py',
            'dspark_sfpu_request_screen.py', 'dspark_request_runtime.py', 'dspark_64k_variants.py'):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != report['sources'].get(name):
            raise ValueError('Audited request implementation changed: ' + name)
    return report['request_checks'][0]


def summarize_timed(requests, audited):
    if len(requests) != 2:
        raise ValueError('Two full-budget timed requests required')
    first = requests[0]
    for value in requests:
        emitted = value.get('emitted', [])
        if (value.get('arm') != 'scatter' or value.get('instrumented_timing') is not False
                or any(value.get(name) is not True for name in ('exact', 'state_exact', 'inactive_exact'))
                or value.get('length') != 65536 or value.get('prompt_tokens') != audited['prompt_tokens']
                or not 16 < len(emitted) <= 256 or emitted[-1] not in value.get('eos_ids', [])
                or emitted[:len(audited['emitted'])] != audited['emitted']
                or value.get('committed_decode_tokens') != len(emitted) - 1):
            raise ValueError('Complete EOS request, audited prefix and exact output/state required')
        if emitted != first['emitted'] or proposal_signature(value) != proposal_signature(first):
            raise ValueError('Repeated requests must reproduce outputs and proposals')
        for field in ('prefill_ms', 'decode_ms'):
            number = value.get(field)
            if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
                raise ValueError('Positive finite full-request timing required')
        draft = value.get('dspark', {})
        norm = value.get('norm_scatter_kernel', {})
        if (draft.get('native_attention') is not True or draft.get('proposal_trace') is not True
                or value.get('commit_only_gdn') is not True
                or value.get('captured_publication', {}).get('enabled') is not True
                or norm.get('restored') is not True or not norm.get('loads')):
            raise ValueError('Same captured native draft/publication/scatter runtime required')
    tokens = sum(value['committed_decode_tokens'] for value in requests)
    summary = dict(pp=1000 * 65536 * 2 / sum(value['prefill_ms'] for value in requests),
        ctx=65536, committed_tg=1000 * tokens / sum(value['decode_ms'] for value in requests),
        committed_tokens=tokens, measured_requests=2, streams=1, output_budget=256,
        timing_boundary='Complete decode loops, including drafting, verification, commit and readback',
        held_out_coding_quality=False, serving_qualified=False)
    return dict(arms=dict(scatter=summary), full_request_passed=True,
        performance_qualified=False, audit_run=SCREEN_RUN,
        qualification_scope='Two repeatable EOS requests; not sustained throughput or held-out coding acceptance')


@contextmanager
def timed_scope(directory):
    import dspark_64k_variants
    import dspark_request_experiment

    audited = qualify(directory)
    original = dspark_request_experiment.run_loaded_requests
    signature = inspect.signature(original)

    def run(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        if bound.arguments.get('max_new_tokens') != 256:
            raise ValueError('Unshortened 256-token output budget required')
        result = original(*args, **kwargs)
        bound.arguments['report'].update(scope=__doc__, full_request_passed=True,
            performance_qualified=False, diagnostic_only=False, sfpu_audit_run=SCREEN_RUN,
            held_out_coding_quality=False, serving_qualified=False)
        return result

    with patch.object(dspark_64k_variants, 'SCHEDULE', (('scatter', False), ('scatter', False))), \
            patch.object(dspark_64k_variants, 'summarize_variants', lambda requests: summarize_timed(requests, audited)), \
            patch.object(dspark_request_experiment, 'run_loaded_requests', run):
        yield
