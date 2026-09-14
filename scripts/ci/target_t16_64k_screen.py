"""Bounded folded-T16 combined-request audit; not a throughput benchmark."""

from contextlib import contextmanager
import inspect
import hashlib
from pathlib import Path
from unittest.mock import patch

from dspark_score_sfpu_request_gate import qualify


def summarize_screen(requests):
    if len(requests) != 1:
        raise ValueError('One fully audited bounded request required')
    value = requests[0]
    if (value.get('arm') != 'scatter' or value.get('instrumented_timing') is not True
            or any(value.get(name) is not True for name in ('exact', 'state_exact', 'inactive_exact'))
            or value.get('length') != 65536 or len(value.get('prompt_tokens', [])) != 65536
            or not 2 <= len(value.get('emitted', [])) <= 17
            or value.get('committed_decode_tokens') != len(value['emitted']) - 1):
        raise ValueError('Exact 64K bounded request outputs and target state required')
    if (any(value.get(name) is not True for name in ('target_attention_t16', 'attention_replay', 'family_routing'))
            or value.get('capture_count') != 5):
        raise ValueError('All five verifier buckets including folded T16 must execute')
    blocks = value['blocks']
    if not any(block['rows'] == 16 for block in blocks):
        raise ValueError('At least one complete T16 block must execute')
    wide = [block for block in blocks if block['rows'] > 1]
    if not wide:
        raise ValueError('At least one speculative verification block required')
    draft = value.get('dspark', {})
    if (draft.get('native_attention') is not True or draft.get('proposal_trace') is not True
            or value.get('commit_only_gdn') is not True):
        raise ValueError('Captured native draft and commit-only target required')
    checks = draft.get('proposal_checks', [])
    if ([check.get('position') for check in checks] != [value['length'], *(block['position'] for block in wide)]
            or any(check.get('exact') is not True or check.get('tensors') != 6 for check in checks)):
        raise ValueError('Every draft block requires exact eager versus trace audit')
    if value.get('gdn_verify_checks') != [dict(position=block['position'], rows=block['rows'], unchanged=True)
            for block in wide]:
        raise ValueError('Every verifier must preserve GDN before commit')
    publication = value.get('captured_publication', {})
    checks = publication.get('checks', [])
    if (publication.get('enabled') is not True or len(checks) != len(blocks) + 1
            or any(check.get('exact') is not True or check.get('tensors') != 20 for check in checks)):
        raise ValueError('Complete publication state audits required')
    norm = value.get('norm_scatter_kernel', {})
    if norm.get('restored') is not True or not norm.get('loads'):
        raise ValueError('Executed and restored norm scatter required')
    return dict(arms=dict(scatter=dict(pp=None, ctx=65536, committed_tg=None)),
        correctness_screen_passed=True, full_request_qualified=False,
        performance_qualified=False, serving_qualified=False, output_limit=17)


@contextmanager
def screen_scope(directory):
    import dspark_64k_variants
    import dspark_request_experiment
    import full_dspark_request

    admission = qualify(directory, Path(directory) / 'dspark-score-sfpu-hardware.json')
    from target_t16_64k_request import qualify as qualify_target
    target_admission = qualify_target(directory)
    dependencies = ('target_t16_64k_request.py', 'target_t16_64k_screen.py', 'dspark_64k_entry.py')

    def fingerprints():
        return {name: hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest()
            for name in dependencies}

    sources = fingerprints()
    original = dspark_request_experiment.run_loaded_requests
    original_measure = full_dspark_request.measure_dspark_request
    signature = inspect.signature(original)

    def run(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        if (bound.arguments.get('max_new_tokens') != 256
                or bound.arguments.get('norm_scatter_variants') is not True
                or bound.arguments.get('captured_publication') is not True):
            raise ValueError('Isolated 64K captured scatter request required')
        result = original(*bound.args, **bound.kwargs)
        if fingerprints() != sources:
            raise ValueError('Folded-verifier integration changed during the request audit')
        bound.arguments['report'].update(scope=__doc__, diagnostic_only=True,
            full_request_passed=False, performance_qualified=False,
            sfpu_numerical_admission=admission, target_t16_hardware_admission=target_admission,
            correctness_screen_passed=True,
            target_t16_integration_sources=sources,
            allocated_output_budget=256, request_output_limit=17,
            pp=None, committed_tg=None)
        return result

    def measure(*args, **kwargs):
        return measure_bounded(original_measure, *args, **kwargs)

    with patch.object(dspark_64k_variants, 'SCHEDULE', (('scatter', True),)), \
            patch.object(dspark_64k_variants, 'summarize_variants', summarize_screen), \
            patch.object(full_dspark_request, 'measure_dspark_request', measure), \
            patch.object(dspark_request_experiment, 'run_loaded_requests', run):
        yield


def measure_bounded(measure, *args, **kwargs):
    if kwargs.get('max_new_tokens') != 256 or kwargs.get('audit_features') is not True:
        raise ValueError('Reserved 256-token capacity and complete feature audits required')
    return measure(*args, **dict(kwargs, max_new_tokens=17))
