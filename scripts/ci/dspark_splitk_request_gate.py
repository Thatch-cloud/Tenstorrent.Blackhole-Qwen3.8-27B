"""Pin the completed combined audit before constructing a clean timing route."""

import hashlib
import json
from pathlib import Path


SCREEN_RUN = 34922265472
SCREEN_SHA256 = 'e9c750907fcd0218656ef510e4339eb264e33f4ba3904b172e996f195b32b4d9'


def qualify(directory, report_path):
    directory = Path(directory)
    payload = Path(report_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != SCREEN_SHA256:
        raise ValueError('Exact completed split-K combined audit required')
    report = json.loads(payload)
    if any(report.get(name) is not True for name in ('passed', 'closed_cleanly', 'correctness_screen_passed')):
        raise ValueError('Completed clean combined audit required')
    if (report.get('ctx_tokens') != 65536 or report.get('streams') != 1
            or report.get('request_output_limit') != 17
            or report['sources'] != report['sources_after']
            or report['native_sources'] != report['native_sources_after']):
        raise ValueError('Stable full-history single-stream screen required')
    combined = report['splitk_combined']
    if len(combined['scopes']) != 1:
        raise ValueError('One combined runtime scope required')
    scope = combined['scopes'][0]
    if scope.get('attention_calls') != 25 or scope.get('kernel_restored') is not True:
        raise ValueError('Executed and restored split-K runtime required')
    for name, expected in combined['sources'].items():
        if Path(name).name != name or hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Audited split-K integration changed: ' + name)
    if len(report['request_checks']) != 1:
        raise ValueError('One complete correctness screen required')
    request = report['request_checks'][0]
    if (any(request.get(name) is not True for name in
            ('exact', 'state_exact', 'inactive_exact', 'target_attention_t16', 'commit_only_gdn'))
            or request.get('committed_decode_tokens') != 16 or len(request.get('emitted', [])) != 17
            or not request.get('gdn_verify_checks')
            or any(check.get('unchanged') is not True for check in request['gdn_verify_checks'])):
        raise ValueError('Exact tokens, state and protected verification required')
    return dict(audit_run=SCREEN_RUN, report_sha256=SCREEN_SHA256,
        request=request, combined_scope=scope, combined_sources=combined['sources'],
        correctness_screen_qualified=True, full_request_qualified=False,
        performance_qualified=False, serving_qualified=False)
