"""Full-response direct-staging timing pinned to the corrected combined audit."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import dspark_sfpu_timed_requests as baseline
import target_t16_64k_timed as target
from dspark_normalization_request_gate import qualify as qualify_numerical
from dspark_normalization_screen import summarize_screen


SCREEN_RUN = 34839119120
SCREEN_SHA256 = '04aa9e0f5298c23141fa11f2a49ea8592a5ea133015a8d52cdee519f62aad7e6'


def qualify(directory, report_directory=None):
    reports = Path(directory if report_directory is None else report_directory)
    admission = qualify_numerical(directory, Path(directory) / 'dspark-normalization-direct-stage-hardware.json')
    payload = (reports / 'dspark-sfpu-request-screen.json').read_bytes()
    if hashlib.sha256(payload).hexdigest() != SCREEN_SHA256:
        raise ValueError('Corrected direct-staging combined audit required')
    report = json.loads(payload)
    comparison = report['request_comparison']
    sources = comparison['normalization_sources']
    for name in ('dspark_direct_fp32_stage.py', 'dspark_normalization_direct_stage.py', 'dspark_normalization_request_gate.py', 'dspark_normalization_screen.py'):
        if sources.get(name) != hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest():
            raise ValueError('Audited direct-staging source changed: ' + name)
    expected = report['request_checks'][0]['native_attention_kernel']['patched']
    if summarize_screen(report['request_checks'], expected=expected, admission=admission, sources=sources) != comparison:
        raise ValueError('Direct-staging audit summary mismatch')
    with patch.object(target, 'SCREEN_RUN', SCREEN_RUN), patch.object(target, 'SCREEN_SHA256', SCREEN_SHA256):
        return target.qualify(directory, reports)


def summarize_timed(requests, audited):
    expected = audited.get('native_attention_kernel')
    if not isinstance(expected, dict) or not all(name in expected for name in ('original', 'patched', 'signature')):
        raise ValueError('Complete audited native kernel identity required')
    for value in requests:
        actual = json.loads(json.dumps(value.get('native_attention_kernel')))
        if actual != json.loads(json.dumps(expected)):
            raise ValueError('Timed request must use the exact audited native kernel')
    with patch.object(target, 'SCREEN_RUN', SCREEN_RUN):
        result = target.summarize_timed(requests, audited)
    result['direct_fp32_stage'] = True
    result['normalization_direct_stage'] = True
    return result


@contextmanager
def timed_scope(directory):
    with patch.object(baseline, 'SCREEN_RUN', SCREEN_RUN), \
            patch.object(baseline, 'SCREEN_SHA256', SCREEN_SHA256), \
            patch.object(baseline, 'qualify', qualify), \
            patch.object(baseline, 'summarize_timed', summarize_timed), baseline.timed_scope(directory):
        yield
