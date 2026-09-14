"""Full-response direct-staging timing pinned to the corrected combined audit."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import dspark_sfpu_timed_requests as baseline
import target_t16_64k_timed as target
from dspark_direct_fp32_request_gate import qualify as qualify_numerical
from dspark_direct_fp32_screen import summarize_screen


SCREEN_RUN = 34835869483
SCREEN_SHA256 = 'a5b53a54dde816f1d738075ef565349a2eac7449ad423bd08b8dc477944f8ac1'


def qualify(directory, report_directory=None):
    reports = Path(directory if report_directory is None else report_directory)
    admission = qualify_numerical(directory, Path(directory) / 'dspark-direct-fp32-stage-hardware.json')
    payload = (reports / 'dspark-sfpu-request-screen.json').read_bytes()
    if hashlib.sha256(payload).hexdigest() != SCREEN_SHA256:
        raise ValueError('Corrected direct-staging combined audit required')
    report = json.loads(payload)
    comparison = report['request_comparison']
    sources = comparison['direct_fp32_sources']
    for name in ('dspark_direct_fp32_stage.py', 'dspark_direct_fp32_request_gate.py', 'dspark_direct_fp32_screen.py'):
        if sources.get(name) != hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest():
            raise ValueError('Audited direct-staging source changed: ' + name)
    expected = report['request_checks'][0]['native_attention_kernel']['patched']
    if summarize_screen(report['request_checks'], expected=expected, admission=admission, sources=sources) != comparison:
        raise ValueError('Direct-staging audit summary mismatch')
    with patch.object(target, 'SCREEN_RUN', SCREEN_RUN), patch.object(target, 'SCREEN_SHA256', SCREEN_SHA256):
        return target.qualify(directory, reports)


def summarize_timed(requests, audited):
    for value in requests:
        if value.get('native_attention_kernel') != audited.get('native_attention_kernel'):
            raise ValueError('Timed request must use the exact audited native kernel')
    with patch.object(target, 'SCREEN_RUN', SCREEN_RUN):
        result = target.summarize_timed(requests, audited)
    result['direct_fp32_stage'] = True
    return result


@contextmanager
def timed_scope(directory):
    with patch.object(baseline, 'SCREEN_RUN', SCREEN_RUN), \
            patch.object(baseline, 'SCREEN_SHA256', SCREEN_SHA256), \
            patch.object(baseline, 'qualify', qualify), \
            patch.object(baseline, 'summarize_timed', summarize_timed), baseline.timed_scope(directory):
        yield
