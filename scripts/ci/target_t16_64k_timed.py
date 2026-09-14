"""Full-response folded T16 measurements after exact combined-request qualification."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import dspark_sfpu_timed_requests as baseline
from dspark_mask_request_gate import qualify as qualify_mask
from target_t16_64k_request import qualify as qualify_target
from target_t16_64k_screen import summarize_screen


SCREEN_RUN = 34831200764
SCREEN_SHA256 = '9731be1bc383c713622a48e9a8abc196ffef111eed88d903f751bf13c5c8b904'
BASE_QUALIFY = baseline.qualify
BASE_SUMMARIZE = baseline.summarize_timed


def qualify(directory, report_directory=None):
    reports = Path(directory if report_directory is None else report_directory)
    qualify_mask(directory, reports / 'dspark-mask-bits-hardware.json')
    qualify_target(directory)
    with patch.object(baseline, 'SCREEN_RUN', SCREEN_RUN), \
            patch.object(baseline, 'SCREEN_SHA256', SCREEN_SHA256), \
            patch.object(baseline, 'summarize_screen', summarize_screen):
        audited = BASE_QUALIFY(directory, reports)
    report = json.loads((reports / 'dspark-sfpu-request-screen.json').read_bytes())
    for name in ('target_t16_64k_request.py', 'target_t16_64k_screen.py'):
        checksum = hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest()
        if report.get('target_t16_integration_sources', {}).get(name) != checksum:
            raise ValueError('Audited folded-verifier implementation changed: ' + name)
    return audited


def summarize_timed(requests, audited):
    for value in requests:
        if (any(value.get(name) is not True for name in ('target_attention_t16', 'attention_replay', 'family_routing'))
                or value.get('max_new_tokens') != 256
                or value.get('capture_count') != 5
                or not any(block.get('rows') == 16 for block in value.get('blocks', []))):
            raise ValueError('Timed requests must execute the audited folded T16 route')
    with patch.object(baseline, 'SCREEN_RUN', SCREEN_RUN):
        result = BASE_SUMMARIZE(requests, audited)
    result['target_attention_t16'] = True
    return result


@contextmanager
def timed_scope(directory):
    with patch.object(baseline, 'SCREEN_RUN', SCREEN_RUN), \
            patch.object(baseline, 'SCREEN_SHA256', SCREEN_SHA256), \
            patch.object(baseline, 'qualify', qualify), \
            patch.object(baseline, 'summarize_timed', summarize_timed), baseline.timed_scope(directory):
        yield
