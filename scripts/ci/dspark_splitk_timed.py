"""Source-matched clean EOS timing after the combined split-K correctness screen."""

from contextlib import contextmanager
import os
from pathlib import Path
from unittest.mock import patch

import dspark_center_fill_timed as center
import dspark_sfpu_timed_requests as baseline
import dspark_splitk_combined_runtime as runtime
from dspark_splitk_combined_build import require_selected
from dspark_splitk_request_gate import SCREEN_RUN, SCREEN_SHA256, qualify as qualify_splitk


def qualify(directory, report_directory=None):
    reports = Path(directory if report_directory is None else report_directory)
    splitk = qualify_splitk(directory, reports / 'dspark-sfpu-request-screen.json')
    with patch.object(center, 'SCREEN_RUN', SCREEN_RUN), patch.object(center, 'SCREEN_SHA256', SCREEN_SHA256):
        audited = center.qualify(directory, reports)
    if audited != splitk['request']:
        raise ValueError('Direct-staging and split-K gates must qualify the same request')
    return audited


def require_timed():
    require_selected()
    expected = dict(QWEN_DSPARK_SFPU_TIMED='1', QWEN_DSPARK_SFPU_REQUEST_SCREEN='0',
        QWEN_TARGET_T16_64K_REQUEST='1', QWEN_DSPARK_CENTER_TILE_FILL='1')
    if (any(os.environ.get(name) != value for name, value in expected.items())
            or os.environ.get('QWEN_SPLITK_FP32_INTERMEDIATES', '0') != '0'):
        raise ValueError('Explicit clean split-K timing configuration required')


def summarize_timed(requests, audited):
    with patch.object(center, 'SCREEN_RUN', SCREEN_RUN):
        result = center.summarize_timed(requests, audited)
    result['splitk_combined'] = True
    return result


@contextmanager
def timed_scope(directory):
    with patch.object(baseline, 'SCREEN_RUN', SCREEN_RUN), \
            patch.object(baseline, 'SCREEN_SHA256', SCREEN_SHA256), \
            patch.object(baseline, 'qualify', qualify), \
            patch.object(baseline, 'summarize_timed', summarize_timed), baseline.timed_scope(directory):
        yield


def validate_execution(records, audited_scope):
    if len(records) != 1 or records[0].get('attention_calls', 0) <= 0:
        raise ValueError('One actually executed split-K runtime required')
    actual = records[0]
    if actual.get('kernel_restored') is not True:
        raise ValueError('Timed kernel must restore cleanly')
    for name in ('component', 'kernel', 'combined_build_binaries'):
        if actual.get(name) != audited_scope.get(name) or name not in actual:
            raise ValueError('Timed runtime differs from combined audit: ' + name)


@contextmanager
def preflight_scope():
    require_timed()
    with patch.object(center, 'timed_scope', timed_scope):
        yield


@contextmanager
def entry_scope(directory, records):
    require_timed()
    admission = qualify_splitk(directory, Path(directory) / 'dspark-sfpu-request-screen.json')
    with patch.object(runtime, 'require_screen', require_timed), \
            patch.object(center, 'timed_scope', timed_scope), runtime.entry_scope(records):
        yield admission
    validate_execution(records, admission['combined_scope'])
