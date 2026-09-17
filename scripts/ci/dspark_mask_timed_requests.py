"""Full-budget mask fast-path measurements after its own numerical and request audits."""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import dspark_sfpu_timed_requests as baseline
from dspark_mask_request_gate import qualify as qualify_numerical


SCREEN_RUN = 34825996484
SCREEN_SHA256 = '428a95d59098e5eecd2c5e366104502bc8dc385fcf77d13e8bea13f9b8cd527d'
BASE_QUALIFY = baseline.qualify


def qualify(directory, report_directory=None):
    reports = Path(directory if report_directory is None else report_directory)
    qualify_numerical(directory, reports / 'dspark-mask-bits-hardware.json')
    with patch.object(baseline, 'SCREEN_RUN', SCREEN_RUN), patch.object(baseline, 'SCREEN_SHA256', SCREEN_SHA256):
        return BASE_QUALIFY(directory, reports)


@contextmanager
def timed_scope(directory):
    with patch.object(baseline, 'SCREEN_RUN', SCREEN_RUN), \
            patch.object(baseline, 'SCREEN_SHA256', SCREEN_SHA256), \
            patch.object(baseline, 'qualify', qualify), baseline.timed_scope(directory):
        yield
