"""Full-budget sum-update measurements after its own numerical and request audits."""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import dspark_sfpu_timed_requests as baseline
from dspark_sum_request_gate import qualify as qualify_numerical


SCREEN_RUN = 34823325684
SCREEN_SHA256 = '935130e1030190b50eaf903090f0610797a22062ce400c8899ab918492a8e569'
BASE_QUALIFY = baseline.qualify


def qualify(directory, report_directory=None):
    reports = Path(directory if report_directory is None else report_directory)
    qualify_numerical(directory, reports / 'dspark-sum-sfpu-hardware.json')
    with patch.object(baseline, 'SCREEN_RUN', SCREEN_RUN), patch.object(baseline, 'SCREEN_SHA256', SCREEN_SHA256):
        return BASE_QUALIFY(directory, reports)


@contextmanager
def timed_scope(directory):
    with patch.object(baseline, 'SCREEN_RUN', SCREEN_RUN), \
            patch.object(baseline, 'SCREEN_SHA256', SCREEN_SHA256), \
            patch.object(baseline, 'qualify', qualify), baseline.timed_scope(directory):
        yield
